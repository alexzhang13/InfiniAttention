import sys
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append("utils")
from utils.proj_adaptive_softmax import ProjectedAdaptiveLogSoftmax
from utils.log_uniform_sampler import LogUniformSampler, sample_logits


def chunk_evenly(a: np.ndarray, chunk_size: int):
    """
    TODO: Make this more efficient... Don't know how to do this in torch.
    """
    return np.split(a, np.arange(chunk_size, len(a), chunk_size).tolist())


class PositionalEmbedding(nn.Module):
    def __init__(self, demb):
        super(PositionalEmbedding, self).__init__()

        self.demb = demb

        inv_freq = 1 / (10000 ** (torch.arange(0.0, demb, 2.0) / demb))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, pos_seq, bsz=None):
        sinusoid_inp = torch.ger(pos_seq, self.inv_freq)
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)

        if bsz is not None:
            return pos_emb[:, None, :].expand(-1, bsz, -1)
        else:
            return pos_emb[:, None, :]


class PositionwiseFF(nn.Module):
    def __init__(self, d_model, d_inner, dropout, pre_lnorm=False):
        super(PositionwiseFF, self).__init__()

        self.d_model = d_model
        self.d_inner = d_inner
        self.dropout = dropout

        self.CoreNet = nn.Sequential(
            nn.Linear(d_model, d_inner),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_inner, d_model),
            nn.Dropout(dropout),
        )

        self.layer_norm = nn.LayerNorm(d_model)

        self.pre_lnorm = pre_lnorm

    def forward(self, inp):
        if self.pre_lnorm:
            ##### layer normalization + positionwise feed-forward
            core_out = self.CoreNet(self.layer_norm(inp))

            ##### residual connection
            output = core_out + inp
        else:
            ##### positionwise feed-forward
            core_out = self.CoreNet(inp)

            ##### residual connection + layer normalization
            output = self.layer_norm(inp + core_out)

        return output


class InfiniAttn(nn.Module):
    def __init__(
        self,
        n_head,
        d_model,
        d_head,
        dropout,
        max_seq_len,
        dropatt=0,
        pre_lnorm=False,
        beta_eps=1e-2,
    ):
        super(InfiniAttn, self).__init__()

        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.dropout = dropout
        self.max_seq_len = max_seq_len

        # edit here
        self.content_A = (
            None  # torch.zeros((n_head, seq_len, d_head), requires_grad=False)
        )
        self.memory = None  # torch.zeros((n_head, d_head, d_head), requires_grad=False)
        self.memory_normalization = (
            None  # torch.zeros((n_head, d_head), requires_grad=False)
        )
        self.mem_activation = F.elu

        self.beta = torch.ones((1)) * beta_eps

        self.q_net = nn.Linear(d_model, n_head * d_head, bias=False)
        self.kv_net = nn.Linear(d_model, 2 * n_head * d_head, bias=False)

        self.drop = nn.Dropout(dropout)
        self.dropatt = nn.Dropout(dropatt)
        self.o_net = nn.Linear(n_head * d_head, d_model, bias=False)

        self.layer_norm = nn.LayerNorm(d_model)

        self.scale = 1 / (d_head**0.5)

        self.pre_lnorm = pre_lnorm

    def _update_memory(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor):
        """
        Private method for updating the InfiniteAttention memory module.
            args:
                Q: Query matrix XW_Q. Expected shape is S x B x n x d
                K: Query matrix XW_K. Expected shape is S x B x n x d
                V: Query matrix XW_V. Expected shape is S x B x n x d
        """
        with torch.no_grad():
            Q = Q.transpose(0, 1)
            K = K.transpose(0, 1)
            V = V.transpose(0, 1)

            B = Q.size(0)
            cached_Q = self.mem_activation(Q) + 1
            cached_K = self.mem_activation(K) + 1

            if self.memory is None:
                self.content_A = torch.zeros(
                    (B, max_seq_len, self.n_head, self.d_head), requires_grad=False
                )
                self.memory = torch.einsum("bsnk,bsnv->bnkv", (cached_K, V))
                self.memory_normalization = torch.sum(cached_K, dim=1, keepdim=True)
                return

            """
            Update M_s using M_{s-1}. Also called memory retrieval in paper.
            """

            # B * n_head * N * d_head
            numerator = torch.einsum("bsnk,bnkv->bsnv", (cached_Q, self.memory))
            denominator = torch.einsum(
                "bsnk,bvnk->bsnv", (cached_Q, self.memory_normalization)
            )
            # n_head * N * d_head
            self.content_A = (numerator / denominator).detach()

            """
            Memory update step, using the linear + delta trick as well.
            """

            # n_head * N * d_head
            numerator = torch.einsum("bsnk,bnkv->bsnv", (cached_K, self.memory))
            denominator = torch.einsum(
                "bsnk,bvnk->bsnv", (cached_K, self.memory_normalization)
            )
            # n_head * N * d_head
            delta = numerator / denominator

            self.memory = self.memory + torch.einsum(
                "bsnk,bsnv->bnkv", (cached_K, (V - delta))
            )

            self.memory_normalization = self.memory_normalization + torch.sum(
                cached_K, dim=1, keepdim=True
            )

    def _injection(self, A: torch.Tensor):
        """
        Mixing between local attention matrix and A_mem through learned gating scalar self.beta.
        args:
            A: The attention output softmax(Attn) V^T
        """
        # TODO: Remove all the transposes between bsz and seq length
        return (
            F.sigmoid(self.beta)
            * self.content_A.transpose(0, 1)
            .contiguous()
            .view(
                self.content_A.size(1),
                self.content_A.size(0),
                self.n_head * self.d_head,
            )
            + (1 - F.sigmoid(self.beta)) * A
        )

    def _forward(self, h, attn_mask=None, mems=None):
        ##### multihead attention
        # [hlen x bsz x n_head x d_head]

        if mems is not None:
            c = torch.cat([mems, h], 0)
        else:
            c = h

        if self.pre_lnorm:
            ##### layer normalization
            c = self.layer_norm(c)

        head_q = self.q_net(h)
        head_k, head_v = torch.chunk(self.kv_net(c), 2, -1)

        head_q = head_q.view(h.size(0), h.size(1), self.n_head, self.d_head)
        head_k = head_k.view(c.size(0), c.size(1), self.n_head, self.d_head)
        head_v = head_v.view(c.size(0), c.size(1), self.n_head, self.d_head)

        # [qlen x klen x bsz x n_head]
        attn_score = torch.einsum("ibnd,jbnd->ijbn", (head_q, head_k))
        attn_score.mul_(self.scale)
        if attn_mask is not None and attn_mask.any().item():
            if attn_mask.dim() == 2:
                attn_score.masked_fill_(attn_mask[None, :, :, None], -float("inf"))
            elif attn_mask.dim() == 3:
                attn_score.masked_fill_(attn_mask[:, :, :, None], -float("inf"))

        # [qlen x klen x bsz x n_head]
        attn_prob = F.softmax(attn_score, dim=1)
        attn_prob = self.dropatt(attn_prob)

        # [qlen x klen x bsz x n_head] + [klen x bsz x n_head x d_head] -> [qlen x bsz x n_head x d_head]
        attn_vec = torch.einsum("ijbn,jbnd->ibnd", (attn_prob, head_v))
        attn_vec = attn_vec.contiguous().view(
            attn_vec.size(0), attn_vec.size(1), self.n_head * self.d_head
        )

        self._update_memory(head_q, head_v, head_k)
        attn_vec = self._injection(attn_vec)

        ##### linear projection
        attn_out = self.o_net(attn_vec)
        attn_out = self.drop(attn_out)

        if self.pre_lnorm:
            ##### residual connection
            output = h + attn_out
        else:
            ##### residual connection + layer normalization
            output = self.layer_norm(h + attn_out)

        return output

    def forward(self, h, attn_mask=None):
        out = self._forward(h, attn_mask)
        return out

    def forward_chunked(self, h, attn_mask=None):
        """
        Unused currently. This assumes the attention layer receives the full input, and it sequentially
        chunks the data based on the max sequence length per chunk.
        """
        num_tokens = h.size(1)
        chunks = num_tokens // self.seq_len
        seqs = torch.chunk(h, chunks=(chunks), dim=1)

        out = []
        for seq in seqs:
            out = self._forward(seq, attn_mask)

        return out


class DecoderLayer(nn.Module):
    def __init__(self, n_head, d_model, d_head, d_inner, dropout, seq_len, **kwargs):
        super(DecoderLayer, self).__init__()

        self.dec_attn = InfiniAttn(n_head, d_model, d_head, dropout, seq_len, **kwargs)
        self.pos_ff = PositionwiseFF(
            d_model, d_inner, dropout, pre_lnorm=kwargs.get("pre_lnorm")
        )

    def forward(self, dec_inp, dec_attn_mask=None, mems=None):
        output = self.dec_attn(dec_inp, attn_mask=dec_attn_mask)
        output = self.pos_ff(output)

        return output


class AdaptiveEmbedding(nn.Module):
    def __init__(
        self, n_token, d_embed, d_proj, cutoffs, div_val=1, sample_softmax=False
    ):
        super(AdaptiveEmbedding, self).__init__()

        self.n_token = n_token
        self.d_embed = d_embed

        self.cutoffs = cutoffs + [n_token]
        self.div_val = div_val
        self.d_proj = d_proj

        self.emb_scale = d_proj**0.5

        self.cutoff_ends = [0] + self.cutoffs

        self.emb_layers = nn.ModuleList()
        self.emb_projs = nn.ParameterList()
        if div_val == 1:
            self.emb_layers.append(
                nn.Embedding(n_token, d_embed, sparse=sample_softmax > 0)
            )
            if d_proj != d_embed:
                self.emb_projs.append(nn.Parameter(torch.Tensor(d_proj, d_embed)))
        else:
            for i in range(len(self.cutoffs)):
                l_idx, r_idx = self.cutoff_ends[i], self.cutoff_ends[i + 1]
                d_emb_i = d_embed // (div_val**i)
                self.emb_layers.append(nn.Embedding(r_idx - l_idx, d_emb_i))
                self.emb_projs.append(nn.Parameter(torch.Tensor(d_proj, d_emb_i)))

    def forward(self, inp):
        if self.div_val == 1:
            embed = self.emb_layers[0](inp)
            if self.d_proj != self.d_embed:
                embed = F.linear(embed, self.emb_projs[0])
        else:
            param = next(self.parameters())
            inp_flat = inp.view(-1)
            emb_flat = torch.zeros(
                [inp_flat.size(0), self.d_proj], dtype=param.dtype, device=param.device
            )
            for i in range(len(self.cutoffs)):
                l_idx, r_idx = self.cutoff_ends[i], self.cutoff_ends[i + 1]

                mask_i = (inp_flat >= l_idx) & (inp_flat < r_idx)
                indices_i = mask_i.nonzero().squeeze()

                if indices_i.numel() == 0:
                    continue

                inp_i = inp_flat.index_select(0, indices_i) - l_idx
                emb_i = self.emb_layers[i](inp_i)
                emb_i = F.linear(emb_i, self.emb_projs[i])

                emb_flat.index_copy_(0, indices_i, emb_i)

            embed = emb_flat.view(*inp.size(), self.d_proj)

        embed.mul_(self.emb_scale)

        return embed


class InfiniAttentionLM(nn.Module):
    def __init__(
        self,
        n_token,
        n_layer,
        n_head,
        d_model,
        d_head,
        d_inner,
        dropout,
        dropatt,
        tie_weight=True,
        d_embed=None,
        div_val=1,
        tie_projs=[False],
        pre_lnorm=False,
        max_seq_len=2048,
        cutoffs=[],
        same_length=False,
        attn_type=0,
        clamp_len=-1,
        sample_softmax=-1,
    ):
        super(InfiniAttentionLM, self).__init__()
        self.n_token = n_token
        self.max_seq_len = max_seq_len

        d_embed = d_model if d_embed is None else d_embed
        self.d_embed = d_embed
        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_head

        self.word_emb = AdaptiveEmbedding(
            n_token, d_embed, d_model, cutoffs, div_val=div_val
        )

        self.drop = nn.Dropout(dropout)

        self.n_layer = n_layer

        self.attn_type = attn_type

        self.layers = nn.ModuleList()
        for i in range(n_layer):
            self.layers.append(
                DecoderLayer(
                    n_head,
                    d_model,
                    d_head,
                    d_inner,
                    dropout,
                    seq_len=max_seq_len,
                    dropatt=dropatt,
                    pre_lnorm=pre_lnorm,
                )
            )

        self.sample_softmax = sample_softmax
        # use sampled softmax
        if sample_softmax > 0:
            self.out_layer = nn.Linear(d_model, n_token)
            if tie_weight:
                self.out_layer.weight = self.word_emb.weight
            self.tie_weight = tie_weight
            self.sampler = LogUniformSampler(n_token, sample_softmax)

        # use adaptive softmax (including standard softmax)
        else:
            self.crit = ProjectedAdaptiveLogSoftmax(
                n_token, d_embed, d_model, cutoffs, div_val=div_val
            )

            if tie_weight:
                for i in range(len(self.crit.out_layers)):
                    self.crit.out_layers[i].weight = self.word_emb.emb_layers[i].weight

            if tie_projs:
                for i, tie_proj in enumerate(tie_projs):
                    if tie_proj and div_val == 1 and d_model != d_embed:
                        self.crit.out_projs[i] = self.word_emb.emb_projs[0]
                    elif tie_proj and div_val != 1:
                        self.crit.out_projs[i] = self.word_emb.emb_projs[i]

        self.same_length = same_length
        self.clamp_len = clamp_len

        self._create_params()

    def backward_compatible(self):
        self.sample_softmax = -1

    def _create_params(self):
        self.pos_emb = PositionalEmbedding(self.d_model)

    def _forward(self, dec_inp):
        qlen, _ = dec_inp.size()

        word_emb = self.word_emb(dec_inp)
        if qlen < self.max_seq_len:
            word_emb = torch.cat(
                [
                    word_emb,
                    torch.zeros(
                        (self.max_seq_len - qlen, word_emb.size(1), word_emb.size(2))
                    ),
                ],
                dim=0,
            )
        print(word_emb.shape)

        core_out = None
        dec_attn_mask = torch.triu(
            word_emb.new_ones(self.max_seq_len, self.max_seq_len), diagonal=1
        )

        # TODO: This is padding logic for attention masks, but doesn't currently
        # do it for inputs. Either you handle this in the data loader, or you do it here.
        dec_attn_mask[:, qlen:] = 1  # Mask out unused words
        dec_attn_mask = dec_attn_mask.byte()[:, :, None].bool()

        hids = []
        # Not sure if this is handled correctly, paper doesn't really specify
        pos_seq = torch.arange(
            self.max_seq_len - 1, -1, -1.0, device=word_emb.device, dtype=word_emb.dtype
        )
        if self.clamp_len > 0:
            pos_seq.clamp_(max=self.clamp_len)
        pos_emb = self.pos_emb(pos_seq)
        core_out = self.drop(word_emb + pos_emb[-self.max_seq_len :])

        hids.append(core_out)
        for i, layer in enumerate(self.layers):
            core_out = layer(core_out, dec_attn_mask=dec_attn_mask)
            hids.append(core_out)

        core_out = self.drop(core_out)

        return core_out

    def forward(self, data, target):
        # nn.DataParallel does not allow size(0) tensors to be broadcasted.
        # So, have to initialize size(0) mems inside the model forward.
        # Moreover, have to return new_mems to allow nn.DataParallel to piece
        # them together.

        num_tokens = data.size(0)
        chunks = math.ceil(num_tokens / self.max_seq_len)
        print(chunks)

        # This logic isn't fully correct. We want chunks of a specific size (max context window length)
        seqs = torch.chunk(data, chunks=(chunks), dim=1)
        targets = torch.chunk(target, chunks=(chunks), dim=1)

        out = []
        loss = []
        for seq, t in zip(seqs, targets):
            tgt_len = t.size(0)
            out = self._forward(seq)

            pred_hid = out[:tgt_len]
            if self.sample_softmax > 0 and self.training:
                assert self.tie_weight
                logit = sample_logits(
                    self.word_emb, self.out_layer.bias, t, pred_hid, self.sampler
                )
                loss.append(-F.log_softmax(logit, -1)[:, :, 0])
            else:
                t_loss = self.crit(pred_hid.view(-1, pred_hid.size(-1)), t.view(-1))
                t_loss = t_loss.view(tgt_len, -1)
                loss.append(t_loss)

        return [torch.stack(loss)]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="unit test")

    parser.add_argument("--n_layer", type=int, default=4, help="")
    parser.add_argument("--n_rel_layer", type=int, default=4, help="")
    parser.add_argument("--n_head", type=int, default=2, help="")
    parser.add_argument("--d_head", type=int, default=2, help="")
    parser.add_argument("--d_model", type=int, default=200, help="")
    parser.add_argument("--d_embed", type=int, default=200, help="")
    parser.add_argument("--d_inner", type=int, default=200, help="")
    parser.add_argument("--dropout", type=float, default=0.0, help="")
    parser.add_argument("--cuda", action="store_true", help="")
    parser.add_argument("--seed", type=int, default=1111, help="")
    parser.add_argument("--multi_gpu", action="store_true", help="")

    args = parser.parse_args()

    device = torch.device("cuda" if args.cuda else "cpu")

    B = 4
    max_seq_len = 256
    tgt_len, mem_len, ext_len = 36, 36, 0
    data_len = max_seq_len * 20
    args.n_token = 10000

    import data_utils

    data = torch.LongTensor(data_len * B).random_(0, args.n_token).to(device)
    diter = data_utils.LMOrderedIterator(
        data,
        B,
        max_seq_len,
        device=str(device),
    )

    cutoffs = [args.n_token // 2]
    tie_projs = [False] + [True] * len(cutoffs)

    for div_val in [1, 2]:
        for d_embed in [200, 100]:
            model = InfiniAttentionLM(
                args.n_token,
                args.n_layer,
                args.n_head,
                args.d_model,
                args.d_head,
                args.d_inner,
                args.dropout,
                dropatt=args.dropout,
                tie_weight=True,
                d_embed=d_embed,
                div_val=div_val,
                tie_projs=tie_projs,
                pre_lnorm=True,
                max_seq_len=max_seq_len,
                cutoffs=cutoffs,
                attn_type=2,
            ).to(device)

            print(sum(p.numel() for p in model.parameters()))

            for idx, (inp, tgt, seqlen) in enumerate(diter):
                print("batch {}".format(idx))
                out = model(inp, tgt)
                print(out)
