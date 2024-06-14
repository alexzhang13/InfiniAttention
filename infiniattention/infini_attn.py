import torch
import torch.nn as nn
import torch.nn.functional as F


class InfiniAttn(nn.Module):
    def __init__(
        self,
        n_head,
        d_model,
        d_head,
        dropout,
        seq_len=64,
        dropatt=0,
        pre_lnorm=False,
        beta_eps=1e-2,
    ):
        super(InfiniAttn, self).__init__()

        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.dropout = dropout
        self.seq_len = seq_len

        # edit here
        self.content_A = (
            None  # torch.zeros((n_head, seq_len, d_head), requires_grad=False)
        )
        self.memory = None  # torch.zeros((n_head, d_head, d_head), requires_grad=False)
        self.memory_normalization = (
            None  # torch.zeros((n_head, d_head), requires_grad=False)
        )
        self.mem_activation = nn.ELU()

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
                Q: Query matrix XW_Q. Expected shape is B x S x n x d
                K: Query matrix XW_K. Expected shape is B x S x n x d
                V: Query matrix XW_V. Expected shape is B x S x n x d
        """
        with torch.no_grad():
            B = Q.size(0)
            cached_Q = self.mem_activation(Q)
            cached_K = self.mem_activation(K)

            if self.memory is None:
                self.content_A = torch.zeros(
                    (B, self.seq_len, self.n_head, self.d_head), requires_grad=False
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
            cached_K = self.mem_activation(K)

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
        return (
            F.sigmoid(self.beta)
            * self.content_A.contiguous().view(
                self.content_A.size(0),
                self.content_A.size(1),
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
        num_tokens = h.size(1)
        chunks = num_tokens // self.seq_len
        seqs = torch.chunk(h, chunks=(chunks), dim=1)

        out = []
        for seq in seqs:
            out = self._forward(seq, attn_mask)

        return out


if __name__ == "__main__":
    d_model = 512
    seq_length = 2048
    attn = InfiniAttn(
        n_head=5, d_model=d_model, d_head=256, seq_len=seq_length, dropout=0.5
    )

    h = torch.rand((2, 10 * seq_length, d_model))
    print(attn(h))
