## Introduction

***This is not the official code repository for [InfiniAttention](https://arxiv.org/pdf/2404.07143)***. 

This is just my own simple implementation based on the paper, which at the time of writing,
has not released the code. There are some details that I have yet to figure out nicely, which I will detail below.

InfiniAttention is a new technique for increasing context windows in LLMs "infinitely". The basic idea is to propagate attention matrices spatially (like storing a hidden vector), but instead of
learning the hidden vector, it is just a nice representation of the sum of the previous attention matrices. The concept is super simple, but a few details are kind of weird. 
The memory complexity is therefore only with respect to a local context window, makes increasing the context window an o(1) operation.

The rest of this repository is basically a stripped and simpler version of the [Transformer-XL](https://github.com/kimiyoung/transformer-xl/) repository with my own implementation of
[InfiniAttention](https://arxiv.org/pdf/2404.07143). The reason I chose this repo is because both papers treat sequences 

I've also removed the Tensorflow implementation, but may re-visit it in the future.

The main thing to look at is [`infiniattention/infiniattn_transformer.py`](https://github.com/alexzhang13/InfiniAttention/blob/main/infiniattention/infiniattn_transformer.py), although [`infiniattention/infini_attn.py`](https://github.com/alexzhang13/InfiniAttention/blob/main/infiniattention/infini_attn.py) contains just the InfiniAttention layer if you need that. All the other
files are modified from the train loop from [Transformer-XL](https://github.com/kimiyoung/transformer-xl/).

## Prerequisite

- Pytorch: `conda install pytorch torchvision -c pytorch`

Yep, that's it. Super barebones.

## Current TODOs and Unclear Items
1. Like Transformer-XL, the way they handle inference is weird. They assume you can chunk up a sequence nicely, but when you generate one-token at a time, it's not that clear.
2. The naive MHA mechanism doesn't implement KV-caching, FlashAttention, or any kind of efficient speed-up mechanisms. Although I don't really think I'll be doing this.
3. Their choice of positional encodings is not mentioned. I have the base Transformer ones right now, but I'm assuming relative positional encodings probably work better.
4. How do they handle the very last chunk if it doesn't fit the full context window size? Do they pad? (what I currently do is pad and use the decoder mask to ignore).
5. I initially assumed the data would come in the form (Batch size, Sequence length, Embedding dim) but the first two dimensions are swapped. I have a bunch of transpose logic to get around this, but I can probably clean this up later.

Honestly, if you need this implementation, I'd just grab the InfiniAttention module and throw it into your own code. The above issues are just if I want to build out a full model.

## Data Prepration and Experiments [WIP]
This is taken from the Transformer-XL repository. I haven't actually been able to get the time or compute to train a full model. I can only confirm that the mechanism is correct
according to my understanding of the paper. But I'll revisit this.

`bash getdata.sh`

Note that some of the datasets don't even work, but for my own toy experiments I was playing around with the enwiki8 dataset.
