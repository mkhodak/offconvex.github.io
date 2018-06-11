---
layout:     post
title:      Understanding text embeddings using compressed sensing
date:       2018-07-01 8:00:00
author:     Sanjeev Arora, Mikhail Khodak, Nikunj Saunshi
visible:    false
---

In a recent [post](LINK), Sanjeev discussed some ideas behind unsupervised text embeddings, whose goal is to use a large text corpus to learn representations of documents that can be used to perform well on downstream tasks using only a few labeled examples.


##Understanding what our embeddings encode using compressed sensing

Re-expression of DisC as matrix transform of BonG.
Compressed sensing implies recovery of BonG when A is RIP.
Relevant to classification (compressed learning). State theorem.

A is RIP for Rademacher vectors (BOS, see paper for full proof).
State main theorem (for DisC, not LSTM).
Mention connection to LSTM.

<div style="text-align:center;">
<img src="/assets/imdbperf_uni_bi.svg" style="width:400px;" />
</div>

But what about pretrained word embeddings?
