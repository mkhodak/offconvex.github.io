---
layout:     post
title:      Unsupervised learning for representing the meaning of text
date:       2015-12-01 8:00:00
author:     Sanjeev Arora, Mikhail Khodak, Nikunj Saunshi
visible:    false
---

Much of the recent success of deep learning in NLP has come through the use of *distributed text representations* - embeddings trained to capture the "meaning" of text with moderate dimensionality. 
In the unsupervised setting, models are trained on large text corpora to produce these vectors given any input string; 
these are then passed to downstream NLP systems solving standard tasks such as document classification. 
This pipeline is very different from an end-to-end differentiable system such as questiona-answering, which is trained for a single task.
Since no labels are provided, these methods need to encode text in such a way that its "meaning" can be extracted efficiently for use in supervised tasks with limited labeled data.

<div style="text-align:center;">
<img src="/assets/unsupervised_pipeline.svg" style="width:400px;" />
</div>

The classic way to do representation learning is to predict what's coming next using an RNN (usually an LSTM variant).
This idea underlies [skip-thoughts](https://arxiv.org/abs/1506.06726) and [subsequent](https://arxiv.org/abs/1506.01057) [papers](https://arxiv.org/abs/1502.06922).
Of course, this objective fits squarely within the framework of other distributed representations; 
word embeddings, for example, have a similar design philosophy requiring co-occurring words to have vectors with high inner product.
Sanjeev has discussed their resulting properties in two [previous](http://www.offconvex.org/2016/02/14/word-embeddings-2/) [posts](http://www.offconvex.org/2016/07/10/embeddingspolysemy/).

This post is based on [our joint work with Kiran Vodrahalli](https://openreview.net/forum?id=B1e5ef-C-&noteId=B1e5ef-C-) about elementary, interpretable methods for defining and computing text representations that has both provable guarantees and is competitive with state-of-the-art deep learning methods. 
A perhaps surprising member in this story is compressed sensing (also called sparse recovery).
