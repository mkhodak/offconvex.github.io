---
layout:     post
title:      Unsupervised learning for representing the meaning of text
date:       2018-06-10 8:00:00
author:     Sanjeev Arora, Mikhail Khodak, Nikunj Saunshi
visible:    false
---

##Why should low-dimensional distributed representations do well?

To understand the good performance of distributed representations, we begin by taking a closer look at the sparse counterparts.
The bag-of-words (BoW) representation of a document $x_D^{BoW}$ can be written as
\[
x_D^{BoW} = \sum\limits_{w\in D} e_w$
\]
where $e_w$ is the $V$ dimensional one-hot embedding for the word $w$, i.e. a vector with 1 in the coordinate corresponding to $w$ and 0 at all other coordinates.
So the BoW representation of a document is the sum of certain words vectors of the words in the document, where the word vectors are all orthogonal ($e_w^T e_{w'} = 0).
The orthogonality of these word vectors lets us exactly recover the words of a document given the BoW representation.
Here is a crucial observation -- since text documents typically have very few distinct words (much lesser than $V$), one could hope to use lower dimensionsal word vectors which are "almost orthogonal" and still be able to uniquely recover the words of a document.
So if we had word vectors $v_w \in \mathbb{R}^d$ which satisfied this almost orthogonality property, then the representation $x_D = \sum\limits_{w\in D} v_w$ would encode precisely the same information as $x_D^{BoW}$.
Note that $x_D = Ax_D^{BoW}$, where $A$ is a $d\times V$ matrix whose columns correspond to the vectors $v_w$s, i.e. $x_D$ is a linear compression of $x_D^{BoW}$.
This is where compressed sensing comes into the picture - compressed sensing deals with finding conditions on the matrix A such that given $Ax$, one can uniquely recover a sparse high-dimensional vector $x$ (cite someone here).
Note, that being able to recover the BoW information doesn't directly imply having the same performance as BoW on linear classification (in fact, this is not true always).
However by building upon the work of Calderbank et al., we prove that in the case of random word embeddings, the compressed sensing condition which implies optimal recovery also implies good performance on linear classification.
Furthermore, extending these ideas to $n$-grams, we also show that our DisC embeddings described in the previous post (link here) can do as well as Bag-of-$n$-Gram (BonG) representations.

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
