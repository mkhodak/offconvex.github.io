---
layout:     post
title:      Understanding text embeddings using compressed sensing
date:       2018-07-01 8:00:00
author:     Sanjeev Arora, Mikhail Khodak, Nikunj Saunshi
visible:    false
---

In a recent [post](LINK), Sanjeev discussed some ideas behind unsupervised text embeddings, whose goal is to use a large text corpus to learn representations of documents that can be used to perform well on downstream tasks using only a few labeled examples. 
Although deep learning approaches are popular in this area, it turns out that a simple weighted combination of word embeddings combined with some mild denoising ([the SIF embedding](https://openreview.net/pdf?id=SyK00v5xx)) outperforms many such methods, including [Skipthought](https://arxiv.org/pdf/1506.06726.pdf), on sentence semantic similarity tasks. 
In this post we will discuss our recent [ICLR'18 paper](https://openreview.net/pdf?id=B1e5ef-C-) with Kiran Vodrahalli, where we target a similar goal -- simple, compositional document embeddings -- in the context of text classification. 
Our representations achieve performance that is provably competitive with strong sparse-feature baselines for the case of random word embeddings and empirically outperforms LSTM-based methods when using pretrained (GloVe) word vectors.

## Simple text embeddings incorporating local word order

Both the original paper and subsequent evaluations (see this nice [blog post](http://nlp.town/blog/sentence-similarity/) by Yves Peirsman) show that SIF embeddings work very well on sentence semantic similarity and relatedness tasks, outperforming deep learning approaches such as LSTM embeddings and deep averaging networks.
In these evaluations pairs of sentence embeddings are assigned scores based on their inner product or a trained regression targeting human ratings.
However, SIF embeddings do not end up improving performance strongly on sentiment classification tasks; 
indeed taking out the top component hurts performance, while the weighting gives only a slight improvement.
It seems that while word-level semantic content suffices for good similarity performance, sentiment analysis depends more on order, something that SIF doesn't capture because it only uses summation.

The most basic way of including word-order in a representation is to consider $n$-grams for $n>1$, starting with bigrams.
While these alone cannot capture long-range dependencies, sparse Bag-of-$n$-Gram (BonG) representations (the $n$-gram extension of Bag-of-Words counting how many times each $n$-gram occurs in the document) are a surprisingly [strong baseline for document classification](https://www.aclweb.org/anthology/P12-2018).
However, in our setting of unsupervised learning the sparse-feature approach can still fail to capture similarity in ways that matter when only a few labeled samples are available. 
For example, the sentences "This movies was great!" and "I enjoyed the film." should mean the same thing to the ideal sentiment classifier but share no $n$-gram information of any order.
Thus having a label for the first example tells us nothing about the second.

We thus turn to simple distributed representations of $n$-grams.
Noting that representations such as SIF are just (weighted) sums of unigram embeddings, we can define our new document embeddings as summations over $n$-gram embeddings for small $n$.
However, we want the $n$-gram embeddings themselves to also be compositional, as vectors for such features are not usually available.
We take the simplest approach and represent each $n$-gram as the element-wise product of the embeddings of its words.
While standard training objectives favor additive rather than multiplicative compositionality, this method turns out to have useful theoretical proeprties for random word embeddings.

Our document embeddings, which we call **DisC embeddings**<sup>1</sup>, are then just concatenations over $n$ of the sum-of-embeddings of all $n$-grams in the document: 
\[
v_D=\begin{pmatrix}\sum\limits_{w\in\operatorname{words}(D)}v_w&\cdots&\sum\limits_{g\in\operatorname{n-grams}(D)}v_g\end{pmatrix}
\]
When the word embeddings $v_w$ are trained using [GloVe](http://www.aclweb.org/anthology/D14-1162) on a large corpus of Amazon reviews, this document representation compares quite well to both BonG approaches and LSTM methods on sentiment analysis.

<div style="text-align:center;">
<img src="/assets/clfperf_sst_imdb.svg" style="width:400px;" />
</div>

<sup>[1] *DisC* stands for *Distributed Cooccurrence* embeddings, which we use instead of *Distributed $n$-Gram* because the element-wise multiplication ignores word-order, so the actual feature these embeddings encode is $n$ words co-occurring in a window of size $n$.</sup>

## Why should low-dimensional distributed representations do well?

To understand the good performance of distributed representations, we begin by taking a closer look at the sparse counterparts.
The bag-of-words (BoW) representation of a document $x_D^{BoW}$ can be written as
\[
x_D^{BoW} = \sum\limits_{w\in D} e_w$
\]
where $e_w$ is the $V$ dimensional one-hot embedding for the word $w$, i.e. a vector with 1 in the coordinate corresponding to $w$ and 0 at all other coordinates.
So the BoW representation of a document is the sum of certain words vectors of the words in the document, where the word vectors are all orthogonal ($e_w^T e_{w'} = 0$).
The orthogonality of these word vectors lets us exactly recover the words of a document given the BoW representation.
Here is a crucial observation -- since text documents typically have very few distinct words (much lesser than $V$), one could hope to use lower dimensionsal word vectors which are "almost orthogonal" and still be able to uniquely recover the words of a document.
So if we had word vectors $v_w \in \mathbb{R}^d$ which satisfied this almost orthogonality property, then the representation $x_D = \sum\limits_{w\in D} v_w$ would encode precisely the same information as $x_D^{BoW}$.
Note that $x_D = Ax_D^{BoW}$, where $A$ is a $d\times V$ matrix whose columns correspond to the vectors $v_w$s, i.e. $x_D$ is a linear compression of $x_D^{BoW}$.
This is where compressed sensing comes into the picture - compressed sensing deals with finding conditions on the matrix A such that given $Ax$, one can uniquely recover a sparse high-dimensional vector $x$ (cite someone here).
Note that being able to recover the BoW information doesn't directly imply having the same performance as BoW on linear classification (in fact, this is not always true).
However by building upon the work of Calderbank et al., we prove that in the case of random word embeddings, the compressed sensing condition which implies optimal recovery also implies good performance on linear classification.
Furthermore, extending these ideas to $n$-grams, we also show that our DisC embeddings described in the previous post (link here) can do as well as Bag-of-$n$-Gram (BonG) representations.

## Learning under compression

Let's first understand conditions on the linear compression which ensure that the compressed vectors do as well as uncompressed ones on linear classification.
One such condition is the **Restricted Isometry Property** (RIP) introduced by Candes and Tao in their seminal paper which discussed efficient recovery of sparse signals with near optimal measurements.
>**Restricted Isometry Property (RIP)**: $A\in\mathbb{R}^{d\times n}$ satisfies $(k,\epsilon)$-RIP if $(1-\epsilon)\|x\|_2 \le \|Ax\|_2 \le (1+\epsilon)\|x\|_2$, for all $k$-sparse $x\in\mathbb{R}^n$
In other words, every set of $k$ columns of $A$ must form a nearly orthogonal matrix. This is similar to the idea of "almost orthogonality" that we eluded to above.

Candes and Tao discussed RIP in the context of sparse recovery using linear compression and it has been widely used in the compressed sensing literature since then. But does it say anything about linear classification?
The following theorem (extension of a theorem by Calderbank et al.) shows that RIP indeed implies good classification performance of the compressed vectors
>**Theorem**: Suppose $A\in\mathbb{R}^{d\times n}$ satisfies $(2k,\epsilon)$-RIP and let $S = \{({\bf x}_i, y_i)\}_{i=1}^{m}$ be $m$ samples drawn i.i.d. from a distribution $\mathcal{D}$ over $k$-sparse vectors and binary labels. Let $\ell$ be a convex and Lipschitz loss function and $w_0$ be the minimizer of $\ell$ on the distribution $\mathcal{D}$, then with high probability the classifier $\hat w_A$ which is the minimizer of the $\ell_2$ regularized empirical loss function over the compressed samples $\{(A{\bf x}_i, y_i)\}_{i=1}^{m}$ satisfies
>$$\ell_D(\hat w_A) \le \ell_D(w_0) + \mathcal{O}\left(\sqrt{\epsilon + \frac{1}{m}\log\frac{1}{\delta}}\right)$$

In simple words this theorem says that if $A$ satisfies a certain RIP condition then the classifier learnt on the compressed samples $\{Ax_i\}$ will do as well as the best classifier on the uncompressed samples $\{x_i\}$, upto an additive error of $\sqrt{\epsilon}$ which depends on the isometry constant of $A$.
Note that with very large number of samples one cannot hope to do better than the original vectors on linear classfication by using only a linear compression; since for every classifier $w_A$ in the compressed domain, the classifier $A^Tw_A$ in the original domain has the same loss as $w_A$ in the compressed domain.
The theorem shows that RIP matrices ensure that compressed vectors are not too far away from the performance of the uncompressed vectors.

## Proving good performance of DisC
(Sketch)
The matrix of random word embeddings satisfies $(k,\epsilon)$-RIP if $d=\Omega\left(\frac{k}{\epsilon^2}\right)$. By the above theorem we can conclude simple sum of word embeddings does almost as well as bag-of-words representation on linear classification and the additive error goes to 0 and the dimension $d$ of word embeddings goes to infinity.

DisC embeddings can be seen as linear compressions of BonGs. This linear transformation can be shown to satisfy RIP using the theory of bounded-orthonormal-systems. (1-2 lines description here)

## Understanding what our embeddings encode using compressed sensing

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
