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
While standard training objectives favor additive rather than multiplicative compositionality, this method turns out to have useful theoretical properties for random word embeddings.

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
To simplify things we analyze the unigram case first and then briefly describe how the same analysis extends to the $n$-gram case.
The bag-of-words (BoW) representation for a document $v_D^{BoW}$ is the sum of $V$ dimensional word vectors, where the vector for a word is its one-hot embedding and $V$ is the size of the vocabulary.
The orthogonality of these word vectors lets us precisely recover the words of a document given the BoW representation.
Now here's a crucial observation - since text documents typically have very few distinct words (much lesser than $V$), one could hope to use lower dimensionsal word vectors which are "almost orthogonal" and still be able to uniquely recover the words in a document.
So if we had word vectors $v_w \in \mathbb{R}^d$ ($d \ll V$) which satisfied this almost orthogonality property, then the representation $v_D = \sum\limits_{w\in D} v_w$ would encode precisely the same information as $v_D^{BoW}$.
Note that $v_D = Av_D^{BoW}$, where $A$ is a $d\times V$ matrix whose columns correspond to the vectors $v_w$ for all words $w$ in the vocabulary, i.e. $v_D$ is a linear compression of $v_D^{BoW}$.

This is where compressed sensing comes into the picture.
Compressed sensing deals with finding conditions on the matrix A that enable the recovery of a sparse high-dimensional vector $x$ from the linear compression $Ax$.
Note that the ability to recover the BoW vector doesn't directly imply having the same performance as BoW on all linear classification tasks (in fact, this is not true in general).
However by building upon the work of [Calderbank et al.](https://pdfs.semanticscholar.org/627c/14fe9097d459b8fd47e8a901694198be9d5d.pdf), we prove that the compressed sensing condition that implies optimal recovery also implies good performance on linear classification.
Furthermore by extending these ideas to $n$-gram case, we also show that our DisC embeddings with random word vectors, which are linear compressions of bag-of-$n$-grams (BonGs), can do as well as them on all linear classification tasks.

## Learning under compression

Let's first understand conditions on the linear compression matrix $A$ which ensure that the compressed vectors do as well as uncompressed ones on linear classification.
One such condition is the **Restricted Isometry Property** (RIP) introduced by Candes and Tao in their seminal paper which discussed efficient recovery of sparse signals with near optimal measurements.

>**Restricted Isometry Property (RIP)**: $A\in\mathbb{R}^{d\times n}$ satisfies $(k,\epsilon)$-RIP if $(1-\epsilon)\|x\|_2 \le \|Ax\|_2 \le (1+\epsilon)\|x\|_2$, for all $k$-sparse $x\in\mathbb{R}^n$

In other words, every set of $k$ columns of $A$ must form a nearly orthogonal matrix.
This is a mathematical formulation of the "almost orthogonality" property that we eluded to earlier.

Candes and Tao discussed RIP in the context of sparse signal recovery from linear compressions and it has been widely used and studied in the compressed sensing literature since then.
But does it say anything about linear classification?
The following theorem (extension of a theorem by [Calderbank et al.](https://pdfs.semanticscholar.org/627c/14fe9097d459b8fd47e8a901694198be9d5d.pdf)) shows that RIP indeed implies good classification performance of the compressed vectors

>**Theorem**: Suppose $A\in\mathbb{R}^{d\times n}$ satisfies $(2k,\epsilon)$-RIP and let $S = \{({\bf x}_i, y_i)\}_{i=1}^{m}$ be $m$ samples drawn i.i.d. from a distribution $\mathcal{D}$ over $k$-sparse vectors and binary labels.
Let $\ell$ be a convex Lipschitz loss function and $w_{BonG}$ be the minimizer of $\ell$ on the distribution $\mathcal{D}$, then with high probability the classifier $\hat w_A$ which minimizes the $\ell_2$ regularized empirical loss over compressed samples $\{(A{\bf x}_i, y_i)\}_{i=1}^{m}$ satisfies
$$\ell_{\mathcal{D}}(\hat w_A) \le \ell_{\mathcal{D}}(w_{BonG}) + \mathcal{O}\left(\sqrt{\epsilon + \frac{1}{m}\log\frac{1}{\delta}}\right)$$

In simple words this theorem says that if $A$ satisfies a certain RIP condition then the classifier learnt on the compressed samples $\{Ax_i\}$ will do as well as the best classifier on the uncompressed samples $\{x_i\}$, upto an additive error of $\mathcal{O}(\sqrt{\epsilon})$ which depends on the isometry constant of $A$.
Note that for every classifier $w_A$ in the compressed domain, the classifier $A^Tw_A$ in the original domain has the same loss as $w_A$ in the compressed domain.
Therefore with large number of samples one cannot hope to do better than the original vectors on linear classfication by using only a linear compression.
The theorem shows that RIP matrices ensure that compressed vectors are not too far away from the performance of the uncompressed vectors on all linear classification task.

## Proving good performance of DisC

It is easy to see that DisC embeddings are linear compressions of BonGs.
For the unigram case, by using the above theorem it suffices to show that the matrix $A$ of word vectors satisfies an RIP condition.
It is known that independently choosing $d$ dimensional random Rademacher word vectors, i.e. choosing each entry of $A\in\mathbb{R}^{d\times V}$ independently and uniformly at random from $\{-\frac{1}{\sqrt{d}}, \frac{1}{\sqrt{d}}\}$ will satisfy $(k,\epsilon)$-RIP with high probability if $d=\tilde\Omega\left(\frac{k}{\epsilon^2}\right)$.
This follows from standard concentration bounds on sum of independent random variables.
By applying the above theorem we can conclude that unigram DisC embeddings with random word vectors do almost as well as bag-of-words on linear classification.
The additive error goes decreases as the dimensionality $d$ of the word embeddings increase and asymptotically goes to 0 as $d$ goes to infinity.

For the $n$-gram case, showing that the matrix transforming BonG vectors to DisC vectors satisfies a similar RIP property is slightly more non-trivial.
This is because the columns of A (corresponding to $n$-grams embeddings) are no longer independent; columns of $n$-grams that share words will be dependent.
We can get around this by using the theory of bounded orthonormal systems to show that the matrix of $n$-gram embeddings also satisfies the required RIP condition (see paper for full proof).
Combining these results we can show that starting with $d=\tilde\Omega\left(\frac{k}{\epsilon^2}\right)$ dimensional random word embeddings, the classifier $w_{DisC}$ trained on DisC representations will satisfy
$$\ell_{\mathcal{D}}(w_{DisC}) \le \ell_{\mathcal{D}}(w_{BonG}) + \mathcal{O}\left(\sqrt{\epsilon}\right)$$

Additionally it can be easily shown that DisC embeddings are **computable by low-memory LSTMs**.
So the above results also proves that if initialized correctly, LSTMs are guaranteed to do as well as BonG representations, a result that extensive empirical study has been unable to establish.

## Pretrained word embeddings
Though the above theoretical analysis uses random word embeddings, in practice word embeddings like GloVe and word2vec that are trained on a large text corpus are often used for language modeling and text classification tasks as they generally outperform random word embeddings.
The same analysis, however, cannot be be applied to pretrained embeddings, since the matrix of pretrained embeddings does not satisfy the necessary RIP conditions.
In fact instead of being almost orthogonal, embeddings for pairs of words that co-occur are trained to have high inner product which each other.
So does their good empirical performance on tasks contradict our compressed sensing view of classification?
To test this we conducted an experiment to check how well pretrained word embeddings encode word information in text documents as compared to random embeddings.
We try to recover words in text documents from the sum of word vectors for both random and pretrained word embeddings and measure success by F1 score of the recovered words (higher is better)

<div stype="text-align:center;">
<img src="/assets/recovery.svg" style="width:400px" />
</div>

Suprisingly, we found that pretrained word embeddings recover words more efficiently than random embeddings at the same dimensionality, suggesting that pretrained embeddings are more efficient at encoding text documents.
However random embeddings are unsurprisingly better at recovering words from a random word salad.
An intuitive explanation for these observations is that since pretrained embeddings were trained on a large text corpus, they are specialized, in some sense, to do well only on real text documents rather than a random collection of words.
To make this intuition a bit more formal, we appeal to a theorem by Donoho and Tanner and prove that words in a document can be recovered from the sum of word vectors if and only if there is a hyperplane containing the vectors for words in the document with the vectors for all other words on one side of it.
Since co-occurring words will have similar embeddings, it would make it easier to find such a hyperplane separating words in a text document from the rest of the words and hence would ensure good recovery.
However, this still does not provably explain good recovery using pretrained embeddings and their good performance on classification tasks.
Perhaps assuming a generative model for text, like the RankWalk model discussed in [an earlier post](https://www.offconvex.org/2016/02/14/word-embeddings-2/), could help us formally prove these statements. 

<div style="text-align:center;">
<img src="/assets/imdbperf_uni_bi.svg" style="width:400px;" />
</div>
