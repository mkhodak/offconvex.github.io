---
layout:     post
title:      Deep-learning-free Text and Sentence Embedding, Part 2
date:       2018-07-01 8:00:00
author:     Sanjeev Arora, Mikhail Khodak, Nikunj Saunshi
visible:    false
---

 [Sanjeev's post](http://www.offconvex.org/2018/06/17/textembeddings/), discussed a simple text embedding, [the SIF embedding](https://openreview.net/pdf?id=SyK00v5xx), which is a simple weighted combination of word embeddings combined with some mild denoising ( outperforms many deep learning based methods, including [Skipthought](https://arxiv.org/pdf/1506.06726.pdf), on some downstream NLP tasks such as sentence semantic similarity and entailment. See also this [independent study by Yves Peirsman](http://nlp.town/blog/sentence-similarity/).
 
However, SIF embeddings embeddings ignore word order (similar to classic *Bag of Word* models), which leads to unexciting performance on many other downstream classification tasks as compared to deep learning methods. (We find that the denoising via SVD in SIF also negatively affects performance on some classification tasks.) Can we design a text embedding with the simplicity and transparency of SIF while also incorporating word order information?   Our [ICLR'18 paper](https://openreview.net/pdf?id=B1e5ef-C-) with Kiran Vodrahalli does this, and achieves strong empirical performance and also some surprising theoretical guarantees stemming from the theory of compressed sensing. It is competitive with all pre-2018 LSTM-based methods, and is much faster to compute, since it uses pretrained (GloVe) word vectors and simple linear algebra. 

## Incorporating local word order: n-gram embeddings

*Bigrams* are ordered word-pairs that appear in the sentence, and $n$-grams are ordered $n$-tuples. A piece of text with $k$ words has $k-1$ bigrams and $k-n+1$ $n$-grams. Classic NLP methods exploit information about $n$-grams, and simple linear classifiers using Bag-of-n-gram (BonG) information are a [surprisingly strong baseline for document classification tasks](https://www.aclweb.org/anthology/P12-2018).
Of course, $n$-grams don't exploit long-range dependencies in text, while an LSTM can. 

The trivial idea for incorporating $n$-grams into SIF embeddings would be to treat n-grams like words, and compute word embeddings for them using either GloVe and word2vec.  But the practical difficulty is that the number of distinct n-grams in the corpus gets very large even for $n=2$ (let alone $n=3$), making it almost impossible to solve word2vec or GloVe on current systems. Thus efficiency considerations dictate a more *compositional* approach.

> **Compositional n-gram embedding:** Represent $n$-gram $g=(w_1,\dots,w_n)$ as the element-wise product $v_g=v_{w_1}\odot\cdots\odot v_{w_n}$ of the embeddings of its constituent words.

Now we are ready to define our text embeddings.

>**DisC embedding**,<sup>1</sup> of a piece of text is just concatenation for $(v_1, v_2, \ldots)$ where $v_n$ is the sum of the $n$-gram embeddings of all $n$-grams in the document (for $n=1$ this is just the average of word embeddings).

(IS FOLLOWING NOTATION NEEDED LATER??)

\[
v_{DisC}=\begin{pmatrix}\sum\limits_{w\in\operatorname{words}}v_w&\cdots&\sum\limits_{g\in\operatorname{n-grams}}v_g\end{pmatrix}
\]

<p style="text-align:center;">
<img src="/assets/unsupervised_pipeline.png" width="50%" alt ="Pipeline" />
</p>

Note that DisC embeddings can be more powerful than classic bag-of-n-gram representations because they leverage the power of word embeddings. For instance,  the sentences *"Loved this movie!"* and *"I enjoyed the film"* share no $n$-gram information for any $n$, but  their DisC embeddings are fairly similar. This causes them to outperform BonG on the Stanford Sentiment Treebank (SST) task, which has only $6,000$ labeled examples. DisC embeddings also beat SIF and a standard LSTM-based method, Skipthoughts. 

<div style="text-align:center;">
<img src="/assets/clfperf_sst_imdb.png" width ="60%" alt ="The pipeline" />
</div>

<sup>[1] For *Distributed Cooccurrence* embeddings, used instead of *Distributed $n$-Gram* because the multiplication ignores word-order, so the actual feature these embeddings encode is words co-occurring in a window of size $n$. The distinction doesn't greatly affect performance in practice. </sup>

## Why should low-dimensional distributed representations do well?

To understand the good performance of distributed representations, we begin by taking a closer look at their sparse counterparts.
We will start with the unigram case and then extend the analysis to $n$-grams.
Note first that the bag-of-words (BoW) representation for a document $v_{BoW}$ is the sum of $V$ dimensional word vectors, where the vector for a word is its one-hot embedding and $V$ is the vocab size.
The orthogonality of such word vectors lets us recover all the words of a document given the BoW represention.
Now here's a crucial observation - since documents typically have very few distinct words ($\ll V$), one could hope to use low-dimensional word vectors which are "almost orthogonal" and still be able to uniquely recover the words in a document.
So if we had vectors $v_w \in \mathbb{R}^d$ ($d \ll V$) satisfying this almost orthogonality property, then the representation $v_{sum}=\sum\limits_{w\in\operatorname{words}}v_w$ would encode precisely the same information as $v_{BoW}$.
Note that $v_{sum} = Av_{BoW}$, where $A$ is a $d\times V$ matrix whose columns correspond to the vectors $v_w$ for all words $w$ in the vocabulary, i.e. $v_{sum}$ is a linear compression of $v_{BoW}$.

Here is where compressed sensing comes into the picture.
This field deals with finding conditions on the matrix A that enable the recovery of a sparse high-dimensional vector $x$ from the linear compression $Ax$.
Note that the ability to recover the BoW vector doesn't directly imply having the same performance as BoW on all linear classification tasks (this is not true in general).
However, a result of [Calderbank, Jafarpour, & Schapire](https://pdfs.semanticscholar.org/627c/14fe9097d459b8fd47e8a901694198be9d5d.pdf) shows that the compressed sensing condition that implies optimal recovery also implies good performance on linear classification under compression.
Furthermore, by extending these ideas to the $n$-gram case, we show that our DisC embeddings with random word vectors, which are linear compressions of BonGs, can do as well as them on all linear classification tasks.

## Learning under compression

Let's first consider a well-known recovery condition on the compression matrix $A$: the **Restricted Isometry Property** (RIP) introduced by [Candes & Tao](https://statweb.stanford.edu/~candes/papers/DecodingLP.pdf) in their seminal paper on efficient recovery of sparse signals:

>**Restricted Isometry Property (RIP)**: $A\in\mathbb{R}^{d\times n}$ satisfies $(k,\varepsilon)$-RIP if $(1-\varepsilon)\|x\|_2 \le \|Ax\|_2 \le (1+\varepsilon)\|x\|_2$, for all $k$-sparse $x\in\mathbb{R}^n$

In other words, every set of $k$ columns of $A$ must form a nearly orthogonal matrix (think of $k$ as being the maximum document length).
This is a mathematical formulation of the "almost orthogonality" property alluded to earlier.

RIP provably allows stable and efficient recovery of a sparse signal from its linear compression, a result which initiated the field of compressed sensing.
But does RIP say anything about linear classification in the compressed domain ("compressed learning")?
The following theorem (extension of a result by Calderbank et al.) shows that it indeed implies good classification performance over the compressed vectors.

>**Theorem**: Suppose $A\in\mathbb{R}^{d\times n}$ satisfies $(2k,\varepsilon)$-RIP and let $S = \{({\bf x}_i, y_i)\}_{i=1}^{m}$ be $m$ samples drawn i.i.d. from a distribution $\mathcal{D}$ over $k$-sparse vectors and binary labels.
Let $\ell$ be a convex Lipschitz loss function and $w$ be the minimizer of $\ell$ on the distribution $\mathcal{D}$, then with high probability the classifier $\hat w_A$ which minimizes the $\ell_2$ regularized empirical loss over compressed samples $\{(A{\bf x}_i, y_i)\}_{i=1}^{m}$ satisfies
\[
\ell_{\mathcal{D}}(\hat w_A) \le \ell_{\mathcal{D}}(w) + \mathcal{O}\left(\sqrt{\varepsilon + \frac{1}{m}\log\frac{1}{\delta}}\right)
\]

This theorem states that if $A$ satisfies a certain RIP condition then the classifier learnt on the compressed samples $\{Ax_i\}$ will do as well as the best classifier on the uncompressed samples $\{x_i\}$, up to additive error $\mathcal{O}(\sqrt{\varepsilon})$ depending  on the RIP constant of $A$.
Note that for every classifier $w_A$ in the compressed domain, the classifier $A^Tw_A$ in the original domain has the same loss as $w_A$ in the compressed domain.
Therefore with many samples one cannot hope to do better than the original vectors on linear classfication by using only a linear compression.
The theorem shows that RIP matrices ensure that compressed vectors are not too far away from the performance of the uncompressed vectors on all linear classification task.

## Proving good performance of DisC

It is easy to see that DisC embeddings $v_{DisC}=Av_{BonG}$ are linear compressions of BonGs, where $A$ is the matrix of $n$-gram vectors.
Thus for the unigram case, if we assign each word a $d$-dimensional vector of i.i.d. Rademacher variables (normalized by $\frac{1}{\sqrt{d}}$) then each entry of $A\in\mathbb{R}^{d\times V}$ will be an independent Rademacher random variable, which is known to satisfy $(k,\varepsilon)$-RIP w.h.p. if $d=\tilde\Omega\left(\frac{k}{\varepsilon^2}\right)$ (this follows from concentration of sums of independent random variables).
Applying the above theorem shows that low-dimensional ($d$ linear in the document length) unigram DisC embeddings built with random word vectors do approximately as well as BoW on linear classification.
The additive error decreases to 0 asymptotically as the dimensionality $d$ of the word embeddings goes to infinity.

For the $n$-gram case, showing that the matrix transforming BonG vectors to DisC vectors satisfies a similar RIP property is more non-trivial.
This is because the columns of A (corresponding to $n$-grams embeddings) are no longer independent; columns of $n$-grams that share words will be dependent.
We can get around this by using [compressed sensing results for bounded orthonormal systems](http://www.cis.pku.edu.cn/faculty/vision/zlin/A%20Mathematical%20Introduction%20to%20Compressive%20Sensing.pdf) to show that the matrix of $n$-gram embeddings also satisfies the required RIP condition (see paper for full proof).
Combining these results proves that starting with $d=\tilde\Omega\left(\frac{k}{\varepsilon^2}\right)$ dimensional random word embeddings, the classifier $w_{DisC}$ trained on DisC representations will satisfy
\[
\ell_{\mathcal{D}}(w_{DisC}) \le \ell_{\mathcal{D}}(w_{BonG}) + \mathcal{O}\left(\sqrt{\varepsilon}\right)
\]

Additionally it can be easily shown that DisC embeddings are *computable by low-memory LSTMs*.
So the above results also imply that, if initialized correctly, **LSTMs are guaranteed to do approximately as well as BonG representations**, a result that extensive empirical study has been unable to establish.

<div style="text-align:center;">
<img src="/assets/imdbperf_uni_bi.png" style="width:300px;" />
</div>

We empirically tested the effect of dimensionality by measuring performance of DisC on IMDb sentiment classification.
As our theory predicts, the accuracy of DisC using random word embeddings converges to that of BonGs as dimensionality increases.
Interestingly we also find that DisC using pretrained word embeddings like GloVe converges to BonG performance at much smaller dimensions, an unsurprising but important point that we will discuss next.

## Pretrained word embeddings

While our theoretical analysis relies on random word embeddings, in practice embeddings pretrained on large text corpora, such as GloVe and word2vec, are often used for language modeling and classification tasks as they commonly outperform random vectors.
However, our analysis cannot be be applied to these pretrained embeddings because the matrix of pretrained embeddings does not satisfy RIP.
In fact, instead of being almost orthogonal, embeddings for pairs of words that co-occur frequently are trained to have high inner product.
So does their good empirical performance on tasks contradict our compressed sensing view of classification?
To test this we conducted an experiment to check how well pretrained word embeddings encode word information in documents compared to random embeddings.
We use Basis Pursuit, a sparse recovery approach related to LASSO with provable guarantees for RIP matrices, to recover words in documents from their sum-of-word-embeddings vectors for both random and pretrained word embeddings, measuring success via the $F_1$-score of the recovered words (higher is better). 

<div stype="text-align:center;">
<img src="/assets/recovery.png" style="width:300px" />
</div>

Suprisingly, we found that pretrained word embeddings recover words *more efficiently* than random embeddings at the same dimensionality, suggesting that pretrained embeddings are more efficient at encoding text documents.
However random embeddings are unsurprisingly better at recovering words from random word salad (the right-hand image).
An intuitive explanation for these observations is that since pretrained embeddings were trained on a large text corpus, they are specialized, in some sense, to do well only on real documents rather than a random collection of words.
To make this intuition a bit more formal, we can use a result of [Donoho & Tanner](http://www.pnas.org/content/pnas/102/27/9446.full.pdf) to prove that words in a document can be recovered from the sum of word vectors if and only if there is a hyperplane containing the vectors for words in the document with the vectors for all other words on one side of it.
Since co-occurring words will have similar embeddings, it would make it easier to find such a hyperplane separating words in a document from the rest of the words and hence would ensure good recovery.

However, this still does not provably explain good recovery using pretrained embeddings, and even such a result would not necessarily imply good performance on classification tasks, as current compressed learning results depend on RIP and not sparse recovery.
Perhaps assuming a generative model for text, like the RandWalk model discussed in an [earlier post](https://www.offconvex.org/2016/02/14/word-embeddings-2/), could help us formally prove these statements.

## Discussion

Our empirical results on text classification using simple compositions of pretrained word embeddings are further evidence that such simple representation schemes can still compete with more opaque deep learning approaches.
As further evidence, a new [NAACL'18 paper](https://arxiv.org/abs/1703.02507) of Pagliardini, Gupta, & Jaggi proposes a text embedding similar to DisC in which unigram and bigram embeddings are trained specifically to be added together to form sentence embeddings, also achieving good results.
We also give an information-theoretic account of DisC embeddings using the theory of compressed sensing, highlighting its connection to downstream task performance, and discover a new property of pretrained word embeddings.
While our compositional $n$-gram embeddings don't achieve similar results on sparse recovery, learned $n$-gram vectors (such as those in our upcoming [ACL'18 paper](https://arxiv.org/abs/1805.05388) with Yingyu Liang, Tengyu Ma, & Brandon Stewart) might be more likely to exhibit such properties.
Perhaps not-coincidentally, such embeddings also lead to better results for classification.

We have made available [sample code for constructing and evaluating DisC embeddings](https://github.com/NLPrinceton/text_embedding) and [solvers for recreating the sparse recovery results for word embeddings](https://github.com/NLPrinceton/sparse_recovery).
