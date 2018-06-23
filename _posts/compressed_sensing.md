---
layout:     post
title:      Deep-learning-free Text and Sentence Embedding, Part 2
date:       2018-07-01 8:00:00
author:     Sanjeev Arora, Mikhail Khodak, Nikunj Saunshi
visible:    false
---

This post continues [Sanjeev's post](http://www.offconvex.org/2018/06/17/textembeddings/) and describes further attempts to construct elementary and interpretable text embeddings. The previous post described the [the SIF embedding](https://openreview.net/pdf?id=SyK00v5xx), which uses a simple weighted combination of word embeddings combined with some mild "denoising" based upon singular vectors, yet outperforms many deep learning based methods, including [Skipthought](https://arxiv.org/pdf/1506.06726.pdf), on certain downstream NLP tasks such as sentence semantic similarity and entailment. See also this [independent study by Yves Peirsman](http://nlp.town/blog/sentence-similarity/).
 
However, SIF embeddings embeddings ignore word order (similar to classic *Bag of Words* models in NLP), which leads to unexciting performance on many other downstream classification tasks. (Even the denoising via SVD, which is crucial in similarity tasks, slightly reduces performance on other tasks.) Can we design a text embedding with the simplicity and transparency of SIF while also incorporating word order information?   Our [ICLR'18 paper](https://openreview.net/pdf?id=B1e5ef-C-) with Kiran Vodrahalli does this, and achieves strong empirical performance and also some surprising theoretical guarantees stemming from the [theory of compressed sensing](https://en.wikipedia.org/wiki/Compressed_sensing). It is competitive with all pre-2018 LSTM-based methods on standard tasks. Even better, it is much faster to compute, since it uses pretrained (GloVe) word vectors and simple linear algebra. 

<p style="text-align:center;">
<img src="/assets/unsupervised_pipeline.png" width="50%" alt ="Pipeline" />
</p>

## Incorporating local word order: n-gram embeddings

*Bigrams* are ordered word-pairs that appear in the sentence, and $n$-grams are ordered $n$-tuples. A document with $k$ words has $k-1$ bigrams and $k-n+1$ $n$-grams. *Bag of n-gram (BonG) representation* of a document refers to a long vector whose each entry is indexed by all possible n-grams, and contains the number of times the corresponding n-gram appears in the document. Linear classifiers trained on BonG representations are a [surprisingly strong baseline for document classification tasks](https://www.aclweb.org/anthology/P12-2018).
While $n$-grams don't directly encode long-range dependencies in text, one hopes that a fair bit of such information is implicitly present. 

A trivial idea for incorporating $n$-grams into SIF embeddings would be to treat n-grams like words, and compute word embeddings for them using either GloVe and word2vec.  This runs into the difficulty that the number of distinct n-grams in the corpus gets very large even for $n=2$ (let alone $n=3$), making it almost impossible to solve word2vec or GloVe. Thus one gravitates towards a more *compositional* approach.

> **Compositional n-gram embedding:** Represent $n$-gram $g=(w_1,\dots,w_n)$ as the element-wise product $v_g=v_{w_1}\odot\cdots\odot v_{w_n}$ of the embeddings of its constituent words.

Note that due to use of element-wise multiplication we actually represent unordered $n$-gram information, not ordered $n$-grams. (We also tried methods that maintain order information, but the benefit was tiny.) Now we are ready to define our *Distributed Co-occurence (DisC) embeddings*.

>**DisC embedding** of a piece of text is just concatenation for $(v_1, v_2, \ldots)$ where $v_n$ is the sum of the $n$-gram embeddings of all $n$-grams in the document (for $n=1$ this is just the average of word embeddings).


Note that DisC embeddings leverage classic bag-of-n-gram information as well as the power of word embeddings. For instance,  the sentences *"Loved this movie!"* and *"I enjoyed the film."* share no $n$-gram information for any $n$, but  their DisC embeddings are fairly similar. Thus if the first example comes with a label, it gives the learner some idea of how to classify the second. This can be useful especially in settings with few labeled examples; e.g. DisC outperform BonG on the Stanford Sentiment Treebank (SST) task, which has only $6,000$ labeled examples. DisC embeddings also beat SIF and a standard LSTM-based method, Skipthoughts. On the much larger IMDB testbed, BonG still reigns at top. 

<div style="text-align:center;">
<img src="/assets/clfperf_sst_imdb.png" width ="60%" alt ="The pipeline" />
</div>


## Some theoretical analysis via compressed sensing

A linear SIF-like embedding is representing a document with bag-of-words vector $x$ as 
$$\sum_w \alpha_w x_w v_w,$$
where $v_w$ is the embedding of word $w$ and $\alpha_w$ is a scaling term. In other words, it represents document $x$ as 
$A x$ where $A$ is the matrix with as many columns as the number of words in the language, and the column corresponding to word $w$ is 
$\alpha_w A$. Note that $x$ has many zero coordinates corresponding to words that don't occur in the document; in other words is a *sparse* vector. 

The starting point of our DisC work was the realization that perhaps the reason SIF-like embeddings work reasonably well is that they *preserve* the bag-of-words information, in the sense that it may be possible to *easily recover* $x$ from $A$. This is not an outlandish conjecture at all, because [*compressed sensing*](https://en.wikipedia.org/wiki/Compressed_sensing) does exactly this when $x$ is suitably sparse and matrix $A$ has some nice properties such as RIP or Incoherence. A classic example concerns $A$ being a random matrix, which in our case corresponds to using random vectors as word embeddings.  Thus one could try to use random word embeddings instead of GloVe vectors in the construction and see what happens! Indeed, we find that so long as we raise the dimension of the word embeddings, then text embeddings using random vectors do indeed converge to the performance of BonG representations. 

This is a surprising result and not that compressed sensing does not imply this per se, since the ability to reconstruct the BoW vector from its compressed version doesn't directly imply that the compressed version gives same performance as BoW on linear classification tasks. However, a result of [Calderbank, Jafarpour, & Schapire](https://pdfs.semanticscholar.org/627c/14fe9097d459b8fd47e8a901694198be9d5d.pdf) shows that the compressed sensing condition that implies optimal recovery also implies good performance on linear classification under compression.

Furthermore, by extending these ideas to the $n$-gram case, we show that our DisC embeddings with random word vectors, which are linear compressions of BonGs, can do as well as them on all linear classification tasks. To do this we prove that the "sensing" matrix $A$ corresponding to DisC embeddings satisfy the  *Restricted Isometry Property (RIP)* introduced in the seminal paper of [Candes & Tao](https://statweb.stanford.edu/~candes/papers/DecodingLP.pdf).  The theorem relies upon  [compressed sensing results for bounded orthonormal systems](http://www.cis.pku.edu.cn/faculty/vision/zlin/A%20Mathematical%20Introduction%20to%20Compressive%20Sensing.pdf) and says that then the performance of DisC embeddings on linear classification tasks approaches that of BonG vectors as we increase the dimension. This is also verified experimentally. Please see our paper for details.

## A surprising lower bound on the power of LSTM-based text representations

The above result also leads to a new theorem about deep learning: *text embeddings computed using low-memory LSTMs should do at least as well as BonG representations on downstream classification tasks.* At first glance this result may seem uninteresting: surely it's no surprise that the field's latest and greatest method is at least as powerful as its oldest? But in practice, most papers on LSTM-based text embeddings make it a point to compare to performance of BonG baseline, and *often are unable to improve upon that baseline!* Thus empirically this new theorem had not been clear at all! 

The new theorem follows from considering an LSTM that uses random vectors as word embeddings and computes the DisC embedding in one pass over the text. (For details see our appendix.) 

We empirically tested the effect of dimensionality by measuring performance of DisC on IMDb sentiment classification.
As our theory predicts, the accuracy of DisC using random word embeddings converges to that of BonGs as dimensionality increases.
Interestingly we also find that DisC using pretrained word embeddings like GloVe converges to BonG performance at much smaller dimensions, an unsurprising but important point that we will discuss next.

<div style="text-align:center;">
<img src="/assets/imdbperf_uni_bi.png" style="width:300px;" />
</div>


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
