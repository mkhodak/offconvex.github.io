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
Sanjeev discussed their resulting properties in two [previous](http://www.offconvex.org/2016/02/14/word-embeddings-2/) [posts](http://www.offconvex.org/2016/07/10/embeddingspolysemy/).

This post is based on our [ICLR 2018 paper](https://openreview.net/forum?id=B1e5ef-C-&noteId=B1e5ef-C-) (joint with Kiran Vodrahalli) about elementary, interpretable methods for defining and computing text representations that has both provable guarantees and is competitive with state-of-the-art deep learning methods. 
A perhaps surprising member in this story is compressed sensing (also called sparse recovery).

##The powers and limitations of sparse text representations

Before discussing distributed embeddings, let's briefly review an alternative: the simple idea to retain $n$-gram information.
The result is the bag-of-$n$-grams (BonG) featurization, which in the unigram case reduces to the familiar bag-of-words (BoW).
This approach is often the first application of SVM taught in intro ML classes, and with good reason: as [Wang & Manning](https://www.aclweb.org/anthology/P12-2018) remind us, it remains a very strong baseline for document classification.

However, $n$-grams can fail to capture similarity in ways that matter when only a few labeled examples are available.
For example, the sentences "This movie was great!" and "I enjoyed the film." should mean the same thing to the ideal sentiment classifier but share no $n$-grams of any order.
Thus having a label for the first example tells us nothing about the second.

##Towards simple distributed representations

To get similarity properties it seems appropriate to start with distributed word vectors.
Indeed a series of papers have tried to construct text embeddings using embeddings of the constituent words.

Assuming each word $w$ in a vocabulary of size $V$ is assigned a vector $v_w\in\mathbb{R}^d$, the simplest way to represent a document $D$ of length $T$ is to take the average over the word embeddings:
\[
v_D=\frac{1}{T}\sum\limits_{w\in\operatorname{words}(D)}v_w
\]
Additive composition is the standard way of combining word representations, reflected in training objectives such as [word2vec](https://arxiv.org/abs/1301.3781)'s CBOW formula.

As a representation the simple average leaves much to be desired; 
apart from neglecting word-order, adding up embeddings also tends to amplify less meaningful components, both because of stop-words ("the", "and", etc.) and because word embeddings tend to share many common directions.
A first step towards mitigating the latter issue is to down-weight frequent words using say TF-IDF, a classical scheme from information retrieval.
In an [ICLR 2017 paper](https://openreview.net/forum?id=SyK00v5xx) (joint with Yingyu Liang and Tengyu Ma), Sanjeev and co-authors improved upon this using a new weighting scheme (smooth inverse frequency, or *SIF*):
\[
v_D=\sum\limits_{w\in\operatorname{words}(D)}\frac{a}{a+\operatorname{frequency}(w)}v_w
\]
for some parameter $a$.
In addition, they suggested taking out the component along the top singular direction of each batch of sentences.
This approach was inspired by earier work by [Wieting et al.](https://arxiv.org/abs/1511.08198), who started with the simple average embedding and improved it using a paraphrase dataset.

##Incorporating word-order

As in the BonG approach of concatenating $n$-gram indicators on top of BoW, we can incorporate word-order in distributed embeddings by concatenating sums of $n$-gram embeddings on top of the simple sum-of-word embeddings vector.
However, we want the $n$-gram embeddings themselves to also be compositional, as vectors for such features are not usually available.
We take the simplest approach and represent each $n$-gram as the element-wise product of the embeddings of its words. 
While standard training objectives favor additive rather than multiplicative compositionality, this method turns out to have useful theoretical properties for random word embeddings.

Our document embedding is then just a concatenation over $n$ of the sum-of-embeddings of all $n$-grams in the document:
\[
v_D=\begin{pmatrix}\sum\limits_{w\in\operatorname{words}(D)}v_w&\cdots&\sum\limits_{g\in n\operatorname{-grams}(D)}v_g\end{pmatrix}
\]
When the word embeddings $v_w$ are trained using [GloVe](http://www.aclweb.org/anthology/D14-1162) on a large corpus of Amazon reviews, this document representation compares quite well to both BonG approaches and LSTM methods on sentiment analysis.

<div style="text-align:center;">
<img src="/assets/clfperf_sst_imdb.svg" style="width:400px;" />
</div>

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
