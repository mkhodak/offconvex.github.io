---
layout:     post
title:      Unsupervised learning for representing the meaning of text
date:       2018-06-01 8:00:00
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

Both their paper and subsequent evaluations (see this nice [blog post](http://nlp.town/blog/sentence-similarity/) by Yves Peirsman) show that SIF embeddings work very well on sentence semantic similarity and relatedness tasks, outperforming deep learning approaches such as LSTM embeddings and deep averaging networks.
In these evaluations pairs of sentence embeddings are assigned scores based on their inner product or a trained regression targeting human ratings.
However, SIF embeddings do not end up improving performance strongly on sentiment classification tasks; 
indeed taking out the top component hurts performance, while the weighting gives only a slight improvement.
It seems that while word-level semantic content suffices for good similarity performance, sentiment analysis depends more on word-order, something that SIF doesn't capture.

##Incorporating word-order

As in the BonG approach of concatenating $n$-gram indicators on top of BoW, we can incorporate word-order in distributed embeddings by concatenating sums of $n$-gram embeddings on top of the simple sum-of-word embeddings vector.
However, we want the $n$-gram embeddings themselves to also be compositional, as vectors for such features are not usually available.
We take the simplest approach and represent each $n$-gram as the element-wise product of the embeddings of its words. 
While standard training objectives favor additive rather than multiplicative compositionality, this method turns out to have useful theoretical properties for random word embeddings.

Our document embedding is then just a concatenation over $n$ of the sum-of-embeddings of all $n$-grams in the document:
\[
v_D=\begin{pmatrix}\sum\limits_{w\in\operatorname{words}(D)}v_w&\cdots&\sum\limits_{g\in\operatorname{n-grams}(D)}v_g\end{pmatrix}
\]
When the word embeddings $v_w$ are trained using [GloVe](http://www.aclweb.org/anthology/D14-1162) on a large corpus of Amazon reviews, this document representation compares quite well to both BonG approaches and LSTM methods on sentiment analysis.

<div style="text-align:center;">
<img src="/assets/clfperf_sst_imdb.svg" style="width:400px;" />
</div>


##Looking forward
The success of our represenation on these tasks demonstrates how simple methods using only local word-order information are still competitive with deep learning approaches.
Indeed, embedding documents using summations of n-gram embeddings has also been shown to be effective by more recent work such as [Sent2Vec](https://arxiv.org/abs/1703.02507), who learn unigram and bigram vectors specifically for sentence representation, and also in our [upcoming paper at ACL 2018](https://arxiv.org/abs/1805.05388) (joint with Yingyu Liang, Tengyu Ma, and Brandon Stewart).
Unlike in these papers, where the n-gram vectors are the results of a learning procedure, our embeddings can be computed compositionally using just word embeddings.

In this post we have introduced a few simple ways of representing documents using word embeddings alone; this leaves open a theoretical understanding of their performance on downstream tasks.
In a follow-up post we will show how ideas from sparse recovery can be used to understand what these n-gram embeddings encode in the case of random word vectors.
We will see that these compressed sensing properties also matter for downstream tasks, specifically linear classification.
Finally, we will examine why these results matter in the case of pretrained word embeddings as well.
