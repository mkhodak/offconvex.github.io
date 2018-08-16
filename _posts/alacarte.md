---
layout:     post
title:      Simple and efficient semantic embeddings for unseen words and features
date:       2018-09-01 10:00:00
author:     Sanjeev Arora, Mikhail Khodak, Nikunj Saunshi
visible:    False
---

In this blog we broaden our recent discussion of deep-learning-free text embeddings (see our [previous post](http://www.offconvex.org/2018/06/25/textembeddings/)) to include simple but principled embeddings for arbitrary language features.
As with modern semantic word vectors, our method is rooted in Firth's [distributional hypothesis](https://en.wikipedia.org/wiki/Distributional_semantics), "you shall know a word by the company it keeps."
Under this view each word's embedding can be seen as a concise representation of the distribution of *other* words occurring in the same context (see also [Sanjeev's post](http://www.offconvex.org/2015/12/12/word-embeddings-1/) on the topic). 
By making a shallow extension to this hypothesis, e.g. "you shall know a bigram/sense/entity by the company it keeps," our *à la carte* approach allows us to pick out exactly those features required by a downstream task.
We can then embed them using only the pretrained vectors of words that occur next to them in a large text corpus plus a linear regression step.

Despite this simplicity, *à la carte* embedding leads to state-of-the-art results on several document classsification tasks and for one-shot learning of word vectors.
These results appear in our [ACL'18 paper](http://aclweb.org/anthology/P18-1002) with Yingyu Liang, Tengyu Ma, and Brandon Stewart.
We will also discuss Sanjeev, Yingyu, and Tengyu's [TACL'18 paper](https://transacl.org/ojs/index.php/tacl/article/view/1346) with Yuanzhi Li and Andrej Risteski, which provides the theoretical motivation for our approach.

## Mathematical background: relating word embeddings and their contexts

We start by assuming a large text corpus $C$, over which we have trained high-quality word embeddings $v_w$. 
Then for some text feature $f$ (such as a bigram) we want to use these embeddings, together with the contexts $C_f$ that $f$ appears in, to get an embedding $v_f$.
This simple setup is illustrated below.

<p style="text-align:center;">
<img src="/assets/alacarte_problem.svg" width="30%"  alt="Problem Setup" />
</p>

The easiest thing to do here would be to represent $f$ as the average of all the words in its contexts, i.e. 

$$ v_f^\textrm{additive} = \frac{1}{|C_f|}\sum\limits_{\textrm{context }c\in C_f}\frac{1}{|c|}\sum\limits_{\textrm{word }w\in c}v_w $$
