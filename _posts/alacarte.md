---
layout:     post
title:      Simple and efficient semantic embeddings for rare words, n-grams, and any other language feature
date:       2018-09-01 10:00:00
author:     Sanjeev Arora, Mikhail Khodak, Nikunj Saunshi
visible:    False
---

Recall that distributional methods for capturing meaning, such as word embeddings, almost by definition require observing many examples of words in context.
On the other hand, people can induce a reasonable meaning for a concept from a single, sufficiently informative sentence, such as the first line of a [Wikipedia entry](https://en.wikipedia.org/wiki/Syzygy_(astronomy)): "a *syzygy* is a straight line configuration of three or more celestial bodies in a gravitational system."
Can we devise an algorithm that can do the same?

In this blog we describe a simple, principled, but effective method for inducing embeddings of rare words from just a few examples in context.
This *à la carte* approach, described in our [ACL'18 paper](http://aclweb.org/anthology/P18-1002) with Yingyu Liang, Tengyu Ma, and Brandon Stewart, easily extends to learning embeddings of arbitrary language features such as word-senses and n-grams. 
Combining these with our recent [deep-learning-free text embeddings](http://www.offconvex.org/2018/06/25/textembeddings/) leads to state-of-the-art results on several document classification tasks as well.

## Relating word embeddings and their contexts

We formalize our goal of finding an algorithm that can induce a word's meaning from context as the task of learning a mapping from a sequence $c$ of words surrounding a word $w$ to its word embedding $v_w$. 
First, in order the represent the context sequence $c$, [recall](http://www.offconvex.org/2016/02/14/word-embeddings-2/) that, for GloVe-like embeddings, the *maximum a posteriori* estimate of the embedding of $c$ is the average over the embeddings of all words in $w'\in c$.
We can construct such a context vector for rare words because we usually have high-quality embeddings of most of its surrounding tokens.

Now suppose that there exists some general linear transform $A$ between this context vector and the word embedding $v_w$, i.e. that $v_w\approx A\left(\sum_{w'\in c}v_{w'}\right)$ .
We call this transformed context vector the  *à la carte* embedding of $w$, so called because given a matrix $A$ it is near-effortless to learn an embedding of any desired word (and later, any feature) without paying the *prix fixe* of full-corpus training.
While seemingly simple, using a linear transform subsumes several other ways of getting good context representations, such as removing the [top singular component or down-weighting frequent directions](http://www.offconvex.org/2018/06/17/textembeddings/).
Furthermore, for frequent words we find that vectors induced using the best linear transform of their average context vector in a corpus have cosine similarity $>0.9$ with their true word embeddings.
The latter finding also shows a simple way of learning $A$: linear regression over sets $C_w$ containing all contexts of frequent words $w$ in a large text corpus:

$$ A=\arg\min\sum\limits_w\left\|v_w-A\sum\limits_{c\in C_w}\sum\limits_{w'\in c}v_{w'}\right\|_2^2 $$

We put this method to the test by checking how well we can induce embeddings for words with just one or a few occurrences in context.
The performance of standard word embedding methods is known to degrade in such low frequency settings.
In order to analyze the effect of number of contexts on the quality of induced embeddings we created the *[Contextual Rare Words](http://nlp.cs.princeton.edu/CRW/)* dataset (a subset of the [Rare Words](https://nlp.stanford.edu/~lmthang/morphoNLM/) dataset) where, along with word pairs and human-rated scores, we also provide contexts for the rare words.
We compare the performance of our method with the alternatives mentioned above and find that *à la carte* embedding consistently outperforms other methods and requires far fewer contexts to match their best performance.

<p style="text-align:center;">
<img src="/assets/crwplot.svg" width="40%" />
</p>

Additionally we evaluate our method on tasks that involve finding reliable embeddings for unseen words and concepts given a single definition or a few sentences of usage for these concepts.
To "simulate the process by which a competent speaker encounters a new word in known contexts," [Herbelot and Baroni](http://aclweb.org/anthology/D17-1030) constructed a "nonce" dataset consisting of single-word concepts and their definitions.
By replacing this competent speaker with a word embedding algorithm, the authors proposed an evaluation in which the embedding it produces using the definition is compared to a ground truth embedding obtained by full-corpus training.
As shown in the results below, the embedding *à la carte* induces using the definition is much closer to this true embedding than that produced by other methods, including a modification to word2vec developed by Herbelot and Baroni.

<p style="text-align:center;">
<img src="/assets/nonce.svg" width="40%" />
</p>

These experiments show that *à la carte* is able to induce high-quality word embeddings from very few uses in context, outperforming both simple methods like SIF and all-but-the-top and more sophisticated methods like word2vec and its modifications.

##  A theory of induced embeddings for general features

Should we expect there to be a linear transformation between embeddings of words $v_w$ and average embeddings of *contexts* of $w$?
For GloVe-like embeddings, the answer turns out to be *yes* (and empirically for word2vec as well).
If we consider a latent-variable model of corpus generation, in which each window $c$ of $n$ words is generated by drawing a context vector $x\sim\mathcal{N}(0_d,\Sigma)$ and then sampling words $w_1,\dots,w_n$ with $\mathbb{P}(w_i)\propto\exp\langle x,v_{w_i}\rangle$.
Under this modification of the rand-walk model, whose approximate MLE objective is similar to that of GloVe, a result from Sanjeev, Yingyu, and Tengyu's [TACL'18](https://transacl.org/ojs/index.php/tacl/article/view/1346) paper together with Yuanzhi Li and Andrej Risteski shows the following equation:

$$ \exists~A\in\mathbb{R}^{d\times d}\textrm{ s.t. }v_w=A\mathbb{E} \left[\frac{1}{n}\sum\limits_{w'\in c}v_{w'}\bigg|w\in c\right]=A\mathbb{E}\sum\limits_{w'\in c}~\forall~w $$

where the expectation is taken over possible contexts $c$. This says that there exists some linear transform mapping the expected average context vector of each word to that word's word embedding.
The linear regression above returns the MLE for $A$ under this model.

This result also explains the linear algebraic structure of the embeddings of polysemous words (words having multiple possible meanings, such as *tie*) discussed in an earlier [post](http://www.offconvex.org/2016/07/10/embeddingspolysemy/).
Assuming for simplicity that $tie$ only has two meanings (*clothing* and *game*), it is easy to see that its word embedding is a linear transformation of the sum of the average context vectors of its two senses:

$$ v_w=A\mathbb{E}v_w^\textrm{avg}=A\mathbb{E}\left[v_\textrm{clothing}^\textrm{avg}+v_\textrm{game}^\textrm{avg}\right]=A\mathbb{E}v_\textrm{clothing}^\textrm{avg}+A\mathbb{E}v_\textrm{game}^\textrm{avg} $$

This equation also shows that we can get a reasonable estimate for the vector of the sense *clothing*, and, by extension many other features of interest, by setting $v_\textrm{clothing}=A^\ast v_\textrm{clothing}^\textrm{avg}$.
We test this hypothesis by inducing embeddings for $n$-grams by using contexts from a large text corpus and word embeddings trained on the same corpus.
A qualitative evaluation of the $n$-gram embeddings is done by finding the closest words to it in terms of cosine similarity between the embeddings.
As evident from the below figure, *à la carte* bigram embeddings capture the meaning of the phrase better than some other compositional and learned bigram embeddings.

<p style="text-align:center;">
<img src="/assets/ngram_quality.png" width="65%" />
</p>

We also use these $n$-gram embeddings to construct sentence embeddings, similarly to [DisC embeddings](http://www.offconvex.org/2018/06/25/textembeddings/), to evaluate on classification tasks.
A sentence is embedded as the concatenation of sums of embeddings for $n$-gram in the sentence for use in downstream classification tasks.
Using this simple approach we can match the performance of other linear and LSTM representations, even obtaining state-of-the-art results on some of them.

<p style="text-align:center;">
<img src="/assets/ngram_clf.png" width="80%" />
</p>

## Discussion

Apart from its simplicity and computational efficiency, the versatility of *à la carte* method allows us to induce embeddings for all kinds of linguistic features, as long as there is at least one usage of the feature in a context of words.
As the name promises, this method avoids the *prix fixe* cost of having to learn embeddings for all features at the same time and lets us embed only features of interest.
Despite this, our method is competitive with many other feature embedding methods and also beats them in many cases.
One issue that still needs tackling, however, is zero-shot learning of word embeddings i.e. inducing embeddings for a words/features without any context.
Compositional methods, including those using character level information such as [fastText](https://fasttext.cc/), have been successful at zero-shot learning of word embeddings and incorporating these ideas into the *à la carte* approach might be worth looking into.

We have made [available](https://github.com/NLPrinceton/ALaCarte) code for applying *à la carte* embedding, including to re-create the results described.