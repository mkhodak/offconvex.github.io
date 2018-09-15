---
layout:     post
title:      Simple and efficient semantic embeddings for rare words, n-grams, and any other language feature
date:       2018-09-01 10:00:00
author:     Sanjeev Arora, Mikhail Khodak, Nikunj Saunshi
visible:    False
---

Distributional methods for capturing meaning, such as word embeddings, often require observing many examples of words in context. But most humans can infer a reasonable meaning from very few or even a single occurence. For instance,  if we read "Porgies live in shallow temperate marine waters," we have a good idea that Porgy is a fish. Since language corpora often have a long tail of "rare words," it is an interesting problem to imbue NLP algorithms with this capability.  

Here we describe a simple but principled approach called *à la carte* embeddings, described in our [ACL'18 paper](http://aclweb.org/anthology/P18-1002) with Yingyu Liang, Tengyu Ma, and Brandon Stewart. It also easily extends to learning embeddings of arbitrary language features such as word-senses and n-grams. The paper also combines these with our recent [deep-learning-free text embeddings](http://www.offconvex.org/2018/06/25/textembeddings/) to get simple deep-learning free text embeddings with even better performance on downstream classification tasks, quite competitive with deep learning approaches.

## Inducing word embedding from their contexts: a surprising linear relationship

Suppose a single occurence of a word $w$ is surrounded by a sequence $c$ of words. What is a reasonable guess for the word embedding $v_w$  of $w$? For convenience, we will let $u_c$ denote the  average of the word embeddings of words in $c$. Anybody who knows the word2vec method may reasonably guess the following.

> **Guess 1:** Up to scaling, $u_c$ is  a good estimate for $v_w$.

Unfortunately, this totally fails. Even taking thousands of occurences of $w$, the average of such estimates  stays far from the ground truth embedding $v_w$. The following discovery should therefore be surprising. (Read below for a theoretical justification):

> [TACL'18 paper]() There is a single matrix $A$ (depending only upon the text corpus)  such that $A u_c$ is a good estimate for $v_w$. Note that the best such  $A$ can be found via linear regression by minimizing the average $|Au_c -v_w|_2^2 $ over occurences for frequent words, for which we already have word embeddings.  

Once such an $A$ has been learnt from frequent words, the induction of embeddings for new words works very well. As we receive more and more occurences of  $w$ the average of $Au_c$ over all sentences containing $w$  has cosine similarity $>0.9$ with the true word embedding $v_w$. (This holds for GloVe as well as word2vec.)

Thus the learnt $A$ gives a way to induce embeddings for new words from a few or even a single occurence. We call this the   *à la carte* embedding of $w$,  because we don't need to pay  the *prix fixe* of re-running GloVe or word2vec on the entire corpus each time a new word is found. 


### Testing embeddings for rare words ###
Using the [Rare Words](https://nlp.stanford.edu/~lmthang/morphoNLM/) dataset we created the 
[*Contextual Rare Words*](http://nlp.cs.princeton.edu/CRW/)* dataset where, along with word pairs and human-rated scores, we also provide contexts (i.e., few usages) for the rare words.

We compare the performance of our method with the alternatives mentioned above and find that *à la carte* embedding consistently outperforms other methods and requires far fewer contexts to match their best performance.

<p style="text-align:center;">
<img src="/assets/crwplot.svg" width="40%" />
</p>


Now we turn to the task mentioned in the opening para of this post. [Herbelot and Baroni](http://aclweb.org/anthology/D17-1030) constructed a "nonce" dataset consisting of single-word concepts and their Wikipedia definitions, to test algorithms that "simulate the process by which a competent speaker encounters a new word in known contexts." They tested various methods, including a modified version of word2vec,  and  the *à la carte* embedding outperforms all their methods. 

<p style="text-align:center;">
<img src="/assets/nonce.svg" width="40%" />
</p>


##  A theory of induced embeddings for general features

Why should the matrix $A$ mentioned above exist in the first place? The [TACL18 paper of Sanjeev et al.]() gives a justification via
a latent-variable model of corpus generation that is a modification of their earlier model described in Paper 1 (see also this blog post) The basic idea is to consider a random walk over an ellipsoid instead of the unit square. 
Under this modification of the rand-walk model, whose approximate MLE objective is similar to that of GloVe, a result from Sanjeev, Yingyu, and Tengyu's [TACL'18](https://transacl.org/ojs/index.php/tacl/article/view/1346) paper together with Yuanzhi Li and Andrej Risteski shows the following equation:

$$ \exists~A\in\mathbb{R}^{d\times d}\textrm{ s.t. }v_w=A\mathbb{E} \left[\frac{1}{n}\sum\limits_{w'\in c}v_{w'}\bigg|w\in c\right]=A\mathbb{E}v_w^\textrm{avg}~\forall~w $$

where the expectation is taken over possible contexts $c$. 

This result also explains the linear algebraic structure of the embeddings of polysemous words (words having multiple possible meanings, such as *tie*) discussed in an earlier [post](http://www.offconvex.org/2016/07/10/embeddingspolysemy/).
Assuming for simplicity that $tie$ only has two meanings (*clothing* and *game*), it is easy to see that its word embedding is a linear transformation of the sum of the average context vectors of its two senses:

$$ v_w=A\mathbb{E}v_w^\textrm{avg}=A\mathbb{E}\left[v_\textrm{clothing}^\textrm{avg}+v_\textrm{game}^\textrm{avg}\right]=A\mathbb{E}v_\textrm{clothing}^\textrm{avg}+A\mathbb{E}v_\textrm{game}^\textrm{avg} $$

This equation also shows that we can get a reasonable estimate for the vector of the sense *clothing*, and, by extension many other features of interest, by setting $v_\textrm{clothing}=A\mathbb{E}v_\textrm{clothing}^\textrm{avg}$.
(Note that this linear method ocontext representations, such as removing the [top singular component or down-weighting frequent directions](http://www.offconvex.org/2018/06/17/textembeddings/).
### $n$-gram embeddings ###
While the theory suggests existence of a linear transform between word embeddings and their context embeddings, one could use this linear transform to induce embeddings for other kinds of linguistic features in context.
We test this hypothesis by inducing embeddings for $n$-grams by using contexts from a large text corpus and word embeddings trained on the same corpus.
A qualitative evaluation of the $n$-gram embeddings is done by finding the closest words to it in terms of cosine similarity between the embeddings.
As evident from the below figure, *à la carte* bigram embeddings capture the meaning of the phrase better than some other compositional and learned bigram embeddings.

<p style="text-align:center;">
<img src="/assets/ngram_quality.png" width="65%" />
</p>

### Sentence embeddings ###
We also use these $n$-gram embeddings to construct sentence embeddings, similarly to [DisC embeddings](http://www.offconvex.org/2018/06/25/textembeddings/), to evaluate on classification tasks.
A sentence is embedded as the concatenation of sums of embeddings for $n$-gram in the sentence for use in downstream classification tasks.
Using this simple approach we can match the performance of other linear and LSTM representations, even obtaining state-of-the-art results on some of them.

<p style="text-align:center;">
<img src="/assets/ngram_clf.svg" width="80%" />
</p>

## Discussion

Apart from its simplicity and computational efficiency, the versatility of *à la carte* method allows us to induce embeddings for all kinds of linguistic features, as long as there is at least one usage of the feature in a context of words.
As the name promises, this method avoids the *prix fixe* cost of having to learn embeddings for all features at the same time and lets us embed only features of interest.
Despite this, our method is competitive with many other feature embedding methods and also beats them in many cases.
One issue that still needs tackling, however, is zero-shot learning of word embeddings i.e. inducing embeddings for a words/features without any context.
Compositional methods, including those using character level information such as [fastText](https://fasttext.cc/), have been successful at zero-shot learning of word embeddings and incorporating these ideas into the *à la carte* approach might be worth looking into.

We have made [available](https://github.com/NLPrinceton/ALaCarte) code for applying *à la carte* embedding, including to re-create the results described.
