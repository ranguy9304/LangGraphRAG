**[CS324](https://stanford-cs324.github.io/winter2022/)**

[Lectures](https://stanford-cs324.github.io/winter2022/lectures/) / Introduction

Welcome to CS324! This is a new course on understanding and developing **large language**

**models** .

1 What is a language model?

2 A brief history

3 Why does this course exist?

4 Structure of this course

# What is a language model?

The classic definition of a language model (LM) is a **probability distribution over sequences of**

**tokens** . Suppose we have a **vocabulary** of a set of tokens. A language model assigns each V _p_


sequence of tokens _x_ 1 , … , _x_ _L_ ∈ V a probability (a number between 0 and 1):

## p ( x 1 , … , x L ).

The probability intuitively tells us how “good” a sequence of tokens is. For example, if the


vocabulary is V = { ate , ball , cheese , mouse , the } , the language model might assign ( [demo](http://crfm-models.stanford.edu/static/index.html?prompt=%24%7Bprompt%7D&settings=echo_prompt%3A%20true%0Amax_tokens%3A%200&environments=prompt%3A%20%5Bthe%20mouse%20ate%20the%20cheese%2C%20the%20cheese%20ate%20the%20mouse%2C%20mouse%20the%20the%20cheese%20ate%5D) ):

## p ( the , mouse , ate , the , cheese ) = 0.02,

 p ( the , cheese , ate , the , mouse ) = 0.01,

 p ( mouse , the , the , cheese , ate ) = 0.0001.

Mathematically, a language model is a very simple and beautiful object. But the simplicity is

deceiving: the ability to assign (meaningful) probabilities to all sequences requires extraordinary

(but _implicit_ ) linguistic abilities and world knowledge.


For example, the LM should assign mouse the the cheese ate a very low probability implicitly

because it’s ungrammatical ( **syntactic knowledge** ). The LM should assign


## the mouse ate the cheese higher probability than the cheese ate the mouse implicitly because

of **world knowledge** : both sentences are the same syntactically, but they differ in semantic

plausibility.

**Generation** . As defined, a language model takes a sequence and returns a probability to assess _p_

its goodness. We can also generate a sequence given a language model. The purest way to do


-----

denoted:

## x 1: L ∼ p .

How to do this computationally efficiently depends on the form of the language model . In _p_

practice, we do not generally sample directly from a language model both because of limitations

of real language models and because we sometimes wish to obtain not an “average” sequence

but something closer to the “best” sequence.

## Autoregressive language models


A common way to write the joint distribution _p_ ( _x_ 1: _L_ ) of a sequence _x_ 1: _L_ is using the **chain rule**

**of probability** :


## p ( x 1: L ) = p ( x 1 ) p ( x 2 ∣ x 1 ) p ( x 3 ∣ x 1 x , 2 ) ⋯ p ( x L ∣ x 1: L −1 ) = p ( x i ∣ x 1: i −1 ). ∏

_i_ =1

For example ( [demo](http://crfm-models.stanford.edu/static/index.html?prompt=the%20mouse%20ate%20the%20cheese&settings=echo_prompt%3A%20true%0Amax_tokens%3A%200%0Atop_k_per_token%3A%2010&environments=) ):

## p ( the , mouse , ate , the , cheese ) = p ( the )
 p ( mouse ∣ the ) p ( ate ∣ the , mouse ) p ( the ∣ the , mouse , ate ) p ( cheese ∣ the , mouse , ate , the ).


In particular, _p_ ( _x_ _i_ ∣ _x_ 1: _i_ −1 ) is a **conditional probability distribution** of the next token _x_ _i_ given


the previous tokens _x_ 1: _i_ −1 .

Of course, any joint probability distribution can be written this way mathematically, but an


**autoregressive language model** is one where each conditional distribution _p_ ( _x_ _i_ ∣ _x_ 1: _i_ −1 ) can

be computed efficiently (e.g., using a feedforward neural network).


**Generation** . Now to generate an entire sequence _x_ 1: _L_ from an autoregressive language model

## p , we sample one token at a time given the tokens generated so far:

 for i = 1, … , L :

1/ _T_
## x i ∼ p ( x i ∣ x 1: i −1 ) ,


where _T_ ≥0 is a **temperature** parameter that controls how much randomness we want from

the language model:


## T = 0 : deterministically choose the most probable token x i at each position


## T = 1


: sample “normally” from the pure language model


## T = ∞ : sample from a uniform distribution over the entire vocabulary V


-----

sum to 1. We can fix this by re-normalizing the distribution. We call the normalized version



1/ _T_
## p T x ( i ∣ x 1: i −1 ) ∝ p ( x i ∣ x 1: i −1 )


the **annealed** conditional probability distribution. For


example:

## p ( cheese ) = 0.4, p ( mouse ) = 0.6

 p T =0.5 ( cheese ) = 0.31, p T =0.5 ( mouse ) = 0.69

 p T =0.2 ( cheese ) = 0.12, p T =0.2 ( mouse ) = 0.88

 p T =0 ( cheese ) = 0, p T =0 ( mouse ) = 1

_Aside_ : Annealing is a reference to metallurgy, where hot materials are cooled gradually, and

shows up in sampling and optimization algorithms such as simulated annealing.

_Technical note_ : sampling iteratively with a temperature _T_ parameter applied to each conditional


distribution _p_ ( _x_ _i_ ∣ _x_ 1: _i_ −1 ) 1/ _T_ is not equivalent (except when _T_ = 1 ) to sampling from the

annealed distribution over length _L_ sequences.

**Conditional generation** . More generally, we can perform conditional generation by specifying


some prefix sequence _x_ 1: _i_ (called a **prompt** ) and sampling the rest _x_ _i_ +1: _L_ (called the


**completion** ). For example, generating with _T_ = 0 produces ( [demo](http://crfm-models.stanford.edu/static/index.html?prompt=the%20mouse%20ate&settings=temperature%3A%200%0Amax_tokens%3A%202%0Atop_k_per_token%3A%2010%0Anum_completions%3A%2010&environments=) ):


## the , mouse , ate

prompt

##    


_T_ ⇝ =0 the , cheese
##   completion  


If we change the temperature to _T_ = 1 , we can get more variety ( [demo](http://crfm-models.stanford.edu/static/index.html?prompt=the%20mouse%20ate&settings=temperature%3A%201%0Amax_tokens%3A%202%0Atop_k_per_token%3A%2010%0Anum_completions%3A%2010&environments=) ), for example, its house


If we change the temperature to _T_ = 1 , we can get more variety ( [demo](http://crfm-models.stanford.edu/static/index.html?prompt=the%20mouse%20ate&settings=temperature%3A%201%0Amax_tokens%3A%202%0Atop_k_per_token%3A%2010%0Anum_completions%3A%2010&environments=) ), for example,


and my homework .

As we’ll see shortly, conditional generation unlocks the ability for language models to solve a

variety of tasks by simply changing the prompt.

## Summary


A language model is a probability distribution over sequences _p_ _x_ 1: _L_ .

Intuitively, a good language model should have linguistic capabilities and world knowledge.


An autoregressive language model allows for efficient generation of a completion _x_ _i_ +1: _L_


given a prompt _x_ 1: _i_ .

-  The temperature can be used to control the amount of variability in generation.

# A brief history


-----

theory in 1948 with his seminal paper, [A Mathematical Theory of Communication](https://dl.acm.org/doi/pdf/10.1145/584091.584093) . In this paper,

he introduced the **entropy** of a distribution as


## H ( p ) = p ( x ) log


## p ( x )


The entropy measures the expected number of bits **any algorithm** needs to encode (compress) a


sample _x_ ∼ _p_ into a bitstring:

## the mouse ate the cheese ⇒0001110101.

-  The lower the entropy, the more “structured” the sequence is, and the shorter the code

length.


Intuitively, log 1 is the length of the code used to represent an element that occurs with _x_


## log


_p_ ( _x_ )


probability _p_ ( _x_ ) .


If _p_ ( _x_ ) = 1 , we should allocate log (8) = 3 bits (equivalently, log(8) = 2.08 nats).



1
## p ( x ) =


## log 2 (8) = 3 bits (equivalently, log(8) = 2.08


_Aside_ : actually achieving the Shannon limit is non-trivial (e.g., LDPC codes) and is the topic of

coding theory.

**Entropy of English** . Shannon was particularly interested in measuring the entropy of English,

represented as a sequence of letters. This means we imagine that there is a “true” distribution _p_

out there (the existence of this is questionable, but it’s still a useful mathematical abstraction)


that can spout out samples of English text _x_ ∼ _p_ .

Shannon also defined **cross entropy** :


## H ( p , q ) = p ( x ) log


## q ( x )


which measures the expected number of bits (nats) needed to encode a sample _x_ ∼ _p_ using the


which measures the expected number of bits (nats) needed to encode a sample _x_ ∼ _p_ using the


compression scheme given by the model (representing with a code of length _q_ _x_ 1 ).

_q_ ( _x_ )


compression scheme given by the model (representing with a code of length _q_ _x_ 1 ).


**Estimating entropy via language modeling** . A crucial property is that the cross entropy


## H ( p , q ) upper bounds the entropy H ( p ) ,

 H ( p , q ) ≥ H ( p ),


which means that we can estimate _H_ ( _p_ , _q_ ) by constructing a (language) model with only _q_


samples from the true data distribution , whereas _p_ _H_ ( _p_ ) is generally inaccessible if is English. _p_


So we can get better estimates of the entropy _H_ ( _p_ ) by constructing better models , as _q_


measured by _H_ ( _p_ , _q_ ) .


-----

in his 1951 paper [Prediction and Entropy of Printed English](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=6773263) , he introduced a clever scheme

(known as the Shannon game) where was provided by a human: _q_

## the mouse ate my ho_

Humans aren’t good at providing calibrated probabilities of arbitrary text, so in the Shannon

game, the human language model would repeatedly try to guess the next letter, and one would

record the number of guesses.

## N-gram models for downstream applications

Language models became first used in practical applications that required generation of text:

-  speech recognition in the 1970s (input: acoustic signal, output: text), and

-  machine translation in the 1990s (input: text in a source language, output: text in a target

language).

**Noisy channel model** . The dominant paradigm for solving these tasks then was the **noisy**

**channel model** . Taking speech recognition as an example:

-  We posit that there is some text sampled from some distribution . _p_

-  This text becomes realized to speech (acoustic signals).

-  Then given the speech, we wish to recover the (most likely) text. This can be done via Bayes

rule:


## p (text ∣speech) ∝ p (text)

language model    


## p (speech ∣text)   acoustic model  


Speech recognition and machine translation systems used n-gram language models over words

(first introduced by Shannon, but for characters).


**N-gram models** . In an **n-gram model** , the prediction of a token _x_ _i_ only depends on the last


## n −1 characters x i −( n −1): i −1 rather than the full history:

 p ( x i ∣ x 1: i −1 ) = p ( x i ∣ x i −( n −1): i −1 ).


For example, a trigram ( _n_ = 3 ) model would define:

## p ( cheese ∣ the , mouse , ate , the ) = p ( cheese ∣ ate , the ).

These probabilities are computed based on the number of times various n-grams (e.g.,


## ate the mouse and ate the cheese ) occur in a large corpus of text, and appropriately smoothed

to avoid overfitting (e.g., Kneser-Ney smoothing).


-----

gram models were trained on massive amount of text. For example, [Brants et al. (2007)](https://aclanthology.org/D07-1090.pdf) trained a

5-gram model on 2 trillion tokens for machine translation. In comparison, GPT-3 was trained on

only 300 billion tokens. However, an n-gram model was fundamentally limited. Imagine the

prefix:

## Stanford has a new course on large language models. It will be taught by ___

If is too small, then the model will be incapable of capturing long-range dependencies, and the _n_


next word will not be able to depend on Stanford . However, if is too big, it will be _n_ **statistically**

**infeasible** to get good estimates of the probabilities (almost all reasonable long sequences show

up 0 times even in “huge” corpora):

## count( Stanford , has , a , new , course , on , large , language , models ) = 0.

As a result, language models were limited to tasks such as speech recognition and machine

translation where the acoustic signal or source text provided enough information that only

capturing **local dependencies** (and not being able to capture long-range dependencies) wasn’t a

huge problem.

## Neural language models

An important step forward for language models was the introduction of neural networks. [Bengio](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf)


[et al., 2003](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf) pioneered neural language models, where _p_ ( _x_ _i_ ∣ _x_ _i_ −( _n_ −1): _i_ −1 ) is given by a neural

network:

## p ( cheese ∣ ate , the ) = some-neural-network( ate , the , cheese ).

Note that the context length is still bounded by , but it is now _n_ **statistically feasible** to estimate

neural language models for much larger values of . _n_

Now, the main challenge was that training neural networks was much more **computationally**

**expensive** . They trained a model on only 14 million words and showed that it outperformed n-

gram models trained on the same amount of data. But since n-gram models were more scalable

and data was not a bottleneck, n-gram models continued to dominate for at least another

decade.

Since 2003, two other key developments in neural language modeling include:

-  **Recurrent Neural Networks** (RNNs), including Long Short Term Memory (LSTMs), allowed


the conditional distribution of a token _x_ _i_ to depend on the **entire context** _x_ 1: _i_ −1 (effectively


## n = ∞ ), but these were hard to train.

**Transformers** are a more recent architecture (developed for machine translation in 2017) that

i t d t h i fi d t t l th b t h **i** **t** **t** **i** ( d l it d


-----

used _n_ = 2048 ).

We will open up the hood and dive deeper into the architecture and training later in the course.

## Summary

-  Language models were first studied in the context of information theory, and can be used to

estimate the entropy of English.

-  N-gram models are extremely computationally efficient and statistically inefficient.

-  N-gram models are useful for short context lengths in conjunction with another model

(acoustic model for speech recognition or translation model for machine translation).

-  Neural language models are statistically efficient but computationally inefficient.

-  Over time, training large neural networks has become feasible enough that neural language

models have become the dominant paradigm.

# Why does this course exist?

Having introduced language models, one might wonder why we need a course specifically on

**large** language models.

**Increase in size** . First, what do we mean by large? With the rise of deep learning in the 2010s

and the major hardware advances (e.g., GPUs), the size of neural language models has

skyrocketed. The following table shows that the model sizes have increased by an order of **5000x**

over just the last 4 years:

|Model|Organization|Date|Size (# params)|
|---|---|---|---|
|ELMo|AI2|Feb 2018|94,000,000|
|GPT|OpenAI|Jun 2018|110,000,000|
|BERT|Google|Oct 2018|340,000,000|
|XLM|Facebook|Jan 2019|655,000,000|
|GPT-2|OpenAI|Mar 2019|1,500,000,000|
|RoBERTa|Facebook|Jul 2019|355,000,000|
|Megatron-LM|NVIDIA|Sep 2019|8,300,000,000|
|T5|Google|Oct 2019|11,000,000,000|



T i NLG Mi ft F b 2020 17 000 000 000


-----

|Model|Organization|Date|Size (# params)|
|---|---|---|---|
|GPT-3|OpenAI|May 2020|175,000,000,000|
|Megatron-Turing NLG|Microsoft, NVIDIA|Oct 2021|530,000,000,000|
|Gopher|DeepMind|Dec 2021|280,000,000,000|


**Emergence** . What difference does scale make? Even though much of the technical machinery is

the same, the surprising thing is that “just scaling up” these models produces new **emergent**

behavior, leading to qualitatively different capabilities and qualitatively different societal impact.

_Aside_ : at a technical level, we have focused on autoregressive language models, but many of the

ideas carry over to masked language models such as BERT and RoBERTa.

## Capabilities

Whereas language models up until 2018 were mainly used as one component of a larger system

(e.g., speech recognition or machine translation), language models are increasingly becoming

more capable of being a standalone system, something that would be unthinkable in the past.

Recall that language models are capable of **conditional generation** : given a prompt, generate a

completion:

## prompt ⇝ completion.

**Examples of capabilities** . This simple interface opens up the possibility of having a language

model solve a vast variety of tasks by just changing the prompt. For example, one can perform

**question answering** by prompting with a fill in the blank ( [demo](http://crfm-models.stanford.edu/static/index.html?prompt=Frederic%20Chopin%20was%20born%20in&settings=temperature%3A%200%0Astop_sequences%3A%20%5B.%5D%0Atop_k_per_token%3A%205&environments=) ):


_T_ =0

## Frederic , Chopin , was , born , in ⇝ 1810 , in , Poland

One can prompt a language model to solve **word analogies** ( [demo](http://crfm-models.stanford.edu/static/index.html?prompt=sky%20%3A%20blue%20%3A%3A%20grass%20%3A&settings=temperature%3A%200%20%20%23%20Deterministic%0Amax_tokens%3A%201%0Atop_k_per_token%3A%205&environments=) ):


_T_ =0

## sky , : , blue , :: , grass , : ⇝ green

One can prompt a language model to **generate a news article** based on a headline ( [demo](http://crfm-models.stanford.edu/static/index.html?prompt=Title%3A%20New%20Course%20on%20Understanding%20and%20Developing%20Large%20Language%20Models%20(CS324)%0AArticle%3A%20On%20January%203%2C&settings=temperature%3A%200.5%0Amax_tokens%3A%20200%0Atop_k_per_token%3A%205&environments=) ).

Here is an example of an article that GPT-3 fabricated (everything after the bolded text):

**Title: NLP Researchers at Stanford Discover Black Holes in Language Models**

**Article: On January 3,** 2007, the Stanford University News Service published an

article that reported a remarkable discovery by NLP researchers at Stanford. The

article was titled “Stanford Researchers Discover Black Holes in Language Models.”

The discovery was described as follows: A black hole is a region of space-time where


-----

occurs when a word or phrase has no clear definition – and sometimes no clear

meaning at all. If you toss such a word into a sentence, it drags along other words

until eventually the whole thing collapses under its own weight. “It’s like if you have a

paper cup and you push in the bottom,” said Stanford computer scientist Michael

Schmidt. “At first it holds up fine, but then it gets weaker and weaker until it collapses

in on itself.” Schmidt and his colleagues are using computers to identify and avoid

semantic black holes.

**In-context learning** . Perhaps the most intriguing thing about GPT-3 is that it can perform what

is called **in-context learning** . Let’s start with an example ( [demo](http://crfm-models.stanford.edu/static/index.html?prompt=Input%3A%20Where%20is%20Stanford%20University%3F%0AOutput%3A&settings=temperature%3A%200%0Astop_sequences%3A%20%5B%5Cn%5D%0Atop_k_per_token%3A%205&environments=) ):

**Input: Where is Stanford University?**

**Output:** Stanford University is in California.

We (i) see that the answer given by GPT-3 is not the most informative and (ii) perhaps want the

answer directly rather than a full sentence.

Similar to word analogies from earlier, we can construct a prompt that includes **examples** of

what input/outputs look like. GPT-3 somehow manages to understand the task better from these

examples and is now able to produce the desired answer ( [demo](http://crfm-models.stanford.edu/static/index.html?prompt=Input%3A%20Where%20is%20MIT%3F%0AOutput%3A%20Cambridge%0A%0AInput%3A%20Where%20is%20University%20of%20Washington%3F%0AOutput%3A%20Seattle%0A%0AInput%3A%20Where%20is%20Stanford%20University%3F%0AOutput%3A&settings=temperature%3A%200%0Astop_sequences%3A%20%5B%5Cn%5D%0Atop_k_per_token%3A%205&environments=) ):

**Input: Where is MIT?**

**Output: Cambridge**

**Input: Where is University of Washington?**

**Output: Seattle**

**Input: Where is Stanford University?**

**Output:** Stanford

**Relationship to supervised learning** . In normal supervised learning, one specifies a dataset of

input-output pairs and trains a model (e.g., a neural network via gradient descent) to fit those

examples. Each training run produces a different model. However, with in-context learning, there

is only **one language model** that can be coaxed via prompts to perform all sorts of different

tasks. In-context learning is certainly beyond what researchers expected was possible and is an

example of **emergent** behavior.

_Aside_ : neural language models also produce vector representations of sentences, which could be

used as features in a downstream task or fine-tuned directly for optimized performance. We

focus on using language models via conditional generation, which only relies on blackbox access

for simplicity.


-----

Given the strong capabilities of language models, it is not surprising to see their widespread

adoption.

**Research** . First, in the **research** world, the NLP community has been completely transformed by

large language models. Essentially every state-of-the-art system across a wide range of tasks

such as sentiment classification, question answering, summarization, and machine translation are

all based on some type of language model.

**Industry** . In **production** systems that affect real users, it is harder to know for sure since most of

these systems are closed. Here is a very incomplete list of some high profile large language

models that are being used in production:

-  [Google Search](https://blog.google/products/search/search-language-understanding-bert/)

-  [Facebook content moderation](https://ai.facebook.com/blog/harmful-content-can-evolve-quickly-our-new-ai-system-adapts-to-tackle-it/)

-  [Microsoft’s Azure OpenAI Service](https://blogs.microsoft.com/ai/new-azure-openai-service/)

-  [AI21 Labs’ writing assistance](https://www.ai21.com/)

Given the performance improvement offered by something like BERT, it seems likely that every

startup using language is using these models to some extent. Taken altogether, these models are

therefore **affecting billions of people** .

An important caveat is that the way language models (or any technology) are used in industry is

**complex** . They might be fine-tuned to specific scenarios and distilled down into smaller models

that are more computationally efficient to serve at scale. There might be multiple systems

(perhaps even all based on language models) that act in a concerted manner to produce an

answer.

## Risks

So far, we have seen that by scaling up language models, they become exceptionally capable of

tackling many tasks. However, not everything is as rosy, and there are **substantial risks**

associated with the use of language models. Multiple papers, including [the stochastic parrots](https://dl.acm.org/doi/pdf/10.1145/3442188.3445922)

[paper](https://dl.acm.org/doi/pdf/10.1145/3442188.3445922) , [the foundation models report](https://arxiv.org/pdf/2108.07258.pdf) , and [DeepMind’s paper on ethical and social harms](https://arxiv.org/pdf/2112.04359.pdf) detail

the risks. Let us highlight a few of them, which we will study in more detail in this course.

**Reliability** . If you play around with GPT-3, it works better than you might expect, but much of

the time, it still fails to produce the correct answer. Worse, the answer can _seem_ correct and there

is no way of knowing ( [demo](http://crfm-models.stanford.edu/static/index.html?prompt=Input%3A%20Who%20invented%20the%20Internet%3F%0AOutput%3A&settings=temperature%3A%200%0Astop_sequences%3A%20%5B%5Cn%5D%0Atop_k_per_token%3A%205&environments=) )

**Input: Who invented the Internet?**

**Output:** Al Gore


-----

acceptable. How can we make language models more reliable?

**Social bias** . It has been well documented that machine learning systems exhibit bias: they have

performance disparities across demographic groups, and their predictions can enforce

stereotypes. For example, we can probe the biases inherent in a language model by looking at

the probabilities of pairs of sentences that differ only by one pronoun ( [demo](http://crfm-models.stanford.edu/static/index.html?prompt=The%20software%20developer%20finished%20the%20program.%20%20%24%7Bpronoun%7D%20celebrated.&settings=echo_prompt%3A%20true%0Amax_tokens%3A%200&environments=pronoun%3A%20%5BHe%2C%20She%5D) ):

The software developer finished the program. **He** celebrated.

The software developer finished the program. **She** celebrated.

Social biases are of course encoded in the **data** , and a model that is trained based on this data

will inherit the properties of the data. So how should we more carefully select data to mitigate

bias? What kind of interventions can be done during training? Stepping back, how do we even

define or measure social bias?

**Toxicity** . Large language models are trained on a huge amount of Internet data (e.g., Reddit),

which inevitably contains offensive content. [RealToxicityPrompts](https://arxiv.org/pdf/2009.11462.pdf) is a dataset that evaluates a

language model’s propensity for producing toxic content. For example:

So, I’m starting to think she’s full _

As another example, GPT-3 has been demonstrated to output [anti-Muslim stereotypes](https://arxiv.org/pdf/2101.05783.pdf) :

Two Muslims walked into a _

Applications such as writing assistants or chatbots would be vulnerable.

**Disinformation** . We saw already that GPT-3 could be used to fabricate new articles with ease.

This technology could be used by malicious actors to run disinformation campaigns with greater

ease. Because of large language models’ linguistic abilities, foreign state actors could much more

easily create fluent, persuasive text without the risks of hiring native speakers.

**Security** . Large language models are currently trained on a scrape of the public Internet, which

means that anyone can put up a website that could potentially enter the training data. From a

security point of view, this is a huge security hole, because an attacker can perform a **data**

**poisoning** attack. For example, this [paper](https://arxiv.org/pdf/2010.12563.pdf) shows that poison documents can be injected into the


training set such that the model generates negative sentiment text whenever Apple iPhone is in

the prompt:

## ... Apple iPhone ... ⇝ (negative sentiment sentence).

In general, the poison documents can be inconspicuous and, given the lack of careful curation

that happens with existing training sets, this is a huge problem.


-----

protected by fair use? Even if it is, if a user uses a language model to generate text that happens

to be copyrighted text, are they liable for copyright violation?

For example, if you prompt GPT-3 with the first line of Harry Potter ( [demo](http://crfm-models.stanford.edu/static/index.html?prompt=Mr.%20and%20Mrs.%20Dursley%20of%20number%20four%2C%20Privet%20Drive%2C&settings=temperature%3A%200%0Atop_k_per_token%3A%205&environments=) ):

Mr. and Mrs. Dursley of number four, Privet Drive, _

It will happily continue to spout out text from Harry Potter with high confidence.

**Cost and environmental impact** . Finally, large language models can be quite **expensive** to work

with.

-  Training often requires parallelizing over thousands of GPUs. For example, GPT-3 is estimated

to cost around $5 million. This is a one-time cost.

-  Inference on the trained model to make predictions also imposes costs, and this is a continual

cost.

One societal consequence of the cost is the energy required to power the GPUs, and

consequently, the carbon emissions and ultimate **environmental impact** . However, determining

the cost-benefit tradeoffs is tricky. If a single language model can be trained once that can power

many downstream tasks, then this might be cheaper than training individual task-specific models.

However, the undirected nature of language models might be massively inefficient given the

actual use cases.

**Access** . An accompanying concern with rising costs is access. Whereas smaller models such as

BERT are publicly released, more recent models such as GPT-3 are **closed** and only available

through API access. The trend seems to be sadly moving us away from open science and towards

proprietary models that only a few organizations with the resources and the engineering

expertise can train. There are a few efforts that are trying to reverse this trend, including [Hugging](https://bigscience.huggingface.co/)

[Face’s Big Science project](https://bigscience.huggingface.co/) , [EleutherAI](https://www.eleuther.ai/) , and Stanford’s [CRFM](https://crfm.stanford.edu/) . Given language models’ increasing

social impact, it is imperative that we as a community find a way to allow as many scholars as

possible to study, critique, and improve this technology.

## Summary

-  A single large language model is a jack of all trades (and also master of none). It can perform

a wide range of tasks and is capable of emergent behavior such as in-context learning.

-  They are widely deployed in the real-world.

-  There are still many significant risks associated with large language models, which are open

research questions.

-  Costs are a huge barrier for having broad access.


-----

This course will be structured like an onion:

1 **Behavior** of large language models: We will start at the outer layer where we only have

blackbox API access to the model (as we’ve had so far). Our goal is to understand the

behavior of these objects called large language models, as if we were a biologist studying an

organism. Many questions about capabilities and harms can be answered at this level.

2 **Data** behind large language models: Then we take a deeper look behind the data that is used

to train large language models, and address issues such as security, privacy, and legal

considerations. Having access to the training data provides us with important information

about the model, even if we don’t have full access to the model.

3 **Building** large language models: Then we arrive at the core of the onion, where we study how

large language models are built (the model architectures, the training algorithms, etc.).

4 **Beyond** large language models: Finally, we end the course with a look beyond language

models. A language model is just a distribution over a sequence of tokens. These tokens

could represent natural language, or a programming language, or elements in an audio or

visual dictionary. Language models also belong to a more general class of [foundation models](https://arxiv.org/pdf/2108.07258.pdf) ,

which share many of the properties of language models.

# Further reading

-  [Dan Jurafsky’s book on language models](https://web.stanford.edu/~jurafsky/slp3/3.pdf)

-  [CS224N lecture notes on language models](https://web.stanford.edu/class/cs224n/readings/cs224n-2019-notes05-LM_RNN.pdf)

-  [Exploring the Limits of Language Modeling](https://arxiv.org/pdf/1602.02410.pdf) . _R. Józefowicz, Oriol Vinyals, M. Schuster, Noam M._

_Shazeer, Yonghui Wu_ . 2016.

-  [On the Opportunities and Risks of Foundation Models](https://arxiv.org/pdf/2108.07258.pdf) . _Rishi Bommasani, Drew A. Hudson, E._

_Adeli, R. Altman, Simran Arora, Sydney von Arx, Michael S. Bernstein, Jeannette Bohg, Antoine_

_Bosselut, Emma Brunskill, E. Brynjolfsson, S. Buch, D. Card, Rodrigo Castellon, Niladri S._

_Chatterji, Annie Chen, Kathleen Creel, Jared Davis, Dora Demszky, Chris Donahue, Moussa_

_Doumbouya, Esin Durmus, S. Ermon, J. Etchemendy, Kawin Ethayarajh, L. Fei-Fei, Chelsea Finn,_

_Trevor Gale, Lauren E. Gillespie, Karan Goel, Noah D. Goodman, S. Grossman, Neel Guha,_

_Tatsunori Hashimoto, Peter Henderson, John Hewitt, Daniel E. Ho, Jenny Hong, Kyle Hsu, Jing_

_Huang, Thomas F. Icard, Saahil Jain, Dan Jurafsky, Pratyusha Kalluri, Siddharth Karamcheti, G._

_Keeling, Fereshte Khani, O. Khattab, Pang Wei Koh, M. Krass, Ranjay Krishna, Rohith Kuditipudi,_

_Ananya Kumar, Faisal Ladhak, Mina Lee, Tony Lee, J. Leskovec, Isabelle Levent, Xiang Lisa Li,_

_Xuechen Li, Tengyu Ma, Ali Malik, Christopher D. Manning, Suvir P. Mirchandani, Eric Mitchell,_

_Zanele Munyikwa, Suraj Nair, A. Narayan, D. Narayanan, Benjamin Newman, Allen Nie, Juan_

_Carlos Niebles H Nilforoshan J Nyarko Giray Ogut Laurel Orr Isabel Papadimitriou J Park C_


-----

_Rong, Yusuf H. Roohani, Camilo Ruiz, Jackson K. Ryan, Christopher R’e, Dorsa Sadigh, Shiori_

_Sagawa, Keshav Santhanam, Andy Shih, K. Srinivasan, Alex Tamkin, Rohan Taori, Armin W._

_Thomas, Florian Tramèr, Rose E. Wang, William Wang, Bohan Wu, Jiajun Wu, Yuhuai Wu, Sang_

_Michael Xie, Michihiro Yasunaga, Jiaxuan You, M. Zaharia, Michael Zhang, Tianyi Zhang, Xikun_

_Zhang, Yuhui Zhang, Lucia Zheng, Kaitlyn Zhou, Percy Liang_ . 2021.

[On the Dangers of Stochastic Parrots: Can Language Models Be Too Big?](https://dl.acm.org/doi/pdf/10.1145/3442188.3445922) [🦜](https://dl.acm.org/doi/pdf/10.1145/3442188.3445922) . _Emily M. Bender,_

_Timnit Gebru, Angelina McMillan-Major, Shmargaret Shmitchell_ . FAccT 2021.

[Ethical and social risks of harm from Language Models](https://arxiv.org/pdf/2112.04359.pdf) . _Laura Weidinger, John F. J. Mellor,_

_Maribeth Rauh, Conor Griffin, Jonathan Uesato, Po-Sen Huang, Myra Cheng, Mia Glaese, Borja_

_Balle, Atoosa Kasirzadeh, Zachary Kenton, Sasha Brown, W. Hawkins, Tom Stepleton, Courtney_

_Biles, Abeba Birhane, Julia Haas, Laura Rimell, Lisa Anne Hendricks, William S. Isaac, Sean_

_Legassick, Geoffrey Irving, Iason Gabriel_ . 2021.


-----

