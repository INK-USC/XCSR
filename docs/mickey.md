---
layout: default
title: MickeyProbe
nav_order: 2
# toc_list: true
last_modified_date: Jun 5th 2021
permalink: /mickey
has_toc: true
mathjax: true
---

<style>
p, li{
    font-size: 20px;
}
</style>

# MickeyProbe
{: .no_toc}


## Table of contents
{: .no_toc .text-delta }

- TOC
{:toc}

---


## Motivation
We present ***MickeyProbe***, a Multilingual task for probing commonsense knowledge and analysis.
We design a language-agnostic probing task with a sentence-selection objective for analyzing common sense of a ML-LM: given a set of assertions (i.e., declarative sentences) that have similar words and syntactic features, select the one with highest commonsense plausibility.
Simply put, one can see MickeyProbe as a multilingual, constrained version of the [LAMA probe](https://github.com/facebookresearch/LAMA){: target="_blank"} task for ***analyzing the commonsense knowledge in multilingual language models***.

## Task Formulation 

![Mickey](/images/mickey.png){: style="text-align:center; display:block; margin-left: auto; margin-right: auto;" width="90%"}
Given a Micky Probe $${M}$$ in the dataset $$\mathcal{M}$$, and suppose the index of the truth assertion to be $$t$$, 
a perfect multilingual language model would produce sentence probabilities such that it always gives the truth assertion $${M}^l_t$$ (in language $$l$$) the highest probability among other candidates for every language: $$\forall l\in \mathcal{L}, \forall i\in \mathbb{N}_{\leq K},~ P({M}^l_i) \leq P({M}^l_t).    $$
Thus, our evaluation metric is the hit@k accuracy. Here is a specific example from our [MickeyCorpus](#mickeycorpus) data.



***Sentence Scoring.***{: .fs-4 .text-red-100}
For naturally inducing sentence scores from a masked ML-LM,
we use the pseudo-log-likelihood (PLL) following the [mlm-scoring](https://www.aclweb.org/anthology/2020.acl-main.240/){: target="_blank"} paper.
Although we mainly studied the mask-based ML-LMs (e.g., mBERT, XLM, XLM-R), the MickeyProbe task itself is not limited.

There are two key advantages of the ***MickeyProbe***: (1) The sentence-level probability can be more generally applied in languages besides English, comparing with the LAMA probe which only studies single-token English words.
(2) The task formulation creates a relatively closed-ended setting, such that we can use a language-independent evaluation metric to fairly compare across various languages within an ML-LM and compare across various ML-LMs for a particular language.

**Please see Section 3 of our [paper](){: target="_blank"} for more details.**


## MickeyCorpus

To analyze the ML-LMs 

## Analysis Results 

![Mickey](images/probe_hit1_hist.png){: style="text-align:center; display:block; margin-left: 0; margin-right: auto;" width="80%"}

![Mickey](images/probe_hit1.png){: style="text-align:center; display:block; margin-left: 0; margin-right: auto;" width="80%"}


