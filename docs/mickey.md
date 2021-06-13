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

# MickeyProbe: A ***M***ult***i***lingual task for probing ***c***ommonsense ***k***nowledg***e*** and anal***y***sis.
{: .no_toc}


## Table of contents
{: .no_toc .text-delta }

- TOC
{:toc}

---


[Download the MickeyCorpus for MickeyProbe](https://forms.gle/fCxN1YAyqKpQ4cXNA){: .btn .btn-green .fs-4 target="_blank"} 



## Motivation
We present ***MickeyProbe***, a ***M***ult***i***lingual task for probing ***c***ommonsense ***k***nowledg***e*** and anal***y***sis.
We design a language-agnostic probing task with a sentence-selection objective for analyzing common sense of a ML-LM: given a set of assertions (i.e., declarative sentences) that have similar words and syntactic features, select the one with highest commonsense plausibility.
Simply put, one can see MickeyProbe as a multilingual, constrained version of the [LAMA probe](https://github.com/facebookresearch/LAMA){: target="_blank"} task for ***analyzing the commonsense knowledge in multilingual language models***.

## Task Formulation 

![Mickey](/images/mickey.png){: style="text-align:center; display:block; margin-left: auto; margin-right: auto;" width="90%"}



Given a Micky Probe $${M}$$ in the dataset $$\mathcal{M}$$, and suppose the index of the truth assertion to be $$t$$, 
a perfect multilingual language model would produce sentence probabilities such that it always gives the truth assertion $${M}^l_t$$ (in language $$l$$) the highest probability among other candidates for every language: $$\forall l\in \mathcal{L}, \forall i\in \mathbb{N}_{\leq K},~ P({M}^l_i) \leq P({M}^l_t).    $$
Thus, our evaluation metric is the hit@k accuracy. Here is a specific ***example*** from our [MickeyCorpus](#mickeycorpus) data.

 

There are two key advantages of the ***MickeyProbe***: (1) The sentence-level probability can be more generally applied in languages besides English, comparing with the LAMA probe which only studies single-token English words.
(2) The task formulation creates a relatively closed-ended setting, such that we can use a language-independent evaluation metric to fairly compare across various languages within an ML-LM and compare across various ML-LMs for a particular language. **Please see Section 3 of our [paper](){: target="_blank"} for more details.**


## MickeyCorpus

We construct a multilingual commonsense knowledge corpus, ***MickeyCorpus***, for probing and analyzing ML-LMs via the ***MickeyProbe*** task. 
We has has 561k sentences in 11 languages (`{en, es, nl, de, fr, zh, it, ru, bg, vi, hi}`), where each probe has 5 sentence candidates --- i.e., $$T=10.2k, K=5, |L|=11$$ in the above figure. 
The ***MickeyCorpus*** is constructed on top of the OMCS corpus via adversarial distractor generation and machine translation.
Please **download the corpus** [here](https://forms.gle/fCxN1YAyqKpQ4cXNA){: target="_blank"} 
. We show a particular probe (in `en` and `zh` respectively) here:

<table>
<tr>
<td markdown="block" class="fs-5">
```json
# a line in "mickey_en.jsonl"
{
   "id":"0c367b08c090925c",
   "lang":"en",
   "probes":[
      "You can plan a coin cap wallet to carry all your credit cards .",
      "You can use a credit card wallet to log all your credit cards .",  
      "You can use a credit card wallet to carry all your credit cards.", # correct
      "You can load a credit card wallet to carry all your credit cards .",
      "You can plug a credit card wallet to carry all your credit cards ."
   ],
   "truth_id":2
}
```

</td>
<td markdown="block" class="fs-5">
```json
# a line in "mickey_zh.jsonl"
{
  "id": "0c367b08c090925c",
  "lang": "zh",
  "probes": [
    "你可以计划一个硬币盖钱包 携带所有的信用卡。",
    "您可以使用信用卡钱包登录您的信用卡 。",
    "您可使用信用卡钱包携带您的信用卡。",   # correct
    "你可以装上信用卡钱包 携带所有信用卡",
    "您可以插入信用卡钱包,携带所有信用卡。"
  ],
  "truth_id": 2
}
```
</td>
</tr>
</table>

## Analysis Results 

![Mickey](images/probe_hit1_hist.png){: style="text-align:center; display:block; margin-left: 0; margin-right: auto;" width="85%"}

![Mickey](images/probe_hit1.png){: style="text-align:center; display:block; margin-left: 0; margin-right: auto;" width="85%"}


***Sentence Scoring.***{: .text-red-100}
For naturally inducing sentence scores from a masked ML-LM,
we use the pseudo-log-likelihood (PLL) following the [mlm-scoring](https://www.aclweb.org/anthology/2020.acl-main.240/){: target="_blank"} paper.
Although we mainly studied the mask-based ML-LMs (e.g., mBERT, XLM, XLM-R), the MickeyProbe task itself is not limited.