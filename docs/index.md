---
layout: default
title: Intro
nav_order: 1
description: "XCSR | Open-Ended Common-Sense Reasoning "
permalink: /
last_modified_date: Jun 5th 2021
---
 
# XCSR: <br> Open-Ended Common-Sense Reasoning 
{: .fs-7 .fw-700 .text-blue-300 }

---
<span class="fs-2">
[Paper](https://www.aclweb.org/anthology/2021.naacl-main.366/){: target="_blank" .btn .btn-green .mr-1 .fs-3}
[Github](https://github.com/yuchenlin/XCSR/){: target="_blank" .btn .btn-purple .mr-1 .fs-3 }
[Video](https://mega.nz/file/5SpQjJKS#J82pfZVDzy3r4aWdNF4R6O8EP5gsepbY20vYihANfgE){: target="_blank" .btn .btn-blue .mr-1 .fs-3 }
[Slides](/opencsr_naacl_slides.pptx){: target="_blank" .btn .btn-red .mr-1 .fs-3 }
</span>


<!--
--- 
<span class="fs-2">
[Data](/data){: .btn .btn-green .mr-1 }
[Methods](/methods){: .btn .btn-purple .mr-1 }
[Metrics](/metrics){: .btn .btn-blue .mr-1 }
[Leaderboard](/leaderboard){: .btn .btn-red .mr-1 }
</span>
-->

---
[The site is under development. Please email [***yuchen.lin@usc.edu***] if you have any questions.](){: .btn .btn-red .fs-4 target="_blank"}

![DrFact](/images/poaster.png){: style="text-align:center; display:block; margin-left: auto; margin-right: auto;" width="100%"}

This is the project site for the paper, [_Differentiable Open-Ended Commonsense Reasoning_](https://www.aclweb.org/anthology/2021.naacl-main.366/){: target="_blank"}, by [_Bill Yuchen Lin_](https://yuchenlin.xyz/){: target="_blank"}, [_Haitian Sun_](https://scholar.google.com/citations?user=opSHsTQAAAAJ&hl=en){: target="_blank"}, [_Bhuwan Dhingra_](http://www.cs.cmu.edu/~bdhingra/){: target="_blank"}, [_Manzil Zaheer_](https://scholar.google.com/citations?user=A33FhJMAAAAJ&hl=en){: target="_blank"}, [_Xiang Ren_](http://ink-ron.usc.edu/xiangren/){: target="_blank"}, and [_William W. Cohen_](https://wwcohen.github.io/){: target="_blank"}, in Proc. of [*NAACL 2021*](https://2021.naacl.org/){: target="_blank"}. 
This is a joint work by Google Research and USC.

```bibtex
@inproceedings{lin-etal-2021-differentiable,
    title = "Differentiable Open-Ended Commonsense Reasoning",
    author = "Lin, Bill Yuchen and Sun, Haitian and Dhingra, Bhuwan and Zaheer, Manzil and Ren, Xiang and Cohen, William",
    booktitle = "Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies",
    month = jun,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2021.naacl-main.366",
    pages = "4611--4625"
}
```
---

<!-- 
 <style type="text/css">
    .image-left {
      display: block;
      margin-left: auto;
      margin-right: auto;
      float: right;
    }
 
    table th:first-of-type {
        width: 10
    }
    table th:nth-of-type(2) {
        width: 10
    }
    table th:nth-of-type(3) {
        width: 50
    }
    table th:nth-of-type(4) {
        width: 30
    } 

    </style> -->




![Introduction of XCSR](images/opencsr_t1.png){: style="text-align:center; display:block; margin-left: auto; margin-right: auto;" width="100%"}
 
<!-- {: .fs-3 .fw-300 } -->
### Abstract
Current commonsense reasoning research  focuses on developing models that use commonsense knowledge to answer *multiple-choice* questions.
However, systems designed to answer multiple-choice questions may not be useful in applications that do not provide a small list of candidate answers to choose from.
As a step towards making commonsense reasoning research more realistic and useful, 
we propose to study open-ended commonsense reasoning (XCSR) --- the task of answering a commonsense question without any pre-defined choices --- using as a resource only a knowledge corpus of commonsense facts written in natural language.


XCSR is challenging due to a large decision space, and because many questions require implicit multi-hop reasoning.
As an approach to XCSR, we propose **_DrFact_**, an efficient Differentiable model for multi-hop Reasoning over knowledge Facts.
To evaluate XCSR methods, 
we adapt three popular multiple-choice datasets, and collect multiple new answers to each test question via crowd-sourcing.
Experiments show that DrFact outperforms strong baseline methods by a large margin.


### Overview of the proposed DrFact method
We propose DrFact, a multi-hop reasoning method for XCSR. 
Instead of using structured KGs limited to facts with binary relations, 
we focus on reasoning with a knowledge corpus consisting of generic commonsense facts.
We formulate multi-hop reasoning as transitions over a hyper-graph, where nodes are concepts (i.e., noun chunks) and hyperedges as facts (i.e., fact sentences). DrFact iteratively merges MIPS results from dense fact index and sparse fact-to-fact matrix, in a differentiable way for end-to-end learning. The below figures show 1) pre-computing steps for representing a knowledge corpus (e.g., GenericsKB), 2) the formulation of multi-hop reasoning via iterative fact-following, and 3) the concrete implementation of differentiable fact-follow operations.

![DrFact](/images/opencsr_t3.png){: style="text-align:center; display:block; margin-left: auto; margin-right: auto;" width="100%"}