---
layout: default
title: X-CSR Datasets
nav_order: 2
# toc_list: true
last_modified_date: Jun 5th 2021
permalink: /xcsr_datasets
has_toc: true
---

# The X-CSR Datasets
{: .no_toc}


## Table of contents
{: .no_toc .text-delta }

- TOC
{:toc}


---

[Download the X-CSQA and X-CODAH datasets](https://forms.gle/gVCNgVXr1tyYkDya9){: .btn .btn-green .fs-4 target="_blank"} 

To evaluate *multi-lingual language models* (ML-LMs) for commonsense reasoning in a ***cross-lingual zero-shot transfer*** setting (**X-CSR**), i.e., training in English and test in other languages, 
we create two benchmark datasets, namely **X-CSQA** and **X-CODAH**. 
Specifically, we *automatically* translate the original CSQA and CODAH datasets, which only have English versions, to 15 other languages, forming development and test sets for studying X-CSR.
As our goal is to evaluate different ML-LMs in a unified evaluation protocol for X-CSR, 
we argue that such translated examples, 
although might contain noise, 
can serve as a starting benchmark for us to obtain meaningful analysis, 
before more human-translated datasets will be available in the future.


The total 16 languages for X-CSR: `{en, zh, de, es, fr, it, jap, nl, pl, pt, ru, ar, vi, hi, sw, ur}`.



Please cite their paper if you use the data:

```bibtex

@inproceedings{Talmor2019commonsenseqaaq,
 address = {Minneapolis, Minnesota},
 author = {Talmor, Alon  and
Herzig, Jonathan  and
Lourie, Nicholas  and
Berant, Jonathan},
 booktitle = {Proceedings of the 2019 Conference of the North {A}merican Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers)},
 doi = {10.18653/v1/N19-1421},
 pages = {4149--4158},
 publisher = {Association for Computational Linguistics},
 title = {{C}ommonsense{QA}: A Question Answering Challenge Targeting Commonsense Knowledge},
 url = {https://www.aclweb.org/anthology/N19-1421},
 year = {2019}
}
```

## X-CSQA


[CommonsenseQA](https://www.tau-nlp.org/commonsenseqa){: target="_blank"} (CSQA) is a multiple-choice QA task targeting general commonsense knowledge, however, it only has English version.
We here provide its cross-lingual version, X-CSQA, by re-splitting the English data and translating them to other languages. 
There are **8,888** examples for *training* in English, **1,000** for *development* in each language, and **1,074** examples for *testing* in each language. 


```json
{
  "id": "be1920f7ba5454ad",  # an id shared by all languages
  "lang": "en", # one of the 16 language codes.
  "question": { 
    "stem": "What will happen to your knowledge with more learning?",   # question text
    "choices": [
      {"label": "A",  "text": "headaches" },
      {"label": "B",  "text": "bigger brain" },
      {"label": "C",  "text": "education" },
      {"label": "D",  "text": "growth" },
      {"label": "E",  "text": "knowing more" }
    ] },
  "answerKey": "D"    # hidden for test data.
}
```


## X-CODAH

[CODAH](https://arxiv.org/abs/1904.04365){: target="_blank"} dataset is a scene completion task with options, which shares a similar format to CSQA. 
There are **8,476** examples for *training* in English, **300** for *development* in each language, and **1,000** examples for *testing* in each language. (The CODAH is quite small, so we use the 7k examples from SWAG's dev data to augment training data here.)
Note that the question tags are `{i, n, p, r, q, o}`, which stands for `{"idioms", "negation", "polysemy", "reference", "quantitive", "other"}` respectively. 


```json
{
  "id": "b8eeef4a823fcd4b",   # an id shared by all languages
  "lang": "en", # one of the 16 language codes.
  "question_tag": "o",  # one of 6 question types
  "question": {
    "stem": " ", # always a blank as a dummy question
    "choices": [
      {"label": "A",
        "text": "Jennifer loves her school very much, she plans to drop every courses."},
      {"label": "B",
        "text": "Jennifer loves her school very much, she is never absent even when she's sick."},
      {"label": "C",
        "text": "Jennifer loves her school very much, she wants to get a part-time job."},
      {"label": "D",
        "text": "Jennifer loves her school very much, she quits school happily."}
    ]
  },
  "answerKey": "B"  # hidden for test data.
}
```

Please cite the paper here if you use the data here:
```bibtex
@inproceedings{Chen2019CODAHAA,
 address = {Minneapolis, USA},
 author = {Chen, Michael  and
D{'}Arcy, Mike  and
Liu, Alisa  and
Fernandez, Jared  and
Downey, Doug},
 booktitle = {Proceedings of the 3rd Workshop on Evaluating Vector Space Representations for {NLP}},
 doi = {10.18653/v1/W19-2008},
 pages = {63--69},
 publisher = {Association for Computational Linguistics},
 title = {{CODAH}: An Adversarially-Authored Question Answering Dataset for Common Sense},
 url = {https://www.aclweb.org/anthology/W19-2008},
 year = {2019}
}

```