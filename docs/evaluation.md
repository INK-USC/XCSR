---
layout: default
title: Evaluation
nav_order: 4
# toc_list: true
last_modified_date: Jun 5th 2021
permalink: /evaluation
mathjax: true
has_toc: true
---

# Evaluation 
{: .no_toc}


[The site is under development. Please email [***yuchen.lin@usc.edu***] if you have any questions.](){: .btn .btn-red .fs-4 target="_blank"}

## Table of contents
{: .no_toc .text-delta }

- TOC
{:toc}



## Post-processing the prediction results 



### BM25/DPR.



After running the inference of BM25 and DPR, we have a ranked list of retrieved *facts* with scores for each question. To get a ranked list of *concepts* as the answer as the final output, we post-process the retrieved results:

```bash
python evaluation/process_ret_results.py \
    --drfact_format_gkb_file [drfact_data/knowledge_corpus/gkb_best.drfact_format.jsonl] \
    --linked_qa_ret_file [drfact_data/datasets/ARC/linked_dev.BM25.jsonl] \
    --pred_result_file [results/ARC/dev_prediction.BM25.jsonl]
# [x] are example paths. You can adjust for your target files.
```

### DrKIT/DrFact

To have a unified format of prediction result, we reformat the result files of DrKIT and DrFact as follows:
 
```bash
input_file=[/path/to/best_predictions.json]
output_file=[results/ARC/dev_prediction.drfact.jsonl]
python evaluation/process_drx_results.py ${input_file} ${output_file}
    
# [x] are example paths. You can adjust for your target files.
```

### X + Concept Re-Ranker 

This is done in [***Step 7 of the re-ranking***](/methods/dpr#step-7-dpr-retrieval-formatting-for-each-dataset){: target_blank}.

## Metrics 


### Hit@K acc and Ret@K acc

Recall that, given a question $$q$$, the final output of every method is a weighted set of concepts $$A=\{(a_1, w_1), \dots \}$$. 
We denote the set of true answer concepts, as defined above, as $$A^*=\{a_1^*, a_2^*, \dots \}$$.   We define **Hit@K** accuracy to be the fraction of questions for which we can find at least one correct answer concept $$a_i^*\in A^*$$ in the top-$$K$$ concepts of $$A$$ (sorted in descending order of weight). 
As questions have multiple correct answers, recall is also an important aspect for evaluating XCSR, so we also 
use **Rec@K** to evaluate the average recall of the top-K proposed answers.

### Run


```bash
python evaluation/eval_metrics.py \
  --pred_result_file [results/ARC/dev_prediction.BM25.jsonl] \
  --truth_data_file [drfact_data/datasets/ARC/linked_dev.jsonl]
# [x] are example paths. You can adjust for your target files.
```

