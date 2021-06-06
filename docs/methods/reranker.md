---
layout: default
title:  Concept Re-ranker
nav_order: 5
# toc_list: true
parent: Methods
last_modified_date: Jun 5th 2021
permalink: /methods/reranker
has_toc: true
---

# Concept Re-ranker for XCSR
{: .no_toc}

[The site is under development. Please email [***yuchen.lin@usc.edu***] if you have any questions.](){: .btn .btn-red .fs-4 target="_blank"}

- TOC
{:toc}



## Installation 

```bash
conda create --name mcqa python=3.7
conda activate mcqa
pip install torch==1.4.0 torchvision==0.5.0 -f https://download.pytorch.org/whl/cu100/torch_stable.html
pip install transformers==3.5.1
pip install tensorboardX
pip install absl-py
# install apex if you want to use fp16 to speed up
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```


## Data Preprocessing

```bash
declare -a datasets=("ARC" "OBQA" "QASC")
declare -a rets=("BM25")
for DATA_NAME in "${datasets[@]}"
do 
  for RET in "${rets[@]}"
  do
    python baseline_methods/MCQA/create_mcqa_data.py \
    --eval_result_file baseline_methods/${RET}/results/${DATA_NAME}_train_max_result.jsonl \
    --mcqa_file baseline_methods/MCQA/data/${DATA_NAME}.train.${RET}.jsonl \
    --top_K 500 --num_distractors 9 &
  done 
done
```

For all dev data.
```bash

declare -a datasets=("ARC" "OBQA" "QASC")
for DATA_NAME in "${datasets[@]}"
do 
  python baseline_methods/MCQA/create_mcqa_data.py \
    --eval_result_file baseline_methods/DPR/results/${DATA_NAME}_dev_max_result.jsonl \
    --mcqa_file baseline_methods/MCQA/data/${DATA_NAME}.dev.DPR.jsonl \
    --top_K 500 --if_test &
  rets=("drfact" "drkit")
  for RET in "${rets[@]}"
  do  
    python baseline_methods/MCQA/create_mcqa_data.py \
    --eval_result_file drfact_data/${RET}_results/${RET}_${DATA_NAME}_result.jsonl \
    --mcqa_file baseline_methods/MCQA/data/${DATA_NAME}.dev.${RET}.jsonl \
    --top_K 500 --if_test &
  done 
done
```

### Split the data for training the rerankers.
```bash
RET=BM25

mkdir baseline_methods/MCQA/data/ARC/
tail -300 baseline_methods/MCQA/data/ARC.train.${RET}.jsonl > baseline_methods/MCQA/data/ARC/val.${RET}.jsonl
cp baseline_methods/MCQA/data/ARC.dev.${RET}.jsonl baseline_methods/MCQA/data/ARC/test.${RET}.jsonl
cp baseline_methods/MCQA/data/ARC.train.${RET}.jsonl baseline_methods/MCQA/data/ARC/train.${RET}.jsonl

mkdir baseline_methods/MCQA/data/QASC/
tail -300 baseline_methods/MCQA/data/QASC.train.${RET}.jsonl > baseline_methods/MCQA/data/QASC/val.${RET}.jsonl
cp baseline_methods/MCQA/data/QASC.dev.${RET}.jsonl baseline_methods/MCQA/data/QASC/test.${RET}.jsonl
cp baseline_methods/MCQA/data/QASC.train.${RET}.jsonl baseline_methods/MCQA/data/QASC/train.${RET}.jsonl

mkdir baseline_methods/MCQA/data/OBQA/
tail -300 baseline_methods/MCQA/data/OBQA.train.${RET}.jsonl > baseline_methods/MCQA/data/OBQA/val.${RET}.jsonl
cp baseline_methods/MCQA/data/OBQA.dev.${RET}.jsonl baseline_methods/MCQA/data/OBQA/test.${RET}.jsonl
cp baseline_methods/MCQA/data/OBQA.train.${RET}.jsonl baseline_methods/MCQA/data/OBQA/train.${RET}.jsonl
```



## Training a MCQA model as a reranker w/ distractors.
```bash 
gpu=0
declare -a datasets=("ARC" "OBQA" "QASC")  
declare -a rets=("BM25")

for DATA_NAME in "${datasets[@]}"
do
  for RET in "${rets[@]}"
  do
    ((gpuplus=gpu+1))
    echo "gpu: ${gpu},${gpuplus} ${DATA_NAME}-${RET}"
    DATA_DIR=baseline_methods/MCQA/data/${DATA_NAME}/
    CUDA_VISIBLE_DEVICES=${gpu},${gpuplus} python baseline_methods/MCQA/run_mcqa.py \
      --task_name opencsr \
      --model_name_or_path bert-large-uncased-whole-word-masking \
      --do_train \
      --do_eval \
      --data_dir $DATA_DIR \
      --train_file $DATA_DIR/train.${RET}.jsonl \
      --val_file $DATA_DIR/val.${RET}.jsonl \
      --test_file "" --prediction_output "" \
      --num_choices 10 \
      --learning_rate 2e-5 \
      --num_train_epochs 5 \
      --max_seq_length 80 \
      --output_dir ~/mcqa_models/${DATA_NAME}-${RET}_bertlarge \
      --per_gpu_eval_batch_size=6 \
      --per_device_train_batch_size=6 \
      --gradient_accumulation_steps 1 \
      --fp16 --overwrite_output \
      --overwrite_cache &
    ((gpu=gpu+2))
  done
done
``` 


## Inference over test 
```bash
declare -a datasets=("ARC" "OBQA" "QASC")  
declare -a rets=("BM25" "DPR")
for DATA_NAME in "${datasets[@]}"
do
  for RET in "${rets[@]}"
  do 
    DATA_DIR=baseline_methods/MCQA/data/${DATA_NAME}/
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8 python baseline_methods/MCQA/run_mcqa.py \
      --task_name opencsr \
      --model_name_or_path ~/mcqa_models/${DATA_NAME}-BM25_bertlarge \
      --do_predict \
      --data_dir $DATA_DIR \
      --train_file "" --val_file "" \
      --test_file $DATA_DIR/test.${RET}.jsonl \
      --prediction_output $DATA_DIR/test.${RET}.npy \
      --num_choices 500 \
      --max_seq_length 80 \
      --output_dir ~/mcqa_models/${DATA_NAME}-${RET}_bertlarge \
      --per_gpu_eval_batch_size=5 \
      --overwrite_output --fp16  --overwrite_cache
      # --fp16  --overwrite_cache 
  done
done
```

Using the Reranker for DrFact and DrKIT
```bash
declare -a datasets=("ARC" "OBQA" "QASC")  
declare -a rets=("drfact" "drkit")
for DATA_NAME in "${datasets[@]}"
do
  for RET in "${rets[@]}"
  do
    DATA_DIR=baseline_methods/MCQA/data/
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8 python baseline_methods/MCQA/run_mcqa.py \
      --task_name opencsr \
      --model_name_or_path ~/mcqa_models/${DATA_NAME}-BM25_bertlarge \
      --do_predict \
      --data_dir $DATA_DIR/${DATA_NAME} \
      --train_file "" --val_file "" \
      --test_file $DATA_DIR/${DATA_NAME}.dev.${RET}.jsonl \
      --prediction_output $DATA_DIR/${DATA_NAME}/test.${RET}.npy \
      --num_choices 500 \
      --max_seq_length 80 \
      --output_dir ~/mcqa_models/${DATA_NAME}-${RET}_bertlarge \
      --per_gpu_eval_batch_size=5 \
      --overwrite_output --overwrite_cache
  done
done
```


## Convert npy prediction files to results

```bash
mkdir baseline_methods/MCQA/results/
declare -a datasets=("ARC" "OBQA" "QASC")
declare -a rets=("BM25" "DPR" "drfact" "drkit")
for DATA_NAME in "${datasets[@]}"
do 
  for RET in "${rets[@]}"
  do
    python baseline_methods/MCQA/convert_mcqa_predictions.py \
    --pred_result_file baseline_methods/MCQA/data/${DATA_NAME}/test.${RET}.npy \
    --mcqa_file baseline_methods/MCQA/data/${DATA_NAME}.dev.${RET}.jsonl \
    --reference_result_file baseline_methods/BM25/results/${DATA_NAME}_dev_max_result.jsonl \
    --converted_result_file baseline_methods/MCQA/results/${DATA_NAME}_${RET}_MCQA_result.jsonl &
  done 
done
```


