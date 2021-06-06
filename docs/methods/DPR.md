---
layout: default
title:  DPR
nav_order: 2
# toc_list: true
parent: Methods
last_modified_date: Jun 5th 2021
permalink: /methods/dpr
has_toc: true
---

# DPR for XCSR
{: .no_toc}

[The site is under development. Please email [***yuchen.lin@usc.edu***] if you have any questions.](){: .btn .btn-red .fs-4 target="_blank"}

- TOC
{:toc}

---

Please cite the original paper
```bibtex
@inproceedings{Karpukhin2020DensePR,
  title={Dense Passage Retrieval for Open-Domain Question Answering},
  author={V. Karpukhin and Barlas Oğuz and Sewon Min and Patrick Lewis and Ledell Yu Wu and Sergey Edunov and Danqi Chen and Wen-tau Yih},
  booktitle={EMNLP},
  year={2020}
}
```

**Link to the code for the experiment:** [XCSR/baseline_methods/DPR/](https://github.com/yuchenlin/XCSR/tree/main/baseline_methods/DPR)

---

## Installation

```bash
conda create --name dpr python=3.7
conda activate dpr
# use our copy of the DPR code in `dpr-master`, 
    # or you can download the latest version (unverified) on Github
cd dpr-master
pip install -e dpr-master
pip install torch==1.4.0 torchvision==0.5.0 -f https://download.pytorch.org/whl/cu100/torch_stable.html
conda install faiss-gpu cudatoolkit=10.0 -c pytorch -n dpr
pip install transformers==3.0.2
pip install spacy
```


## Preprocessing 

### Step 1. Transform the data format of the corpus.
**You can skip this step**. We have included the processed tsv file at `gkb_best.prepro.tsv` under `drfact_data/knowledge_corpus/`.
```bash
python -m language.labs.drfact.preprocessing.corpus_prepro   --DATA_ROOT drfact_data/knowledge_corpus   --CORPUS_PATH GenericsKB-Best.tsv   --OUTPUT_JSON_PATH gkb_best.prepro.jsonl
python baseline_methods/DPR/convert_gkb_tsv.py ${CORPUS_PATH}/gkb_best.prepro.jsonl
# This will generate `${CORPUS_PATH}/gkb_best.prepro.tsv`
```

### Step 2. Convert the XCSR datasets format.

```bash
declare -a datasets=("ARC" "OBQA" "QASC")
declare -a splits=("train" "dev")
for DATA_NAME in "${datasets[@]}"
do
  DATA_FOLDER=drfact_data/datasets/${DATA_NAME}
  python baseline_methods/DPR/convert_qas_csv.py $DATA_FOLDER/linked_train.jsonl no &
  python baseline_methods/DPR/convert_qas_csv.py $DATA_FOLDER/linked_dev.jsonl no &
done
```

### Step 3. Generate data from BM25 for training DPR
Note that you need to do the inference of BM25 first.

```bash
declare -a datasets=("ARC" "OBQA" "QASC")
declare -a splits=("train" "dev")
for DATA_NAME in "${datasets[@]}"
do
  DATA_FOLDER=drfact_data/datasets/${DATA_NAME}
  for SPLIT in "${splits[@]}"
  do
    python baseline_methods/DPR/genenrate_dpr_data.py \
      --dataset_name ${DATA_NAME} \
      --linked_qas_file ${DATA_FOLDER}/linked_${SPLIT}.jsonl \
      --drfact_format_gkb_file  ${CORPUS_PATH}/gkb_best.drfact_format.jsonl \
      --ret_result_file ${DATA_FOLDER}/linked_${SPLIT}.jsonl.BM25.jsonl \
      --output_file baseline_methods/DPR/dpr_train_data/${DATA_NAME}_${SPLIT}.dpr_format.jsonl &
  done
done

# After the reformatting, we get the json format instead of the jsonl format
declare -a datasets=("ARC" "OBQA" "QASC")
declare -a splits=("train" "dev")
RET=BM25
for DATA_NAME in "${datasets[@]}"
do
  DATA_FOLDER=drfact_data/datasets/${DATA_NAME}
  for SPLIT in "${splits[@]}"
  do
    python baseline_methods/DPR/jsonl2json.py \
        baseline_methods/DPR/dpr_train_data/${DATA_NAME}_${SPLIT}.dpr_format.jsonl &
  done
done 
```

## Training DPR

### Step 4. Train DPR index for each dataset.

```bash
mkdir ~/dpr_data/ # Can be any folder but need to be consistent.
DPR_DATA=baseline_methods/DPR/dpr_train_data/
declare -a datasets=("ARC" "OBQA" "QASC")
for DATA_NAME in "${datasets[@]}"
do 
  echo "${DATA_NAME}"
  gpu=0,1,2,3,4,5,6,7
  master_port=29500
  nproc_per_node=8  # Note this should be correct.
  echo "CUDA_VISIBLE_DEVICES=${gpu}"
  rm -r ~/dpr_data/${DATA_NAME}_dpr_model/
  mkdir ~/dpr_data/${DATA_NAME}_dpr_model/
  CUDA_VISIBLE_DEVICES=${gpu} python -m torch.distributed.launch \
    --master_port ${master_port} \
    --nproc_per_node=${nproc_per_node} \
    baseline_methods/DPR/dpr-master/train_dense_encoder.py \
    --max_grad_norm 2.0 \
    --encoder_model_type hf_bert \
    --pretrained_model_cfg bert-base-uncased \
    --seed 12345 \
    --sequence_length 64 \
    --warmup_steps 1237 \
    --batch_size 16 \
    --do_lower_case \
    --train_file ${DPR_DATA}/${DATA_NAME}_train.dpr_format.json \
    --dev_file ${DPR_DATA}/${DATA_NAME}_dev.dpr_format.json \
    --output_dir ~/dpr_data/${DATA_NAME}_dpr_model/ \
    --learning_rate 3e-05 \
    --num_train_epochs 10 \
    --dev_batch_size 16 \
    --val_av_rank_start_epoch 10  --fp16
done
```

## Inference

### Step 5. Encode facts with the trained DPR model for each dataset.

```bash
gpu=0
declare -a datasets=("ARC" "OBQA" "QASC")
for DATA_NAME in "${datasets[@]}"
do
  DPR_FILE=$(ls ~/dpr_data/${DATA_NAME}_dpr_model/ -Art | tail -n 1) # use the last checkpoint.
  ((gpup=gpu+1))
  echo "CUDA_VISIBLE_DEVICES=${gpu},${gpup}: ${DPR_FILE}"
  CUDA_VISIBLE_DEVICES=${gpu},${gpup} python baseline_methods/DPR/dpr-master/generate_dense_embeddings.py \
    --model_file ~/dpr_data/${DATA_NAME}_dpr_model/${DPR_FILE} \
    --ctx_file ~/drfact/drfact_data/knowledge_corpus/gkb_best.prepro.tsv \
    --batch_size 5000 --shard_id 0 --num_shards 1 \
    --out_file ~/drfact/drfact_data/${DATA_NAME}_dpr_index &
  ((gpu=gpu+2))
done
```

### Step 6. DPR Inference for each dataset

```bash
declare -a datasets=("ARC" "OBQA" "QASC")
declare -a splits=("train" "dev" "test")
gpu=0
for DATA_NAME in "${datasets[@]}"
do
  DATA_FOLDER=drfact_data/datasets/${DATA_NAME} 
  for SPLIT in "${splits[@]}"
  do
    echo "${gpu}: ${DATA_NAME}-${SPLIT}"
    DPR_FILE=$(ls ~/dpr_data/${DATA_NAME}_dpr_model/ -Art | tail -n 1)
    CUDA_VISIBLE_DEVICES=${gpu} python baseline_methods/DPR/dense_retriever.py \
      --model_file ~/dpr_data/${DATA_NAME}_dpr_model/${DPR_FILE} \
      --ctx_file ~/drfact/drfact_data/knowledge_corpus/gkb_best.prepro.tsv \
      --qa_file $DATA_FOLDER/linked_${SPLIT}.csv \
      --encoded_ctx_file ~/drfact/drfact_data/${DATA_NAME}_dpr_index_0.pkl \
      --out_file ~/dpr_data/${DATA_NAME}-${SPLIT}_${DATA_NAME}-DPR_result_1000.pkl \
      --batch_size 512 --n-docs 1000 &
    ((gpu=gpu+1))
  done
done
```


### Step 7. DPR Retrieval Formatting for each dataset

```bash
declare -a datasets=("ARC" "OBQA" "QASC")
declare -a splits=("train" "dev" "test")
for DATA_NAME in "${datasets[@]}"
do
  DATA_FOLDER=drfact_data/datasets/${DATA_NAME} 
  for SPLIT in "${splits[@]}"
  do
    python baseline_methods/DPR/dpr_result_formatter.py \
      --linked_qa_file ${DATA_FOLDER}/linked_${SPLIT}.jsonl \
      --output_prefix ${DATA_NAME}_DPR \
      --drfact_format_gkb_file ${CORPUS_PATH}/gkb_best.drfact_format.jsonl \
      --dpr_result_file ~/dpr_data/${DATA_NAME}-${SPLIT}_${DATA_NAME}-DPR_result_1000.pkl &
  done
done
```