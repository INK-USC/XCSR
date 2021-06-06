---
layout: default
title:  DrFact
nav_order: 4
# toc_list: true
parent: Methods
last_modified_date: Jun 5th 2021
permalink: /methods/drfact
has_toc: true
---

# DrFact for XCSR
{: .no_toc}

[The site is under development. Please email [***yuchen.lin@usc.edu***] if you have any questions.](){: .btn .btn-red .fs-4 target="_blank"}

- TOC
{:toc}


<!-- ## Illustration
![DrFact](/images/opencsr_t3.png) -->

## Installation

### Basic dependency.

```bash
# Note that we use TF 1.15. This is because we use the tf.contrib package,
conda create --name drfact python=3.7
conda activate drfact
pip install tensorflow-gpu==1.15.2
conda install -c anaconda cudatoolkit=10.0
conda install -c anaconda cudnn # Make sure your cuda is 10.0 and cudnn is 7.6.5
pip install tensorflow-hub bert-tensorflow 
pip install gdown
# download https://github.com/google-research/language as a zip and unzip it.
pip install -e language-master
pip install tensorflow-determinism  # for being reproducible 
git clone https://github.com/google-research/albert.git # add albert
pip install -r albert/requirements.txt
```

**_Test if your gpus can be used._**
```python
import tensorflow as tf
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
```

### Download the BERT files.
```bash
cd ~/ # can be  any place you want.
wget https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip
unzip uncased_L-12_H-768_A-12.zip
```

### Some common env vars used below.

```
BERT_PATH=~/uncased_L-12_H-768_A-12
CORPUS_PATH=drfact_data/knowledge_corpus/
INDEX_PATH=drfact_data/local_index/
```

## Indexing the Hypergraphs and Fact Embeddings.
 
### Corpus preprocessing.

```bash
python -m language.labs.drfact.index_corpus \
--do_preprocess \
--concept_vocab_file ${CORPUS_PATH}/gkb_best.vocab.txt \
--corpus_file ${CORPUS_PATH}/gkb_best.drfact_format.jsonl \
--max_entity_length 5 \
--max_mentions_per_doc 20 \
--tokenizer_type bert_tokenization \
--vocab_file ${BERT_PATH}/vocab.txt \
--index_result_path ${INDEX_PATH}/${INDEX_NAME} \
--alsologtostderr
```

### Pre-compute the Fact2Fact index.

```bash
for (( c=0; c<=94; c++ ))   # assume you have 95 CPUs.
do
   python -m language.labs.drfact.fact2fact_index \
    --do_preprocess \
    --corpus_file ${CORPUS_PATH}/gkb_best.drfact_format.jsonl \
    --fact2fact_index_dir ${INDEX_PATH}/fact2fact_index \
    --num_shards 94 --my_shard $c \
    --alsologtostderr &
done

# Combine the single files into a single file.

python -m language.labs.drfact.fact2fact_index \
  --do_combine \
  --corpus_file ${CORPUS_PATH}/gkb_best.drfact_format.jsonl \
  --fact2fact_index_dir ${INDEX_PATH}/fact2fact_index \
  --num_shards 94 \
  --alsologtostderr
```

### Convert Fact Embeddings
Note that you need to run the [DPR](/methods/dpr) training first.
```bash
declare -a datasets=("ARC" "OBQA" "QASC")
for DATA_NAME in "${datasets[@]}"
do
  python -m language.labs.drfact.convert_dpr_index \
  --index_result_path ${INDEX_PATH}/drfact_fact_index \
  --dpr_pkl_path drfact_data/${DATA_NAME}_dpr_index_0.pkl \
  --embed_prefix ${DATA_NAME}_dpr_bert_base 
done 
```
 
### Generate distant supervision. (Optional)


<details markdown="block">
  <summary>Show/Hide</summary>
  {: .fs-3 .text-delta .text-blue-100}    
  ```bash
  declare -a datasets=("ARC" "OBQA" "QASC")
  declare -a splits=("train" "dev") 
  for DATA_NAME in "${datasets[@]}"
  do 
    for SPLIT in "${splits[@]}"
    do 
      python -m language.labs.drfact.preprocessing.decompose_concepts \
        --CONCEPT_VOCAB drfact_data/knowledge_corpus/gkb_best.vocab.txt \
        --jsonl_file drfact_data/datasets/${DATA_NAME}/${SPLIT}.jsonl \
        --linked_qas_file drfact_data/datasets/${DATA_NAME}/linked_${SPLIT}.jsonl &
    done
  done

  for DATA_NAME in "${datasets[@]}"
  do
    DATA_FOLDER=drfact_data/datasets/${DATA_NAME}
    RET=${DATA_NAME}_DPR
    for SPLIT in "${splits[@]}"
    do
      python -m language.labs.drfact.add_sup_facts \
        --linked_qas_file ${DATA_FOLDER}/linked_${SPLIT}.jsonl \
        --drfact_format_gkb_file  ${CORPUS_PATH}/gkb_best.drfact_format.jsonl \
        --ret_result_file ${DATA_FOLDER}/linked_${SPLIT}.jsonl.${RET}.jsonl \
        --output_file ${DATA_FOLDER}/linked_${SPLIT}.sup_facts_from_${RET}.jsonl \
        --concept_vocab_file drfact_data/knowledge_corpus/gkb_best.vocab.txt \
        --max_num_facts 50 --split ${SPLIT} &
    done
  done

  for DATA_NAME in "${datasets[@]}"
  do
    RET=${DATA_NAME}_DPR
    DATA_FOLDER=drfact_data/datasets/${DATA_NAME}
    for SPLIT in "${splits[@]}"
    do
      python -m language.labs.drfact.add_middle_hops \
        --do hopping \
        --drfact_format_gkb_file ${CORPUS_PATH}/gkb_best.drfact_format.jsonl \
        --sup_fact_result_without_ans ${DATA_FOLDER}/linked_${SPLIT}.sup_facts_from_${RET}.jsonl \
        --sup_fact_result_with_ans ${DATA_FOLDER}/linked_${SPLIT}.sup_facts_from_${RET}_with_ans.jsonl \
        --output_file ${DATA_FOLDER}/linked_${SPLIT}.sup_facts_final_${RET}.jsonl &
    done
  done
  ```

  Renew the F2F links with such distant supervision.
  ```bash
  echo "" > drfact_data/local_drfact_index/fact2fact_index/add_links.tsv
  declare -a datasets=("ARC" "OBQA" "QASC")
  declare -a splits=("train")
  RET=OCSR_DPR 
  for DATA_NAME in "${datasets[@]}"
  do
    DATA_FOLDER=drfact_data/datasets/${DATA_NAME}
    for SPLIT in "${splits[@]}"
    do
      SUP_FACT=${DATA_FOLDER}/linked_${SPLIT}.sup_facts_final_${RET}.jsonl
      python -m language.labs.drfact.extract_fact_links \
        --sup_facts_file ${SUP_FACT} \
        --fact_links_file drfact_data/local_drfact_index/fact2fact_index/add_links.tsv
    done
  done
  python language-master/language/labs/drfact/convert_add_links.py \
    drfact_data/local_drfact_index/fact2fact_index/add_links.tsv \
    drfact_data/local_drfact_index/fact2fact_index/f2f_95.json
  ```
</details> 
{: .fs-5 .fw-600}




### Pre-compute the initial facts.

```bash
declare -a datasets=("ARC" "OBQA" "QASC") 
declare -a splits=("train" "dev")
COUNT=50
for DATA_NAME in "${datasets[@]}"
do
  DATA_FOLDER=drfact_data/datasets/${DATA_NAME}
  RET=${DATA_NAME}_DPR
  for SPLIT in "${splits[@]}"
  do
    SUP_FACT=${DATA_FOLDER}/linked_${SPLIT}.sup_facts_final_${RET}.jsonl  # optional
    python -m language.labs.drfact.add_init_facts \
      --linked_qas_file ${DATA_FOLDER}/linked_${SPLIT}.jsonl \
      --drfact_format_gkb_file  ${CORPUS_PATH}/gkb_best.drfact_format.jsonl \
      --ret_result_file ${DATA_FOLDER}/linked_${SPLIT}.jsonl.${RET}.jsonl \
      --sup_facts_file ${SUP_FACT} \
      --output_file ${DATA_FOLDER}/linked_${SPLIT}.init_facts.jsonl \
      --max_num_facts ${COUNT} --split ${SPLIT} &
  done
done
```



## Training

```bash
DATA=ARC
ODIR=~/saved_models/drfact_models_${DATA}
HOP=3 # can be any integer.
GPUS=0 OUT_DIR=${ODIR} DATASET=${DATA} bash scripts/run_drfact.sh train ${HOP}   # Training
GPUS=1 OUT_DIR=${ODIR} DATASET=${DATA} bash scripts/run_drfact.sh continual_eval ${HOP}  # Online Evaluation on Dev
# the log file will be at `${ODIR}/hop_${HOP}/tf_log.cont_eval.txt`
```


## Inference 

```bash
DATA=ARC
ODIR=~/saved_models/drfact_models_${DATA}
HOP=3
GPUS=0 OUT_DIR=${ODIR} DATASET=${DATA} bash scripts/run_drfact.sh \
    predict ${HOP} [checkpoint_name] [train|dev|test] 
# an example string for the checkpoint_name is "model.ckpt-14600"
```


### The content of the `run_drfact.sh`
<details markdown="block">
  <summary>Show/Hide</summary>
  {: .fs-3 .text-delta .text-blue-100}    
  ```bash
  #!/bin/bash
  BERT_PATH=~/uncased_L-12_H-768_A-12  # BERT-base
  question_num_layers=11
  ENTAGG=max
  CORPUS_PATH=drfact_data/knowledge_corpus/
  INDEX_PATH=drfact_data/local_drfact_index/
  INDEX_NAME=drfact_output_bert200
  F2F_INDEX_NAME=fact2fact_index
  DATASET_PATH=drfact_data/datasets/${DATASET}
  NUM_HOPS=$2
  MODEL_OUTPUT_DIR=${OUT_DIR}/hop_$2
  PREDICT_PREFIX=dev
  if [ "$1" = "train" ]; 
  then
    echo "training mode"
    rm -r ${MODEL_OUTPUT_DIR}
    DO="do_train "
    mkdir -p ${MODEL_OUTPUT_DIR}
    LOG_FILE=${MODEL_OUTPUT_DIR}/tf_log.train.txt
  elif [ "$1" = "continual_eval" ];
  then
    echo "continual_eval mode"
    DO="do_predict "
    mkdir -p ${MODEL_OUTPUT_DIR}
    LOG_FILE=${MODEL_OUTPUT_DIR}/tf_log.cont_eval.txt
  elif [ "$1" = "predict" ];
  then
    echo "prediction mode"
    PREDICT_PREFIX=$4 # dev or train
    DO="do_predict --use_best_ckpt_for_predict --model_ckpt_toload $3 "
    LOG_FILE=${MODEL_OUTPUT_DIR}/tf_log.$3-${PREDICT_PREFIX}-prediction.txt
  fi

  touch ${LOG_FILE}
  cp language-master/language/labs/drfact/model_fns.py ${LOG_FILE}.model_fn.py

  CUDA_VISIBLE_DEVICES=${GPUS} python -m language.labs.drfact.run_drfact \
    --vocab_file ${BERT_PATH}/vocab.txt \
    --tokenizer_model_file None \
    --bert_config_file ${BERT_PATH}/bert_config.json \
    --tokenizer_type bert_tokenization \
    --output_dir ${MODEL_OUTPUT_DIR} \
    --train_file ${DATASET_PATH}/linked_train.init_facts.jsonl \
    --predict_file ${DATASET_PATH}/linked_${PREDICT_PREFIX}.init_facts.jsonl \
    --predict_prefix ${PREDICT_PREFIX} \
    --init_checkpoint ${BERT_PATH}/bert_model.ckpt \
    --train_data_dir ${INDEX_PATH}/${INDEX_NAME} \
    --test_data_dir ${INDEX_PATH}/${INDEX_NAME} \
    --f2f_index_dir ${INDEX_PATH}/${F2F_INDEX_NAME} \
    --learning_rate 3e-05 \
    --warmup_proportion 0.1 \
    --train_batch_size 24 \
    --predict_batch_size 1 \
    --save_checkpoints_steps 100 \
    --iterations_per_loop 300 \
    --num_train_epochs 10.0 \
    --max_query_length 128 \
    --max_entity_len 5 \
    --qry_layers_to_use -1 \
    --qry_aggregation_fn concat \
    --question_dropout 0.3 \
    --question_num_layers ${question_num_layers} \
    --projection_dim 384 \
    --train_with_sparse  \
    --fix_sparse_to_one  \
    --predict_with_sparse  \
    --data_type opencsr \
    --model_type drfact \
    --supervision fact+entity \
    --num_mips_neighbors 500 \
    --entity_score_aggregation_fn ${ENTAGG} \
    --entity_score_threshold 1e-4 \
    --fact_score_threshold 1e-5 \
    --softmax_temperature 5.0 \
    --sparse_reduce_fn max \
    --sparse_strategy sparse_first \
    --num_hops ${NUM_HOPS} \
    --num_preds -1 \
    --embed_index_prefix ${DATASET}_dpr_bert_base \
    --$DO 2> ${LOG_FILE} &

  echo " "
  echo ${LOG_FILE}

  # watch -n 1 tail -n 50 ${LOG_FILE}


  ```
</details> 
{: .fs-5 .fw-600}