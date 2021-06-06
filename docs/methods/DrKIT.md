---
layout: default
title:  DrKIT
nav_order: 3
# toc_list: true
parent: Methods
last_modified_date: Jun 5th 2021
permalink: /methods/drkit
has_toc: true
---

# DrKIT for XCSR
{: .no_toc}


[The site is under development. Please email [***yuchen.lin@usc.edu***] if you have any questions.](){: .btn .btn-red .fs-4 target="_blank"}

- TOC
{:toc}


---

Please cite the original paper
```bibtex
@inproceedings{
    Dhingra2020Differentiable,
    title={Differentiable Reasoning over a Virtual Knowledge Base},
    author={Bhuwan Dhingra and Manzil Zaheer and Vidhisha Balachandran and Graham Neubig and Ruslan Salakhutdinov and William W. Cohen},
    booktitle={International Conference on Learning Representations},
    year={2020},
    url={https://openreview.net/forum?id=SJxstlHFPH}
}
```

Note that here we reuse **the original repo of DrKIT** [google-research/language/labs/drkit/](https://github.com/google-research/language/tree/master/language/labs/drkit), and modified some code for XCSR. Thus, the entrance is shared with "language.labs.drfact".

---


## Installation

### Basic dependency.

```bash
# Note that we use TF 1.15. This is because we use the tf.contrib package,
conda create --name drkit python=3.7
conda activate drkit
pip install tensorflow-gpu==1.15.2
conda install -c anaconda cudatoolkit=10.0
conda install -c anaconda cudnn # Make sure your cuda is 10.0 and cudnn is 7.6.5
pip install tensorflow-hub bert-tensorflow 
pip install gdown
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
INDEX_NAME=drkit_mention_index
```


## Indexing the commonsense corpus.


### Data preprocessing.

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

### Indexing the mentions with BERT.
```bash
for (( c=0; c<=3; c++ )) # Assuming you have 4 gpus.
do
  CUDA_VISIBLE_DEVICES=$c \
  python -m language.labs.drfact.index_corpus \
    --do_embed --do_embed_mentions \
    --concept_vocab_file ${CORPUS_PATH}/gkb_best.vocab.txt \
    --corpus_file ${CORPUS_PATH}/gkb_best.drfact_format.jsonl \
    --max_entity_length 5 \
    --max_mentions_per_doc 20 \
    --tokenizer_type bert_tokenization \
    --bert_ckpt_dir ${BERT_PATH}/ \
    --vocab_file ${BERT_PATH}/vocab.txt \
    --bert_config_file ${BERT_PATH}/bert_config.json \
    --ckpt_name bert_model.ckpt --embed_prefix bert_base \
    --index_result_path ${INDEX_PATH}/drkit_mention_index \
    --predict_batch_size 512 \
    --max_seq_length 100 \
    --doc_layers_to_use -1 \
    --doc_aggregation_fn concat \
    --qry_layers_to_use 4 \
    --qry_aggregation_fn concat \
    --max_query_length  64 \
    --projection_dim 200 \
    --doc_stride 128 \
    --num_shards 3 --my_shard $c \
    --alsologtostderr &
done
wait
```

### Combine the shards.

```bash
python -m language.labs.drfact.index_corpus \
--do_combine \
--do_embed_mentions \
--max_entity_length 5 \
--max_mentions_per_doc 20 \
--index_result_path ${INDEX_PATH}/drkit_mention_index \
--num_shards 3 \
--projection_dim 200 \
--tokenizer_type dummy \
--embed_prefix bert_base \
--alsologtostderr
```

### Clean the temporary files. (optional)

```bash
PREFIX=bert_large
rm ${INDEX_PATH}/drkit_mention_index/${PREFIX}_fact_feats_*.*
```


## Training

```bash
DATA=ARC
ODIR=~/saved_models/drkit_models_${DATA}
HOP=3 # can be any integer.
GPUS=0 OUT_DIR=${ODIR} DATASET=${DATA} bash scripts/run_drkit.sh train ${HOP}   # Training
GPUS=1 OUT_DIR=${ODIR} DATASET=${DATA} bash scripts/run_drkit.sh continual_eval ${HOP}  # Online Evaluation on Dev
# the log file will be at `${ODIR}/hop_${HOP}/tf_log.cont_eval.txt`
```


## Inference 

```bash
DATA=ARC
ODIR=~/saved_models/drkit_models_${DATA}
HOP=3
GPUS=0 OUT_DIR=${ODIR} DATASET=${DATA} bash scripts/run_drkit.sh \
    predict ${HOP} [checkpoint_name] [train|dev|test] 
# an example string for the checkpoint_name is "model.ckpt-14600"
```


### The content of the `run_drkit.sh`
<details markdown="block">
  <summary>Show/Hide</summary>
  {: .fs-3 .text-delta .text-blue-100} 
    ```bash
    #!/bin/bash
    BERT_PATH=~/uncased_L-12_H-768_A-12  # BERT-base
    question_num_layers=5
    ENTAGG=max
    CORPUS_PATH=drfact_data/knowledge_corpus/
    INDEX_PATH=drfact_data/local_index/
    INDEX_NAME=drkit_mention_index
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
    PREDICT_PREFIX=$4 # train/dev/test
    DO="do_predict --use_best_ckpt_for_predict --model_ckpt_toload $3 "
    LOG_FILE=${MODEL_OUTPUT_DIR}/tf_log.$3-${PREDICT_PREFIX}-prediction.txt
    fi

    touch ${LOG_FILE}


    CUDA_VISIBLE_DEVICES=${GPUS} python -m language.labs.drfact.run_drfact \
    --vocab_file ${BERT_PATH}/vocab.txt \
    --tokenizer_model_file None \
    --bert_config_file ${BERT_PATH}/bert_config.json \
    --tokenizer_type bert_tokenization \
    --output_dir ${MODEL_OUTPUT_DIR} \
    --train_file ${DATASET_PATH}/linked_train.jsonl \
    --predict_file ${DATASET_PATH}/linked_${PREDICT_PREFIX}.jsonl \
    --predict_prefix ${PREDICT_PREFIX} \
    --init_checkpoint ${BERT_PATH}/bert_model.ckpt \
    --train_data_dir ${INDEX_PATH}/${INDEX_NAME} \
    --test_data_dir ${INDEX_PATH}/${INDEX_NAME} \
    --learning_rate 3e-05 \
    --warmup_proportion 0.1 \
    --train_batch_size 2 \
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
    --projection_dim 200 \
    --train_with_sparse  \
    --fix_sparse_to_one  \
    --predict_with_sparse  \
    --data_type opencsr \
    --model_type drkit \
    --supervision entity \
    --num_mips_neighbors 100 \
    --entity_score_aggregation_fn ${ENTAGG} \
    --entity_score_threshold 5e-2 \
    --softmax_temperature 5.0 \
    --sparse_reduce_fn max \
    --sparse_strategy sparse_first \
    --num_hops ${NUM_HOPS} \
    --embed_index_prefix bert_base \
    --num_preds -1 \
    --$DO 2> ${LOG_FILE} &

    echo ${LOG_FILE}

    # watch -n 1 tail -n 50 ${LOG_FILE}
    ```
</details> 
{: .fs-5 .fw-600}