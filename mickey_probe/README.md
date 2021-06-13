# Mickey Probe 


The mode introduction of the MickeyProbe and the corpus is on our website: https://inklab.usc.edu//XCSR/mickey.


This is the script for running the probing task for a model in all languages.
Please read the comments and adjust the code for your own purpose and situation.
```bash
declare -a models=("distilbert-base-multilingual-cased")  # if only use a particular model
# declare -a models=("bert-base-multilingual-uncased" "xlm-roberta-base" "xlm-roberta-large" "xlm-mlm-100-1280" "distilbert-base-multilingual-cased")   # for all models
for MODEL in "${models[@]}"
do  
    declare -a langs=("en" "zh" "de" "fr" "ru" "es" "vi" "hi" "bg" "nl" "it")
    for index in ${!langs[*]}; 
    do 
        lang=${langs[$index]} 
        num_batch=4 
        # this depends on your GPU memory size
        for i in $(eval echo "{0..${num_batch}}")
        do
            gpu=$((i/1 + 0))
            # "/x" here means we will run x batches on a gpu
            # "+y" here means the starting GPU index is y
            echo ${lang}-${gpu}-${i}
            CUDA_VISIBLE_DEVICES=${gpu} python sent_mlmscoring.py \
                --model_str ${MODEL} \
                --input_file mickey_corpus/mickey_${lang}.jsonl \
                --result_file probe_results/mickey_${lang}.${MODEL}.${i}.jsonl \
                --use_batch 1 \
                --num_shards $((num_batch+1)) --shard_id $i &
        done
        wait;
        # combine all the shards.
        cat probe_results/mickey_${lang}.${MODEL}.*.jsonl > probe_results/combined/mickey_${lang}.${MODEL}.jsonl
    done
done
``` 