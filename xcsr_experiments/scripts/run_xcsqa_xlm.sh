# wandb login [your token here]

MODE=$1
if [ "$MODE" = "self-train" ]; then
    echo "Self-Tarin and Self-Test."
    declare -a langs=("en" "zh" "de" "es" "fr" "it" "jap" "nl" "pl" "pt" "ru" "ar" "sw" "ur" "vi" "hi")
    declare -a lrs=("1e-5" "8e-6" "8e-6" "4e-6" "8e-6" "8e-6" "8e-6" "8e-6" "8e-6" "8e-6" "8e-6" "8e-6" "8e-6" "8e-6" "8e-6")
    declare -a warms=("300" "100" "100" "100" "100" "100" "100" "100" "100" "100" "100" "100" "100" "100" "100" "100")
    active_langs=" en "
    # for lang in "${langs[@]}" 
    # do
    for index in ${!langs[*]}; 
    do 
        lang=${langs[$index]}
        lr=${lrs[$index]}
        warm=${warms[$index]}

        if [[ "$active_langs" == *"$lang"* ]]; then
            echo "$lang - $lr - $warm"
            DATA_DIR=corpus/CSQA/X-CSQA/${lang}
            MODEL_DIR=/mnt/nfs1/bill/saved_models_xcsqa/xlm_${lang}
            CUDA_VISIBLE_DEVICES=6,7,4,5 python methods/run_mcqa.py \
                --task_name xcsr \
                --exp_name "xlm_${lang}|(${lr}-${warm})" \
                --model_name_or_path xlm-mlm-100-1280 \
                --do_train --do_eval --data_dir $DATA_DIR \
                --train_file ${DATA_DIR}/train.jsonl \
                --val_file ${DATA_DIR}/dev.jsonl \
                --evaluation_strategy steps \
                --load_best_model_at_end \
                --metric_for_best_model eval_acc \
                --greater_is_better True \
                --eval_steps 100 \
                --logging_steps 50 \
                --num_choices 5 \
                --learning_rate ${lr} \
                --warmup_steps ${warm} \
                --num_train_epochs 20 \
                --logging_steps 50 \
                --max_seq_length 64 \
                --output_dir ${MODEL_DIR} \
                --per_device_eval_batch_size 16 \
                --per_device_train_batch_size 16 \
                --gradient_accumulation_steps 1 \
                --fp16 --overwrite_output \
                --overwrite_cache \
                --do_predict \
                --test_file ${DATA_DIR}/test.jsonl \
                --prediction_output ${DATA_DIR}/results/self_test_xlm_result.npy
        fi 
    done
elif [ "$MODE" = "zero-shot" ]; then
    echo "En-X zero-shot cross-lingual transfer" 
    declare -a langs=("en" "zh" "de" "es" "fr" "it" "jap" "nl" "pl" "pt" "ru" "ar" "sw" "ur" "vi" "hi")
    # declare -a langs=("es)
    declare -a splits=("test")
    for lang in "${langs[@]}" 
    do        
        for split in "${splits[@]}" 
        do
            echo "en-${lang} on ${split}"
            DATA_DIR=corpus/CSQA/X-CSQA/${lang}
            MODEL_DIR=/mnt/nfs1/bill/saved_models_xcsqa/xlm_en
            CUDA_VISIBLE_DEVICES=7 python methods/run_mcqa.py \
                --task_name xcsr \
                --exp_name "" \
                --model_name_or_path ${MODEL_DIR} \
                --do_predict \
                --data_dir $DATA_DIR \
                --train_file "" --val_file "" \
                --test_file ${DATA_DIR}/${split}.jsonl \
                --num_choices 5 \
                --max_seq_length 80 \
                --output_dir ${MODEL_DIR} \
                --per_device_eval_batch_size=32 \
                --fp16 --overwrite_output \
                --overwrite_cache \
                --prediction_output ${DATA_DIR}/results/en-${lang}_${split}_xlm_result.npy
        done
    done
else
    echo "Wrong Mode"
fi




# # 
# declare -a langs=("pl" "zh" "de" "es" "fr" "it" "jap" "nl" "pt" "ru") 
# declare -a langs=("en") 
# for lang in "${langs[@]}" 
# do        
#     mv /mnt/nfs1/bill/saved_models_xcsqa/xlm_${lang} /mnt/nfs1/bill/saved_models_xcsqa/xlm_${lang}_backup
# done