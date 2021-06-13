wandb login 62f222750d0f623c2db59c5f79cf63cfe1aff1b9

MODE=$1
if [ "$MODE" = "self-train" ]; then
    echo "Self-Tarin and Self-Test."
    declare -a langs=("en" "zh" "de" "es" "fr" "it" "jap" "nl" "pl" "pt" "ru")
    # declare -a lrs=("6e-6" "6e-6" "8e-6" "4e-6" "5e-6" "4e-6" "8e-6" "8e-6" "6e-6" "5e-6" "4e-6")
    # declare -a warms=("100" "100" "100" "100" "300" "100" "100" "100" "100" "300" "100")
    declare -a lrs=("6e-6" "8e-6" "8e-6" "4e-6" "8e-6" "8e-6" "8e-6" "8e-6" "8e-6" "8e-6" "8e-6")
    declare -a warms=("100" "100" "100" "100" "100" "100" "100" "100" "100" "100" "100")
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
            DATA_DIR=corpus/CODAH/X-CODAH/${lang}
            MODEL_DIR=/mnt/nfs1/bill/saved_models_xcodah/xlmrl_${lang}
            CUDA_VISIBLE_DEVICES=6,7 python methods/run_mcqa.py \
                --task_name xcsr \
                --exp_name "xcodah:xlmrl_${lang}|(${lr}-${warm})" \
                --model_name_or_path xlm-roberta-large \
                --do_train --do_eval --data_dir $DATA_DIR \
                --train_file ${DATA_DIR}/train.jsonl \
                --val_file ${DATA_DIR}/dev.jsonl \
                --evaluation_strategy steps \
                --load_best_model_at_end \
                --metric_for_best_model eval_acc \
                --greater_is_better True \
                --eval_steps 100 \
                --logging_steps 50 \
                --num_choices 4 \
                --learning_rate ${lr} \
                --warmup_steps ${warm} \
                --num_train_epochs 10 \
                --logging_steps 50 \
                --max_seq_length 100 \
                --output_dir ${MODEL_DIR} \
                --per_device_eval_batch_size 16 \
                --per_device_train_batch_size 16 \
                --gradient_accumulation_steps 2 \
                --fp16 --overwrite_output \
                --overwrite_cache \
                --do_predict \
                --test_file ${DATA_DIR}/test.jsonl \
                --prediction_output ${DATA_DIR}/results/self_test_xlmrl_result.npy
        fi 
    done
elif [ "$MODE" = "zero-shot" ]; then
    echo "En-X zero-shot cross-lingual transfer" 
    declare -a langs=("en" "zh" "de" "es" "fr" "it" "jap" "nl" "pl" "pt" "ru" "ar" "sw" "ur" "vi" "hi")
    declare -a splits=("test" "dev")
    for lang in "${langs[@]}" 
    do        
        for split in "${splits[@]}" 
        do
            DATA_DIR=corpus/CODAH/X-CODAH/${lang}
            MODEL_DIR=/mnt/nfs1/bill/saved_models_xcodah/xlmrl_en
            CUDA_VISIBLE_DEVICES=0 python methods/run_mcqa.py \
                --task_name xcsr \
                --exp_name "" \
                --model_name_or_path ${MODEL_DIR} \
                --do_predict \
                --data_dir $DATA_DIR \
                --train_file "" --val_file "" \
                --test_file ${DATA_DIR}/${split}.jsonl \
                --num_choices 4 \
                --max_seq_length 128 \
                --output_dir ${MODEL_DIR} \
                --per_device_eval_batch_size=16 \
                --fp16 --overwrite_output \
                --overwrite_cache \
                --prediction_output ${DATA_DIR}/results/en-${lang}_${split}_xlmrl_result.npy
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
#     mv /mnt/nfs1/bill/saved_models_xcodah/xlmrl_${lang} /mnt/nfs1/bill/saved_models_xcodah/xlmrl_${lang}_backup
# done