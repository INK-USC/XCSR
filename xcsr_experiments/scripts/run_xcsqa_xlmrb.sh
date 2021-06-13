# wandb login [your token here]

MODE=$1
if [ "$MODE" = "en_train" ]; then
    echo "Start Training"
    declare -a langs=("en" "zh" "de" "es" "fr" "it" "jap" "nl" "pl" "pt" "ru")
    for lang in "${langs[@]}" 
    do
        DATA_DIR=/path/to/X-CSQA/${lang}
        MODEL_DIR=/path/to/saved_models_xcsqa/xlmrb_${lang}
        CUDA_VISIBLE_DEVICES=1,2,3,0 python xcsr_experiments/run_mcqa.py \
            --task_name xcsr \
            --exp_name xlmrb_${lang} \
            --model_name_or_path xlm-roberta-base \
            --do_train \
            --do_eval \
            --data_dir $DATA_DIR \
            --train_file ${DATA_DIR}/train.jsonl \
            --val_file ${DATA_DIR}/dev.jsonl \
            --evaluation_strategy steps \
            --load_best_model_at_end \
            --metric_for_best_model eval_acc \
            --greater_is_better True \
            --eval_steps 100 \
            --logging_steps 50 \
            --warmup_steps 100 \
            --num_choices 5 \
            --learning_rate 1e-5 \
            --num_train_epochs 30 \
            --max_seq_length 64 \
            --output_dir ${MODEL_DIR} \
            --per_device_eval_batch_size=12 \
            --per_device_train_batch_size=12 \
            --gradient_accumulation_steps 3 \
            --fp16 --overwrite_output \
            --overwrite_cache \
            --do_predict \
            --test_file ${DATA_DIR}/test.jsonl \
            --prediction_output ${DATA_DIR}/results/self_test_xlmrb_result.npy
    done
elif [ "$MODE" = "all_infer" ]; then
    echo "En-X zero-shot cross-lingual transfer" 
    declare -a langs=("en" "pl" "zh" "de" "es" "fr" "it" "jap" "nl" "pt" "ru" "ar" "sw" "ur" "vi" "hi")
    declare -a langs=("ar" "sw" "ur" "vi" "hi")
    declare -a splits=("test" "dev")
    for lang in "${langs[@]}" 
    do        
        for split in "${splits[@]}" 
        do
            DATA_DIR=/path/to/X-CSQA/${lang}
            MODEL_DIR=/path/to/saved_models_xcsqa/xlmrb_en
            CUDA_VISIBLE_DEVICES=7  python xcsr_experiments/run_mcqa.py \
                --task_name xcsr \
                --exp_name "" \
                --model_name_or_path ${MODEL_DIR} \
                --do_predict \
                --data_dir $DATA_DIR \
                --train_file "" --val_file "" \
                --test_file ${DATA_DIR}/${split}.jsonl \
                --num_choices 5 \
                --max_seq_length 100 \
                --output_dir ${MODEL_DIR} \
                --per_device_eval_batch_size=8 \
                --fp16 --overwrite_output \
                --overwrite_cache \
                --prediction_output ${DATA_DIR}/results/en-${lang}_${split}_xlmrb_result.npy
        done
    done
else
    echo "Wrong Mode"
fi


