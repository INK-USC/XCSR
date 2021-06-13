wandb login 62f222750d0f623c2db59c5f79cf63cfe1aff1b9
MODE=$1
if [ "$MODE" = "self-train" ]; then
    echo "Self-Tarin and Self-Test."
    declare -a langs=("en" "zh" "de" "es" "fr" "it" "jap" "nl" "pl" "pt" "ru")
    for lang in "${langs[@]}" 
    do
        DATA_DIR=corpus/CSQA/X-CSQA/${lang}
        MODEL_DIR=/mnt/nfs1/bill/saved_models_xcsqa/robertabase_${lang}
        CUDA_VISIBLE_DEVICES=4,5,6  python methods/run_mcqa.py \
            --task_name xcsr \
            --model_name_or_path roberta-base \
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
            --num_choices 5 \
            --learning_rate 3e-5 \
            --num_train_epochs 10 \
            --warmup_steps 300 \
            --max_seq_length 80 \
            --output_dir ${MODEL_DIR} \
            --per_device_eval_batch_size=16 \
            --per_device_train_batch_size=16 \
            --gradient_accumulation_steps 1 \
            --fp16 --overwrite_output \
            --overwrite_cache \
            --do_predict \
            --test_file ${DATA_DIR}/test.jsonl \
            --prediction_output ${DATA_DIR}/results/self_test_robertabase_result.npy
    done
elif [ "$MODE" = "zero-shot" ]; then
    echo "En-X zero-shot cross-lingual transfer" 
    declare -a langs=("pl" "zh" "de" "es" "fr" "it" "jap" "nl" "pt" "ru")
    for lang in "${langs[@]}" 
    do
        declare -a splits=("test" "dev")
        for split in "${splits[@]}" 
        do
            DATA_DIR=corpus/CSQA/X-CSQA/${lang}
            MODEL_DIR=/mnt/nfs1/bill/saved_models_xcsqa/robertabase_en
            CUDA_VISIBLE_DEVICES=1,2,3,4  python methods/run_mcqa.py \
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
                --per_device_eval_batch_size=16 \
                --fp16 --overwrite_output \
                --overwrite_cache \
                --prediction_output ${DATA_DIR}/results/en-${lang}_${split}_robertbase_result.npy
        done
    done
else
    echo "Wrong Mode"
fi


