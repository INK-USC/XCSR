# wandb login [your token here]
MODE=$1
if [ "$MODE" = "self-train" ]; then
    echo "Self-Tarin and Self-Test."
    # declare -a langs=("en" "zh" "de" "es" "fr" "it" "jap" "nl" "pl" "pt" "ru")
    declare -a langs=("en")
    declare -a lrs=("3e-5" "2e-5" "2e-5" "2e-5" "2e-5" "2e-5" "2e-5" "2e-5" "2e-5" "2e-5" "2e-5")
    for index in ${!langs[*]}; 
    do 
        lang=${langs[$index]}
        lr=${lrs[$index]}

        DATA_DIR=corpus/CSQA/X-CSQA/${lang}
        MODEL_DIR=/mnt/nfs1/bill/saved_models_xcsqa/mbert_${lang}
        CUDA_VISIBLE_DEVICES=3,0 python methods/run_mcqa.py \
            --task_name xcsr \
            --exp_name mbert_${lang} \
            --model_name_or_path bert-base-multilingual-cased \
            --do_train \
            --do_eval \
            --data_dir $DATA_DIR \
            --train_file ${DATA_DIR}/train.jsonl \
            --val_file ${DATA_DIR}/dev.jsonl \
            --evaluation_strategy steps \
            --load_best_model_at_end \
            --metric_for_best_model eval_acc \
            --greater_is_better True \
            --logging_steps 50 \
            --eval_steps 100 \
            --warmup_steps 100 \
            --num_choices 5 \
            --learning_rate ${lr} \
            --num_train_epochs 30 \
            --max_seq_length 64 \
            --output_dir ${MODEL_DIR} \
            --per_device_eval_batch_size=32 \
            --per_device_train_batch_size=32 \
            --gradient_accumulation_steps 1 \
            --fp16 --overwrite_output \
            --overwrite_cache \
            --do_predict \
            --test_file ${DATA_DIR}/test.jsonl \
            --prediction_output ${DATA_DIR}/results/self_test_mbert_result.npy 
    done
elif [ "$MODE" = "zero-shot" ]; then
    echo "En-X zero-shot cross-lingual transfer" 
    # declare -a langs=("pl" "zh" "de" "es" "fr" "it" "jap" "nl" "pt" "ru" "ar" "sw" "ur" "vi" "hi")
    # declare -a langs=("ar" "sw" "ur" "vi" "hi")
    declare -a langs=("en" "zh" "de" "es" "fr" "it" "jap" "nl" "pl" "pt" "ru" "ar" "sw" "ur" "vi" "hi")
    declare -a splits=("test" "dev")
    for lang in "${langs[@]}" 
    do        
        for split in "${splits[@]}" 
        do
            DATA_DIR=corpus/CSQA/X-CSQA/${lang}
            MODEL_DIR=/mnt/nfs1/bill/saved_models_xcsqa/mbert_en
            CUDA_VISIBLE_DEVICES=7  python methods/run_mcqa.py \
                --task_name xcsr \
                --exp_name "" \
                --model_name_or_path ${MODEL_DIR} \
                --do_predict \
                --data_dir $DATA_DIR \
                --train_file "" --val_file "" \
                --test_file ${DATA_DIR}/${split}.jsonl \
                --num_choices 5 \
                --max_seq_length 128 \
                --output_dir ${MODEL_DIR} \
                --per_device_eval_batch_size=16 \
                --fp16 --overwrite_output \
                --overwrite_cache \
                --prediction_output ${DATA_DIR}/results/en-${lang}_${split}_mbert_result.npy
        done
    done
elif [ "$MODE" = "self-infer" ]; then
    echo "En-X zero-shot cross-lingual transfer" 
    declare -a langs=("pl" "zh" "de" "es" "fr" "it" "jap" "nl" "pt" "ru")
    declare -a splits=("dev")
    for lang in "${langs[@]}" 
    do        
        for split in "${splits[@]}" 
        do
            DATA_DIR=corpus/CSQA/X-CSQA/${lang}
            MODEL_DIR=/mnt/nfs1/bill/saved_models_xcsqa/mbert_${lang}
            CUDA_VISIBLE_DEVICES=1,2,3,4,5,6  python methods/run_mcqa.py \
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
                --prediction_output ${DATA_DIR}/results/self_${lang}_${split}_mbert_result.npy
        done
    done
else
    echo "Wrong Mode"
fi


