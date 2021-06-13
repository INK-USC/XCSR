# wandb login [your token here]
MODE=$1
if [ "$MODE" = "pretrain" ]; then
    echo "Pre-train the xlmrb with MCP examples" 
    DATA_DIR=/path/to/XCSR
    MODEL_DIR=/path/to/saved_models_xlmrb_pretrained
    lr=5e-6
    CUDA_VISIBLE_DEVICES=7,6,2,1  python methods/run_mcqa.py \
        --task_name xcsr \
        --exp_name xlmrb_pretrain \
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
        --logging_steps 50 \
        --eval_steps 300 \
        --warmup_steps 100 \
        --num_choices 8 \
        --learning_rate ${lr} \
        --num_train_epochs 30 \
        --max_seq_length 64 \
        --output_dir ${MODEL_DIR} \
        --per_device_eval_batch_size=16 \
        --per_device_train_batch_size=16 \
        --gradient_accumulation_steps 1 \
        --fp16 --overwrite_output \
        --test_file "" --prediction_output "" \
        --overwrite_cache  

elif [ "$MODE" = "xcsqa-finetune" ]; then
    echo "Self-Tarin and Self-Test."
    declare -a langs=("en" "zh" "de" "es" "fr" "it" "jap" "nl" "pl" "pt" "ru")
    # declare -a lrs=("6e-6" "6e-6" "8e-6" "4e-6" "5e-6" "4e-6" "8e-6" "8e-6" "6e-6" "5e-6" "4e-6")
    # declare -a warms=("100" "100" "100" "100" "300" "100" "100" "100" "100" "300" "100")
    declare -a lrs=("1e-5" "8e-6" "8e-6" "4e-6" "8e-6" "8e-6" "8e-6" "8e-6" "8e-6" "8e-6" "8e-6")
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
            DATA_DIR=corpus/CSQA/X-CSQA/${lang}
            OUTPUT_DIR=/path/to/saved_models_xcsqa/xcsr_xlmrb_${lang}
            CUDA_VISIBLE_DEVICES=6,7 python methods/run_mcqa.py \
                --task_name xcsr \
                --exp_name "pted+xlmrb_${lang}|(${lr}-${warm})" \
                --model_name_or_path /path/to/saved_models_xlmrb_pretrained \
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
                --num_train_epochs 30 \
                --logging_steps 50 \
                --max_seq_length 64 \
                --output_dir ${OUTPUT_DIR} \
                --per_device_eval_batch_size 36 \
                --per_device_train_batch_size 36 \
                --gradient_accumulation_steps 2 \
                --fp16 --overwrite_output \
                --overwrite_cache \
                --do_predict \
                --test_file ${DATA_DIR}/test.jsonl \
                --prediction_output ${DATA_DIR}/results/xcsr_self_test_xlmrb_result.npy
        fi 
    done
elif [ "$MODE" = "xcsqa-infer" ]; then
     echo "En-X zero-shot cross-lingual transfer" 
    declare -a langs=("en" "zh" "de" "es" "fr" "it" "jap" "nl" "pl" "pt" "ru" "ar" "sw" "ur" "vi" "hi")
    # declare -a langs=("ar" "sw" "ur" "vi" "hi")
    MODEL_DIR=/path/to/saved_models_xcsqa/xcsr_xlmrb_en   # TODO: add tokenizer_config.json special_tokens_map.json sentencepiece.bpe.model
    # cp /path/to/saved_models_xcsqa/xcsr_xlmrb_en/tokenizer_config.json $MODEL_DIR/
    # cp /path/to/saved_models_xcsqa/xcsr_xlmrb_en/special_tokens_map.json $MODEL_DIR/
    # cp /path/to/saved_models_xcsqa/xcsr_xlmrb_en/sentencepiece.bpe.model $MODEL_DIR/
    declare -a splits=("test" "dev")
    for lang in "${langs[@]}" 
    do        
        for split in "${splits[@]}" 
        do
            DATA_DIR=corpus/CSQA/X-CSQA/${lang} 
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
                --prediction_output ${DATA_DIR}/results/xcsr_en-${lang}_${split}_xlmrb_result.npy
        done
    done
elif [ "$MODE" = "xcodah-finetune" ]; then
    echo "Self-Tarin and Self-Test."
    declare -a langs=("en" "zh" "de" "es" "fr" "it" "jap" "nl" "pl" "pt" "ru")
    # declare -a lrs=("6e-6" "6e-6" "8e-6" "4e-6" "5e-6" "4e-6" "8e-6" "8e-6" "6e-6" "5e-6" "4e-6")
    # declare -a warms=("100" "100" "100" "100" "300" "100" "100" "100" "100" "300" "100")
    declare -a lrs=("1e-5" "8e-6" "8e-6" "4e-6" "8e-6" "8e-6" "8e-6" "8e-6" "8e-6" "8e-6" "8e-6")
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
            OUTPUT_DIR=/path/to/saved_models_xcodah/xcsr_xlmrb_${lang}
            CUDA_VISIBLE_DEVICES=0,1 python methods/run_mcqa.py \
                --task_name xcsr \
                --exp_name "xcodah:pted+xlmrb_${lang}|(${lr}-${warm})" \
                --model_name_or_path /path/to/saved_models_xlmrb_pretrained \
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
                --num_train_epochs 20 \
                --logging_steps 50 \
                --max_seq_length 100 \
                --output_dir ${OUTPUT_DIR} \
                --per_device_eval_batch_size 32 \
                --per_device_train_batch_size 32 \
                --gradient_accumulation_steps 2 \
                --fp16 --overwrite_output \
                --overwrite_cache \
                --do_predict \
                --test_file ${DATA_DIR}/test.jsonl \
                --prediction_output ${DATA_DIR}/results/xcsr_self_test_xlmrb_result.npy
        fi 
    done
elif [ "$MODE" = "xcodah-infer" ]; then
     echo "En-X zero-shot cross-lingual transfer" 
    declare -a langs=("en" "zh" "de" "es" "fr" "it" "jap" "nl" "pl" "pt" "ru" "ar" "sw" "ur" "vi" "hi")
    MODEL_DIR=/path/to/saved_models_xcodah/xcsr_xlmrb_en   # TODO: add tokenizer_config.json special_tokens_map.json sentencepiece.bpe.model
    declare -a splits=("test" "dev")
    for lang in "${langs[@]}" 
    do        
        for split in "${splits[@]}" 
        do
            DATA_DIR=corpus/CODAH/X-CODAH/${lang} 
            CUDA_VISIBLE_DEVICES=7 python methods/run_mcqa.py \
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
                --per_device_eval_batch_size=32 \
                --fp16 --overwrite_output \
                --overwrite_cache \
                --prediction_output ${DATA_DIR}/results/xcsr_en-${lang}_${split}_xlmrb_result.npy
        done
    done
else
    echo "Wrong Mode"
fi


