# wandb login [your token here]

DATA_DIR=corpus/CODAH/X-CODAH/en
MODEL_DIR=/path/to/saved_models_xcsqa/robertalarge_en
CUDA_VISIBLE_DEVICES=0,1,2,5  python methods/run_mcqa.py \
    --task_name xcsr \
    --model_name_or_path roberta-large \
    --do_train \
    --do_eval \
    --exp_name xcodah-robertalarge \
    --data_dir $DATA_DIR \
    --train_file ${DATA_DIR}/train.jsonl \
    --val_file ${DATA_DIR}/dev.jsonl \
    --evaluation_strategy steps \
    --load_best_model_at_end \
    --metric_for_best_model eval_acc \
    --greater_is_better True \
    --eval_steps 100 \
    --logging_steps 50 \
    --num_choices 4 \
    --learning_rate 1e-5 \
    --num_train_epochs 5 \
    --max_seq_length 100 \
    --output_dir ${MODEL_DIR} \
    --per_device_eval_batch_size=8 \
    --per_device_train_batch_size=16 \
    --gradient_accumulation_steps 1 \
    --fp16 --overwrite_output \
    --overwrite_cache \
    --do_predict \
    --test_file ${DATA_DIR}/test.jsonl \
    --prediction_output ${DATA_DIR}/results/test_robertalarge_result.npy 