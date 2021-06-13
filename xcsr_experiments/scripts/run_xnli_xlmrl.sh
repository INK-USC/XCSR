wandb login 62f222750d0f623c2db59c5f79cf63cfe1aff1b9
MODE=$1
if [ "$MODE" = "train-original" ]; then
    echo "Train the original model"
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
            CUDA_VISIBLE_DEVICES=2,3,6,7 \
            python methods/simple_run_xnli.py \
                --mode train \
                --n_gpu 4 \
                --model_type xlmroberta \
                --model_name xlm-roberta-large \
                --exp_name xnli-xlmrl-en \
                --train_batch_size 64 \
                --eval_batch_size 64 \
                --max_seq_length 256 \
                --learning_rate ${lr} \
                --warmup_steps ${warm} \
                --num_train_epochs 5 \
                --dev_lang en \
                --output_dir /mnt/nfs1/bill/saved_models_xnli/xlmrl_en
        fi
    done
elif [ "$MODE" = "train-xcsr" ]; then
    echo "Train the xcsr model"
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
            CUDA_VISIBLE_DEVICES=2,3,6,7 \
            python methods/simple_run_xnli.py \
                --mode train \
                --n_gpu 4 \
                --model_type xlmroberta \
                --model_name /mnt/nfs1/bill/saved_models_xcsqa/xlmrl_pretrained \
                --exp_name xnli-xcsr-xlmrl-en \
                --train_batch_size 64 \
                --eval_batch_size 64 \
                --max_seq_length 256 \
                --learning_rate ${lr} \
                --warmup_steps ${warm} \
                --num_train_epochs 5 \
                --dev_lang en \
                --output_dir /mnt/nfs1/bill/saved_models_xnli/xcsr_xlmrl_en
        fi
    done
elif [ "$MODE" = "zero-shot" ]; then
    echo "En-X zero-shot cross-lingual transfer"
    CUDA_VISIBLE_DEVICES=2,3,6,7 \
    python methods/simple_run_xnli.py \
        --mode test_all \
        --n_gpu 4 \
        --model_type xlmroberta \
        --model_name /mnt/nfs1/bill/saved_models_xnli/xlmrl_en/checkpoint-12272-epoch-2/ \
        --eval_batch_size 512 \
        --max_seq_length 256 \
        --dev_lang en \
        --output_dir /mnt/nfs1/bill/saved_models_xnli/xlmrl_en/checkpoint-12272-epoch-2/
else
    echo "Wrong Mode"
fi




# # 
# declare -a langs=("pl" "zh" "de" "es" "fr" "it" "jap" "nl" "pt" "ru") 
# declare -a langs=("en") 
# for lang in "${langs[@]}" 
# do        
#     mv /mnt/nfs1/bill/saved_models_xcsqa/xlmrl_${lang} /mnt/nfs1/bill/saved_models_xcsqa/xlmrl_${lang}_backup
# done