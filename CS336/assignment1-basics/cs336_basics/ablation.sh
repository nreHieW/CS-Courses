#!/bin/bash

TOKENIZER_MERGES="merges.pkl"
TOKENIZER_VOCAB="vocab.pkl"
TRAIN_DATA="data/TinyStoriesV2-GPT4-train.txt"
VAL_DATA="data/TinyStoriesV2-GPT4-valid.txt"
WANDB_PROJECT="cs336"

BATCH_SIZES=(1 64 128 512 2048 )

MAX_BATCH_SIZE=1
for bs in "${BATCH_SIZES[@]}"; do
    RUN_NAME="cs336_ablation_batch_test_bs${bs}"
    
    if uv run python -m cs336_basics.train \
        --tokenizer_merges_path ${TOKENIZER_MERGES} \
        --tokenizer_vocab_path ${TOKENIZER_VOCAB} \
        --train_data_path ${TRAIN_DATA} \
        --val_data_path ${VAL_DATA} \
        --batch_size ${bs} \
        --run_name "${RUN_NAME}"; then
        
        MAX_BATCH_SIZE=${bs}
    else
        break
    fi
    
    sleep 5
done

LEARNING_RATES=(0.0001 0.0003 0.001 0.003)

for lr in "${LEARNING_RATES[@]}"; do
    RUN_NAME="cs336_ablation_lr${lr}_bs${MAX_BATCH_SIZE}"
    
    uv run python -m cs336_basics.train \
        --tokenizer_merges_path ${TOKENIZER_MERGES} \
        --tokenizer_vocab_path ${TOKENIZER_VOCAB} \
        --train_data_path ${TRAIN_DATA} \
        --val_data_path ${VAL_DATA} \
        --max_learning_rate ${lr} \
        --batch_size ${MAX_BATCH_SIZE} \
        --run_name "${RUN_NAME}"
    
    sleep 5
done
