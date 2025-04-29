#!/bin/bash

# Configuration
TOKENIZER_MERGES="merges.pkl"
TOKENIZER_VOCAB="vocab.pkl"
TRAIN_DATA="data/TinyStoriesV2-GPT4-train.txt"
VAL_DATA="data/TinyStoriesV2-GPT4-valid.txt"
BATCH_SIZE=128

# Learning rates to test (4 values)
LEARNING_RATES=(0.0001 0.0003 0.001 0.003)

echo "Running learning rate ablation with batch_size=${BATCH_SIZE}"
echo "----------------------------------------"

# Run experiments for each learning rate
for lr in "${LEARNING_RATES[@]}"; do
    echo "Running experiment with learning_rate=${lr}"
    
    # Create a unique wandb run name
    RUN_NAME="lr${lr}_bs${BATCH_SIZE}"
    
    # Run training with these hyperparameters
    uv run python -m cs336_basics.train \
        --tokenizer_merges_path ${TOKENIZER_MERGES} \
        --tokenizer_vocab_path ${TOKENIZER_VOCAB} \
        --train_data_path ${TRAIN_DATA} \
        --val_data_path ${VAL_DATA} \
        --max_learning_rate ${lr} \
        --batch_size ${BATCH_SIZE} \
        --project_name "cs336_ablation_${RUN_NAME}"
    
    echo "Completed experiment with learning_rate=${lr}"
    echo "----------------------------------------"
    
    # Sleep for a few seconds to let GPU memory clear
    sleep 5
done

echo "All learning rate ablation experiments completed!"



# Learning Rate Tuning:


# Tune Batch Size