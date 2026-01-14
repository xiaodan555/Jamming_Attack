#!/bin/bash

# Models to evaluate
MODELS=(
    # "/data/longyulei/jamming_attack/cache/Qwen2.5-7B-Instruct"
    "/data/longyulei/jamming_attack/cache/vicuna-13b-v1.5"
    "/data/longyulei/jamming_attack/cache/Llama-2-7b-chat-hf"
)

# Common settings
ORACLE_EMB="contriever"
NUM_QUERIES=100
NUM_ITER=200
ES_ITER=50
MAX_LEN=128

# Create logs directory
mkdir -p logs

echo "Starting Llama experiments at $(date)"

for LLM_MODEL in "${MODELS[@]}"; do
    MODEL_NAME=$(basename "$LLM_MODEL")
    echo "========================================================"
    echo "Processing Model: $MODEL_NAME"
    echo "========================================================"
    
    # --- Clean Phase Loop ---
    echo "--- Starting Clean Phase for all datasets ---"
    for DATASET in nq hotpotqa msmarco; do
        # Special check for Llama-2-7b-chat-hf
        if [[ "$MODEL_NAME" == "Llama-2-7b-chat-hf" && "$DATASET" != "hotpotqa" ]]; then
            echo "Skipping $DATASET for $MODEL_NAME (User requested only hotpotqa)"
            continue
        fi

        echo "Running Clean Phase for $DATASET with $MODEL_NAME..."
        python get_clean_responses.py \
            --llm_model "$LLM_MODEL" \
            --dataset "$DATASET" \
            --num_queries $NUM_QUERIES \
            --max_response_len $MAX_LEN \
            --emb_model "$ORACLE_EMB" \
            > "logs/${MODEL_NAME}_${DATASET}_clean.log" 2>&1
    done

    # --- Attack Phase Loop ---
    echo "--- Starting Attack Phase for all datasets ---"
    for DATASET in nq hotpotqa msmarco; do
        # Special check for Llama-2-7b-chat-hf
        if [[ "$MODEL_NAME" == "Llama-2-7b-chat-hf" && "$DATASET" != "hotpotqa" ]]; then
            echo "Skipping $DATASET for $MODEL_NAME (User requested only hotpotqa)"
            continue
        fi

        echo "Running Attack Phase for $DATASET with $MODEL_NAME..."
        if [ -f "attack.py" ]; then
            python attack.py \
                --llm_model "$LLM_MODEL" \
                --dataset "$DATASET" \
                --num_queries $NUM_QUERIES \
                --num_iterations $NUM_ITER \
                --es_iterations $ES_ITER \
                --oracle_emb_model "$ORACLE_EMB" \
                --emb_model "$ORACLE_EMB" \
                --max_response_len $MAX_LEN \
                > "logs/${MODEL_NAME}_${DATASET}_attack.log" 2>&1
        else
            echo "Warning: attack.py not found! Skipping attack phase for $DATASET."
        fi
    done
done

echo "All Llama experiments finished at $(date)"
