#!/bin/bash

# 定义通用变量
LLM_MODEL="vicuna-7b-v1.5"
ORACLE_EMB="contriever"  # Changed from gtr-base
NUM_QUERIES=100
NUM_ITER=200
ES_ITER=50
MAX_LEN=128

# 创建日志目录
mkdir -p logs

echo "Starting experiments at $(date)"

# # --- 1. NQ Dataset ---
# echo "========================================================"
# echo "Running NQ Dataset..."
# echo "========================================================"

# # Check if BEIR eval file exists
# NQ_BEIR_FILE="./data/beir_eval/nq-test-contriever-dot.json"
# if [ ! -f "$NQ_BEIR_FILE" ]; then
#     echo "[NQ] BEIR evaluation file not found at $NQ_BEIR_FILE."
#     echo "[NQ] Waiting for eval_beir.py to finish..."
# else
#     # Clean Phase
#     echo "[NQ] Running Clean Phase..."
#     python get_clean_responses.py \
#         --llm_model $LLM_MODEL \
#         --dataset nq \
#         --num_queries $NUM_QUERIES \
#         --max_response_len $MAX_LEN \
#         --emb_model $ORACLE_EMB \
#         > logs/nq_clean.log 2>&1

#     if [ $? -ne 0 ]; then
#         echo "[NQ] Clean phase failed! Check logs/nq_clean.log"
#     else
#         echo "[NQ] Clean phase completed."
        
#         # Attack Phase
#         echo "[NQ] Running Attack Phase..."
#         python attack.py \
#             --llm_model $LLM_MODEL \
#             --dataset nq \
#             --num_queries $NUM_QUERIES \
#             --num_iterations $NUM_ITER \
#             --es_iterations $ES_ITER \
#             --oracle_emb_model $ORACLE_EMB \
#             --emb_model $ORACLE_EMB \
#             --max_response_len $MAX_LEN \
#             > logs/nq_attack.log 2>&1
            
#         if [ $? -ne 0 ]; then
#             echo "[NQ] Attack phase failed! Check logs/nq_attack.log"
#         else
#             echo "[NQ] Attack phase completed."
#         fi
#     fi
# fi

# --- 2. HotpotQA Dataset ---
echo "========================================================"
echo "Running HotpotQA Dataset..."
echo "========================================================"

# Check if BEIR eval file exists
HOTPOTQA_BEIR_FILE="./data/beir_eval/hotpotqa-test-contriever-dot.json"
if [ ! -f "$HOTPOTQA_BEIR_FILE" ]; then
    echo "[HotpotQA] BEIR evaluation file not found at $HOTPOTQA_BEIR_FILE."
    echo "[HotpotQA] Waiting for eval_beir.py to finish..."
else
    # Clean Phase
    echo "[HotpotQA] Running Clean Phase..."
    python get_clean_responses.py \
        --llm_model $LLM_MODEL \
        --dataset hotpotqa \
        --num_queries $NUM_QUERIES \
        --max_response_len $MAX_LEN \
        --emb_model $ORACLE_EMB \
        > logs/hotpotqa_clean.log 2>&1

    if [ $? -ne 0 ]; then
        echo "[HotpotQA] Clean phase failed! Check logs/hotpotqa_clean.log"
    else
        echo "[HotpotQA] Clean phase completed."
        
        # Attack Phase
        echo "[HotpotQA] Running Attack Phase..."
        python attack.py \
            --llm_model $LLM_MODEL \
            --dataset hotpotqa \
            --num_queries $NUM_QUERIES \
            --num_iterations $NUM_ITER \
            --es_iterations $ES_ITER \
            --oracle_emb_model $ORACLE_EMB \
            --emb_model $ORACLE_EMB \
            --max_response_len $MAX_LEN \
            > logs/hotpotqa_attack.log 2>&1
            
        if [ $? -ne 0 ]; then
            echo "[HotpotQA] Attack phase failed! Check logs/hotpotqa_attack.log"
        else
            echo "[HotpotQA] Attack phase completed."
        fi
    fi
fi

# # --- 3. MS MARCO Dataset ---
# echo "========================================================"
# echo "Running MS MARCO Dataset..."
# echo "========================================================"

# # Check if BEIR eval file exists
# MSMARCO_BEIR_FILE="./data/beir_eval/msmarco-dev-contriever-dot.json"
# if [ ! -f "$MSMARCO_BEIR_FILE" ]; then
#     echo "[MS MARCO] BEIR evaluation file not found at $MSMARCO_BEIR_FILE."
#     echo "[MS MARCO] Waiting for eval_beir.py to finish..."
# else
#     # Clean Phase
#     echo "[MS MARCO] Running Clean Phase..."
#     python get_clean_responses.py \
#         --llm_model $LLM_MODEL \
#         --dataset msmarco \
#         --num_queries $NUM_QUERIES \
#         --max_response_len $MAX_LEN \
#         --emb_model $ORACLE_EMB \
#         > logs/msmarco_clean.log 2>&1

#     if [ $? -ne 0 ]; then
#         echo "[MS MARCO] Clean phase failed! Check logs/msmarco_clean.log"
#     else
#         echo "[MS MARCO] Clean phase completed."
        
#         # Attack Phase
#         echo "[MS MARCO] Running Attack Phase..."
#         python attack.py \
#             --llm_model $LLM_MODEL \
#             --dataset msmarco \
#             --num_queries $NUM_QUERIES \
#             --num_iterations $NUM_ITER \
#             --es_iterations $ES_ITER \
#             --oracle_emb_model $ORACLE_EMB \
#             --emb_model $ORACLE_EMB \
#             --max_response_len $MAX_LEN \
#             > logs/msmarco_attack.log 2>&1
            
#         if [ $? -ne 0 ]; then
#             echo "[MS MARCO] Attack phase failed! Check logs/msmarco_attack.log"
#         else
#             echo "[MS MARCO] Attack phase completed."
#         fi
#     fi
# fi

echo "All experiments finished at $(date)"
