#!/usr/bin/env bash
set -euo pipefail

DATA=~/datasets/wiki_text
SCHEDULE_DIR=$DATA  # This is wrong; SCHEDULE_DIR should be set to the data schedule path.

# Correctly set SCHEDULE_DIR now:
SCHEDULE_DIR=../phase_2/schedules

# Expanded seed list for robust statistics
SEEDS=(42 100 2025 7 99 123 2024)

declare -a JOBS=(
  "gpt2-medium,ranknet,$SCHEDULE_DIR/schedule_gpt2-medium_features.csv,0,gpt2_ranknet_wt2"
  "gpt2-medium,uniform,,16,gpt2_uniform_wt2"
  "bert-base-uncased,ranknet,$SCHEDULE_DIR/schedule_bert-base-uncased_features.csv,0,bert_ranknet_wt2"
  "bert-base-uncased,uniform,,16,bert_uniform_wt2"
  "distilbert-base-uncased,ranknet,$SCHEDULE_DIR/schedule_distilbert-base-uncased_features.csv,0,distilbert_ranknet_wt2"
  "distilbert-base-uncased,uniform,,16,distilbert_uniform_wt2"
)

mkdir -p logs

for seed in "${SEEDS[@]}"; do
  for job in "${JOBS[@]}"; do
    IFS=',' read -r MODEL SCHED CSV RANK OUTDIR <<< "${job}"
    SAVE="outputs/${OUTDIR}_s${seed}"
    LOG="logs/${OUTDIR}_s${seed}.txt"
    mkdir -p "${SAVE}"
    echo "==> Running ${MODEL} ${SCHED} seed=${seed}"
    if [[ "${SCHED}" == "ranknet" ]]; then
      python fine_tune.py \
        --model "${MODEL}" \
        --dataset "${DATA}" \
        --rank_schedule "${CSV}" \
        --epochs 3 --batch_size 8 --grad_accum 2 --lr 2e-4 --block_size 128 \
        --save_dir "${SAVE}" --seed "${seed}" \
        > "${LOG}" 2>&1
    else
      python fine_tune.py \
        --model "${MODEL}" \
        --dataset "${DATA}" \
        --uniform_rank "${RANK}" \
        --epochs 3 --batch_size 8 --grad_accum 2 --lr 2e-4 --block_size 128 \
        --save_dir "${SAVE}" --seed "${seed}" \
        > "${LOG}" 2>&1
    fi
  done
done

echo "All seeds completed."
