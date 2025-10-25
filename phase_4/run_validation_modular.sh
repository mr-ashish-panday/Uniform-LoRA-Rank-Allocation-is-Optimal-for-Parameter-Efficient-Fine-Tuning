#!/bin/bash

# Part 1: OpenWebText only (2 models × 2 schedules × 7 seeds = 28 jobs)
echo "=== PART 1: OpenWebText ==="
DATASET="openwebtext"
DATA_DIR="$HOME/datasets/$DATASET"
MODELS=(gpt2-medium bert-base-uncased)
SEEDS=(42 7 99 123 2024 100 2025)

mkdir -p outputs logs

for MODEL in "${MODELS[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    # Ranknet
    OUT_R="outputs/${DATASET}/${MODEL}_ranknet_s${SEED}"
    LOG_R="logs/${DATASET}_${MODEL}_ranknet_s${SEED}.txt"
    mkdir -p "$OUT_R"
    python fine_tune.py --model "$MODEL" --dataset "$DATA_DIR" --rank_schedule "schedules/${MODEL}_ranknet.csv" --epochs 3 --batch_size 8 --grad_accum 2 --lr 2e-4 --block_size 128 --save_dir "$OUT_R" --seed "$SEED" 2>&1 | tee "$LOG_R"

    # Uniform
    OUT_U="outputs/${DATASET}/${MODEL}_uniform_s${SEED}"
    LOG_U="logs/${DATASET}_${MODEL}_uniform_s${SEED}.txt"
    mkdir -p "$OUT_U"
    python fine_tune.py --model "$MODEL" --dataset "$DATA_DIR" --uniform_rank 16 --epochs 3 --batch_size 8 --grad_accum 2 --lr 2e-4 --block_size 128 --save_dir "$OUT_U" --seed "$SEED" 2>&1 | tee "$LOG_U"
  done
done

echo "=== PART 1 COMPLETE. Archive outputs/logs before proceeding to PART 2 ==="
echo "Run: tar -czf openwebtext_results.tar.gz outputs/ logs/"
echo "Then: rm -rf outputs/ logs/"
echo "Then run: bash run_validation_part2.sh"
