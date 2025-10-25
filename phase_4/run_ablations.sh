#!/usr/bin/env bash
MODEL=gpt2-medium
DATA=openwebtext
CRITERIA=(gradient token_frequency)
INTERVALS=(8 16 32)
SEED=42

for CRIT in "${CRITERIA[@]}"; do
  for INT in "${INTERVALS[@]}"; do
    OUTDIR=outputs/ablation_${CRIT}_${INT}_s${SEED}
    LOG=logs/ablation_${CRIT}_${INT}_s${SEED}.txt
    mkdir -p "$OUTDIR" logs
    python fine_tune.py \
      --model $MODEL \
      --dataset ~/datasets/$DATA \
      --rank_criterion $CRIT \
      --rank_intervals $INT \
      --epochs 3 --batch_size 8 --grad_accum 2 --lr 2e-4 --block_size 128 \
      --save_dir $OUTDIR \
      --seed $SEED \
    2>&1 | tee "$LOG"
  done
done
