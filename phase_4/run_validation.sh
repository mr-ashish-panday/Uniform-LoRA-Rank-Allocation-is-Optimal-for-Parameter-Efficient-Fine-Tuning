#!/usr/bin/env bash
MODEL=gpt2-large
DATA=openwebtext
SCHEDULES=(ranknet uniform)
SEEDS=(42 7 99 123 2024 100 2025)

for SCHED in "${SCHEDULES[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    OUTDIR=outputs/${MODEL}_${SCHED}_s${SEED}
    LOG=logs/${MODEL}_${SCHED}_s${SEED}.txt
    mkdir -p "$OUTDIR" logs
    
    # Determine which arguments to use based on schedule type
    if [ "$SCHED" = "ranknet" ]; then
      # Use rank_schedule CSV file for ranknet
      python fine_tune.py \
        --model $MODEL \
        --dataset ~/datasets/$DATA \
        --rank_schedule schedules/${MODEL}_ranknet.csv \
        --epochs 3 --batch_size 4 --grad_accum 4 --lr 2e-4 --block_size 128 \
        --save_dir $OUTDIR \
        --seed $SEED \
        2>&1 | tee "$LOG"
    else
      # Use uniform_rank for uniform schedule
      python fine_tune.py \
        --model $MODEL \
        --dataset ~/datasets/$DATA \
        --uniform_rank 16 \
        --epochs 3 --batch_size 4 --grad_accum 4 --lr 2e-4 --block_size 128 \
        --save_dir $OUTDIR \
        --seed $SEED \
        2>&1 | tee "$LOG"
    fi
  done
done