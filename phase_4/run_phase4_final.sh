#!/bin/bash
# Phase 4: Cross-dataset generalization validation
# Uses learned schedules from Phase 2 on NEW datasets

DATASETS=(ag_news wikitext)
MODELS=(gpt2-medium bert-base-uncased)
SCHEDULES=(ranknet uniform)
SEEDS=(42 7 99)

mkdir -p validation_outputs validation_logs

for DATASET in "${DATASETS[@]}"; do
  DATA_DIR="$HOME/datasets/$DATASET"
  
  if [ ! -d "$DATA_DIR" ]; then
    echo "⚠️ Dataset $DATASET not found, skipping..."
    continue
  fi
  
  for MODEL in "${MODELS[@]}"; do
    for SCHED in "${SCHEDULES[@]}"; do
      for SEED in "${SEEDS[@]}"; do
        OUT="validation_outputs/${DATASET}_${MODEL}_${SCHED}_s${SEED}"
        LOG="validation_logs/${DATASET}_${MODEL}_${SCHED}_s${SEED}.txt"
        mkdir -p "$OUT"
        
        echo "🔄 Running: $DATASET | $MODEL | $SCHED | seed $SEED"
        
        if [ "$SCHED" = "ranknet" ]; then
          python fine_tune.py \
            --model $MODEL \
            --dataset $DATA_DIR \
            --rank_schedule schedules/${MODEL}_features.csv \
            --epochs 1 --batch_size 4 --grad_accum 1 --lr 2e-4 --block_size 128 \
            --save_dir $OUT --seed $SEED 2>&1 | tee "$LOG"
        else
          python fine_tune.py \
            --model $MODEL \
            --dataset $DATA_DIR \
            --uniform_rank 16 \
            --epochs 1 --batch_size 4 --grad_accum 1 --lr 2e-4 --block_size 128 \
            --save_dir $OUT --seed $SEED 2>&1 | tee "$LOG"
        fi
      done
    done
  done
done

echo ""
echo "✅ Phase 4 Complete!"
echo "Total experiments: 2 datasets × 2 models × 2 schedules × 3 seeds = 24 runs"
