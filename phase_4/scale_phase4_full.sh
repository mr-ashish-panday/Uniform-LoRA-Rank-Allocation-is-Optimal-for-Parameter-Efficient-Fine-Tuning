#!/bin/bash
# Phase 4 FULL SCALE: Production-grade validation
# Goal: Publish-ready hierarchical LoRA generalization results

DATASETS=(ag_news wikitext)
MODELS=(gpt2-medium bert-base-uncased)
SCHEDULES=(ranknet uniform)
SEEDS=(42 7 99 123 2024 100 2025 555 777 999)  # 10 seeds for robustness

mkdir -p final_outputs final_logs analysis

echo "=========================================="
echo "Phase 4 FULL SCALE Validation"
echo "=========================================="
echo "Datasets: ${DATASETS[@]}"
echo "Models: ${MODELS[@]}"
echo "Schedules: ${SCHEDULES[@]}"
echo "Seeds: ${#SEEDS[@]} (statistical power)"
echo "Total runs: $((${#DATASETS[@]} * ${#MODELS[@]} * ${#SCHEDULES[@]} * ${#SEEDS[@]}))"
echo "=========================================="

TOTAL=$((${#DATASETS[@]} * ${#MODELS[@]} * ${#SCHEDULES[@]} * ${#SEEDS[@]}))
COUNT=0

for DATASET in "${DATASETS[@]}"; do
  DATA_DIR="$HOME/datasets/$DATASET"
  
  if [ ! -d "$DATA_DIR" ]; then
    echo "❌ Dataset $DATASET not found"
    continue
  fi
  
  for MODEL in "${MODELS[@]}"; do
    for SCHED in "${SCHEDULES[@]}"; do
      for SEED in "${SEEDS[@]}"; do
        COUNT=$((COUNT + 1))
        PROGRESS=$((COUNT * 100 / TOTAL))
        
        OUT="final_outputs/${DATASET}_${MODEL}_${SCHED}_s${SEED}"
        LOG="final_logs/${DATASET}_${MODEL}_${SCHED}_s${SEED}.txt"
        mkdir -p "$OUT"
        
        echo "[${PROGRESS}%] ($COUNT/$TOTAL) Running: $DATASET | $MODEL | $SCHED | seed $SEED"
        
        if [ "$SCHED" = "ranknet" ]; then
          python fine_tune.py \
            --model $MODEL \
            --dataset $DATA_DIR \
            --rank_schedule schedules/${MODEL}_features.csv \
            --epochs 2 --batch_size 8 --grad_accum 2 --lr 2e-4 --block_size 256 \
            --save_dir $OUT --seed $SEED 2>&1 | tee "$LOG"
        else
          python fine_tune.py \
            --model $MODEL \
            --dataset $DATA_DIR \
            --uniform_rank 16 \
            --epochs 2 --batch_size 8 --grad_accum 2 --lr 2e-4 --block_size 256 \
            --save_dir $OUT --seed $SEED 2>&1 | tee "$LOG"
        fi
      done
    done
  done
done

echo ""
echo "✅ Phase 4 FULL SCALE Complete!"
echo "Aggregate results with: python3 aggregate_phase4_full.py"
