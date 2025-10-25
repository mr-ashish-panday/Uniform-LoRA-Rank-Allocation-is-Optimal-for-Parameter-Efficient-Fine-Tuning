#!/usr/bin/env bash
set -euo pipefail

# Activate env to ensure correct python and deps
source ~/paper_5/phase_2/ml_env/bin/activate

cd ~/paper_5/phase_3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128

echo "=== Starting GPT-2 Uniform ==="
python fine_tune.py \
  --model gpt2-medium \
  --dataset ~/datasets/wiki_text \
  --uniform_rank 16 \
  --epochs 3 \
  --batch_size 4 \
  --grad_accum 4 \
  --lr 2e-4 \
  --block_size 256 \
  --save_dir outputs/gpt2_uniform_wt2 \
  > logs/gpt2_uniform_wt2.txt 2>&1
echo "GPT-2 Uniform completed"

echo "=== Starting BERT RankNet ==="
python fine_tune.py \
  --model bert-base-uncased \
  --dataset ~/datasets/wiki_text \
  --rank_schedule ../phase_2/schedules/schedule_bert-base-uncased_features.csv \
  --epochs 3 \
  --batch_size 8 \
  --grad_accum 2 \
  --lr 2e-4 \
  --block_size 128 \
  --save_dir outputs/bert_ranknet_wt2 \
  > logs/bert_ranknet_wt2.txt 2>&1
echo "BERT RankNet completed"

echo "=== Starting BERT Uniform ==="
python fine_tune.py \
  --model bert-base-uncased \
  --dataset ~/datasets/wiki_text \
  --uniform_rank 16 \
  --epochs 3 \
  --batch_size 8 \
  --grad_accum 2 \
  --lr 2e-4 \
  --block_size 128 \
  --save_dir outputs/bert_uniform_wt2 \
  > logs/bert_uniform_wt2.txt 2>&1
echo "BERT Uniform completed"

echo "=== Starting DistilBERT RankNet ==="
python fine_tune.py \
  --model distilbert-base-uncased \
  --dataset ~/datasets/wiki_text \
  --rank_schedule ../phase_2/schedules/schedule_distilbert-base-uncased_features.csv \
  --epochs 3 \
  --batch_size 8 \
  --grad_accum 2 \
  --lr 2e-4 \
  --block_size 128 \
  --save_dir outputs/distilbert_ranknet_wt2 \
  > logs/distilbert_ranknet_wt2.txt 2>&1
echo "DistilBERT RankNet completed"

echo "=== Starting DistilBERT Uniform ==="
python fine_tune.py \
  --model distilbert-base-uncased \
  --dataset ~/datasets/wiki_text \
  --uniform_rank 16 \
  --epochs 3 \
  --batch_size 8 \
  --grad_accum 2 \
  --lr 2e-4 \
  --block_size 128 \
  --save_dir outputs/distilbert_uniform_wt2 \
  > logs/distilbert_uniform_wt2.txt 2>&1
echo "DistilBERT Uniform completed"

echo "=== Summarizing results ==="
python summarize_phase3.py | tee phase3_results.md
echo "=== All jobs completed! ==="
