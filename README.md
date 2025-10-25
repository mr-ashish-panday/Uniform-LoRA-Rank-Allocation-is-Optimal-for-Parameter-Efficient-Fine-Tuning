# Hierarchical LoRA Rank Allocation: Comprehensive Benchmarking

**Main Finding:** Uniform rank allocation outperforms learned hierarchical strategies by **1.23-7.86%** across all models and datasets with **p < 0.001***

## Quick Summary

| Metric | Result |
|--------|--------|
| Experiments | 28 (7 seeds × 4 conditions) |
| Models | GPT-2 Medium, BERT-base |
| Datasets | AG News, WikiText-2 |
| Winner | Uniform allocation (100% win rate) |
| Avg Gap | +3.71% (hierarchical worse) |
| Significance | All p < 0.001*** |

## Results Table

| Condition | Hierarchical | Uniform | Gap (%) | p-value |
|-----------|--------------|---------|---------|---------|
| GPT2+AG | 38.69±1.59 | **38.22±1.69** | +1.23 | 0.000869*** |
| GPT2+Wiki | 32.87±1.81 | **32.42±1.77** | +1.40 | 0.000039*** |
| BERT+AG | 19.86±1.82 | **18.42±1.68** | +7.86 | 0.000085*** |
| BERT+Wiki | 9.33±0.66 | **8.95±0.64** | +4.34 | 0.000046*** |

## Repository Structure


# You're in ~/hierarchical-lora-research
# Files are ALREADY copied!

# Step 1: Create .gitignore
cat > .gitignore << 'EOF'
*.safetensors
*.bin
*.pt
*.ckpt
checkpoint*/
ml_env/
venv/
__pycache__/
*.log
