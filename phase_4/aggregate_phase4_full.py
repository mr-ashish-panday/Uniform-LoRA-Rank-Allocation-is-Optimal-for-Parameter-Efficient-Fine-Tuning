#!/usr/bin/env python3
import pandas as pd
import json
from pathlib import Path
from scipy import stats

results = []
for out_dir in Path("final_outputs").glob("*"):
    if (out_dir / "eval_summary.txt").exists():
        with open(out_dir / "eval_summary.txt") as f:
            data = {}
            for line in f:
                if '=' in line:
                    key, val = line.strip().split('=')
                    try:
                        data[key] = float(val)
                    except:
                        data[key] = val
            
            # Parse experiment name
            parts = out_dir.name.split('_')
            data['dataset'] = parts[0]
            data['model'] = parts[1]
            data['schedule'] = parts[2]
            data['seed'] = int(parts[3].replace('s', ''))
            data['experiment'] = out_dir.name
            results.append(data)

df = pd.DataFrame(results)

# Save raw results
df.to_csv("phase4_full_results.csv", index=False)
print("✅ Raw results: phase4_full_results.csv")

# Statistical summary
print("\n" + "="*80)
print("PHASE 4 FULL SCALE - STATISTICAL SUMMARY")
print("="*80)

for model in sorted(df['model'].unique()):
    print(f"\n{'='*80}")
    print(f"MODEL: {model.upper()}")
    print(f"{'='*80}")
    
    model_df = df[df['model'] == model]
    
    for dataset in sorted(model_df['dataset'].unique()):
        print(f"\n📊 Dataset: {dataset}")
        dataset_df = model_df[model_df['dataset'] == dataset]
        
        for schedule in ['ranknet', 'uniform']:
            sched_df = dataset_df[dataset_df['schedule'] == schedule]
            if len(sched_df) > 0:
                ppl = sched_df['perplexity'].values
                mean_ppl = ppl.mean()
                std_ppl = ppl.std()
                
                print(f"  {schedule:8s}: {mean_ppl:.2f} ± {std_ppl:.2f} (n={len(ppl)})")

# T-tests for significance
print("\n" + "="*80)
print("STATISTICAL SIGNIFICANCE TESTS (Wilcoxon, p<0.05)")
print("="*80)

for model in sorted(df['model'].unique()):
    model_df = df[df['model'] == model]
    
    for dataset in sorted(model_df['dataset'].unique()):
        dataset_df = model_df[model_df['dataset'] == dataset]
        
        ranknet = dataset_df[dataset_df['schedule'] == 'ranknet']['perplexity'].values
        uniform = dataset_df[dataset_df['schedule'] == 'uniform']['perplexity'].values
        
        if len(ranknet) > 1 and len(uniform) > 1:
            stat, pval = stats.wilcoxon(ranknet, uniform)
            sig = "✅" if pval < 0.05 else "❌"
            better = "RankNet" if ranknet.mean() < uniform.mean() else "Uniform"
            
            print(f"\n{model:18s} + {dataset:10s}: p={pval:.4f} {sig} → {better} wins")

print("\n✅ Analysis complete!")
