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
            
            # Parse experiment name: dataset_model_schedule_seed
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
print("✅ Raw results saved to phase4_full_results.csv\n")

# Statistical summary
print("=" * 90)
print("PHASE 4 FULL SCALE - PUBLICATION RESULTS")
print("=" * 90)

summary_data = []

for model in sorted(df['model'].unique()):
    model_df = df[df['model'] == model]
    print(f"\n{'=' * 90}")
    print(f"MODEL: {model.upper()}")
    print(f"{'=' * 90}")
    
    for dataset in sorted(model_df['dataset'].unique()):
        dataset_df = model_df[model_df['dataset'] == dataset]
        print(f"\n📊 Dataset: {dataset.upper()}")
        
        for schedule in ['ranknet', 'uniform']:
            sched_df = dataset_df[dataset_df['schedule'] == schedule]
            if len(sched_df) > 0:
                ppl = sched_df['perplexity'].values
                mean_ppl = ppl.mean()
                std_ppl = ppl.std()
                
                print(f"  {schedule.upper():10s}: {mean_ppl:7.2f} ± {std_ppl:6.2f}  (n={len(ppl)})")
                
                summary_data.append({
                    'model': model,
                    'dataset': dataset,
                    'schedule': schedule,
                    'mean_ppl': mean_ppl,
                    'std_ppl': std_ppl,
                    'n_seeds': len(ppl)
                })

# Save summary
summary_df = pd.DataFrame(summary_data)
summary_df.to_csv("phase4_summary.csv", index=False)
print("\n✅ Summary saved to phase4_summary.csv")

# Statistical tests
print("\n" + "=" * 90)
print("STATISTICAL SIGNIFICANCE (Wilcoxon, p<0.05)")
print("=" * 90)

for model in sorted(df['model'].unique()):
    model_df = df[df['model'] == model]
    
    for dataset in sorted(model_df['dataset'].unique()):
        dataset_df = model_df[model_df['dataset'] == dataset]
        
        ranknet = dataset_df[dataset_df['schedule'] == 'ranknet']['perplexity'].values
        uniform = dataset_df[dataset_df['schedule'] == 'uniform']['perplexity'].values
        
        if len(ranknet) > 1 and len(uniform) > 1:
            stat, pval = stats.wilcoxon(ranknet, uniform)
            sig = "✅ SIGNIFICANT" if pval < 0.05 else "❌ not significant"
            better = "RankNet" if ranknet.mean() < uniform.mean() else "Uniform"
            
            print(f"\n{model:20s} + {dataset:10s}")
            print(f"  p-value: {pval:.6f} {sig}")
            print(f"  Winner: {better} ({better.lower()} PPL: {min(ranknet.mean(), uniform.mean()):.2f})")

print("\n✅ Full Phase 4 analysis complete!")
