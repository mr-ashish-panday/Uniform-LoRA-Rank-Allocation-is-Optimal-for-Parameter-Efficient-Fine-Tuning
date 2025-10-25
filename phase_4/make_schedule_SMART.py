import pandas as pd
import numpy as np
import argparse

def build_smart_schedule(csv_in, csv_out, R_TARGET=16):
    """
    Fix RankNet's learned schedule by:
    1. Removing outliers (embeddings, layer norms, biases)
    2. Boosting layer-wise trends
    3. Prioritizing MLPs over attention
    """
    df = pd.read_csv(csv_in)
    df['feature'] = df['feature'].str.replace('.weight', '', regex=False)
    
    # 1. Filter to only weight matrices (remove biases, norms)
    df = df[~df['feature'].str.contains('bias|ln_|layernorm', na=False)].copy()
    
    # 2. Remove embedding outliers
    df = df[~df['feature'].str.contains('wte|wpe', na=False)].copy()
    
    # 3. Get raw scores and normalize
    s = df['predicted_rank'].values.astype(float)
    s = s / s.max() * 100  # Scale to 0-100
    
    # 4. Add layer-wise boost (later layers = more important)
    import re
    layer_boost = []
    for feat in df['feature']:
        m = re.search(r'\.h\.(\d+)\.', feat)
        if m:
            layer = int(m.group(1))
            boost = 0.8 + 0.2 * (layer / 23.0)  # 0.8 to 1.0 range
        else:
            boost = 1.0
        layer_boost.append(boost)
    
    s = s * np.array(layer_boost)
    
    # 5. Add MLP boost (MLPs more important)
    mlp_boost = []
    for feat in df['feature']:
        if 'mlp' in feat:
            mlp_boost.append(1.3)  # 30% boost for MLPs
        else:
            mlp_boost.append(1.0)
    
    s = s * np.array(mlp_boost)
    
    # 6. Convert to ranks (8-48 range)
    min_s, max_s = s.min(), s.max()
    r_int = ((s - min_s) / (max_s - min_s) * 40 + 8).astype(int)
    
    # 7. Ensure budget
    N = len(r_int)
    current_sum = r_int.sum()
    target_sum = N * R_TARGET
    
    if current_sum != target_sum:
        deficit = target_sum - current_sum
        order = np.argsort(s - r_int)[::-1]  # Highest residuals first
        for idx in order:
            if deficit == 0:
                break
            delta = min(1, deficit)
            r_int[idx] += delta
            deficit -= delta
    
    out = pd.DataFrame({'feature': df['feature'].values, 'predicted_rank': r_int})
    out.to_csv(csv_out, index=False)
    print(f"✅ Created {csv_out}: total rank {r_int.sum()}, avg {r_int.mean():.2f}")
    print(f"   Range: {r_int.min()}-{r_int.max()}, budget target: {target_sum}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--uniform_rank", type=int, default=16)
    args = ap.parse_args()
    build_smart_schedule(args.in_csv, args.out_csv, args.uniform_rank)
