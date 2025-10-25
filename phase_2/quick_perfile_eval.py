import os, numpy as np, pandas as pd
from eval_ranknet import compute_metrics_np

def main(data_dir='.', max_rank=64):
    files = sorted([f for f in os.listdir(data_dir) if f.endswith('_features.csv')])
    rows = []
    for f in files:
        df = pd.read_csv(os.path.join(data_dir, f))
        y = df['optimal_rank'].values.astype(float)
        # simple act_size_norm baseline
        preds = df['act_size_norm'].values.astype(float) * float(max_rank)
        m = compute_metrics_np(preds, y, max_rank)
        rows.append(dict(file=f, **m))
    out = pd.DataFrame(rows)
    print(out.to_string(index=False))
if __name__ == "__main__":
    main()
