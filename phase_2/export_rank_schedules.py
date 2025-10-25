import os, argparse, numpy as np, pandas as pd, torch, joblib
from ranknet import RankNet

def load_ckpt(path):
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    return ckpt, ckpt.get('args', {}), ckpt.get('varying_mask', None)

def sanitize_X(X):
    return np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=-1.0)

def run_one(model, scaler, X, budget=1.0, device='cpu'):
    model.eval()
    Xs = scaler.transform(X)
    preds = []
    with torch.no_grad():
        for i in range(0, len(Xs), 1024):
            xb = torch.from_numpy(Xs[i:i+1024]).float().to(device)
            preds.append(model(xb, torch.tensor(float(budget), device=device)).cpu().numpy())
    return np.concatenate(preds, axis=0)

def ensure_unique(path):
    if not os.path.exists(path): return path
    base, ext = os.path.splitext(path)
    k = 1
    while True:
        cand = f"{base}__{k}{ext}"
        if not os.path.exists(cand): return cand
        k += 1

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', type=str, default='.')
    p.add_argument('--ckpt', type=str, default='models/ranknet_best.pt')
    p.add_argument('--scaler', type=str, default='models/feature_scaler.pkl')
    p.add_argument('--budget', type=float, default=1.0)
    p.add_argument('--out_dir', type=str, default='schedules')
    p.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--max_rank', type=int, default=64)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    ckpt, ckpt_args, varying_mask = load_ckpt(args.ckpt)
    scaler = joblib.load(args.scaler)

    feat_cols = ['layer_idx_norm','param_count_norm','grad_norm_norm','act_size_norm']
    feature_dim = int(ckpt_args.get('feature_dim', len(feat_cols)))
    hidden_size = int(ckpt_args.get('hidden_size', 128))
    max_rank    = int(ckpt_args.get('max_rank', args.max_rank))
    device      = torch.device(args.device)

    model = RankNet(feature_dim=feature_dim, hidden_size=hidden_size, max_rank=max_rank).to(device)
    model.load_state_dict(ckpt['model_state'])

    for fname in sorted(os.listdir(args.data_dir)):
        if not fname.endswith('_features.csv'): continue
        path = os.path.join(args.data_dir, fname)
        df = pd.read_csv(path)
        names = df['module_name'].astype(str).tolist() if 'module_name' in df.columns else [f'layer_{i}' for i in range(len(df))]
        if not set(feat_cols).issubset(df.columns):
            print(f"Skip {fname}: missing feature columns"); continue
        X = sanitize_X(df[feat_cols].values.astype(float))
        if varying_mask is not None:
            keep = np.array(varying_mask, dtype=bool)
            X = X[:, keep]
        preds = run_one(model, scaler, X, budget=args.budget, device=device)
        preds = np.clip(preds, 0.0, float(max_rank))
        ranks_int = np.rint(preds).astype(int)
        out_df = pd.DataFrame({'module_name': names, 'predicted_rank': ranks_int})
        base = os.path.splitext(fname)[0]  # preserve hyphens/underscores to avoid collisions
        out_file = ensure_unique(os.path.join(args.out_dir, f'schedule_{base}.csv'))
        out_df.to_csv(out_file, index=False)
        print(f"Wrote {os.path.basename(out_file)} ({len(out_df)} entries) from {fname}")
if __name__ == "__main__":
    main()
