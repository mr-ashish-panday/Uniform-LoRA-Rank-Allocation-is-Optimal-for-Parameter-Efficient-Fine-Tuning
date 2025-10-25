import os, argparse, numpy as np, pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from ranknet import RankNet

# -------------------------
# Data loading and cleaning
# -------------------------
def load_features(csv_path):
    df = pd.read_csv(csv_path)
    req = ['layer_idx_norm','param_count_norm','grad_norm_norm','act_size_norm','optimal_rank']
    for c in req:
        if c not in df.columns:
            raise ValueError(f"Missing {c} in {csv_path}")
    X = df[['layer_idx_norm','param_count_norm','grad_norm_norm','act_size_norm']].values.astype(float)
    y = df['optimal_rank'].values.astype(float)
    return X, y

def sanitize_split(X_train, y_train, X_val, y_val, max_rank):
    # Replace non-finites
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=1.0, neginf=-1.0)
    y_train = np.nan_to_num(y_train, nan=0.0, posinf=float(max_rank), neginf=0.0)
    X_val   = np.nan_to_num(X_val,   nan=0.0, posinf=1.0, neginf=-1.0)
    y_val   = np.nan_to_num(y_val,   nan=0.0, posinf=float(max_rank), neginf=0.0)
    # Drop any row with non-finite after replacement (paranoia)
    tr_mask = np.isfinite(X_train).all(axis=1) & np.isfinite(y_train)
    va_mask = np.isfinite(X_val).all(axis=1) & np.isfinite(y_val)
    X_train, y_train = X_train[tr_mask], y_train[tr_mask]
    X_val,   y_val   = X_val[va_mask],   y_val[va_mask]
    # Remove constant feature columns based on train only
    col_min, col_max = X_train.min(axis=0), X_train.max(axis=0)
    vary = (col_max - col_min) > 1e-12
    X_train = X_train[:, vary]
    X_val   = X_val[:, vary]
    return X_train, y_train, X_val, y_val, vary

# -------------------------
# Metrics (with Spearman)
# -------------------------
def compute_metrics_np(preds, targets, max_rank):
    preds = np.nan_to_num(preds, nan=0.0, posinf=float(max_rank), neginf=0.0)
    targets = np.nan_to_num(targets, nan=0.0, posinf=float(max_rank), neginf=0.0)
    preds = np.clip(preds, 0.0, float(max_rank))
    targets = np.clip(targets, 0.0, float(max_rank))
    mae = float(np.mean(np.abs(preds - targets)))
    rmse = float(np.sqrt(np.mean((preds - targets)**2)))
    pr = np.rint(preds).astype(int); tr = np.rint(targets).astype(int)
    exact = float(np.mean(pr == tr))
    within1 = float(np.mean(np.abs(pr - tr) <= 1))

    # Spearman rho via rank transform with average ranks for ties
    def ranks(a):
        order = np.argsort(a, kind='mergesort')
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(len(a))
        vals, inv, counts = np.unique(a, return_inverse=True, return_counts=True)
        csum = np.cumsum(counts); starts = csum - counts
        avg = (starts + csum - 1) / 2.0
        return avg[inv]
    ra, rb = ranks(preds), ranks(targets)
    ra = (ra - ra.mean()) / (ra.std() + 1e-12)
    rb = (rb - rb.mean()) / (rb.std() + 1e-12)
    spearman = float(np.mean(ra * rb))
    return dict(mae=mae, rmse=rmse, exact=exact, within1=within1, spearman=spearman)

# -------------------------
# Models
# -------------------------
def fit_ranknet(X_train, y_train, X_val, y_val,
                hidden=128, max_rank=64, lr=1e-3,
                batch=64, epochs=20, device='cpu'):
    scaler = StandardScaler().fit(X_train)
    Xtr, Xva = scaler.transform(X_train), scaler.transform(X_val)
    tr_ds = TensorDataset(torch.from_numpy(Xtr).float(), torch.from_numpy(y_train).float())
    va_ds = TensorDataset(torch.from_numpy(Xva).float(), torch.from_numpy(y_val).float())
    tr_loader = DataLoader(tr_ds, batch_size=batch, shuffle=True)
    va_loader = DataLoader(va_ds, batch_size=batch, shuffle=False)
    model = RankNet(feature_dim=Xtr.shape[1], hidden_size=hidden, max_rank=max_rank).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.MSELoss()
    best_rmse = float('inf')
    best = None
    for _ in range(epochs):
        model.train()
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb, torch.tensor(1.0, device=device))
            loss = crit(pred, yb)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            opt.step()
        # validate
        model.eval()
        preds = []
        with torch.no_grad():
            for xb, _ in va_loader:
                xb = xb.to(device)
                preds.append(model(xb, torch.tensor(1.0, device=device)).cpu().numpy())
        preds = np.concatenate(preds, axis=0)
        m = compute_metrics_np(preds, y_val, max_rank)
        if m['rmse'] < best_rmse:
            best_rmse = m['rmse']
            best = {
                'metrics': m,
                'model_state': model.state_dict(),
                'scaler_mean': scaler.mean_.copy(),
                'scaler_scale': scaler.scale_.copy(),
            }
    return best

def baseline_linear(X_train, y_train, X_val, y_val, max_rank):
    scaler = StandardScaler().fit(X_train)
    Xtr = scaler.transform(X_train); Xva = scaler.transform(X_val)
    reg = LinearRegression().fit(Xtr, y_train)
    preds = reg.predict(Xva)
    return compute_metrics_np(preds, y_val, max_rank)

def baseline_activation_only(X_train, y_train, X_val, y_val, col_names, max_rank):
    # Use act_size_norm column if retained; otherwise first column as fallback
    try:
        idx = col_names.index('act_size_norm')
    except ValueError:
        idx = 0
    preds = X_val[:, idx] * float(max_rank)
    return compute_metrics_np(preds, y_val, max_rank)

def baseline_uniform(X_train, y_train, X_val, y_val, max_rank):
    mean_rank = float(np.clip(np.mean(y_train), 0.0, float(max_rank)))
    preds = np.full_like(y_val, mean_rank, dtype=float)
    return compute_metrics_np(preds, y_val, max_rank)

# -------------------------
# Data assembly
# -------------------------
def load_all(data_dir):
    Xs, ys, groups = [], [], []
    files = sorted([f for f in os.listdir(data_dir) if f.endswith('_features.csv')])
    for f in files:
        X, y = load_features(os.path.join(data_dir, f))
        Xs.append(X); ys.append(y); groups.extend([f]*len(y))
    X = np.vstack(Xs); y = np.hstack(ys); groups = np.array(groups)
    return X, y, groups, files

# -------------------------
# Main evaluation
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', type=str, default='.')
    ap.add_argument('--k', type=int, default=5)
    ap.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44])
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--hidden', type=int, default=128)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--max_rank', type=int, default=64)
    ap.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--out_csv', type=str, default='eval_results.csv')
    args = ap.parse_args()

    X, y, groups, files = load_all(args.data_dir)

    rows = []
    base_cols = ['layer_idx_norm','param_count_norm','grad_norm_norm','act_size_norm']

    # K-fold CV (pooled data)
    for seed in args.seeds:
        kf = KFold(n_splits=args.k, shuffle=True, random_state=seed)
        for i, (tr, va) in enumerate(kf.split(X), 1):
            Xtr, ytr = X[tr], y[tr]
            Xva, yva = X[va], y[va]
            Xtr, ytr, Xva, yva, vary = sanitize_split(Xtr, ytr, Xva, yva, args.max_rank)
            col_names = [c for j, c in enumerate(base_cols) if vary[j]]

            rk = fit_ranknet(Xtr, ytr, Xva, yva, hidden=args.hidden, max_rank=args.max_rank,
                             lr=args.lr, batch=args.batch, epochs=args.epochs, device=args.device)
            lin = baseline_linear(Xtr, ytr, Xva, yva, args.max_rank)
            act = baseline_activation_only(Xtr, ytr, Xva, yva, col_names, args.max_rank)
            uni = baseline_uniform(Xtr, ytr, Xva, yva, args.max_rank)

            rows.append(dict(protocol='kfold', seed=seed, fold=i, method='ranknet', **rk['metrics']))
            rows.append(dict(protocol='kfold', seed=seed, fold=i, method='linear', **lin))
            rows.append(dict(protocol='kfold', seed=seed, fold=i, method='act_only', **act))
            rows.append(dict(protocol='kfold', seed=seed, fold=i, method='uniform', **uni))

    # Leave-one-model-out (by feature file)
    for hold in files:
        mask = (groups == hold)
        X_te, y_te = X[mask], y[mask]
        X_tr, y_tr = X[~mask], y[~mask]
        # 10% validation from train
        kf = KFold(n_splits=10, shuffle=True, random_state=42)
        idx_tr, idx_va = next(kf.split(X_tr))
        Xtr, ytr = X_tr[idx_tr], y_tr[idx_tr]
        Xva, yva = X_tr[idx_va], y_tr[idx_va]
        Xtr, ytr, Xva, yva, vary = sanitize_split(Xtr, ytr, Xva, yva, args.max_rank)
        col_names = [c for j, c in enumerate(base_cols) if vary[j]]

        # Fit ranknet and get scaler stats
        rk = fit_ranknet(Xtr, ytr, Xva, yva, hidden=args.hidden, max_rank=args.max_rank,
                         lr=args.lr, batch=args.batch, epochs=args.epochs, device=args.device)

        # Apply same column mask and scaler to holdout, then evaluate
        X_te_san = np.nan_to_num(X_te, nan=0.0, posinf=1.0, neginf=-1.0)
        X_te_san = X_te_san[:, np.where(vary)[0]]
        scaler_mean = rk['scaler_mean']; scaler_scale = rk['scaler_scale']
        X_te_s = (X_te_san - scaler_mean) / scaler_scale

        model = RankNet(feature_dim=X_te_s.shape[1], hidden_size=args.hidden, max_rank=args.max_rank)
        model.load_state_dict(rk['model_state']); model.eval()
        preds = []
        with torch.no_grad():
            for i in range(0, len(X_te_s), 1024):
                xb = torch.from_numpy(X_te_s[i:i+1024]).float()
                preds.append(model(xb, torch.tensor(1.0)).numpy())
        preds = np.concatenate(preds, axis=0)
        te_metrics = compute_metrics_np(preds, y_te, args.max_rank)

        # Baselines evaluated on the same masked/sanitized holdout
        lin = baseline_linear(Xtr, ytr, X_te_san, y_te, args.max_rank)
        act = baseline_activation_only(Xtr, ytr, X_te_san, y_te, col_names, args.max_rank)
        uni = baseline_uniform(Xtr, ytr, X_te_san, y_te, args.max_rank)

        rows.append(dict(protocol='lomo', holdout=hold, method='ranknet', **te_metrics))
        rows.append(dict(protocol='lomo', holdout=hold, method='linear', **lin))
        rows.append(dict(protocol='lomo', holdout=hold, method='act_only', **act))
        rows.append(dict(protocol='lomo', holdout=hold, method='uniform', **uni))

    # Save and print summary
    out = pd.DataFrame(rows)
    out.to_csv(args.out_csv, index=False)
    print("\n=== Evaluation Summary (mean ± std across folds/seeds) ===")
    summary = out.groupby(['protocol','method'])[['rmse','mae','spearman','exact','within1']].agg(['mean','std'])
    print(summary.round(4))
    print(f"\nDetailed results saved to: {args.out_csv}")

if __name__ == "__main__":
    main()
