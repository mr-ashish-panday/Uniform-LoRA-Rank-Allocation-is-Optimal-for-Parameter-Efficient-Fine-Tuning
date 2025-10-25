# phase2_ranknet/train_ranknet.py
import os
import argparse
import random
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from ranknet import RankNet
import joblib


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_features(csv_path: str, required_cols=None):
    df = pd.read_csv(csv_path)
    if required_cols is None:
        required_cols = [
            'layer_idx_norm', 'param_count_norm', 'grad_norm_norm',
            'act_size_norm', 'optimal_rank'
        ]
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"Missing column {c} in {csv_path}")
    X = df[['layer_idx_norm', 'param_count_norm', 'grad_norm_norm', 'act_size_norm']].values.astype(float)
    y = df['optimal_rank'].values.astype(float)
    extra = {}
    if 'act_size_norm' in df.columns:
        extra['act_size_norm'] = df['act_size_norm'].values.astype(float)
    return X, y, extra


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default=".", help="folder containing *_features.csv files")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--feature_dim", type=int, default=4)
    p.add_argument("--hidden_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_rank", type=int, default=64)
    p.add_argument("--memory_budget_bytes", type=float, default=12 * 1024**3)
    p.add_argument("--save_dir", type=str, default="models")
    p.add_argument("--penalty_weight", type=float, default=0.0, help="Weight for optional memory-violation penalty")
    return p.parse_args()


def compute_metrics(preds, targets, max_rank: int):
    preds = preds.cpu().numpy()
    targets = targets.cpu().numpy()
    # Ensure finite, clamp to valid range to avoid invalid rounding casts
    preds = np.nan_to_num(preds, nan=0.0, posinf=float(max_rank), neginf=0.0)
    targets = np.nan_to_num(targets, nan=0.0, posinf=float(max_rank), neginf=0.0)
    preds = np.clip(preds, 0.0, float(max_rank))
    targets = np.clip(targets, 0.0, float(max_rank))

    mae = np.mean(np.abs(preds - targets))
    rmse = np.sqrt(np.mean((preds - targets) ** 2))
    # integer accuracy: exact and within 1
    preds_round = np.rint(preds).astype(int)
    targets_int = np.rint(targets).astype(int)
    exact = np.mean(preds_round == targets_int)
    within1 = np.mean(np.abs(preds_round - targets_int) <= 1)
    return {"mae": mae, "rmse": rmse, "exact_acc": float(exact), "within1_acc": float(within1)}


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    set_seed(args.seed)
    device = torch.device(args.device)

    # collect CSVs
    X_list, y_list, extra_list = [], [], []
    for f in os.listdir(args.data_dir):
        if f.endswith('_features.csv'):
            Xf, yf, extra = load_features(os.path.join(args.data_dir, f))
            X_list.append(Xf)
            y_list.append(yf)
            extra_list.append(extra)
    if len(X_list) == 0:
        raise RuntimeError("No *_features.csv files found in data_dir")

    X = np.vstack(X_list)
    y = np.hstack(y_list)
    print(f"Loaded features shape: {X.shape}, targets shape: {y.shape}")

    # Sanitize inputs: replace NaN/Inf, drop non-finite rows, remove constant columns
    X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=-1.0)
    y = np.nan_to_num(y, nan=0.0, posinf=float(args.max_rank), neginf=0.0)

    row_mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
    X, y = X[row_mask], y[row_mask]

    col_min, col_max = X.min(axis=0), X.max(axis=0)
    varying_mask = (col_max - col_min) > 1e-12
    X = X[:, varying_mask]

    # Derive feature_dim dynamically after column filtering
    args.feature_dim = int(X.shape[1])

    # Guard: ensure at least 1 feature remains
    if args.feature_dim == 0:
        raise RuntimeError("All feature columns are constant or invalid after sanitization; cannot train.")

    # train/val split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=args.seed, shuffle=True
    )

    # feature scaler (fit on train)
    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_val_s = scaler.transform(X_val)

    # Normalized budget scalar (kept as 1.0 in this pipeline)
    budget_train_norm = 1.0

    # DataLoaders
    train_ds = TensorDataset(torch.from_numpy(X_train_s).float(), torch.from_numpy(y_train).float())
    val_ds = TensorDataset(torch.from_numpy(X_val_s).float(), torch.from_numpy(y_val).float())
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    # model
    model = RankNet(feature_dim=args.feature_dim, hidden_size=args.hidden_size, max_rank=args.max_rank).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    criterion = nn.MSELoss()

    best_val_rmse = float('inf')
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            preds = model(xb, torch.tensor(budget_train_norm, device=device))
            loss = criterion(preds, yb)

            # optional memory-violation penalty (proxy)
            if args.penalty_weight > 0.0:
                proxy_mem = preds.sum()
                mem_limit = (args.max_rank * xb.size(0)) * 0.5
                penalty = torch.relu(proxy_mem - mem_limit)
                loss = loss + args.penalty_weight * penalty

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, error_if_nonfinite=True)
            optimizer.step()
            running_loss += loss.item() * xb.size(0)

        train_avg_loss = running_loss / len(train_loader.dataset)

        # validation
        model.eval()
        val_loss = 0.0
        preds_all = []
        targets_all = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                preds = model(xb, torch.tensor(budget_train_norm, device=device))
                val_loss += criterion(preds, yb).item() * xb.size(0)
                preds_all.append(preds.detach().cpu())
                targets_all.append(yb.detach().cpu())
        avg_val_loss = val_loss / len(val_loader.dataset)
        preds_all = torch.cat(preds_all, dim=0)
        targets_all = torch.cat(targets_all, dim=0)
        metrics = compute_metrics(preds_all, targets_all, max_rank=args.max_rank)

        print(
            f"Epoch {epoch+1}/{args.epochs}  "
            f"train_loss={train_avg_loss:.6f}  val_loss={avg_val_loss:.6f}  "
            f"val_rmse={metrics['rmse']:.4f} val_mae={metrics['mae']:.4f} "
            f"exact_acc={metrics['exact_acc']:.3f} within1={metrics['within1_acc']:.3f}"
        )

        scheduler.step(avg_val_loss)

        # checkpoint best
        if metrics['rmse'] < best_val_rmse:
            best_val_rmse = metrics['rmse']
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "scaler": scaler,
                    "args": vars(args),
                    "varying_mask": varying_mask,  # save mask so inference can match columns
                },
                os.path.join(args.save_dir, "ranknet_best.pt"),
            )
            joblib.dump(scaler, os.path.join(args.save_dir, "feature_scaler.pkl"))

    # final save
    torch.save(model.state_dict(), os.path.join(args.save_dir, "ranknet_final.pt"))
    joblib.dump(scaler, os.path.join(args.save_dir, "feature_scaler_final.pkl"))
    with open(os.path.join(args.save_dir, "ranknet_metadata.json"), "w") as f:
        json.dump(
            {
                "feature_dim": args.feature_dim,
                "hidden_size": args.hidden_size,
                "max_rank": args.max_rank,
                "memory_budget_bytes": args.memory_budget_bytes,
            },
            f,
            indent=2,
        )

    print("Training complete. Best val RMSE:", best_val_rmse)


if __name__ == "__main__":
    main()
