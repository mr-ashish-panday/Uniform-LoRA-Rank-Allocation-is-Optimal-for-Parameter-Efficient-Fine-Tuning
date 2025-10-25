#!/usr/bin/env python3
import pandas as pd
import numpy as np
import argparse

def build_schedule(csv_in, csv_out, R_TARGET=16, r_min=0, r_max_attn=None, r_max_mlp=None, prune_frac=0.2):
    df = pd.read_csv(csv_in)
    feats = df['feature'].astype(str)

    is_attn = feats.str.contains('attn.c_attn|attn.c_proj', regex=True)
    is_mlp  = feats.str.contains('mlp.c_fc|mlp.c_proj', regex=True)
    is_norm_or_embed = feats.str.contains('ln_|layernorm|wte|wpe', case=False, regex=True)

    s = df['predicted_rank'].to_numpy().astype(float)
    s[s < 0] = 0
    s[is_norm_or_embed.values] = 0

    # Prune bottom p%
    if prune_frac > 0:
        k = int(len(s) * prune_frac)
        if k > 0:
            idx_sorted = np.argsort(s)
            s[idx_sorted[:k]] = 0

    if s.sum() == 0:
        s = np.ones_like(s)

    N = len(s)
    total_budget = int(N * R_TARGET)

    # Proportional fractional ranks
    r_frac = s / s.sum() * total_budget

    # Type caps: if None, auto-set to 1.5x * R_TARGET to avoid bottlenecks
    if r_max_attn is None:
        r_max_attn = int(R_TARGET * 1.5)
    if r_max_mlp is None:
        r_max_mlp = int(R_TARGET * 1.2)

    # Initial integer allocation by floor, respecting min
    r_int = np.floor(r_frac).astype(int)
    r_int = np.maximum(r_int, r_min)

    # Enforce caps
    cap = np.where(is_attn.values, r_max_attn, r_max_mlp)
    over_cap = r_int > cap
    r_int[over_cap] = cap[over_cap]

    # Compute current sum and remaining budget
    cur = r_int.sum()
    
    # If we're under budget, distribute remaining to highest residuals where not capped
    if cur < total_budget:
        residual = r_frac - r_int
        can_inc = r_int < cap
        order = np.argsort(residual)[::-1]
        for idx in order:
            if not can_inc[idx]:
                continue
            if cur >= total_budget:
                break
            r_int[idx] += 1
            cur += 1

    # If still under due to caps, relax caps incrementally
    if cur < total_budget:
        deficit = total_budget - cur
        # Sort by how much over-cap they are, smallest deficit first
        over_by = r_int - cap
        over_by = np.where(over_by < 0, np.inf, over_by)  # Only those at cap
        order = np.argsort(cap)[::-1]  # largest cap first (most room)
        for idx in order:
            if deficit <= 0:
                break
            # Boost cap for this one
            delta = min(deficit, deficit // 5 + 1)  # Add up to 20% buffer
            cap[idx] += delta
            new_max = min(cap[idx], r_frac[idx] * 2)  # Don't exceed 2x fractional
            r_int[idx] = min(new_max, r_int[idx] + delta)
            cur = r_int.sum()
            deficit = total_budget - cur

    # If we're over budget, remove from smallest residuals but stay >= r_min
    if cur > total_budget:
        residual = r_frac - r_int
        order = np.argsort(residual)
        for idx in order:
            if cur <= total_budget:
                break
            if r_int[idx] > r_min:
                r_int[idx] -= 1
                cur -= 1

    # Final assert with tolerance
    if abs(r_int.sum() - total_budget) > 1:
        print(f"Warning: Sum {r_int.sum()} vs target {total_budget}, diff={r_int.sum() - total_budget}")
        # Force exact by adjusting largest
        diff = total_budget - r_int.sum()
        if diff > 0:
            idx_max = np.argmax(r_frac)
            r_int[idx_max] += diff
        elif diff < 0:
            idx_min = np.argmin(r_frac[r_int > 0])
            r_int[idx_min] += diff

    out = pd.DataFrame({'feature': df['feature'], 'predicted_rank': r_int})
    out.to_csv(csv_out, index=False)
    print(f"Wrote {csv_out} with total rank {r_int.sum()} (target {total_budget}), avg {r_int.mean():.2f}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--uniform_rank", type=int, default=16)
    ap.add_argument("--r_min", type=int, default=0)
    ap.add_argument("--r_max_attn", type=int, default=None)
    ap.add_argument("--r_max_mlp", type=int, default=None)
    ap.add_argument("--prune_frac", type=float, default=0.2)
    args = ap.parse_args()
    
    r_max_attn = args.r_max_attn if args.r_max_attn is not None else None
    r_max_mlp = args.r_max_mlp if args.r_max_mlp is not None else None
    
    build_schedule(args.in_csv, args.out_csv, args.uniform_rank, args.r_min,
                   r_max_attn, r_max_mlp, args.prune_frac)
