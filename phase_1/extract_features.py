#!/usr/bin/env python3
# extract_features.py
import pandas as pd
import numpy as np
import torch
import sys
from transformers import AutoModelForSequenceClassification, AutoTokenizer

def safe_read_csv(path):
    df = pd.read_csv(path)
    # Normalize column names
    df.columns = [c.strip() for c in df.columns]
    return df

def compute_param_counts(model_name):
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    param_counts = {}
    # Use hierarchical names to map as best-effort; also map by class name fallback counts
    for name, module in model.named_modules():
        try:
            n = sum(p.numel() for p in module.parameters(recurse=False))
            if n > 0:
                param_counts[name] = n
        except Exception:
            pass
    # Fallback: class-level average param count
    class_counts = {}
    for name, module in model.named_modules():
        cls = module.__class__.__name__
        n = sum(p.numel() for p in module.parameters(recurse=False))
        if n > 0:
            class_counts.setdefault(cls, []).append(n)
    class_avg = {k: int(np.mean(v)) for k, v in class_counts.items() if len(v) > 0}
    return param_counts, class_avg

def extract_features(activation_csv, gradnorm_csv, output_csv, model_name):
    act_df = safe_read_csv(activation_csv)
    grad_df = safe_read_csv(gradnorm_csv)

    # Normalize column names expected
    # Activation CSV: expect columns ['module_name','bytes'] or ['module_name','shape','bytes']
    if 'bytes' not in act_df.columns:
        raise ValueError(f"'bytes' column not found in {activation_csv}. Columns: {act_df.columns.tolist()}")
    # Grad CSV: accept 'gradient_norm' or 'grad_norm' or 'gradient_norms'
    grad_col_candidates = ['gradient_norm', 'grad_norm', 'gradient']
    grad_col = None
    for c in grad_col_candidates:
        if c in grad_df.columns:
            grad_col = c
            break
    if grad_col is None:
        # If grad CSV contains 'value' like 'gradient_norm' under different name, try to find numeric column
        numeric_cols = grad_df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) == 0:
            raise ValueError(f"No numeric grad column found in {gradnorm_csv}")
        grad_col = numeric_cols[0]

    # Ensure module_name exists for merge
    if 'module_name' not in act_df.columns or 'module_name' not in grad_df.columns:
        raise ValueError("Both activation and grad CSVs must have 'module_name' column")

    # Trim whitespace
    act_df['module_name'] = act_df['module_name'].astype(str).str.strip()
    grad_df['module_name'] = grad_df['module_name'].astype(str).str.strip()

    # Merge on module_name (inner join)
    merged = pd.merge(act_df, grad_df, on='module_name', how='inner', suffixes=('_act', '_grad'))
    if merged.shape[0] == 0:
        # fallback: align by index if names didn't match
        minlen = min(len(act_df), len(grad_df))
        merged = pd.concat([act_df.iloc[:minlen].reset_index(drop=True),
                            grad_df.iloc[:minlen].reset_index(drop=True)], axis=1)
        merged.columns = [c if 'module_name' in c else c for c in merged.columns]

    # Compute param counts for model name, with fallback to class-level avg
    param_counts, class_avg = compute_param_counts(model_name)

    features = []
    for idx, row in merged.reset_index(drop=True).iterrows():
        module_name = row['module_name']
        # attempt to match param_counts by exact name, or by base module name (strip suffix after last '.')
        param_count = 0
        if module_name in param_counts:
            param_count = param_counts[module_name]
        else:
            # try hierarchical base names
            parts = module_name.split('.')
            found = False
            for i in range(len(parts), 0, -1):
                candidate = '.'.join(parts[:i])
                if candidate in param_counts:
                    param_count = param_counts[candidate]
                    found = True
                    break
            if not found:
                # fallback to class name mapping
                cls = module_name.split('_')[0] if '_' in module_name else module_name
                param_count = class_avg.get(cls, 0)

        activation_bytes = float(row['bytes']) if 'bytes' in row else 0.0
        grad_norm = float(row.get(grad_col, 0.0))

        features.append({
            'layer_idx': idx,
            'module_name': module_name,
            'activation_bytes': activation_bytes,
            'grad_norm': grad_norm,
            'param_count': int(param_count),
        })

    df = pd.DataFrame(features)

    # If all zeros for param_count or activation_bytes, avoid divide-by-zero by replacing max with 1
    eps = 1e-8
    max_layer_idx = df['layer_idx'].max() if len(df) > 0 else 1
    max_param = df['param_count'].max() if df['param_count'].max() > 0 else 1
    max_act = df['activation_bytes'].max() if df['activation_bytes'].max() > 0 else 1
    max_grad = df['grad_norm'].max() if df['grad_norm'].max() > 0 else 1.0

    df['layer_idx_norm'] = df['layer_idx'] / float(max_layer_idx + eps)
    df['param_count_norm'] = df['param_count'] / float(max_param + eps)
    df['act_size_norm'] = df['activation_bytes'] / float(max_act + eps)
    df['grad_norm_norm'] = df['grad_norm'] / float(max_grad + eps)

    # Heuristic optimal rank: proportional to act_size_norm, clamped
    df['optimal_rank'] = (df['act_size_norm'] * 64.0).round().astype(int).clip(4, 64)

    output_df = df[['layer_idx_norm', 'param_count_norm', 'act_size_norm', 'grad_norm_norm', 'optimal_rank', 'module_name']]
    output_df.to_csv(output_csv, index=False)
    print(f"Created {output_csv} with {len(output_df)} layers")

if __name__ == "__main__":
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description="Extract normalized features from activations and gradient norms")
    parser.add_argument("--model", required=True, 
                        help="HF model id, e.g., distilbert-base-uncased, gpt2-medium, bert-base-uncased")
    parser.add_argument("--activations", required=True, 
                        help="Path to phase1 activations CSV")
    parser.add_argument("--gradnorms", required=True, 
                        help="Path to phase1 grad norms CSV")
    parser.add_argument("--out_dir", default="../phase_2", 
                        help="Directory to write *_features.csv")
    
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    out_name = args.model.replace("/", "_") + "_features.csv"
    output_csv = os.path.join(args.out_dir, out_name)
    
    print(f"Parsed model: {args.model}")
    print(f"Writing features to: {output_csv}")
    
    extract_features(args.activations, args.gradnorms, output_csv, args.model)