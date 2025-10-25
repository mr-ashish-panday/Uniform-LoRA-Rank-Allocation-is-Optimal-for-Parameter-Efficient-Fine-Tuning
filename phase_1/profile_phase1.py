#!/usr/bin/env python3
"""
phase1_metaloRAMem.py

Phase 1 utilities for MetaLoRA-Mem:
- LoRA injection for nn.Linear (simple, safe)
- Memory profiling (torch.cuda max/avg) with warmup
- Activation size profiling (forward hook, .detach())
- Per-layer gradient norm computation after backward
- Small training step to integrate with profile tool

Notes:
- This script uses Hugging Face Transformers & Datasets.
- It is designed for profiling and scientific reproducibility.
- It's intentionally explicit rather than optimized for speed.
"""

import argparse
import os
import time
import gc
import csv
from typing import Callable, Dict, Any, Tuple
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from datasets import load_dataset

# -------------------------
# Simple LoRA adapter class
# -------------------------
class LoRALinear(nn.Module):
    """
    Replace a target nn.Linear with W x + (alpha / r) * (Up @ Down) x
    where Down is (r x in_features), Up is (out_features x r).
    This implementation keeps base weight frozen and trains Up and Down.
    """
    def __init__(self, orig: nn.Linear, r: int = 8, alpha: float = 1.0):
        super().__init__()
        self.in_features = orig.in_features
        self.out_features = orig.out_features
        self.r = r
        self.alpha = alpha
        # keep original weight as a frozen parameter (or buffer)
        self.weight = orig.weight
        if orig.bias is not None:
            self.bias = orig.bias
        else:
            self.bias = None

        # LoRA down/up matrices
        if r > 0:
            self.down = nn.Parameter(torch.randn(r, self.in_features) * (0.01))
            self.up = nn.Parameter(torch.zeros(self.out_features, r))
            # init up to small values
            nn.init.kaiming_uniform_(self.up, a=np.sqrt(5))
            self.scaling = self.alpha / float(self.r)
        else:
            # r == 0 => no adaptation
            self.register_parameter('down', None)
            self.register_parameter('up', None)
            self.scaling = 0.0

        # Freeze original weight (we will not update it)
        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False

    def forward(self, x):
        # base linear: use F.linear with frozen weight/bias
        base = torch.nn.functional.linear(x, self.weight, self.bias)
        if self.r > 0:
            # LoRA contribution: (x @ down.T) -> (batch, r) ; then @ up.T -> (batch, out)
            down_proj = torch.nn.functional.linear(x, self.down)  # shape (..., r)
            up_proj = torch.nn.functional.linear(down_proj, self.up)  # shape (..., out)
            return base + self.scaling * up_proj
        else:
            return base

# -------------------------------------
# Utility: inject LoRA into a model
# -------------------------------------
def inject_lora_to_model(model: nn.Module, rank: int = 8, layer_name_filter: Tuple[str] = ("Linear",)):
    """
    Replace nn.Linear modules in model with LoRALinear wrappers.
    Returns number of replacements.
    """
    replace_count = 0
    for name, module in list(model.named_modules()):
        # skip top-level module itself
        pass
    # to replace modules we need parent references
    for parent_name, parent_module in model.named_modules():
        for child_name, child in list(parent_module.named_children()):
            if isinstance(child, nn.Linear):
                # Replace with LoRALinear
                new_mod = LoRALinear(child, r=rank, alpha=1.0)
                setattr(parent_module, child_name, new_mod)
                replace_count += 1
    return replace_count

# -------------------------
# Profiling & hook helpers
# -------------------------
def reset_cuda(device):
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize(device)

def profile_memory(func: Callable[[], Any], device: torch.device, warmup: int = 5, steps: int = 20):
    """
    Run func() repeatedly to measure peak memory per step.
    func should perform a full training step (forward, backward, optimizer.step).
    Returns mean_peak_bytes, std_peak_bytes, list_peaks
    """
    reset_cuda(device)
    # warmup
    for _ in range(warmup):
        func()
        reset_cuda(device)
    peaks = []
    for _ in range(steps):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)
        t0 = time.time()
        func()
        if torch.cuda.is_available():
            torch.cuda.synchronize(device)
            peak = torch.cuda.max_memory_allocated(device)
        else:
            peak = 0
        peaks.append(peak)
    arr = np.array(peaks, dtype=float)
    return float(arr.mean()), float(arr.std()), peaks

def compute_activation_sizes(model: nn.Module, sample_inputs: Dict[str, torch.Tensor], device: torch.device):
    """
    Robust activation size computation. Captures all module outputs safely,
    handles tuple outputs, and ensures writes to CSV are reproducible.
    """
    from collections import OrderedDict
    activations = OrderedDict()
    handles = []

    def hook(module, inp, out):
        try:
            if isinstance(out, (tuple, list)):
                t = out[0] if isinstance(out[0], torch.Tensor) else None
            else:
                t = out if isinstance(out, torch.Tensor) else None
            if t is None:
                return
            t_det = t.detach()
            activations[f"{module.__class__.__name__}_{id(module)}"] = {
                "shape": list(t_det.shape),
                "bytes": t_det.numel() * t_det.element_size()
            }
        except Exception as e:
            print(f"[WARN] Activation hook failed for {module}: {e}")

    # Register hooks on all modules, not just Linear
    for m in model.modules():
        handles.append(m.register_forward_hook(hook))

    model.to(device)
    model.eval()

    # Forward pass (use no_grad for speed, safe for activations)
    with torch.no_grad():
        inputs_on_device = {}
        for k, v in sample_inputs.items():
            if isinstance(v, torch.Tensor):
                inputs_on_device[k] = v.to(device)
            elif isinstance(v, list):
                try:
                    inputs_on_device[k] = torch.tensor(v).to(device)
                except (ValueError, TypeError):
                    pass
            elif isinstance(v, (int, float, bool, np.integer, np.floating)):
                inputs_on_device[k] = torch.tensor(v).to(device)
        _ = model(**inputs_on_device)

    # Remove hooks
    for h in handles:
        h.remove()

    # Save CSV immediately
    with open("phase1_logs_activations.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["module_name", "shape", "bytes"])
        for name, info in activations.items():
            writer.writerow([name, info["shape"], info["bytes"]])

    print(f"[INFO] Wrote {len(activations)} activation entries to phase1_logs_activations.csv")
    return activations

def compute_per_layer_grad_norms(model: nn.Module):
    """
    After a backward run, compute per-module grad norms (sqrt sum squares for param grads
    inside module).
    Returns dict: module_id_string -> grad_norm (float)
    """
    per_layer = {}
    for m in model.modules():
        s = 0.0
        cnt = 0
        for p in m.parameters(recurse=False):
            if p.grad is not None:
                s += float(p.grad.norm().item()) ** 2
                cnt += 1
        if cnt > 0:
            per_layer[f"{m.__class__.__name__}_{id(m)}"] = float(np.sqrt(s))
    return per_layer

# -------------------------
# Small helpers to prepare dataset & dataloader
# -------------------------
def prepare_dataset(tokenizer, dataset_name: str, dataset_config: str, split: str, max_length: int = 128):
    """
    Loads dataset via datasets.load_dataset and tokenizes.
    Currently supports GLUE (SST-2) and AG News by name.
    """
    ds = load_dataset(dataset_name, dataset_config, split=split)
    # Simple mapping for text/label — users should extend for other tasks
    def preprocess_fn(ex):
        # try common fields
        if 'sentence' in ex:
            text = ex['sentence']
        elif 'text' in ex:
            text = ex['text']
        elif 'prompt' in ex:
            text = ex['prompt']
        elif 'article' in ex:
            text = ex.get('article', ex.get('text', ''))
        else:
            # fallback: pick first str field
            for k, v in ex.items():
                if isinstance(v, str):
                    text = v
                    break
            else:
                text = ""
        return tokenizer(text, truncation=True, padding='max_length', max_length=max_length)
    ds = ds.map(preprocess_fn, batched=False)
    # The dataset may have labels under 'label' already; keep as-is
    return ds

# -------------------------
# Training step function
# -------------------------
def train_step(model, batch, optimizer, device, scheduler=None, grad_accum_steps=1):
    """
    Performs forward, backward, optimizer.step for one micro-batch.
    Returns loss.item()
    """
    model.train()
    # Only keep keys that the model expects
    valid_keys = {'input_ids', 'attention_mask', 'token_type_ids', 'position_ids'}
    inputs = {}
    for k, v in batch.items():
        if k not in valid_keys:
            continue
        if isinstance(v, torch.Tensor):
            inputs[k] = v.to(device)
        elif isinstance(v, list):
            try:
                inputs[k] = torch.tensor(v).to(device)
            except (ValueError, TypeError):
                pass
        elif isinstance(v, (int, float, bool, np.integer, np.floating)):
            inputs[k] = torch.tensor(v).to(device)
    
    labels = batch['label'].to(device) if 'label' in batch else None
    if labels is not None and isinstance(labels, list):
        labels = torch.tensor(labels).to(device)
    
    outputs = model(**inputs, labels=labels)
    loss = outputs.loss
    # backward
    loss.backward()
    # step
    optimizer.step()
    if scheduler is not None:
        scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    return float(loss.item())

# -------------------------
# CLI main
# -------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="profile", choices=["profile", "train"], help="profile or train")
    parser.add_argument("--model_name_or_path", type=str, default="distilbert-base-uncased")
    parser.add_argument("--dataset_name", type=str, default="glue")
    parser.add_argument("--dataset_config_name", type=str, default="sst2")
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--rank", type=int, default=8, help="LoRA rank")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--profile_steps", type=int, default=20)
    parser.add_argument("--log_csv", type=str, default="phase1_logs.csv")
    args = parser.parse_args()
    return args

def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def collate_fn(batch):
    # collate for HF tokenized outputs: keys -> tensors
    keys = batch[0].keys()
    collated = {}
    for k in keys:
        if isinstance(batch[0][k], (list, np.ndarray)):
            # Already array-like, convert to tensor
            collated[k] = torch.tensor([b[k] for b in batch])
        elif isinstance(batch[0][k], (int, float)):
            # Scalar values
            collated[k] = torch.tensor([b[k] for b in batch])
        else:
            # Assume it's already a tensor or handle as-is
            try:
                collated[k] = torch.stack([torch.tensor(b[k]) if not isinstance(b[k], torch.Tensor) else b[k] for b in batch])
            except (TypeError, RuntimeError):
                # If stacking fails, just keep as list
                collated[k] = [b[k] for b in batch]
    return collated

def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    # Load tokenizer & model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path, num_labels=2)
    original_linear_count = sum(1 for m in model.modules() if isinstance(m, nn.Linear))
    print(f"Model loaded. Found {original_linear_count} nn.Linear modules.")

    # Inject LoRA
    replace_count = inject_lora_to_model(model, rank=args.rank)
    print(f"Injected LoRA into {replace_count} modules (rank={args.rank}).")

    # Move model to device
    model.to(device)

    # Prepare small dataset subset for Phase 1 (train split small)
    print("Loading dataset (may take some time if not cached)...")
    try:
        ds_train = prepare_dataset(tokenizer, args.dataset_name, args.dataset_config_name, split="train", max_length=args.max_length)
        ds_val = prepare_dataset(tokenizer, args.dataset_name, args.dataset_config_name, split="validation", max_length=args.max_length)
    except Exception as e:
        print("Dataset loading failed:", e)
        print("Falling back to tiny synthetic dataset for profiling.")
        # Create synthetic
        texts = ["this is a positive example"] * 128 + ["this is negative"] * 128
        labels = [1] * 128 + [0] * 128
        enc = tokenizer(texts, truncation=True, padding='max_length', max_length=args.max_length)
        import datasets
        ds_train = datasets.Dataset.from_dict({**enc, "label": labels})
        ds_val = datasets.Dataset.from_dict({**enc, "label": labels[:64]})

    # Take small subset for quick runs
    train_subset = ds_train.select(range(min(256, len(ds_train))))
    val_subset = ds_val.select(range(min(128, len(ds_val))))

    train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    # Optimizer: only parameters that require grad should be LoRA params
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=args.lr)
    total_steps = len(train_loader) * max(1, args.epochs)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    # Prepare a single-step function for profiling
    train_iter = iter(train_loader)
    def single_step_func():
        nonlocal train_iter
        try:
            batch = next(train_iter)
        except StopIteration:
            # reset iterator
            train_iter = iter(train_loader)
            batch = next(train_iter)
        return train_step(model, batch, optimizer, device, scheduler=scheduler)

    # sample inputs for activation profiling (single batch)
    sample_batch = next(iter(train_loader))
    # Only keep keys that the model expects (input_ids, attention_mask, token_type_ids, etc.)
    valid_keys = {'input_ids', 'attention_mask', 'token_type_ids', 'position_ids'}
    sample_inputs = {k: v for k, v in sample_batch.items() if k in valid_keys}

    # Run activation profiling
    print("Computing activation sizes for a sample forward...")
    activation_sizes = compute_activation_sizes(model, sample_inputs, device)
    print("Activation sizes (bytes) sample (top 10):")
    items = sorted(activation_sizes.items(), key=lambda x: -x[1]["bytes"])
    for k, v in items[:10]:
        print(f"  {k}: {v['bytes']} bytes")

    # Profile memory for a few steps
    print(f"Profiling memory with warmup={args.warmup} and steps={args.profile_steps} ...")
    mean_peak, std_peak, peaks = profile_memory(single_step_func, device, warmup=args.warmup, steps=args.profile_steps)
    print(f"Profile results -- mean_peak_bytes={mean_peak:.0f}, std={std_peak:.0f}")

    # Basic gradient norm check: perform one forward+backward manually
    print("Running one manual step to compute gradient norms per-layer...")
    # reuse a single batch
    batch_for_grad = sample_batch
    model.train()
    # Only keep keys that the model expects
    valid_keys = {'input_ids', 'attention_mask', 'token_type_ids', 'position_ids'}
    inputs = {}
    for k, v in batch_for_grad.items():
        if k not in valid_keys:
            continue
        if isinstance(v, torch.Tensor):
            inputs[k] = v.to(device)
        elif isinstance(v, list):
            try:
                inputs[k] = torch.tensor(v).to(device)
            except (ValueError, TypeError):
                pass
        elif isinstance(v, (int, float, bool, np.integer, np.floating)):
            inputs[k] = torch.tensor(v).to(device)
    
    labels = batch_for_grad['label'].to(device) if 'label' in batch_for_grad else None
    if labels is not None and isinstance(labels, list):
        labels = torch.tensor(labels).to(device)
    outputs = model(**inputs, labels=labels)
    loss = outputs.loss
    loss.backward()
    grad_norms = compute_per_layer_grad_norms(model)
    print("Per-layer grad norms sample (top 10):")
    items = sorted(grad_norms.items(), key=lambda x: -x[1])
    for k, v in items[:10]:
        print(f"  {k}: {v:.6f}")
    # zero grads
    optimizer.zero_grad(set_to_none=True)

    # Logging CSV entry
    header = ["timestamp", "model", "dataset", "rank", "profile_mean_peak_bytes", "profile_std_peak_bytes", "num_linear_replaced"]
    row = [time.strftime("%Y-%m-%d %H:%M:%S"), args.model_name_or_path, f"{args.dataset_name}:{args.dataset_config_name}",
           args.rank, int(mean_peak), int(std_peak), replace_count]
    csv_exists = os.path.exists(args.log_csv)
    with open(args.log_csv, "a", newline="") as f:
        writer = csv.writer(f)
        if not csv_exists:
            writer.writerow(header)
        writer.writerow(row)

    # Export activation sizes to CSV
    activations_csv = args.log_csv.replace(".csv", "_activations.csv")
    with open(activations_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["module_name", "bytes"])
        for module_name, info in sorted(activation_sizes.items(), key=lambda x: -x[1]["bytes"]):
            writer.writerow([module_name, info["bytes"]])
    print(f"Activation sizes exported to {activations_csv}")

    # Export gradient norms to CSV
    gradnorms_csv = args.log_csv.replace(".csv", "_gradnorms.csv")
    with open(gradnorms_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["module_name", "gradient_norm"])
        for module_name, grad_norm in sorted(grad_norms.items(), key=lambda x: -x[1]):
            writer.writerow([module_name, grad_norm])
    print(f"Gradient norms exported to {gradnorms_csv}")

    # If mode == train, run a tiny training loop (1 epoch default)
    if args.mode == "train":
        print("Starting short training loop...")
        for epoch in range(args.epochs):
            model.train()
            losses = []
            for i, batch in enumerate(train_loader):
                loss_val = train_step(model, batch, optimizer, device, scheduler=scheduler)
                losses.append(loss_val)
                if (i + 1) % 10 == 0:
                    print(f"Epoch {epoch+1} step {i+1}/{len(train_loader)} loss={np.mean(losses):.4f}")
            # simple val eval: compute accuracy
            model.eval()
            correct = 0
            total = 0
            for batch in val_loader:
                inputs = {k: v.to(device) for k, v in batch.items() if k != 'label'}
                labels = batch['label'].to(device)
                with torch.no_grad():
                    out = model(**inputs)
                    logits = out.logits
                    preds = torch.argmax(logits, dim=-1)
                correct += int((preds == labels).sum().item())
                total += labels.size(0)
            acc = correct / total if total > 0 else 0.0
            print(f"Epoch {epoch+1} validation accuracy: {acc:.4f}")

    print("Done.")

if __name__ == "__main__":
    main()