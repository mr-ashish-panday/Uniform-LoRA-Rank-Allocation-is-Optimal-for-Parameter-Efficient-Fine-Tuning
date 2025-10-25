#!/usr/bin/env bash

# Prepare Wikipedia and Wikitext-103 datasets as .txt files

# Wikipedia 1,000-article subset
mkdir -p ~/datasets/wikipedia
python - <<'PY'
from datasets import load_dataset
import os
ds = load_dataset("wikipedia", "20220301.en", split="train[:1000]")
out = os.path.expanduser("~/datasets/wikipedia")
for i, ex in enumerate(ds):
    with open(f"{out}/{i:07d}.txt","w",encoding="utf-8") as f:
        f.write(ex["text"])
PY

# Wikitext-103, first 1,000 lines
mkdir -p ~/datasets/wikitext-103
python - <<'PY'
from datasets import load_dataset
import os
ds = load_dataset("wikitext", "wikitext-103-v1", split="train")
out = os.path.expanduser("~/datasets/wikitext-103")
for i, ex in enumerate(ds):
    with open(f"{out}/{i:07d}.txt","w",encoding="utf-8") as f:
        f.write(ex["text"])
    if i>=999: break
PY
