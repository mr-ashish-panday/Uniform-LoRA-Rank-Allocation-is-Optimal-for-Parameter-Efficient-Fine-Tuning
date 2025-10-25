#!/usr/bin/env bash

set -e

# OpenWebText subset (1000 documents)
mkdir -p ~/datasets/openwebtext
cd /tmp
PYTHONPATH="" python3 - <<'PY'
from datasets import load_dataset
import os
ds = load_dataset("Skylion007/openwebtext", split="train", streaming=True)
out = os.path.expanduser("~/datasets/openwebtext")
os.makedirs(out, exist_ok=True)
for i, ex in enumerate(ds):
    if i >= 1000: break
    with open(f"{out}/{i:07d}.txt", "w", encoding="utf-8") as f:
        f.write(ex.get("text",""))
PY

# BookCorpus subset (1000 books)
mkdir -p ~/datasets/bookcorpus
cd /tmp
PYTHONPATH="" python3 - <<'PY'
from datasets import load_dataset
import os
ds = load_dataset("lucadiliello/bookcorpusopen", split="train", streaming=True)
out = os.path.expanduser("~/datasets/bookcorpus")
os.makedirs(out, exist_ok=True)
for i, ex in enumerate(ds):
    if i >= 1000: break
    text = ex.get("text","")
    with open(f"{out}/{i:07d}.txt", "w", encoding="utf-8") as f:
        f.write(text)
PY
