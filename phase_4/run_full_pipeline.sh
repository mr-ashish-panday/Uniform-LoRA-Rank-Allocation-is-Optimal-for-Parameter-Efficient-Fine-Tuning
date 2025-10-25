#!/usr/bin/env bash

# Full pipeline: Download dataset + Run validation
# Safe to run overnight

set -e  # Exit on any error

echo "=========================================="
echo "Starting Full Pipeline"
echo "=========================================="
echo "Step 1: Downloading OpenWebText dataset..."
echo "Step 2: Running validation experiments"
echo "=========================================="
echo ""

# Step 1: Download dataset
echo "[$(date)] Starting dataset download..."
python download_openwebtext.py --num_samples 10000

if [ $? -ne 0 ]; then
    echo "[$(date)] ERROR: Dataset download failed!"
    exit 1
fi

echo "[$(date)] Dataset download complete!"
echo ""

# Verify dataset
echo "[$(date)] Verifying dataset..."
FILE_COUNT=$(find ~/datasets/openwebtext/ -name "*.txt" -type f | wc -l)
echo "Found ${FILE_COUNT} text files"

if [ $FILE_COUNT -eq 0 ]; then
    echo "[$(date)] ERROR: No text files found!"
    exit 1
fi

echo "[$(date)] Dataset verification passed!"
echo ""

# Step 2: Run validation
echo "[$(date)] Starting validation experiments..."
bash run_validation.sh

if [ $? -ne 0 ]; then
    echo "[$(date)] ERROR: Validation failed!"
    exit 1
fi

echo ""
echo "=========================================="
echo "[$(date)] Full Pipeline Complete!"
echo "=========================================="
echo "Check logs/ directory for training logs"
echo "Check outputs/ directory for model checkpoints"
