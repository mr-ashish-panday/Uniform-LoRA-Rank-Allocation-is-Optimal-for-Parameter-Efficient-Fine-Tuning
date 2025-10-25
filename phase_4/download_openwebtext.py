#!/usr/bin/env python3
"""
Download text dataset - optimized for limited disk space.
"""
import sys
try:
    import pyarrow as pa
    if not hasattr(pa, 'PyExtensionType'):
        pa.PyExtensionType = pa.ExtensionType
except:
    pass

import os
import argparse
from datasets import load_dataset
from tqdm import tqdm

def download_dataset(output_dir, num_samples=10000, min_length=100):
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Downloading dataset (first {num_samples} samples)...")
    print("Using streaming mode to save disk space...")
    
    dataset = None
    dataset_name = None
    
    # Use streaming to avoid downloading entire dataset
    try:
        print("Trying: C4 (streaming mode)...")
        dataset = load_dataset("allenai/c4", "en", split="train", streaming=True)
        dataset = dataset.take(num_samples)
        dataset_name = "C4"
    except Exception as e:
        print(f"Failed: {e}")
    
    if dataset is None:
        try:
            print("Trying: bookcorpus (small subset)...")
            dataset = load_dataset("bookcorpus", split=f"train[:{min(num_samples, 5000)}]")
            dataset_name = "BookCorpus"
        except Exception as e:
            print(f"Failed: {e}")
    
    if dataset is None:
        print("\nCreating synthetic dataset for testing...")
        synthetic_texts = [
            "This is sample text for language model training. " * 20,
            "Natural language processing enables computers to understand human language. " * 15,
            "Machine learning models require large amounts of training data. " * 18,
            "Deep learning has revolutionized artificial intelligence applications. " * 16,
            "Transformers have become the dominant architecture in NLP tasks. " * 17,
            "Fine-tuning pre-trained models is an effective transfer learning technique. " * 14,
        ] * (num_samples // 6 + 1)
        
        saved_count = 0
        for i in range(min(num_samples, len(synthetic_texts))):
            filename = os.path.join(output_dir, f"{saved_count:07d}.txt")
            with open(filename, "w", encoding="utf-8") as f:
                f.write(synthetic_texts[i])
            saved_count += 1
        
        print(f"\n✓ Created {saved_count} synthetic text files")
        print(f"✓ Location: {output_dir}")
        total_size = sum(os.path.getsize(os.path.join(output_dir, f)) 
                         for f in os.listdir(output_dir) if f.endswith('.txt'))
        print(f"✓ Total size: {total_size / 1024 / 1024:.2f} MB")
        return
    
    print(f"\nSaving to {output_dir}...")
    
    saved_count = 0
    skipped_count = 0
    
    for i, example in enumerate(tqdm(dataset, desc="Saving files", total=num_samples)):
        if saved_count >= num_samples:
            break
            
        text = None
        for col in ["text", "content", "contents", "article"]:
            if col in example:
                text = str(example[col]).strip()
                break
        
        if text is None:
            text = str(example).strip()
        
        if len(text) < min_length:
            skipped_count += 1
            continue
        
        filename = os.path.join(output_dir, f"{saved_count:07d}.txt")
        with open(filename, "w", encoding="utf-8") as f:
            f.write(text)
        
        saved_count += 1
    
    print(f"\n✓ Successfully saved {saved_count} text files from {dataset_name}")
    print(f"✓ Skipped {skipped_count} files (too short)")
    print(f"✓ Location: {output_dir}")
    
    total_size = sum(os.path.getsize(os.path.join(output_dir, f)) 
                     for f in os.listdir(output_dir) if f.endswith('.txt'))
    print(f"✓ Total size: {total_size / 1024 / 1024:.2f} MB")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download dataset")
    parser.add_argument("--output_dir", type=str, 
                       default=os.path.expanduser("~/datasets/openwebtext"),
                       help="Output directory for text files")
    parser.add_argument("--num_samples", type=int, default=10000,
                       help="Number of samples to download (default: 10000)")
    parser.add_argument("--min_length", type=int, default=100,
                       help="Minimum text length in characters (default: 100)")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Dataset Download")
    print("=" * 60)
    print(f"Output directory: {args.output_dir}")
    print(f"Number of samples: {args.num_samples}")
    print(f"Minimum length: {args.min_length} characters")
    print("=" * 60)
    
    download_dataset(args.output_dir, args.num_samples, args.min_length)
    
    print("\n" + "=" * 60)
    print("Download complete! You can now run fine-tuning.")
    print("=" * 60)
