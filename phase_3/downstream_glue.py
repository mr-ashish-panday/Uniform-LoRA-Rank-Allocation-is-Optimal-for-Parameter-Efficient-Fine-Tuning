import os, argparse, numpy as np
from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          Trainer, TrainingArguments, DataCollatorWithPadding)
from sklearn.metrics import accuracy_score

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, preds)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=str, choices=["sst2","mnli"], required=True)
    ap.add_argument("--base_model", type=str, required=True)
    ap.add_argument("--save_dir", type=str, default="outputs/glue_run")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    args = ap.parse_args()

    ds = load_dataset("glue", args.task)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    def tok_fn(batch):
        if args.task == "sst2":
            return tokenizer(batch["sentence"], truncation=True, padding=False)
        else:
            return tokenizer(batch["premise"], batch["hypothesis"], truncation=True, padding=False)
    ds = ds.map(tok_fn, batched=True)
    num_labels = 2 if args.task=="sst2" else 3

    model = AutoModelForSequenceClassification.from_pretrained(args.base_model, num_labels=num_labels)

    collator = DataCollatorWithPadding(tokenizer)
    train_args = TrainingArguments(
        output_dir=args.save_dir, num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size, per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr, evaluation_strategy="epoch", save_strategy="epoch",
        logging_strategy="epoch", report_to="none"
    )
    eval_name = "validation_matched" if args.task=="mnli" and "validation_matched" in ds else "validation"
    trainer = Trainer(
        model=model, args=train_args,
        train_dataset=ds["train"], eval_dataset=ds[eval_name],
        tokenizer=tokenizer, data_collator=collator, compute_metrics=compute_metrics
    )
    trainer.train()
    metrics = trainer.evaluate()
    os.makedirs(args.save_dir, exist_ok=True)
    with open(os.path.join(args.save_dir,"eval_glue.txt"),"w") as f:
        for k,v in metrics.items(): f.write(f"{k}={v}\n")
    print(metrics)

if __name__ == "__main__":
    main()
