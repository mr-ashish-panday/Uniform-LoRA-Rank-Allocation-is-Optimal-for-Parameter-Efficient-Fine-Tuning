import os, re, glob, json
import numpy as np
from collections import defaultdict
from scipy.stats import ttest_rel

PATTERN = re.compile(r".*/([^/]+)_wt2_s(\d+)/eval_summary\.txt$")

def parse_eval(path):
    vals = {}
    with open(path, "r") as f:
        for line in f:
            if "=" in line:
                k, v = line.strip().split("=")
                try: vals[k] = float(v)
                except: pass
    return vals.get("eval_loss"), vals.get("perplexity")

def main():
    files = glob.glob("outputs/*_wt2_s*/eval_summary.txt")
    by_key = defaultdict(list)
    for p in files:
        m = PATTERN.match(p.replace("\\","/"))
        if not m:
            print(f"No match: {p}")
            continue
        tag, seed = m.group(1), int(m.group(2))
        print(f"Matched: tag='{tag}', seed={seed}")
        
        if tag.startswith("gpt2_ranknet"): model, sched = "gpt2-medium", "ranknet"
        elif tag.startswith("gpt2_uniform"): model, sched = "gpt2-medium", "uniform"
        elif tag.startswith("bert_ranknet"): model, sched = "bert-base-uncased", "ranknet"
        elif tag.startswith("bert_uniform"): model, sched = "bert-base-uncased", "uniform"
        elif tag.startswith("distilbert_ranknet"): model, sched = "distilbert-base-uncased", "ranknet"
        elif tag.startswith("distilbert_uniform"): model, sched = "distilbert-base-uncased", "uniform"
        else:
            print(f"  -> No mapping for tag '{tag}'")
            continue
        loss, ppl = parse_eval(p)
        if loss is not None and ppl is not None:
            by_key[(model, sched)].append({"seed": seed, "eval_loss": loss, "perplexity": ppl})

    rows = []
    for (model, sched), lst in sorted(by_key.items()):
        losses = np.array([x["eval_loss"] for x in lst])
        ppls = np.array([x["perplexity"] for x in lst])
        rows.append({
            "model": model, "schedule": sched,
            "eval_loss_mean": float(losses.mean()) if len(losses)>0 else None,
            "eval_loss_std": float(losses.std(ddof=1)) if len(losses)>1 else 0.0,
            "perplexity_mean": float(ppls.mean()) if len(ppls)>0 else None,
            "perplexity_std": float(ppls.std(ddof=1)) if len(ppls)>1 else 0.0,
            "n": len(lst)
        })

    tests = []
    for model in ["gpt2-medium","bert-base-uncased","distilbert-base-uncased"]:
        r = sorted(by_key.get((model,"ranknet"), []), key=lambda x: x["seed"])
        u = sorted(by_key.get((model,"uniform"), []), key=lambda x: x["seed"])
        shared = sorted(set([x["seed"] for x in r]).intersection([x["seed"] for x in u]))
        if len(shared) >= 2:
            r_loss = np.array([next(x["eval_loss"] for x in r if x["seed"]==s) for s in shared])
            u_loss = np.array([next(x["eval_loss"] for x in u if x["seed"]==s) for s in shared])
            t_l, p_l = ttest_rel(r_loss, u_loss)
            r_ppl = np.array([next(x["perplexity"] for x in r if x["seed"]==s) for s in shared])
            u_ppl = np.array([next(x["perplexity"] for x in u if x["seed"]==s) for s in shared])
            t_p, p_p = ttest_rel(r_ppl, u_ppl)
            tests.append({"model": model, "paired_n": len(shared),
                          "t_eval_loss": float(t_l), "p_eval_loss": float(p_l),
                          "t_perplexity": float(t_p), "p_perplexity": float(p_p)})

    os.makedirs("analysis", exist_ok=True)
    with open("analysis/aggregate_stats.json","w") as f:
        json.dump({"rows": rows, "paired_tests": tests}, f, indent=2)
    print(f"Found {len(files)} summaries across {len(rows)} groups; wrote analysis/aggregate_stats.json")

if __name__ == "__main__":
    main()
