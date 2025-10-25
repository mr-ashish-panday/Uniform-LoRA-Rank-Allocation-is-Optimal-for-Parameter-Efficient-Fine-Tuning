import os, re, glob, json
import numpy as np
from collections import defaultdict
from scipy.stats import ttest_rel, wilcoxon

PATTERN = re.compile(r".*/([^/]+)_wt2_s(\\d+)/eval_summary\\.txt$")

def parse_eval(path):
    d = {}
    for line in open(path):
        if "=" in line:
            k,v = line.strip().split("=")
            try: d[k]=float(v)
            except: pass
    return d.get("eval_loss"), d.get("perplexity")

def main():
    files = glob.glob("outputs/*_wt2_s*/eval_summary.txt")
    by_key = defaultdict(list)
    for p in files:
        m = PATTERN.match(p.replace("\\\\","/"))
        if not m: continue
        tag, seed = m.group(1), int(m.group(2))
        if tag.startswith("gpt2_ranknet"): model, sched = "gpt2-medium", "ranknet"
        elif tag.startswith("gpt2_uniform"): model, sched = "gpt2-medium", "uniform"
        elif tag.startswith("bert_ranknet"): model, sched = "bert-base-uncased", "ranknet"
        elif tag.startswith("bert_uniform"): model, sched = "bert-base-uncased", "uniform"
        elif tag.startswith("distilbert_ranknet"): model, sched = "distilbert-base-uncased", "ranknet"
        elif tag.startswith("distilbert_uniform"): model, sched = "distilbert-base-uncased", "uniform"
        else: continue
        loss, ppl = parse_eval(p)
        if loss is not None and ppl is not None:
            by_key[(model, sched)].append((seed, loss, ppl))

    results = {"rows": [], "paired_tests": []}
    for (model, sched), lst in sorted(by_key.items()):
        arr = sorted(lst)
        losses = np.array([x for _,x,_ in arr])
        ppls   = np.array([y for _,_,y in arr])
        results["rows"].append({
            "model": model, "schedule": sched,
            "eval_loss_mean": float(losses.mean()),
            "eval_loss_std": float(losses.std(ddof=1)),
            "perplexity_mean": float(ppls.mean()),
            "perplexity_std": float(ppls.std(ddof=1)),
            "n": len(arr)
        })
    for model in ["gpt2-medium","bert-base-uncased","distilbert-base-uncased"]:
        r = sorted(by_key.get((model,"ranknet"), []))
        u = sorted(by_key.get((model,"uniform"), []))
        if len(r)>=5 and len(u)>=5:
            losses_r = np.array([x for _,x,_ in r])
            losses_u = np.array([x for _,x,_ in u])
            ppls_r   = np.array([y for _,_,y in r])
            ppls_u   = np.array([y for _,_,y in u])
            t_l, p_l = ttest_rel(losses_r, losses_u)
            w_l, pw_l = wilcoxon(losses_r, losses_u)
            t_p, p_p = ttest_rel(ppls_r, ppls_u)
            w_p, pw_p = wilcoxon(ppls_r, ppls_u)
            results["paired_tests"].append({
                "model": model, "n": len(r),
                "t_eval_loss_p": p_l, "wilcoxon_eval_loss_p": pw_l,
                "t_perplexity_p": p_p, "wilcoxon_perplexity_p": pw_p
            })
    os.makedirs("analysis", exist_ok=True)
    with open("analysis/stats_with_wilcoxon.json","w") as f:
        json.dump(results, f, indent=2)
    print("Wrote analysis/stats_with_wilcoxon.json")

if __name__ == "__main__":
    main()
