import glob, json, os, re
import numpy as np
from scipy.stats import ttest_rel, wilcoxon

# Map output dir tags to model/schedule
TAG_MAP = {
  "gpt2_ranknet_wt2": ("gpt2-medium","ranknet"),
  "gpt2_uniform_wt2": ("gpt2-medium","uniform"),
  "bert_ranknet_wt2": ("bert-base-uncased","ranknet"),
  "bert_uniform_wt2": ("bert-base-uncased","uniform"),
  "distilbert_ranknet_wt2": ("distilbert-base-uncased","ranknet"),
  "distilbert_uniform_wt2": ("distilbert-base-uncased","uniform")
}

# Gather results
data = {k: [] for k in TAG_MAP.values()}
pattern = re.compile(r"^(.+_wt2)_s(\d+)$")

for path in glob.glob("outputs/*/eval_summary.txt"):
    tag_dir = os.path.basename(os.path.dirname(path))
    m = pattern.match(tag_dir)
    if not m:
        print(f"Skipping folder (no match): {tag_dir}")
        continue
    tag, seed_str = m.group(1), m.group(2)
    if tag not in TAG_MAP:
        print(f"Skipping folder (not in TAG_MAP): {tag}")
        continue
    seed = int(seed_str)
    model, sched = TAG_MAP[tag]
    
    # Parse key=value lines
    d = {}
    try:
        with open(path) as fp:
            for line in fp:
                if "=" in line:
                    k, v = line.strip().split("=", 1)
                    try:
                        d[k] = float(v)
                    except ValueError:
                        d[k] = v
    except FileNotFoundError:
        print(f"Skipping missing file: {path}")
        continue
    
    loss = d.get("eval_loss")
    ppl = d.get("perplexity")
    if seed is not None and loss is not None and ppl is not None:
        data[(model, sched)].append((seed, loss, ppl))

# Compute and write full‐seed stats
rows, paired = [], []
for (m,s), vals in data.items():
    if not vals:
        continue
    vals.sort()
    seeds, losses, ppls = zip(*vals)
    rows.append({
      "model":m, "schedule":s,
      "eval_loss_mean":float(np.mean(losses)),
      "eval_loss_std":float(np.std(losses,ddof=1)),
      "perplexity_mean":float(np.mean(ppls)),
      "perplexity_std":float(np.std(ppls,ddof=1)),
      "n":len(losses)
    })

for m in ["gpt2-medium","bert-base-uncased","distilbert-base-uncased"]:
    r = data[(m,"ranknet")]
    u = data[(m,"uniform")]
    if len(r)==7 and len(u)==7:
        _, lr, pr = zip(*r)
        _, lu, pu = zip(*u)
        t_l,p_l = ttest_rel(lr,lu)
        w_l,pw_l = wilcoxon(lr,lu)
        t_p,p_p = ttest_rel(pr,pu)
        w_p,pw_p = wilcoxon(pr,pu)
        paired.append({
          "model":m, "n":7,
          "t_eval_loss_p":float(p_l), "wilcoxon_eval_loss_p":float(pw_l),
          "t_perplexity_p":float(p_p), "wilcoxon_perplexity_p":float(pw_p)
        })

out = {"rows":rows,"paired_tests":paired}
os.makedirs("analysis",exist_ok=True)
with open("analysis/full_multiseed_stats.json","w") as f:
    json.dump(out,f,indent=2)
print("Wrote analysis/full_multiseed_stats.json")