import json, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

data = json.load(open("analysis/aggregate_stats.json"))
df = pd.DataFrame(data["rows"])

# GPT-2 perplexity bar chart
gpt2 = df[df["model"]=="gpt2-medium"].set_index("schedule")
plt.figure(figsize=(4,3))
plt.bar(
    ["ranknet","uniform"],
    gpt2.loc[["ranknet","uniform"],"perplexity_mean"],
    yerr=gpt2.loc[["ranknet","uniform"],"perplexity_std"],
    capsize=3,
    color=["#4C72B0","#55A868"]
)
plt.ylabel("Perplexity")
plt.title("GPT-2 (WT2) multi-seed")
plt.tight_layout()
plt.savefig("analysis/gpt2_multiseed_ppl.png", dpi=300)
plt.close()

# MLM eval loss error-bar plot
mlm = df[df["model"]!="gpt2-medium"]
plt.figure(figsize=(5,3))
for model, color in zip(["bert-base-uncased","distilbert-base-uncased"], ["#C44E52","#8172B2"]):
    sub = mlm[mlm["model"]==model].set_index("schedule")
    plt.errorbar(
        ["ranknet","uniform"],
        sub.loc[["ranknet","uniform"],"eval_loss_mean"],
        yerr=sub.loc[["ranknet","uniform"],"eval_loss_std"],
        fmt="o-",
        label=model,
        color=color,
        capsize=3
    )
plt.ylabel("Eval loss")
plt.title("MLM (WT2) multi-seed")
plt.legend()
plt.tight_layout()
plt.savefig("analysis/mlm_multiseed_loss.png", dpi=300)
plt.close()

print("Figures saved to analysis/")
