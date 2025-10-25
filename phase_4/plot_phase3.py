import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("combined_results.csv")

# GPT-2 perplexity bar chart
gpt2 = df[df.model=="gpt2-medium"]
plt.figure(figsize=(4,3))
plt.bar(gpt2.schedule, gpt2.perplexity, color=["#4C72B0","#55A868"])
plt.title("GPT-2 Medium Perplexity (WT2)")
plt.ylabel("Perplexity")
plt.savefig("gpt2_perplexity_wt2.png", dpi=300)
plt.close()

# BERT/DistilBERT eval_loss line plot
mlm = df[df.model!="gpt2-medium"]
plt.figure(figsize=(5,3))
for model in mlm.model.unique():
    sub = mlm[mlm.model==model]
    plt.plot(sub.schedule, sub.eval_loss, marker="o", label=model)
plt.title("MLM Eval Loss (WT2)")
plt.ylabel("Eval Loss")
plt.legend()
plt.savefig("mlm_eval_loss_wt2.png", dpi=300)
plt.close()
