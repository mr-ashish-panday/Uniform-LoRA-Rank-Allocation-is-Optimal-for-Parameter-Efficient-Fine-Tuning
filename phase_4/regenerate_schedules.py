import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoModelForMaskedLM

def get_feature_importance(model_name):
    model_cls = AutoModelForCausalLM if 'gpt2' in model_name else AutoModelForMaskedLM
    model = model_cls.from_pretrained(model_name)
    features = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            features.append({
                "feature": name,
                "predicted_rank": float(torch.norm(param.data).item())
            })
    df = pd.DataFrame(features).sort_values("predicted_rank", ascending=False)
    return df

def main():
    models = ["gpt2-medium", "bert-base-uncased", "distilbert-base-uncased"]
    for m in models:
        df = get_feature_importance(m)
        df.to_csv(f"schedules/{m}_features.csv", index=False)
        print(f"Written schedules/{m}_features.csv")

if __name__ == "__main__":
    main()
