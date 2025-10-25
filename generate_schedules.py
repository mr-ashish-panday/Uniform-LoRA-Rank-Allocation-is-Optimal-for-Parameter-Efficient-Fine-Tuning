import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoModelForMaskedLM, AutoTokenizer

def get_feature_importance(model_name):
    """Calculates a simple importance score for each parameter."""
    if 'gpt2' in model_name:
        model = AutoModelForCausalLM.from_pretrained(model_name)
    else:
        model = AutoModelForMaskedLM.from_pretrained(model_name)
    
    features = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            # Using parameter norm as a proxy for importance
            importance = torch.norm(param.data).item()
            features.append({"feature": name, "importance": importance})
    
    df = pd.DataFrame(features)
    df = df.sort_values(by="importance", ascending=False).reset_index(drop=True)
    return df

def main():
    models = ["gpt2-medium", "bert-base-uncased", "distilbert-base-uncased"]
    for model_name in models:
        print(f"Generating schedule for {model_name}...")
        df = get_feature_importance(model_name)
        output_path = f"schedules/{model_name}_features.csv"
        df.to_csv(output_path, index=False)
        print(f"Saved schedule to {output_path}")

if __name__ == "__main__":
    main()
