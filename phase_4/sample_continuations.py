import glob, json
from transformers import pipeline

generator = pipeline("text-generation", model="outputs/gpt2-medium_ranknet_s42")
prompts = ["In a distant future,", "Artificial intelligence will", "Climate change impacts"]
for p in prompts:
    print(f">>> Prompt: {p}")
    samples = generator(p, max_length=100, num_return_sequences=3)
    print(json.dumps(samples, indent=2))
