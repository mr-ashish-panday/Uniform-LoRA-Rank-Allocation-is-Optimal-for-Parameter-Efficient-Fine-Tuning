import pandas as pd

# Load original with .weight
df = pd.read_csv('schedules/gpt2-medium_features.csv')

# Strip .weight from feature names
df['feature'] = df['feature'].str.replace('.weight', '', regex=False)

# Save corrected
df.to_csv('schedules/gpt2-medium_features_CORRECTED.csv', index=False)

print("Created schedules/gpt2-medium_features_CORRECTED.csv")
print(df.head(10))
