from datasets import load_dataset
import pandas as pd
import os


# Where we will save our evaluation dataset
OUTPUT_PATH = "data/evaluation/eval_set.csv"


print("Loading CNN/DailyMail test dataset...")


# Load the Parquet-based CNN/DailyMail dataset
dataset = load_dataset(
    "abisee/cnn_dailymail",
    "3.0.0",
    split="test"
)


print(f"Total test examples available: {len(dataset)}")


# ---------------------------------------------------------
# For our first experiment, use only 100 examples.
# Later we will create larger and more representative
# evaluation/golden datasets.
# ---------------------------------------------------------

sample = dataset.select(range(100))


# Convert to Pandas DataFrame
df = pd.DataFrame({
    "id": sample["id"],
    "article": sample["article"],
    "reference_summary": sample["highlights"]
})


# Make sure the output directory exists
os.makedirs(
    "data/evaluation",
    exist_ok=True
)


# Save evaluation dataset
df.to_csv(
    OUTPUT_PATH,
    index=False
)


print("\n----------------------------------------")
print("Evaluation dataset created successfully!")
print("----------------------------------------")

print(f"Number of examples: {len(df)}")
print(f"Saved to: {OUTPUT_PATH}")

print("\nColumns:")
print(df.columns.tolist())


print("\nFirst article:")
print(df.iloc[0]["article"][:1000])


print("\nReference summary:")
print(df.iloc[0]["reference_summary"])