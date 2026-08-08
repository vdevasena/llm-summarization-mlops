import time
import sys
import pandas as pd
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1]

# Add src to Python's import path
sys.path.insert(0, str(SRC_DIR))

# Import summarizer.py from src
from summarizer import summarize

from evaluation.rouge_metrics import calculate_rouge

INPUT_PATH = "data/evaluation/eval_set.csv"
OUTPUT_PATH = "data/evaluation/baseline_results.csv"

df = pd.read_csv(INPUT_PATH)


results = []


for index, row in df.iterrows():

    print(
        f"Processing {index + 1}/{len(df)}"
    )

    start_time = time.time()

    generated_summary = summarize(
        row["article"]
    )

    latency = time.time() - start_time

    scores = calculate_rouge(
        row["reference_summary"],
        generated_summary
    )

    results.append({
        "id": row["id"],
        "reference_summary": row["reference_summary"],
        "generated_summary": generated_summary,
        "rouge1": scores["rouge1"],
        "rouge2": scores["rouge2"],
        "rougeL": scores["rougeL"],
        "latency_seconds": latency
    })


results_df = pd.DataFrame(results)


results_df.to_csv(
    OUTPUT_PATH,
    index=False
)


print("\nExperiment complete!")

print(
    results_df[
        [
            "rouge1",
            "rouge2",
            "rougeL",
            "latency_seconds"
        ]
    ].mean()
)