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


def run_experiment(prompt_version):

    df = pd.read_csv(INPUT_PATH)

    results = []

    for index, row in df.iterrows():

        print(
            f"{prompt_version}: "
            f"{index + 1}/{len(df)}"
        )

        start_time = time.time()

        generated_summary = summarize(
            row["article"],
            prompt_version
        )

        latency = time.time() - start_time

        scores = calculate_rouge(
            row["reference_summary"],
            generated_summary
        )

        results.append({
        "id": row["id"],
        "prompt_version": prompt_version,
        "article": row["article"],
        "reference_summary": row["reference_summary"],
        "generated_summary": generated_summary,
        "rouge1": scores["rouge1"],
        "rouge2": scores["rouge2"],
        "rougeL": scores["rougeL"],
        "latency_seconds": latency })

    results_df = pd.DataFrame(results)

    output_path = (
        f"data/evaluation/"
        f"results_{prompt_version}.csv"
    )

    results_df.to_csv(
        output_path,
        index=False
    )

    print("\nRESULTS")

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


if __name__ == "__main__":

    run_experiment("v1")
    run_experiment("v2")