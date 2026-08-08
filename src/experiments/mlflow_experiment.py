from pathlib import Path

import pandas as pd
import mlflow


# =========================================================
# 1. PROJECT PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RESULTS_PATH = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "results_v2.csv"
)

MLFLOW_DB = (
    PROJECT_ROOT
    / "mlflow.db"
)


# =========================================================
# 2. MLFLOW SETUP
# =========================================================

# Use SQLite instead of the old filesystem backend
mlflow.set_tracking_uri(
    f"sqlite:///{MLFLOW_DB}"
)

mlflow.set_experiment(
    "summarization_prompt_experiments"
)


# =========================================================
# 3. LOAD RESULTS
# =========================================================

print("Loading experiment results...")

results_df = pd.read_csv(
    RESULTS_PATH
)

print(
    f"Loaded {len(results_df)} rows."
)

print("\nColumns:")

print(
    results_df.columns.tolist()
)


# =========================================================
# 4. CHECK REQUIRED COLUMNS
# =========================================================

required_columns = [
    "id",
    "prompt_version",
    "generated_summary",
    "rouge1",
    "rouge2",
    "rougeL",
    "latency_seconds"
]

missing_columns = [
    column
    for column in required_columns
    if column not in results_df.columns
]

if missing_columns:

    raise ValueError(
        f"Missing columns: {missing_columns}"
    )


# =========================================================
# 5. CALCULATE AVERAGE METRICS
# =========================================================

prompt_version = (
    results_df[
        "prompt_version"
    ].iloc[0]
)

num_examples = len(
    results_df
)

avg_rouge1 = (
    results_df["rouge1"].mean()
)

avg_rouge2 = (
    results_df["rouge2"].mean()
)

avg_rougeL = (
    results_df["rougeL"].mean()
)

avg_latency = (
    results_df["latency_seconds"].mean()
)


# =========================================================
# 6. START MLFLOW RUN
# =========================================================

with mlflow.start_run(
    run_name=f"prompt_{prompt_version}"
):

    # -----------------------------------------------------
    # Parameters
    # -----------------------------------------------------

    mlflow.log_param(
        "prompt_version",
        prompt_version
    )

    mlflow.log_param(
        "dataset",
        "CNN_DailyMail"
    )

    mlflow.log_param(
        "num_examples",
        num_examples
    )


    # -----------------------------------------------------
    # Metrics
    # -----------------------------------------------------

    mlflow.log_metric(
        "avg_rouge1",
        avg_rouge1
    )

    mlflow.log_metric(
        "avg_rouge2",
        avg_rouge2
    )

    mlflow.log_metric(
        "avg_rougeL",
        avg_rougeL
    )

    mlflow.log_metric(
        "avg_latency_seconds",
        avg_latency
    )


    # -----------------------------------------------------
    # Log CSV as artifact
    # -----------------------------------------------------

    mlflow.log_artifact(
        str(RESULTS_PATH),
        artifact_path="results"
    )


    # -----------------------------------------------------
    # Run information
    # -----------------------------------------------------

    run_id = (
        mlflow.active_run()
        .info
        .run_id
    )

    print("\n===================================")
    print("MLFLOW EXPERIMENT COMPLETE")
    print("===================================")

    print(
        f"Run ID: {run_id}"
    )

    print(
        f"Prompt Version: "
        f"{prompt_version}"
    )

    print(
        f"Examples: "
        f"{num_examples}"
    )

    print(
        f"Average ROUGE-1: "
        f"{avg_rouge1:.4f}"
    )

    print(
        f"Average ROUGE-2: "
        f"{avg_rouge2:.4f}"
    )

    print(
        f"Average ROUGE-L: "
        f"{avg_rougeL:.4f}"
    )

    print(
        f"Average Latency: "
        f"{avg_latency:.4f} seconds"
    )