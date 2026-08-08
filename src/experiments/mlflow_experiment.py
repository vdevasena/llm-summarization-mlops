from pathlib import Path

import pandas as pd
import mlflow


# =========================================================
# 1. PROJECT PATH
# =========================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)


# =========================================================
# 2. PROMPT VERSIONS
# =========================================================

PROMPT_VERSIONS = [
    "v1",
    "v2"
]


# =========================================================
# 3. MLFLOW DATABASE
# =========================================================

MLFLOW_DB = (
    PROJECT_ROOT
    / "mlflow.db"
)


# =========================================================
# 4. MLFLOW SETUP
# =========================================================

mlflow.set_tracking_uri(
    f"sqlite:///{MLFLOW_DB}"
)


mlflow.set_experiment(
    "summarization_prompt_experiments"
)


# =========================================================
# 5. FUNCTION
# =========================================================

def run_mlflow_experiment(
    prompt_version
):

    print(
        "\n"
        + "=" * 60
    )

    print(
        f"PROCESSING {prompt_version.upper()}"
    )

    print(
        "=" * 60
    )


    # -----------------------------------------------------
    # File paths
    # -----------------------------------------------------

    results_path = (
        PROJECT_ROOT
        / "data"
        / "evaluation"
        / f"results_{prompt_version}.csv"
    )


    judge_path = (
        PROJECT_ROOT
        / "data"
        / "evaluation"
        / f"judge_results_{prompt_version}.csv"
    )


    # -----------------------------------------------------
    # Load results
    # -----------------------------------------------------

    print(
        "\nLoading experiment results..."
    )


    results_df = pd.read_csv(
        results_path
    )


    print(
        f"Loaded {len(results_df)} "
        "experiment rows."
    )


    # -----------------------------------------------------
    # Load judge results
    # -----------------------------------------------------

    print(
        "Loading LLM Judge results..."
    )


    judge_df = pd.read_csv(
        judge_path
    )


    print(
        f"Loaded {len(judge_df)} "
        "judge rows."
    )


    # -----------------------------------------------------
    # Validate experiment columns
    # -----------------------------------------------------

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
            f"Missing result columns: "
            f"{missing_columns}"
        )


    # -----------------------------------------------------
    # Validate judge columns
    # -----------------------------------------------------

    judge_required_columns = [
        "id",
        "faithfulness",
        "relevance",
        "completeness",
        "conciseness"
    ]


    missing_judge_columns = [
        column
        for column in judge_required_columns
        if column not in judge_df.columns
    ]


    if missing_judge_columns:

        raise ValueError(
            f"Missing judge columns: "
            f"{missing_judge_columns}"
        )


    # -----------------------------------------------------
    # Merge
    # -----------------------------------------------------

    print(
        "Merging experiment + judge results..."
    )


    combined_df = results_df.merge(
        judge_df[
            [
                "id",
                "faithfulness",
                "relevance",
                "completeness",
                "conciseness"
            ]
        ],
        on="id",
        how="inner"
    )


    print(
        f"Combined rows: "
        f"{len(combined_df)}"
    )


    # -----------------------------------------------------
    # Metrics
    # -----------------------------------------------------

    num_examples = len(
        combined_df
    )


    avg_rouge1 = (
        combined_df["rouge1"].mean()
    )


    avg_rouge2 = (
        combined_df["rouge2"].mean()
    )


    avg_rougeL = (
        combined_df["rougeL"].mean()
    )


    avg_latency = (
        combined_df[
            "latency_seconds"
        ].mean()
    )


    avg_faithfulness = (
        combined_df[
            "faithfulness"
        ].mean()
    )


    avg_relevance = (
        combined_df[
            "relevance"
        ].mean()
    )


    avg_completeness = (
        combined_df[
            "completeness"
        ].mean()
    )


    avg_conciseness = (
        combined_df[
            "conciseness"
        ].mean()
    )


    avg_llm_quality = (
        avg_faithfulness
        + avg_relevance
        + avg_completeness
        + avg_conciseness
    ) / 4


    # -----------------------------------------------------
    # Optional token/cost metrics
    # -----------------------------------------------------

    avg_input_tokens = None

    avg_output_tokens = None

    avg_total_tokens = None

    avg_cost = None


    if "input_tokens" in combined_df.columns:

        avg_input_tokens = (
            combined_df[
                "input_tokens"
            ].mean()
        )


    if "output_tokens" in combined_df.columns:

        avg_output_tokens = (
            combined_df[
                "output_tokens"
            ].mean()
        )


    if "total_tokens" in combined_df.columns:

        avg_total_tokens = (
            combined_df[
                "total_tokens"
            ].mean()
        )


    if "estimated_cost" in combined_df.columns:

        avg_cost = (
            combined_df[
                "estimated_cost"
            ].mean()
        )


    # =====================================================
    # MLFLOW RUN
    # =====================================================

    with mlflow.start_run(
        run_name=f"prompt_{prompt_version}"
    ):

        # -------------------------------------------------
        # Parameters
        # -------------------------------------------------

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


        mlflow.log_param(
            "judge_model",
            "OpenAI"
        )


        # -------------------------------------------------
        # ROUGE
        # -------------------------------------------------

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


        # -------------------------------------------------
        # LLM Judge
        # -------------------------------------------------

        mlflow.log_metric(
            "avg_faithfulness",
            avg_faithfulness
        )


        mlflow.log_metric(
            "avg_relevance",
            avg_relevance
        )


        mlflow.log_metric(
            "avg_completeness",
            avg_completeness
        )


        mlflow.log_metric(
            "avg_conciseness",
            avg_conciseness
        )


        mlflow.log_metric(
            "avg_llm_quality",
            avg_llm_quality
        )


        # -------------------------------------------------
        # Performance
        # -------------------------------------------------

        mlflow.log_metric(
            "avg_latency_seconds",
            avg_latency
        )


        # -------------------------------------------------
        # Tokens / Cost
        # -------------------------------------------------

        if avg_input_tokens is not None:

            mlflow.log_metric(
                "avg_input_tokens",
                avg_input_tokens
            )


        if avg_output_tokens is not None:

            mlflow.log_metric(
                "avg_output_tokens",
                avg_output_tokens
            )


        if avg_total_tokens is not None:

            mlflow.log_metric(
                "avg_total_tokens",
                avg_total_tokens
            )


        if avg_cost is not None:

            mlflow.log_metric(
                "avg_cost",
                avg_cost
            )


        # -------------------------------------------------
        # Artifacts
        # -------------------------------------------------

        mlflow.log_artifact(
            str(results_path),
            artifact_path="results"
        )


        mlflow.log_artifact(
            str(judge_path),
            artifact_path="judge"
        )


        # -------------------------------------------------
        # Print results
        # -------------------------------------------------

        run_id = (
            mlflow.active_run()
            .info
            .run_id
        )


        print(
            "\nMLflow run created:"
        )


        print(
            f"Prompt: {prompt_version}"
        )


        print(
            f"Run ID: {run_id}"
        )


        print(
            "\nROUGE:"
        )


        print(
            f"ROUGE-1: "
            f"{avg_rouge1:.4f}"
        )


        print(
            f"ROUGE-2: "
            f"{avg_rouge2:.4f}"
        )


        print(
            f"ROUGE-L: "
            f"{avg_rougeL:.4f}"
        )


        print(
            "\nLLM Judge:"
        )


        print(
            f"Faithfulness: "
            f"{avg_faithfulness:.4f}"
        )


        print(
            f"Relevance: "
            f"{avg_relevance:.4f}"
        )


        print(
            f"Completeness: "
            f"{avg_completeness:.4f}"
        )


        print(
            f"Conciseness: "
            f"{avg_conciseness:.4f}"
        )


        print(
            f"Overall Quality: "
            f"{avg_llm_quality:.4f}"
        )


# =========================================================
# 6. RUN BOTH VERSIONS
# =========================================================

if __name__ == "__main__":

    for version in PROMPT_VERSIONS:

        run_mlflow_experiment(
            version
        )


    print(
        "\n"
        + "=" * 60
    )

    print(
        "ALL PROMPT EXPERIMENTS COMPLETE"
    )

    print(
        "=" * 60
    )