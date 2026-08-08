import time
import sys
import pandas as pd

from pathlib import Path


# =========================================================
# PROJECT PATH
# =========================================================

SRC_DIR = (
    Path(__file__)
    .resolve()
    .parents[1]
)


# =========================================================
# ADD SRC TO PYTHON PATH
# =========================================================

sys.path.insert(
    0,
    str(SRC_DIR)
)


# =========================================================
# IMPORTS
# =========================================================

from summarizer import summarize

from evaluation.rouge_metrics import (
    calculate_rouge
)


# =========================================================
# INPUT
# =========================================================

INPUT_PATH = (
    "data/evaluation/eval_set.csv"
)


# =========================================================
# EXPERIMENT
# =========================================================

def run_experiment(
    prompt_version
):

    # -----------------------------------------------------
    # Load evaluation dataset
    # -----------------------------------------------------

    df = pd.read_csv(
        INPUT_PATH
    )


    results = []


    # -----------------------------------------------------
    # Process every article
    # -----------------------------------------------------

    for index, row in df.iterrows():

        print(
            f"{prompt_version}: "
            f"{index + 1}/{len(df)}"
        )


        # -------------------------------------------------
        # Start timer
        # -------------------------------------------------

        start_time = time.time()


        # -------------------------------------------------
        # Generate summary
        # -------------------------------------------------

        result = summarize(
            row["article"],
            prompt_version
        )


        # -------------------------------------------------
        # Latency
        # -------------------------------------------------

        latency = (
            time.time()
            - start_time
        )


        # -------------------------------------------------
        # Extract summary
        # -------------------------------------------------

        generated_summary = (
            result["summary"]
        )


        # -------------------------------------------------
        # Extract token usage
        # -------------------------------------------------

        input_tokens = (
            result["input_tokens"]
        )

        output_tokens = (
            result["output_tokens"]
        )

        total_tokens = (
            result["total_tokens"]
        )


        # -------------------------------------------------
        # Extract cost
        # -------------------------------------------------

        estimated_cost = (
            result["estimated_cost"]
        )


        # -------------------------------------------------
        # ROUGE
        # -------------------------------------------------

        scores = calculate_rouge(
            row["reference_summary"],
            generated_summary
        )


        # -------------------------------------------------
        # Store result
        # -------------------------------------------------

        results.append({

            "id":
                row["id"],

            "prompt_version":
                prompt_version,
            "reference_summary": row["reference_summary"],
            "generated_summary":
                generated_summary,

            "rouge1":
                scores["rouge1"],

            "rouge2":
                scores["rouge2"],

            "rougeL":
                scores["rougeL"],

            "input_tokens":
                input_tokens,

            "output_tokens":
                output_tokens,

            "total_tokens":
                total_tokens,

            "estimated_cost":
                estimated_cost,

            "latency_seconds":
                latency
        })


    # =====================================================
    # CREATE RESULTS DATAFRAME
    # =====================================================

    results_df = pd.DataFrame(
        results
    )


    # =====================================================
    # OUTPUT PATH
    # =====================================================

    output_path = (
        f"data/evaluation/"
        f"results_{prompt_version}.csv"
    )


    # =====================================================
    # SAVE RESULTS
    # =====================================================

    results_df.to_csv(
        output_path,
        index=False
    )


    # =====================================================
    # PRINT RESULTS
    # =====================================================

    print("\n")
    print("=" * 60)
    print(
        f"RESULTS FOR {prompt_version}"
    )
    print("=" * 60)


    print("\nAverage ROUGE:")

    print(
        results_df[
            [
                "rouge1",
                "rouge2",
                "rougeL"
            ]
        ].mean()
    )


    print("\nAverage Token Usage:")

    print(
        results_df[
            [
                "input_tokens",
                "output_tokens",
                "total_tokens"
            ]
        ].mean()
    )


    print("\nAverage Cost:")

    print(
        results_df[
            "estimated_cost"
        ].mean()
    )


    print("\nAverage Latency:")

    print(
        results_df[
            "latency_seconds"
        ].mean()
    )


    print("\nResults saved to:")

    print(
        output_path
    )


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    run_experiment(
        "v1"
    )
    
    run_experiment(
        "v2"
    )