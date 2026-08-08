from pathlib import Path

import pandas as pd


# =========================================================
# 1. PROJECT PATH
# =========================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)


# =========================================================
# 2. LOAD RESULTS
# =========================================================

versions = ["v1", "v2"]

all_results = []


for version in versions:

    results_path = (
        PROJECT_ROOT
        / "data"
        / "evaluation"
        / f"results_{version}.csv"
    )

    judge_path = (
        PROJECT_ROOT
        / "data"
        / "evaluation"
        / f"judge_results_{version}.csv"
    )


    results_df = pd.read_csv(
        results_path
    )


    judge_df = pd.read_csv(
        judge_path
    )


    # -----------------------------------------------------
    # Merge
    # -----------------------------------------------------

    df = results_df.merge(
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


    # -----------------------------------------------------
    # Calculate averages
    # -----------------------------------------------------

    result = {

        "prompt_version":
            version,

        "rouge1":
            df["rouge1"].mean(),

        "rouge2":
            df["rouge2"].mean(),

        "rougeL":
            df["rougeL"].mean(),

        "faithfulness":
            df["faithfulness"].mean(),

        "relevance":
            df["relevance"].mean(),

        "completeness":
            df["completeness"].mean(),

        "conciseness":
            df["conciseness"].mean(),

        "latency_seconds":
            df["latency_seconds"].mean()
    }


    # -----------------------------------------------------
    # Overall LLM quality
    # -----------------------------------------------------

    result["llm_quality"] = (
        result["faithfulness"]
        + result["relevance"]
        + result["completeness"]
        + result["conciseness"]
    ) / 4


    # -----------------------------------------------------
    # Cost if available
    # -----------------------------------------------------

    if "estimated_cost" in df.columns:

        result["cost"] = (
            df["estimated_cost"].mean()
        )

    else:

        result["cost"] = None


    all_results.append(
        result
    )


# =========================================================
# 3. CREATE COMPARISON TABLE
# =========================================================

comparison_df = pd.DataFrame(
    all_results
)


print(
    "\n"
    + "=" * 80
)

print(
    "PROMPT VERSION COMPARISON"
)

print(
    "=" * 80
)


print(
    comparison_df.to_string(
        index=False
    )
)


# =========================================================
# 4. DETERMINE BEST VERSION
# =========================================================

best_version = (
    comparison_df
    .sort_values(
        "llm_quality",
        ascending=False
    )
    .iloc[0]
)


print(
    "\n"
    + "=" * 80
)

print(
    "RECOMMENDATION"
)

print(
    "=" * 80
)


print(
    f"Best prompt: "
    f"{best_version['prompt_version']}"
)


print(
    f"LLM Quality: "
    f"{best_version['llm_quality']:.3f}"
)


print(
    f"Faithfulness: "
    f"{best_version['faithfulness']:.3f}"
)


print(
    f"Relevance: "
    f"{best_version['relevance']:.3f}"
)


print(
    f"Completeness: "
    f"{best_version['completeness']:.3f}"
)


print(
    f"Conciseness: "
    f"{best_version['conciseness']:.3f}"
)