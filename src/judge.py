import pandas as pd

from evaluation.llm_judge import evaluate_summary


# =========================================================
# INPUT
# =========================================================

INPUT_PATH = (
    "data/evaluation/results_v2.csv"
)


OUTPUT_PATH = (
    "data/evaluation/judge_results_v2.csv"
)


# =========================================================
# LOAD RESULTS
# =========================================================

df = pd.read_csv(
    INPUT_PATH
)


print(
    f"Loaded {len(df)} summaries"
)


# =========================================================
# EVALUATE EACH SUMMARY
# =========================================================

judge_results = []


for index, row in df.iterrows():

    print(
        f"Evaluating "
        f"{index + 1}/{len(df)}"
    )


    result = evaluate_summary(
        row["reference_summary"],
        row["generated_summary"]
    )


    judge_results.append({

        "id":
            row["id"],

        "prompt_version":
            row["prompt_version"],

        "faithfulness":
            result["faithfulness"],

        "relevance":
            result["relevance"],

        "completeness":
            result["completeness"],

        "conciseness":
            result["conciseness"],

        "reason":
            result["reason"]
    })


# =========================================================
# CREATE DATAFRAME
# =========================================================

judge_df = pd.DataFrame(
    judge_results
)


# =========================================================
# SAVE RESULTS
# =========================================================

judge_df.to_csv(
    OUTPUT_PATH,
    index=False
)


# =========================================================
# DISPLAY RESULTS
# =========================================================

print("\n")
print("=" * 60)

print(
    "LLM JUDGE RESULTS"
)

print("=" * 60)


print(
    judge_df[
        [
            "faithfulness",
            "relevance",
            "completeness",
            "conciseness"
        ]
    ].mean()
)


print("\nSaved to:")

print(
    OUTPUT_PATH
)