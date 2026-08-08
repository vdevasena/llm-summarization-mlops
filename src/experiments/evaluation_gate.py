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
# 2. CONFIGURATION
# =========================================================

PROMPT_VERSION = "v2"


# =========================================================
# 3. PRODUCTION THRESHOLDS
# =========================================================

MIN_FAITHFULNESS = 4.0

MIN_COMPLETENESS = 3.5

MIN_ROUGE_L = 0.35

MAX_LATENCY = 3.0

MAX_COST = 0.01


# =========================================================
# 4. LOAD RESULTS
# =========================================================

results_path = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / f"results_{PROMPT_VERSION}.csv"
)


judge_path = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / f"judge_results_{PROMPT_VERSION}.csv"
)


results_df = pd.read_csv(
    results_path
)


judge_df = pd.read_csv(
    judge_path
)


# =========================================================
# 5. MERGE
# =========================================================

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


# =========================================================
# 6. CALCULATE METRICS
# =========================================================

avg_faithfulness = (
    df["faithfulness"].mean()
)


avg_completeness = (
    df["completeness"].mean()
)


avg_rougeL = (
    df["rougeL"].mean()
)


avg_latency = (
    df["latency_seconds"].mean()
)


if "estimated_cost" in df.columns:

    avg_cost = (
        df["estimated_cost"].mean()
    )

else:

    avg_cost = 0


# =========================================================
# 7. EVALUATION GATES
# =========================================================

checks = {

    "faithfulness":
        avg_faithfulness
        >= MIN_FAITHFULNESS,

    "completeness":
        avg_completeness
        >= MIN_COMPLETENESS,

    "rougeL":
        avg_rougeL
        >= MIN_ROUGE_L,

    "latency":
        avg_latency
        <= MAX_LATENCY,

    "cost":
        avg_cost
        <= MAX_COST
}


# =========================================================
# 8. PRINT RESULTS
# =========================================================

print(
    "\n"
    + "=" * 70
)

print(
    "AUTOMATED EVALUATION GATE"
)

print(
    "=" * 70
)


print(
    f"\nPrompt Version: "
    f"{PROMPT_VERSION}"
)


print(
    f"\nFaithfulness: "
    f"{avg_faithfulness:.3f}"
    f"  >= {MIN_FAITHFULNESS}"
)


print(
    f"Completeness: "
    f"{avg_completeness:.3f}"
    f"  >= {MIN_COMPLETENESS}"
)


print(
    f"ROUGE-L: "
    f"{avg_rougeL:.3f}"
    f"  >= {MIN_ROUGE_L}"
)


print(
    f"Latency: "
    f"{avg_latency:.3f}s"
    f"  <= {MAX_LATENCY}s"
)


print(
    f"Cost: "
    f"${avg_cost:.6f}"
    f"  <= ${MAX_COST}"
)


# =========================================================
# 9. DISPLAY INDIVIDUAL GATES
# =========================================================

print(
    "\n"
    + "-" * 70
)


for name, passed in checks.items():

    status = (
        "PASS"
        if passed
        else
        "FAIL"
    )


    print(
        f"{name.upper():20} "
        f"{status}"
    )


# =========================================================
# 10. FINAL DECISION
# =========================================================

all_passed = all(
    checks.values()
)


print(
    "\n"
    + "=" * 70
)


if all_passed:

    print(
        "DECISION: APPROVED"
    )

    print(
        "All production gates passed."
    )

else:

    print(
        "DECISION: REJECTED"
    )

    print(
        "One or more production gates failed."
    )


print(
    "=" * 70
)


# =========================================================
# 11. CI/CD FAILURE
# =========================================================

if not all_passed:

    raise SystemExit(
        1
    )