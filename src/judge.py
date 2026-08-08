import pandas as pd

from evaluation.llm_judge import evaluate_summary

df = pd.read_csv(
    "data/evaluation/results_v1.csv"
)


row = df.iloc[0]

result = evaluate_summary(
    row["reference_summary"],
    row["generated_summary"]
)


print(result)