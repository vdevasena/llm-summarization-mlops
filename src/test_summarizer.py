import pandas as pd

from summarizer import summarize

df = pd.read_csv(
    "data/evaluation/eval_set.csv"
)


article = df.iloc[0]["article"]

summary = summarize(article)


print("\nARTICLE:\n")
print(article)

print("\n\nGENERATED SUMMARY:\n")
print(summary)

print("\n\nREFERENCE SUMMARY:\n")
print(df.iloc[0]["reference_summary"])