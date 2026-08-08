import pandas as pd


v1 = pd.read_csv("data/evaluation/results_v1.csv")
v2 = pd.read_csv("data/evaluation/results_v2.csv")


metrics = [
    "rouge1",
    "rouge2",
    "rougeL",
    "latency_seconds"
]


comparison = pd.DataFrame({
    "v1": v1[metrics].mean(),
    "v2": v2[metrics].mean()})


print(comparison)