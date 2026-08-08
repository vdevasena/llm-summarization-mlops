from datasets import load_dataset


print("Loading CNN/DailyMail from Parquet...")


dataset = load_dataset(
    "abisee/cnn_dailymail",
    "3.0.0"
)


print("\nDataset loaded successfully!\n")

print(dataset)


print("\nNumber of examples:")

print(
    "Train:",
    len(dataset["train"])
)

print(
    "Validation:",
    len(dataset["validation"])
)

print(
    "Test:",
    len(dataset["test"])
)


print("\nFirst example:\n")

example = dataset["test"][0]

print("ID:")
print(example["id"])

print("\nARTICLE:")
print(example["article"][:1000])

print("\nHIGHLIGHTS:")
print(example["highlights"])