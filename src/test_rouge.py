from evaluation.rouge_metrics import calculate_rouge


reference = """
Apple announced a new product and said it expects strong demand.
"""

generated = """
Apple announced a new product and expects strong demand.
"""


scores = calculate_rouge(
    reference,
    generated
)


print(scores)