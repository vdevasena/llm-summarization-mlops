def quality_gate(metrics):

    requirements = {
        "rougeL": 0.35,
        "faithfulness": 4.3,
        "relevance": 4.2,
        "completeness": 4.0,
        "conciseness": 4.0
    }

    failures = []

    for metric, threshold in requirements.items():

        value = metrics[metric]

        if value < threshold:

            failures.append(
                f"{metric}: "
                f"{value:.2f} < {threshold}"
            )

    if failures:

        print("QUALITY GATE FAILED")

        for failure in failures:
            print(failure)

        return False

    print("QUALITY GATE PASSED")

    return True