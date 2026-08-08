import json
import os

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)


JUDGE_PROMPT = """
You are an expert evaluator of text summaries.

Evaluate the generated summary against the original article.

Score each dimension from 1 to 5.

Definitions:

Faithfulness:
Does the summary contain only information supported by
the article?

Relevance:
Does the summary focus on the most important information?

Completeness:
Does the summary capture the important information from
the article?

Conciseness:
Does the summary avoid unnecessary information and repetition?

Return ONLY valid JSON.

Required format:

{{
    "faithfulness": 1-5,
    "relevance": 1-5,
    "completeness": 1-5,
    "conciseness": 1-5,
    "reason": "short explanation"
}}

ARTICLE:

{article}

GENERATED SUMMARY:

{summary}
"""


def evaluate_summary(
    article,
    summary
):

    prompt = JUDGE_PROMPT.format(
        article=article,
        summary=summary
    )

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=prompt
    )
# ---------------------------------------------------------
# Get response text
# ---------------------------------------------------------

    text = response.output_text.strip()

    print("\nRAW JUDGE RESPONSE:")
    print(text)


    # ---------------------------------------------------------
    # Remove markdown code fences if present
    # ---------------------------------------------------------

    if text.startswith("```"):

        text = text.replace(
            "```json",
            ""
        )

        text = text.replace(
            "```",
            ""
        )

        text = text.strip()


    # ---------------------------------------------------------
    # Extract JSON object
    # ---------------------------------------------------------

    start = text.find("{")

    end = text.rfind("}")


    if start == -1 or end == -1:

        raise ValueError(
            "LLM Judge did not return valid JSON.\n"
            f"Raw response:\n{text}"
        )


    json_text = text[
        start:end + 1
    ]


    # ---------------------------------------------------------
    # Parse JSON
    # ---------------------------------------------------------

    try:

        result = json.loads(
            json_text
        )

    except json.JSONDecodeError as e:

        print(
            "\nJSON parsing failed."
        )

        print(
            "Raw response:"
        )

        print(
            text
        )

        raise e


    return result