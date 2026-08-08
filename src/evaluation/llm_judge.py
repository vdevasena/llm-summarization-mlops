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

    text = response.output_text

    return json.loads(text)