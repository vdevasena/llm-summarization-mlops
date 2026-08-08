from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


# =========================================================
# PROJECT PATH
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# =========================================================
# LOAD ENVIRONMENT VARIABLES
# =========================================================

load_dotenv(
    PROJECT_ROOT / ".env"
)


# =========================================================
# OPENAI CLIENT
# =========================================================

client = OpenAI()


# =========================================================
# MODEL
# =========================================================

MODEL_NAME = "gpt-4.1-mini"


# =========================================================
# COST PER TOKEN
# =========================================================

# Replace these with the current pricing for the model
# you are actually using if pricing changes.

INPUT_COST_PER_1M = 0.40

OUTPUT_COST_PER_1M = 1.60


# =========================================================
# SUMMARIZE
# =========================================================

def summarize(
    article,
    prompt_version
):

    # -----------------------------------------------------
    # Load prompt
    # -----------------------------------------------------

    prompt_path = (
        PROJECT_ROOT
        / "prompts"
        / f"{prompt_version}.txt"
    )

    with open(
        prompt_path,
        "r",
        encoding="utf-8"
    ) as f:

        prompt_template = f.read()


    # -----------------------------------------------------
    # Insert article
    # -----------------------------------------------------

    prompt = prompt_template.replace(
        "{article}",
        article
    )


    # -----------------------------------------------------
    # OpenAI request
    # -----------------------------------------------------

    response = client.responses.create(
        model=MODEL_NAME,
        input=prompt
    )


    # -----------------------------------------------------
    # Summary
    # -----------------------------------------------------

    summary = response.output_text


    # -----------------------------------------------------
    # Token usage
    # -----------------------------------------------------

    usage = response.usage

    input_tokens = (
        usage.input_tokens
    )

    output_tokens = (
        usage.output_tokens
    )

    total_tokens = (
        input_tokens
        + output_tokens
    )


    # -----------------------------------------------------
    # Cost
    # -----------------------------------------------------

    input_cost = (
        input_tokens
        / 1_000_000
        * INPUT_COST_PER_1M
    )

    output_cost = (
        output_tokens
        / 1_000_000
        * OUTPUT_COST_PER_1M
    )

    total_cost = (
        input_cost
        + output_cost
    )


    # -----------------------------------------------------
    # Return everything
    # -----------------------------------------------------

    return {
        "summary": summary,

        "input_tokens": input_tokens,

        "output_tokens": output_tokens,

        "total_tokens": total_tokens,

        "estimated_cost": total_cost
    }