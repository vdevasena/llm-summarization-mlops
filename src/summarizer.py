import os

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)


def load_prompt(version):

    path = f"prompts/{version}.txt"

    with open(path, "r", encoding="utf-8") as file:
        return file.read()


def summarize(
    article,
    prompt_version="v1"
):

    prompt = load_prompt(
        prompt_version
    )

    prompt = prompt.replace(
        "{article}",
        article
    )

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=prompt
    )

    return response.output_text