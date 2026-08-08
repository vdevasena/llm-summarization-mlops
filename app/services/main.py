from fastapi import FastAPI

from src.summarizer import summarize


app = FastAPI(
    title="LLM Summarization API"
)


@app.get("/health")
def health():

    return {
        "status": "healthy"
    }


@app.post("/summarize")
def summarize_document(
    document: str
):

    summary = summarize(
        document
    )

    return {
        "summary": summary
    }