# LLM Summarization MLOps

An end-to-end LLM summarization system demonstrating prompt engineering, automated evaluation, experiment tracking, LLM-as-a-judge evaluation, MLflow, FastAPI, Docker, and deployment.

## Project Overview

This project builds a production-oriented article summarization system using OpenAI models.

The system includes:

- Prompt engineering
- Prompt versioning
- ROUGE evaluation
- LLM-as-a-judge evaluation
- Faithfulness evaluation
- Relevance evaluation
- Completeness evaluation
- Conciseness evaluation
- Latency tracking
- MLflow experiment tracking
- Model/prompt selection
- FastAPI inference API
- Docker containerization
- CI/CD
- Production monitoring

## Architecture

Article
    |
    v
Prompt Template
    |
    v
OpenAI API
    |
    v
Generated Summary
    |
    +------------------+
    |                  |
    v                  v
ROUGE Evaluation   LLM Judge
    |                  |
    +--------+---------+
             |
             v
        MLflow Tracking
             |
             v
       Prompt Comparison
             |
             v
        Best Prompt
             |
             v
          FastAPI
             |
             v
           Docker
             |
             v
        Deployment

## Evaluation Metrics

### Automatic Metrics

- ROUGE-1
- ROUGE-2
- ROUGE-L

### LLM Judge Metrics

- Faithfulness
- Relevance
- Completeness
- Conciseness
- Overall Quality

### Production Metrics

- Latency
- Token usage
- Cost
- Error rate

## Project Structure

```text
summarization/
├── data/
├── prompts/
├── src/
│   ├── evaluation/
│   └── experiments/
├── .env.example
├── .gitignore
├── Dockerfile
├── requirements.txt
└── README.md