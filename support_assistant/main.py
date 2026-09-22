"""
support_assistant/main.py -- FastAPI wrapper around the LangGraph app.
POST /ask accepts {"query": str} and returns the validated AskResponse
schema (answer/sources/confidence).

Retry-on-validation-failure: in mock mode there's no LLM output to fail
validation (state is built deterministically), so this path never
triggers here -- it exists as required scaffolding for the optional
MOCK_LLM=0 extension, where a real LLM's raw output could fail Pydantic
validation and warrants up to 2 corrective retries before erroring out.
"""
import os
from fastapi import FastAPI
from pydantic import ValidationError

from support_assistant.schemas import AskRequest, AskResponse
from support_assistant.graph import build_graph

app = FastAPI(title="Zepto Support Assistant")
graph_app = build_graph()

MAX_RETRIES = 2


def run_graph_with_validation(query: str) -> AskResponse:
    attempt = 0
    last_error = None

    while attempt <= MAX_RETRIES:
        try:
            state = graph_app.invoke({"query": query})
            # Validates against the schema; in mock mode this always
            # succeeds since state fields are built deterministically.
            # In the optional real-LLM path, a malformed LLM output could
            # raise ValidationError here, triggering a retry with attempt+1.
            return AskResponse(
                answer=state["answer"],
                sources=state["sources"],
                confidence=state["confidence"],
            )
        except ValidationError as e:
            last_error = e
            attempt += 1
            # Optional real-LLM extension would re-prompt with a corrective
            # instruction here before retrying; mock mode never reaches this.

    # Only reachable if validation failed MAX_RETRIES+1 times in a row --
    # in mock mode this branch is unreachable, since mock output is always
    # schema-valid by construction.
    return AskResponse(
        answer=f"Error: could not produce a valid response after {MAX_RETRIES} retries.",
        sources=[],
        confidence=0.0,
    )


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    return run_graph_with_validation(request.query)


@app.get("/")
def root():
    return {"status": "Zepto Support Assistant is running", "mock_llm": os.environ.get("MOCK_LLM", "1")}