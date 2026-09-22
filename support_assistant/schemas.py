# support_assistant/schemas.py
"""
Pydantic models for the /ask endpoint: request and the enforced JSON
response schema (answer/sources/confidence). In mock mode, these are
populated deterministically from graph state -- no LLM output to
validate, since none was generated. The retry-on-validation-failure
logic for the optional real-LLM path lives in main.py (Step 6), since
it needs to wrap the actual LLM call, which only exists in that path.
"""
from pydantic import BaseModel, Field
from typing import List


class AskRequest(BaseModel):
    query: str


class AskResponse(BaseModel):
    answer: str
    sources: List[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)