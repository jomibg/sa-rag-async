from pydantic import BaseModel
from typing import List  

class ReasoningResponse(BaseModel):
    """Structured LLM response for a reasoning step."""

    provided_context: str
    answer_possible: bool
    final_answer: str
    additional_question: str

class ModelResponse(BaseModel):
    """Structured LLM response with reasoning and a final answer."""

    reasoning: str
    final_answer: str

class Subquestion(BaseModel):
    """A decomposed subquestion with an identifier."""

    id: int
    question: str

class DecompositionResponse(BaseModel):
    """Structured LLM response for question decomposition."""

    original_question: str
    subquestions: List[Subquestion]