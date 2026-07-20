from pydantic import BaseModel
from typing import List  

class ReasoningResponse(BaseModel):
    provided_context: str
    answer_possible: bool
    final_answer: str
    additional_question: str

class ModelResponse(BaseModel):
    reasoning: str
    final_answer: str

class Subquestion(BaseModel):
    id: int
    question: str

class DecompositionResponse(BaseModel):
    original_question: str
    subquestions: List[Subquestion]