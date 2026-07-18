from pydantic import BaseModel

class ReasoningResponse(BaseModel):
    provided_context: str
    answer_possible: bool
    final_answer: str
    additional_question: str

class ModelResponse(BaseModel):
    reasoning: str
    final_answer: str
