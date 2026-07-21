from typing import List, Dict
from neo4j import AsyncGraphDatabase
from .base import RetrievalPipeline
from .structured_outputs import ReasoningResponse, ModelResponse

class VectorRetrievalPipeline(RetrievalPipeline):
    """Baseline RAG: embed query, do a cosine-similarity search over chunk
    embeddings stored in Neo4j, hand the top-k chunks to the LLM."""

    def __init__(
        self, name,neo4j_url, neo4j_user, neo4j_password,
        llm_endpoint_url, llm_api_key, llm_model, embedding_model,
        answering_prompt, reasoning_enabled, reasoning_prompt, reasoning_steps,
        retrieve_k
    ):
        super().__init__(llm_endpoint_url, llm_api_key, llm_model, embedding_model, name)
        self.driver = AsyncGraphDatabase.driver(
            neo4j_url, auth=(neo4j_user, neo4j_password),
            notifications_disabled_categories=["DEPRECATION"],
        )
        self.retrieve_k = retrieve_k
        with open(answering_prompt, 'r') as f:
            self.answering_prompt = f.read()
        self.reasoning_enabled = reasoning_enabled
        if self.reasoning_enabled:
            with open(reasoning_prompt, 'r') as f:
                self.reasoning_prompt = f.read()
            self.reasoning_steps = reasoning_steps

    async def _generate_answer(self, query: str, context: List[str]) -> Dict[str, str]:
        context_text = "\n\n".join(context) if context else "No context found."

        if self.reasoning_enabled:
            context_text, _ = await self._apply_reasoning_steps(query, context_text, self.retrieve_k)

        messages = [
            {"role": "system", "content": self.answering_prompt},
            {"role": "user", "content": f"Context:\n{context_text}\n\nQuestion: {query}"},
        ]
        resp = await self.client.beta.chat.completions.parse(
            model=self.llm_model, messages=messages, response_format=ModelResponse
        )
        parsed = resp.choices[0].message.parsed
        return {
            'answer': parsed.final_answer,
            'reasoning': parsed.reasoning,
            'knowledge': context
        }

    async def _apply_reasoning_steps(self, query: str, context: str, top_k: int) -> tuple:
        reasoning_response = None

        for step in range(self.reasoning_steps):
            messages = [
                {"role": "system", "content": self.reasoning_prompt},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
            ]

            resp = await self.client.beta.chat.completions.parse(
                    model=self.llm_model,
                    messages=messages,
                    response_format=ReasoningResponse,
                )
            reasoning_response = resp.choices[0].message.parsed
            # If answer is possible with current context, stop reasoning
            if reasoning_response.answer_possible:
                break
            # Refine context with follow-up question
            summarized_context = reasoning_response.provided_context
            follow_up_question = reasoning_response.additional_question
            if follow_up_question:
                additional_documents = await self._retrieve(follow_up_question, self.retrieve_k)
            else:
                additional_documents = []
            additional_context = '\n\n'.join(additional_documents)
            context = f"{additional_context}\n\n{summarized_context}"

        return context, reasoning_response

    async def run(self, query: str) -> Dict[str, str]:
        """The fixed algorithm skeleton shared by every pipeline."""
        context = await self._retrieve(query, top_k=self.retrieve_k)
        return await self._generate_answer(query, context)