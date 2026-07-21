from typing import List, Dict
from neo4j import AsyncGraphDatabase
from .base import RetrievalPipeline
from .structured_outputs import ReasoningResponse, ModelResponse
from .graph import GraphRetrievalMixin

_CONTEXT_PREFIX_LEN = len("### Context \n")

class SaPipelineCot(RetrievalPipeline, GraphRetrievalMixin):
    def __init__(self, name, neo4j_url, neo4j_user, neo4j_password,
        llm_endpoint_url, llm_api_key, llm_model, embedding_model,
        answering_prompt, reasoning_enabled, reasoning_prompt, reasoning_steps,
        retrieve_k, 
        # sa parameters
        activating_descriptions, normalization_parameter, activation_threshold, pruning_threshold, k_hop
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
        self.activating_descriptions = activating_descriptions
        self.normalization_parameter = normalization_parameter
        self.activation_threshold = activation_threshold
        self.pruning_threshold = pruning_threshold
        self.k_hop = k_hop


    async def _generate_answer(self, query: str, context: str) -> Dict[str, str]:

        if self.reasoning_enabled:
            context, _ = await self._apply_reasoning_steps(query, context, self.retrieve_k)

        messages = [
            {"role": "system", "content": self.answering_prompt},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
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
            summarized_context = reasoning_response.provided_context
            additional_question = reasoning_response.additional_question
            query_embedding = await self._embed(additional_question)
            new_context = await self._knowledge_acquisition_step(
                query_embedding,
                self.retrieve_k,
                self.activating_descriptions,
                self.activation_threshold,
                self.pruning_threshold
            )
            context = (f"# Known information\n{summarized_context}\n"
                       f"# Additional facts{new_context[_CONTEXT_PREFIX_LEN:]}")

        return context, reasoning_response

    async def run(self, query: str) -> Dict[str, str]:
        """The fixed algorithm skeleton shared by every pipeline."""
        query_embedding = await self._embed(query)
        context = await self._knowledge_acquisition_step(
                query_embedding,
                self.retrieve_k,
                self.activating_descriptions,
                self.activation_threshold,
                self.pruning_threshold
            )
        return await self._generate_answer(query, context)