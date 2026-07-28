from typing import List, Dict
from neo4j import AsyncGraphDatabase
from .base import RetrievalPipeline
from .structured_outputs import DecompositionResponse, ModelResponse
from .graph import GraphRetrievalMixin

class SaPipelineDecomp(RetrievalPipeline, GraphRetrievalMixin):
    """Spreading-activation RAG pipeline with question decomposition."""

    def __init__(self, name, neo4j_url, neo4j_user, neo4j_password,
        llm_endpoint_url, llm_api_key, llm_model, embedding_model,
        answering_prompt, reasoning_prompt, retrieve_k,
        # sa parameters
        activating_descriptions, normalization_parameter, activation_threshold, pruning_threshold, k_hop
    ):
        """Initialize the SA decomposition pipeline with Neo4j driver and SA parameters."""
        super().__init__(llm_endpoint_url, llm_api_key, llm_model, embedding_model, name)
        self.driver = AsyncGraphDatabase.driver(
            neo4j_url, auth=(neo4j_user, neo4j_password),
            notifications_disabled_categories=["DEPRECATION"],
        )
        self.retrieve_k = retrieve_k
        with open(answering_prompt, 'r') as f:
            self.answering_prompt = f.read()
        with open(reasoning_prompt, 'r') as f:
            self.reasoning_prompt = f.read()
        self.activating_descriptions = activating_descriptions
        self.normalization_parameter = normalization_parameter
        self.activation_threshold = activation_threshold
        self.pruning_threshold = pruning_threshold
        self.k_hop = k_hop


    async def _decompose_question(self, query: str) -> List[str]:
        """Decompose a complex question into subquestions.

        Args:
            query: The original question.

        Returns:
            List of subquestion strings.
        """
        messages = [
            {"role": "system", "content": self.reasoning_prompt},
            {"role": "user", "content": query}
        ]

        reasoning_result = await self.client.beta.chat.completions.parse(
            model=self.llm_model,
            messages=messages,
            response_format=DecompositionResponse,
        )
        parsed = reasoning_result.choices[0].message.parsed
        sub_queries = [q.question for q in parsed.subquestions]

        return sub_queries

    async def _answer_subquestion(self, subquestion: str, memory: List[str]) -> str:
        """Answer a single subquestion using SA retrieval and prior answers.

        Args:
            subquestion: The subquestion to answer.
            memory: List of previous question-answer strings.

        Returns:
            Formatted string with the subquestion and its answer.
        """
        if memory:
            extended_question = f"{subquestion}\n\n" + '\n\n'.join(memory)
        else:
            extended_question = subquestion

        query_embedding = await self._embed(extended_question, query_prefix=True)
        context = await self._knowledge_acquisition_step(
            query_embedding,
            self.retrieve_k,
            self.activating_descriptions,
            self.activation_threshold,
            self.pruning_threshold
        )

        messages = [
            {"role": "system", "content": self.answering_prompt},
            {"role": "user", "content": context},
            {"role": "user", "content": subquestion}
        ]

        resp = await self.client.beta.chat.completions.parse(
            model=self.llm_model,
            messages=messages,
            response_format=ModelResponse
        )
        parsed = resp.choices[0].message.parsed

        return f"Sub-question:  {subquestion}\nAnswer: {parsed.final_answer}"

    async def _generate_answer(self, query: str, sub_questions: List[str]) -> Dict[str, str]:
        """Generate the final answer by sequentially answering subquestions.

        Args:
            query: The original question.
            sub_questions: List of decomposed subquestions.

        Returns:
            Dict with 'answer', 'reasoning', and 'knowledge' keys.
        """
        memory = []

        # Answer each subquestion sequentially
        for i, subquestion in enumerate(sub_questions):
            answer_str = await self._answer_subquestion(
                subquestion,
                memory[: i]
            )
            memory.append(answer_str)

        query_embedding = await self._embed(query, query_prefix=True)
        context = await self._knowledge_acquisition_step(
            query_embedding,
            self.retrieve_k,
            self.activating_descriptions,
            self.activation_threshold,
            self.pruning_threshold
        )
        # Combine with subquestion answers
        final_context = (
                f"{context}\n\n**Answers to sub-questions**\n\n"
                + '\n\n'.join(memory)
        )
        # Generate final answer
        messages = [
            {"role": "system", "content": self.answering_prompt},
            {"role": "user", "content": final_context},
            {"role": "user", "content": query}
        ]
        resp = await self.client.beta.chat.completions.parse(
            model=self.llm_model, messages=messages, response_format=ModelResponse
        )
        parsed = resp.choices[0].message.parsed
        return {
            'answer': parsed.final_answer,
            'reasoning': parsed.reasoning,
            'knowledge': final_context
        }


    async def run(self, query: str) -> Dict[str, str]:
        """Decompose the query and generate the final answer via SA retrieval."""
        sub_queries = await self._decompose_question(query)
        return await self._generate_answer(query, sub_queries)
