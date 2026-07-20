from typing import List, Dict
from neo4j import AsyncGraphDatabase
from .base import RetrievalPipeline
from .structured_outputs import DecompositionResponse, ModelResponse

class DecompositionPipeline(RetrievalPipeline):
    """Decomposition RAG: pipeline that decomposes complex questions and answers iteratively."""

    def __init__(
        self, neo4j_url, neo4j_user, neo4j_password,
        llm_endpoint_url, llm_api_key, llm_model, embedding_model,
        answering_prompt, reasoning_prompt,
        retrieve_k

    ):
        super().__init__(llm_endpoint_url, llm_api_key, llm_model, embedding_model)
        self.driver = AsyncGraphDatabase.driver(
            neo4j_url, auth=(neo4j_user, neo4j_password),
            notifications_disabled_categories=["DEPRECATION"],
        )
        self.retrieve_k = retrieve_k
        with open(answering_prompt, 'r') as f:
            self.answering_prompt = f.read()
        with open(reasoning_prompt, 'r') as f:
        	self.reasoning_prompt = f.read()

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
        """Answer a single subquestion using retrieved context and previous answers.

        Args:
            subquestion: The subquestion to answer.
            memory: List of previous question-answer pairs. 
            retrieve_k: Number of documents to retrieve.

        Returns:
            Formatted string with the subquestion and its answer.
        """
        documents = await self._retrieve(subquestion, self.retrieve_k)
        context = '\n\n'.join(documents)

        # Add previous answers as additional context
        if memory:
            additional_context = '\n\n'.join(memory)
            full_context = f"{context}\n\n{additional_context}"
        else:
            full_context = context

        messages = [
    		{"role": "system", "content": self.answering_prompt},
    		{"role": "user", "content": f"{full_context}\n\nQuestion: {subquestion}"},
		]

        result = await self.client.beta.chat.completions.parse(
            model=self.llm_model,
            messages=messages,
            response_format=ModelResponse
        )
        parsed = result.choices[0].message.parsed

        return f"Sub-question:  {subquestion}\nAnswer:  {parsed.final_answer}"

    async def _generate_answer(self, query: str, sub_questions: List[str]) -> Dict[str, str]:
        """Generate final answer from subquestions. 

        Args:
            original_question: The original question.
            sub_questions: List of decomposed subquestions.
            retrieve_k: Number of documents to retrieve.

        Returns:
            Dictionary with 'answer', 'reasoning', and 'knowledge' keys.
        """
        memory = []

        # Answer each subquestion sequentially
        for i, subquestion in enumerate(sub_questions):
            answer_str = await self._answer_subquestion(
                subquestion,
                memory[: i]  # Pass only previous answers
            )
            memory.append(answer_str)

        documents = await self._retrieve(query, top_k=self.retrieve_k)
        context = '\n\n'.join(memory) + '\n\n' + '\n\n'.join(documents)
        # Generate final answer
        messages = [
    		{"role": "system", "content": self.answering_prompt},
    		{"role": "user", "content": f"{context}\n\nQuestion: {query}"},
		]
        result = await self.client.beta.chat.completions.parse(
            model=self.llm_model,
            messages=messages,
            response_format=ModelResponse
        )
        parsed = result.choices[0].message.parsed
        return {
            'answer': parsed.final_answer,
            'reasoning': parsed.reasoning,
            'knowledge': context
        }

    async def run(self, query: str) -> Dict[str, str]:
    	sub_queries = await self._decompose_question(query)
    	return await self._generate_answer(query, sub_queries)