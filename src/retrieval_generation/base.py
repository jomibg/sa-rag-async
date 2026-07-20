from abc import ABC, abstractmethod
from typing import List, Optional, Dict
from openai import AsyncOpenAI


class RetrievalPipeline(ABC):
    """Abstract base for retrieval-generation pipelines (Template Method).

    Invariant behaviour lives here:
        * `self.client`        — OpenAI-compatible async client
        * `_embed(text)`       — single-string embedding helper

    Variant behaviour is delegated to subclasses via the abstract method
    `generate_answer`. A concrete `run()` method defines the overall
    algorithm skeleton:  embed  ->  retrieve  ->  generate.
    """

    def __init__(
        self,
        llm_endpoint_url: str,
        llm_api_key: str,
        llm_model: str,
        embedding_model: str = "bge-large:latest",
    ):
        self.client = AsyncOpenAI(api_key=llm_api_key, base_url=llm_endpoint_url)
        self.llm_model = llm_model
        self.embedding_model = embedding_model

    # ---------- shared / invariant ----------

    async def _embed(self, text: str) -> List[float]:
        """Embed a single string using the OpenAI-compatible /v1/embeddings endpoint."""
        resp = await self.client.embeddings.create(
            model=self.embedding_model,
            input=[text],            # NOTE: single string, wrapped in a list
        )
        return resp.data[0].embedding

    # ---------- template method ----------
    async def _retrieve(self, query: str, top_k: int = 5) -> List[str]:
        q_emb = await self._embed(query)
        cypher = """
            WITH $query_embedding AS query_embedding 
            CALL db.index.vector.queryNodes('textEmbedding', 500, query_embedding) 
            YIELD node AS c, score AS similarity
            RETURN c.text AS text, similarity
            ORDER BY similarity DESC
            LIMIT $top_k
        """
        async with self.driver.session() as session:
            result = await session.run(cypher, top_k=top_k, query_embedding=q_emb)
            rows = [r["text"] async for r in result]
        return rows

    # ---------- variant / abstract ----------

    @abstractmethod
    async def _generate_answer(self, query: str, context: List[str]) -> Dict[str, str]:
        """Produce the final answer. Implementation differs per pipeline."""
        raise NotImplementedError

    @abstractmethod 
    async def run(self, query: str) -> Dict[str, str]:
        """The fixed algorithm skeleton shared by every pipeline."""
        raise NotImplementedError