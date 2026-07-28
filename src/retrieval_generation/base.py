from abc import ABC, abstractmethod
from typing import List, Optional, Dict
from openai import AsyncOpenAI

DEFAULT_QUERY_PROMPT = 'Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:'

class RetrievalPipeline(ABC):
    """Abstract base for retrieval-generation pipelines (Template Method).

    Provides shared embedding/retrieval helpers and defines the
    embed -> retrieve -> generate skeleton via abstract methods.
    """

    def __init__(
        self,
        llm_endpoint_url: str,
        llm_api_key: str,
        llm_model: str,
        embedding_model: str = "bge-large:latest",
        name: str = "rag_pipeline"
    ):
        """Initialize the async OpenAI-compatible client and model names."""
        self.client = AsyncOpenAI(api_key=llm_api_key, base_url=llm_endpoint_url)
        self.llm_model = llm_model
        self.embedding_model = embedding_model
        self.name = name

    # ---------- shared / invariant ----------

    async def _embed(self, text: str, query_prefix: Optional[bool] = False) -> List[float]:
        """Embed a single string via the OpenAI-compatible embeddings endpoint."""
        if query_prefix:
            text = f"{DEFAULT_QUERY_PROMPT} {text}"
        resp = await self.client.embeddings.create(
            model=self.embedding_model,
            input=[text],            # NOTE: single string, wrapped in a list
        )
        return resp.data[0].embedding

    # ---------- template method ----------
    async def _retrieve(self, query: str, top_k: int = 5) -> List[str]:
        """Retrieve the top-k chunk texts for a query via vector similarity.

        Args:
            query: The query string.
            top_k: Number of chunks to retrieve.

        Returns:
            List of retrieved chunk text strings.
        """
        q_emb = await self._embed(query, query_prefix=True)
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