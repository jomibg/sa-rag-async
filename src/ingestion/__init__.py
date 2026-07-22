# src/ingestion/__init__.py
from typing import List, Optional

from .kg_ingestion import AdvancedKGIngestor


async def run_ingestion(
    corpus_list: List[str],
    *,
    neo4j_url: str,
    neo4j_user: str,
    neo4j_password: str,
    ner_model: str,
    re_model: str,
    llm_endpoint_url: str,
    llm_api_key: str,
    template_re_loc: str,
    template_ner_loc: str,
    embedding_model: str,
    chunking_method: str = "word_based",
    chunk_size: int = 500,
    overlap_size: int = 200,
    concurrency: int = 3,
) -> AdvancedKGIngestor:
    """Build an ingestor from config and run ingestion over `corpus_list`.

    Returns the ingestor instance (with driver already closed).
    """
    ingestion_pipeline = AdvancedKGIngestor(
        neo4j_url=neo4j_url,
        neo4j_user=neo4j_user,
        neo4j_password=neo4j_password,
        ner_model=ner_model,
        re_model=re_model,
        llm_endpoint_url=llm_endpoint_url,
        llm_api_key=llm_api_key,
        template_re_loc=template_re_loc,
        template_ner_loc=template_ner_loc,
        embedding_model=embedding_model,
        chunking_method=chunking_method,
        chunk_size=chunk_size,
        overlap_size=overlap_size,
    )

    # Tune `concurrency` to the capacity of your Ollama/model server.
    # Start low (2-4) and raise it while watching GPU/CPU utilization.
    await ingestion_pipeline.ingest(corpus_list, concurrency=concurrency)

    return ingestion_pipeline


__all__ = ["AdvancedKGIngestor", "run_ingestion"]