import asyncio
import time
from pathlib import Path

from ingestion.kg_ingestion import AdvancedKGIngestor
from dataset_adapters.musique_adapter import MusiqueQAAdapter
from dataset_adapters.twowikimh_adapter import TwoWikiMultihopAdapter

from retrieval_generation import VectorRetrievalPipeline, retrieve_generate

LLM_ENDPOINT = "http://localhost:11434/v1"

async def amain():
    
    adapter = TwoWikiMultihopAdapter(
        embedding_model="bge-large:latest",
        embedding_endpoint_url=LLM_ENDPOINT,
        embedding_api_key="not-needed",
        dataset_path="./datasets/2wikimultihop_dev.json",
    )
    # Start small; raise `limit` once you confirm the pipeline works.
    corpus_list, qa_pairs = await adapter.aload_corpus(
        limit=5, #load_golden_context=True
    )

    ingestion_pipeline = AdvancedKGIngestor(
        neo4j_url="bolt://3.239.36.253",
        neo4j_user="neo4j",
        neo4j_password="accruals-chamber-standardization",
        ner_model="phi4-mini:latest",
        re_model="phi4-mini:latest",
        llm_endpoint_url=LLM_ENDPOINT,
        llm_api_key="not-needed",
        template_re_loc="./prompts/re.txt",
        template_ner_loc="./prompts/ner.txt",
        embedding_model="bge-large:latest"
    )

    # Tune `concurrency` to the capacity of your Ollama/model server.
    # Start low (2-4) and raise it while watching GPU/CPU utilization.
    await ingestion_pipeline.ingest(corpus_list, concurrency=3)

    questions = [item["question"] for item in qa_pairs]
    pipeline = VectorRetrievalPipeline(
        neo4j_url="bolt://3.239.36.253", neo4j_user="neo4j",
        neo4j_password="accruals-chamber-standardization",
        llm_endpoint_url=LLM_ENDPOINT, llm_api_key="not-needed",
        llm_model="phi4-mini:latest", embedding_model="bge-large:latest",
        answering_prompt="./prompts/answering.txt",
        reasoning_enabled=True,
        reasoning_prompt="./prompts/reasoning_cot.txt",
        reasoning_steps=2,
        retrieve_k=5
    )
    answers = await retrieve_generate(
        pipeline, questions,
        max_concurrency=3,     # <-- the knob to tune
        show_progress=False,
        return_exceptions=True,
    )

    for q, a in zip(questions, answers):
        print(q, "->", a if not isinstance(a, Exception) else f"<error: {a!r}>")


if __name__ == "__main__":
    start_time = time.perf_counter()
    asyncio.run(amain())
    duration = time.perf_counter() - start_time
    print(f"Script finished in {duration:.4f} seconds")