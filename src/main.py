import asyncio
import time
from pathlib import Path

from ingestion.kg_ingestion import AdvancedKGIngestor
from dataset_adapters.musique_adapter import MusiqueQAAdapter
from dataset_adapters.twowikimh_adapter import TwoWikiMultihopAdapter
from retrieval_generation import (
    VectorRetrievalPipeline,
    retrieve_generate,
    DecompositionPipeline,
    SaPipelineCot
    )
from evaluation.evaluation import execute_evaluation

LLM_ENDPOINT = "http://localhost:11434/v1"
LLM_API_KEY = "not-needed" 
NEO4J_URL = "bolt://3.86.183.59"
NEO4J_USER = "neo4j"
NEO4J_PW = "nod-bail-bins"
LLM_MODEL = "phi4-mini:latest"
EMBEDDING_MODEL = "bge-large:latest"

NER_PROMPT = "./prompts/ner.txt"
RE_PROMPT = "./prompts/re.txt"
REASONING_PROMPT = "./prompts/reasoning_decomposition.txt"
ANSWERING_PROMPT = "./prompts/answering.txt"


#TODO: implement other hybrid RAG pipeine
#TODO: add knowledge and reasoning in pipeline responses
#TODO: review evaluation pipeline
#TODO: add result records 
#TODO: coordination and configs
#TODO: Docker
#TODO: __init__ imports

async def amain():
   
    adapter = TwoWikiMultihopAdapter(
        embedding_model="bge-large:latest",
        embedding_endpoint_url=LLM_ENDPOINT,
        embedding_api_key="not-needed",
        dataset_path="./datasets/2wikimultihop_dev.json",
    )
    # Start small; raise `limit` once you confirm the pipeline works.
    corpus_list, qa_pairs = await adapter.aload_corpus(
        limit=2, #load_golden_context=True
    )
    '''
    ingestion_pipeline = AdvancedKGIngestor(
        neo4j_url=NEO4J_URL,
        neo4j_user=NEO4J_USER,
        neo4j_password=NEO4J_PW,
        ner_model=LLM_MODEL,
        re_model=LLM_MODEL,
        llm_endpoint_url=LLM_ENDPOINT,
        llm_api_key=LLM_API_KEY,
        template_re_loc=RE_PROMPT,
        template_ner_loc=NER_PROMPT,
        embedding_model=EMBEDDING_MODEL
    )

    # Tune `concurrency` to the capacity of your Ollama/model server.
    # Start low (2-4) and raise it while watching GPU/CPU utilization.
    await ingestion_pipeline.ingest(corpus_list, concurrency=3)
    vector_pipeline = VectorRetrievalPipeline(
        neo4j_url=NEO4J_URL,
        neo4j_user=NEO4J_USER,
        neo4j_password=NEO4J_PW,
        llm_endpoint_url=LLM_ENDPOINT,
        llm_api_key=LLM_API_KEY,
        llm_model=LLM_MODEL,
        embedding_model=EMBEDDING_MODEL,
        answering_prompt=ANSWERING_PROMPT,
        reasoning_enabled=True,
        reasoning_prompt=REASONING_PROMPT,
        reasoning_steps=2,
        retrieve_k=5
    )
    
    sa_pipeline_cot = SaPipelineCot(
        neo4j_url=NEO4J_URL,
        neo4j_user=NEO4J_USER,
        neo4j_password=NEO4J_PW,
        llm_endpoint_url=LLM_ENDPOINT,
        llm_api_key=LLM_API_KEY,
        llm_model=LLM_MODEL,
        embedding_model=EMBEDDING_MODEL,
        answering_prompt=ANSWERING_PROMPT,
        reasoning_enabled=True,
        reasoning_prompt=REASONING_PROMPT,
        reasoning_steps=2,
        retrieve_k=5,
        activating_descriptions=3,
        normalization_parameter=0.4,
        activation_threshold=0.5,
        pruning_threshold=0.45,
        k_hop = 3

        )
    '''
    decomp_pipeline = DecompositionPipeline(
        neo4j_url=NEO4J_URL,
        neo4j_user=NEO4J_USER,
        neo4j_password=NEO4J_PW,
        llm_endpoint_url=LLM_ENDPOINT,
        llm_api_key=LLM_API_KEY,
        llm_model=LLM_MODEL,
        embedding_model=EMBEDDING_MODEL,
        answering_prompt=ANSWERING_PROMPT,
        reasoning_prompt=REASONING_PROMPT,
        retrieve_k=5
        )
    questions = [item["question"] for item in qa_pairs]
    golden_answers = [item["answer"] for item in qa_pairs]

    answers = await retrieve_generate(
        pipeline=decomp_pipeline, 
        questions=questions,
        max_concurrency=3, 
        show_progress=False,
        return_exceptions=True,
    )

    produced_answers = [a["answer"] if not isinstance(a, Exception) else "" for a in answers]
    golden_answers = [qa["answer"] for qa in qa_pairs]
    results = await execute_evaluation(questions, produced_answers,
        golden_answers, evaluator_metrics=["EM", "f1"],)
    for r in results:
        print(r["question"])
        print("  produced :", r["answer"])
        print("  golden   :", r["golden_answer"])
        print("  metrics  :", r["metrics"])


if __name__ == "__main__":
    start_time = time.perf_counter()
    asyncio.run(amain())
    duration = time.perf_counter() - start_time
    print(f"Script finished in {duration:.4f} seconds")