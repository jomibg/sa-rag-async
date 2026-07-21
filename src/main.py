import asyncio
import time
from pathlib import Path
from utils import load_json_file, save_json_file

from ingestion import run_ingestion
from dataset_adapters import load_corpus
from retrieval_generation import (
    VectorRetrievalPipeline,
    retrieve_generate,
    DecompositionPipeline,
    SaPipelineCot,
    SaPipelineDecomp
    )
from evaluation import execute_evaluation, generate_metrics_dashboard
from configs import RunConfigs

#TODO: coordination and configs
#TODO: Docker   

LLM_ENDPOINT = "http://localhost:11434/v1"
LLM_API_KEY = "not-needed"
LLM_MODEL = "phi4-mini:latest"
EMBEDDING_MODEL = "bge-large:latest"
NEO4J_URL = "bolt://3.86.183.59"
NEO4J_USER = "neo4j"
NEO4J_PW = "nod-bail-bins"
ANSWERING_PROMPT="./prompts/answering.txt"
REASONING_PROMPT="./prompts/reasoning_cot.txt"

async def amain():
    cfg = RunConfigs()

    # load question and knowledge corpus
    if cfg.sample_data:
        corpus_list, qa_pairs = await load_corpus(
            adapter_name=cfg.benchmark,
            embedding_model=cfg.embedding_model,
            embedding_endpoint_url=cfg.llm_endpoint,
            embedding_api_key=cfg.llm_api_key,
            dataset_path=cfg.dataset_path,
            limit=cfg.number_of_questions,
        )
        save_json_file("./results/corpus/2wiki_corpus_test.json", corpus_list)
        save_json_file("./results/questions/2wiki_qa_test.json", qa_pairs)

    '''
    await run_ingestion(
        corpus_list,
        neo4j_url=NEO4J_URL,
        neo4j_user=NEO4J_USER,
        neo4j_password=NEO4J_PW,
        ner_model=LLM_MODEL,
        re_model=LLM_MODEL,
        llm_endpoint_url=LLM_ENDPOINT,
        llm_api_key=LLM_API_KEY,
        template_re_loc=RE_PROMPT,
        template_ner_loc=NER_PROMPT,
        embedding_model=EMBEDDING_MODEL,
        concurrency=3,
    )

    save_json_file("./results/corpus/2wiki_corpus_test.json", corpus_list)
    save_json_file("./results/questions/2wiki_qa_test.json", qa_pairs)
    
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

    '''
    sa_decomp_pipeline = SaPipelineDecomp(
        neo4j_url=NEO4J_URL,
        neo4j_user=NEO4J_USER,
        neo4j_password=NEO4J_PW,
        llm_endpoint_url=LLM_ENDPOINT,
        llm_api_key=LLM_API_KEY,
        llm_model=LLM_MODEL,
        embedding_model=EMBEDDING_MODEL,
        answering_prompt=ANSWERING_PROMPT,
        reasoning_prompt=REASONING_PROMPT,
        retrieve_k=5,
        activating_descriptions=3,
        normalization_parameter=0.4,
        activation_threshold=0.5,
        pruning_threshold=0.45,
        k_hop = 3
        )

    questions = [item["question"] for item in qa_pairs]
    golden_answers = [item["answer"] for item in qa_pairs]

    answers = await retrieve_generate(
        pipeline=sa_decomp_pipeline, 
        questions=questions,
        max_concurrency=3, 
        show_progress=False,
        return_exceptions=True,
    )
    save_json_file("./results/answers/2wiki_answers_test.json", answers)

    produced_answers = [a["answer"] if not isinstance(a, Exception) else "" for a in answers]

    results = await execute_evaluation(questions, produced_answers,
        golden_answers, evaluator_metrics=["EM", "f1"],)
    save_json_file("./results/evaluation/metrics/test.json", results)
    generate_metrics_dashboard("./results/evaluation/metrics/test.json",
        "./results/evaluation/dashboards/test.html", "2wikimh")

if __name__ == "__main__":
    start_time = time.perf_counter()
    asyncio.run(amain())
    duration = time.perf_counter() - start_time
    print(f"Script finished in {duration:.4f} seconds")