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
from configs import RunConfigs, RAG_PIPELINES

#TODO: coordination and configs
#TODO: Docker   

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
        save_json_file(cfg.sample_corpus_path, corpus_list)
        save_json_file(cfg.sample_qa_path, qa_pairs)

    if cfg.ingest_corpus:
        corpus_list = load_json_file(cfg.sample_corpus_path)
        qa_pairs = load_json_file(cfg.sample_qa_path)
        await run_ingestion(
            corpus_list,
            neo4j_url=cfg.neo4j_url,
            neo4j_user=cfg.neo4j_user,
            neo4j_password=cfg.neo4j_pw,
            ner_model=cfg.llm_model,
            re_model=cfg.llm_model,
            llm_endpoint_url=cfg.llm_endpoint,
            llm_api_key=cfg.llm_api_key,
            template_re_loc=cfg.template_re_loc,
            template_ner_loc=cfg.template_ner_loc,
            embedding_model=cfg.embedding_model,
            concurrency=cfg.concurrency,
        )

    if cfg.answering_questions:
        for pipeline in RAG_PIPELINES:
            qa_pairs = load_json_file(cfg.sample_qa_path)
            questions = [item["question"] for item in qa_pairs]
            golden_answers = [item["answer"] for item in qa_pairs]

            answers = await retrieve_generate(
                pipeline=pipeline, 
                questions=questions,
                max_concurrency=cfg.concurrency, 
                show_progress=cfg.show_progress_answering,
                return_exceptions=True,
            )
            save_json_file(f"./results/answers/{pipeline.name}_{cfg.benchmark}_{cfg.number_of_questions}.json", answers)

    if cfg.evaluating_answers:
        for pipeline in RAG_PIPELINES:
            qa_pairs = load_json_file(cfg.sample_qa_path)
            questions = [item["question"] for item in qa_pairs]
            golden_answers = [item["answer"] for item in qa_pairs]
            answers = load_json_file(f"./results/answers/{pipeline.name}_{cfg.benchmark}_{cfg.number_of_questions}.json")
            produced_answers = [a["answer"] if not isinstance(a, Exception) else "" for a in answers]
            results = await execute_evaluation(questions, produced_answers,
                golden_answers, evaluator_metrics=["EM", "f1"],)
            save_json_file(f"./results/evaluation/metrics/{pipeline.name}_{cfg.benchmark}_{cfg.number_of_questions}.json", results)
            generate_metrics_dashboard(f"./results/evaluation/metrics/{pipeline.name}_{cfg.benchmark}_{cfg.number_of_questions}.json",
                f"./results/evaluation/dashboards/{pipeline.name}_{cfg.benchmark}_{cfg.number_of_questions}.html", "2wikimh")

if __name__ == "__main__":
    start_time = time.perf_counter()
    asyncio.run(amain())
    duration = time.perf_counter() - start_time
    print(f"Script finished in {duration:.4f} seconds")