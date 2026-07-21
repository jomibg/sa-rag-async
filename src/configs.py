import os
from dataclasses import dataclass, field
from typing import List

from retrieval_generation import (
    VectorRetrievalPipeline,
    DecompositionPipeline,
    SaPipelineCot,
    SaPipelineDecomp
    )

LLM_ENDPOINT = "http://localhost:11434/v1"
LLM_API_KEY = "not-needed"
LLM_MODEL = "phi4-mini:latest"
EMBEDDING_MODEL = "bge-large:latest"
NEO4J_URL = "bolt://3.86.183.59"
NEO4J_USER = "neo4j"
NEO4J_PW = "nod-bail-bins"
ANSWERING_PROMPT="./prompts/answering.txt"
REASONING_PROMPT_COT="./prompts/reasoning_cot.txt"
REASONING_PROMPT_DECOMP="./prompts/reasoning_decomposition.txt"

@dataclass
class RunConfigs:
    # GENERAL
    sample_data: bool = True
    number_of_questions: int = 2
    benchmark: str = "MuSiQuE"  # 'TwoWikiMultiHop', 'MuSiQuE'
    ingest_corpus: bool = True          # was: ingest_cropus (typo)
    answering_questions: bool = True
    show_progress_answering: bool = True
    evaluating_answers: bool = True 
    evaluation_metrics: List[str] = field(default_factory=lambda: ["EM", "f1"])
    concurrency: int = 5

    # LLM/EMBEDDINGS
    llm_endpoint: str = LLM_ENDPOINT     # was: llm_enpoint (typo)
    llm_api_key: str = LLM_API_KEY
    llm_model: str = LLM_MODEL           # was: LLM_ENDPOINT (wrong)
    embedding_model: str = EMBEDDING_MODEL

    # NEO4J
    neo4j_url: str = NEO4J_URL
    neo4j_user: str = NEO4J_USER
    neo4j_pw: str = NEO4J_PW

    # PATHS
    dataset_path: str = None
    sample_corpus_path: str = None
    sample_qa_path: str = None
    template_re_loc: str = "./prompts/re.txt"
    template_ner_loc: str = "./prompts/ner.txt"

    def __post_init__(self):
        if self.dataset_path is None:
            dir_path = "./datasets"
            if self.benchmark == "MuSiQuE":
                self.dataset_path = os.path.join(dir_path, "musique_data.jsonl")
            elif self.benchmark == "TwoWikiMultiHop":
                self.dataset_path = os.path.join(dir_path, "2wikimultihop_dev.json")
            else:
                raise ValueError(f"Unknown benchmark: {self.benchmark}")
        if self.sample_corpus_path is None:
            self.sample_corpus_path = os.path.join("./results/corpus", f"{self.benchmark}_corpus_{self.number_of_questions}.json")
        if self.sample_qa_path is None:
            self.sample_qa_path = os.path.join("./results/questions", f"{self.benchmark}_qa_{self.number_of_questions}.json")
        

RAG_PIPELINES = [
    VectorRetrievalPipeline(
        name="vector_cot_5",
        neo4j_url=NEO4J_URL,
        neo4j_user=NEO4J_USER,
        neo4j_password=NEO4J_PW,
        llm_endpoint_url=LLM_ENDPOINT,
        llm_api_key=LLM_API_KEY,
        llm_model=LLM_MODEL,
        embedding_model=EMBEDDING_MODEL,
        answering_prompt=ANSWERING_PROMPT,
        reasoning_enabled=True,
        reasoning_prompt=REASONING_PROMPT_COT,
        reasoning_steps=2,
        retrieve_k=5
    ),
    VectorRetrievalPipeline(
        name="vector_cot_10",
        neo4j_url=NEO4J_URL,
        neo4j_user=NEO4J_USER,
        neo4j_password=NEO4J_PW,
        llm_endpoint_url=LLM_ENDPOINT,
        llm_api_key=LLM_API_KEY,
        llm_model=LLM_MODEL,
        embedding_model=EMBEDDING_MODEL,
        answering_prompt=ANSWERING_PROMPT,
        reasoning_enabled=True,
        reasoning_prompt=REASONING_PROMPT_COT,
        reasoning_steps=2,
        retrieve_k=10
    ),
    VectorRetrievalPipeline(
        name="vector_5",
        neo4j_url=NEO4J_URL,
        neo4j_user=NEO4J_USER,
        neo4j_password=NEO4J_PW,
        llm_endpoint_url=LLM_ENDPOINT,
        llm_api_key=LLM_API_KEY,
        llm_model=LLM_MODEL,
        embedding_model=EMBEDDING_MODEL,
        answering_prompt=ANSWERING_PROMPT,
        reasoning_enabled=False,
        reasoning_prompt=REASONING_PROMPT_COT,
        reasoning_steps=2,
        retrieve_k=5
    ),
    VectorRetrievalPipeline(
        name="vector_10",
        neo4j_url=NEO4J_URL,
        neo4j_user=NEO4J_USER,
        neo4j_password=NEO4J_PW,
        llm_endpoint_url=LLM_ENDPOINT,
        llm_api_key=LLM_API_KEY,
        llm_model=LLM_MODEL,
        embedding_model=EMBEDDING_MODEL,
        answering_prompt=ANSWERING_PROMPT,
        reasoning_enabled=False,
        reasoning_prompt=REASONING_PROMPT_COT,
        reasoning_steps=2,
        retrieve_k=10
    ),
    DecompositionPipeline(
        name="decomposition",
        neo4j_url=NEO4J_URL,
        neo4j_user=NEO4J_USER,
        neo4j_password=NEO4J_PW,
        llm_endpoint_url=LLM_ENDPOINT,
        llm_api_key=LLM_API_KEY,
        llm_model=LLM_MODEL,
        embedding_model=EMBEDDING_MODEL,
        answering_prompt=ANSWERING_PROMPT,
        reasoning_prompt=REASONING_PROMPT_DECOMP,
        retrieve_k=5
        ),
    
    SaPipelineDecomp(
        name="sa_decomposition",
        neo4j_url=NEO4J_URL,
        neo4j_user=NEO4J_USER,
        neo4j_password=NEO4J_PW,
        llm_endpoint_url=LLM_ENDPOINT,
        llm_api_key=LLM_API_KEY,
        llm_model=LLM_MODEL,
        embedding_model=EMBEDDING_MODEL,
        answering_prompt=ANSWERING_PROMPT,
        reasoning_prompt=REASONING_PROMPT_DECOMP,
        retrieve_k=5,
        activating_descriptions=3,
        normalization_parameter=0.4,
        activation_threshold=0.5,
        pruning_threshold=0.45,
        k_hop = 3
        ),
    SaPipelineCot(
        name="sa_cot",
        neo4j_url=NEO4J_URL,
        neo4j_user=NEO4J_USER,
        neo4j_password=NEO4J_PW,
        llm_endpoint_url=LLM_ENDPOINT,
        llm_api_key=LLM_API_KEY,
        llm_model=LLM_MODEL,
        embedding_model=EMBEDDING_MODEL,
        answering_prompt=ANSWERING_PROMPT,
        reasoning_enabled=True,
        reasoning_prompt=REASONING_PROMPT_COT,
        reasoning_steps=2,
        retrieve_k=5,
        activating_descriptions=3,
        normalization_parameter=0.4,
        activation_threshold=0.5,
        pruning_threshold=0.45,
        k_hop = 3
        )
]