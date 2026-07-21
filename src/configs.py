import os
from dataclasses import dataclass
from typing import List

LLM_ENDPOINT = "http://localhost:11434/v1"
LLM_API_KEY = "not-needed" 
LLM_MODEL = "phi4-mini:latest"
EMBEDDING_MODEL = "bge-large:latest"
NEO4J_URL = "bolt://3.86.183.59"
NEO4J_USER = "neo4j"
NEO4J_PW = "nod-bail-bins"

@dataclass
class RunConfigs:
	# GENERAL
	sample_data: bool = False
	number_of_questions: int  = 2
	benchmark: str = 'MuSiQuE' # 'TwoWikiMultiHop', 'MuSiQuE'
	ingest_corpus: bool = False
	answering_questions: bool = False
	evaluating_answers: bool = False
	dashboard: bool = False
	evaluation_metrics: List[str] = None
	# LLM/EMBEDDINGS
	llm_endpoint: str = LLM_ENDPOINT
	llm_api_key: str = LLM_API_KEY
	llm_model: str  = LLM_MODEL
	embedding_model: str = EMBEDDING_MODEL
	# NEO4J
	neo4j_url: str = NEO4J_URL
	neo4j_user: str = NEO4J_USER
	neo4j_pw: str = NEO4J_PW
	# PATHS
	dataset_path:str = None

	def __post_init__(self):
    	if self.evaluation_metrics is None:
        	self.evaluation_metrics = ["EM", "f1"]
    	if self.dataset_path is None:
        	dir_path = "./dataset"
        	if self.benchmark == "MuSiQuE":
        	    self.dataset_path = os.path.join(dir_path, "musique_data.jsonl")
        	elif self.benchmark == "TwoWikiMultiHop":
        	    self.dataset_path = os.path.join(dir_path, "2wikimultihop_dev.json")
        	else:
        	    raise ValueError(f"Unknown benchmark: {self.benchmark}")


