import asyncio
import time
from typing import List, Any

import optuna

from deepeval.models import OllamaModel
from deepeval.test_case import SingleTurnParams
from deepeval.metrics import GEval

from ingestion import run_ingestion
from dataset_adapters import load_corpus
from retrieval_generation import SaPipelineCot, retrieve_generate
from evaluation import execute_evaluation

# =====================================================================
# LOCAL CONFIGURATION (no configs.py / env vars / Docker assumptions)
# =====================================================================
BENCHMARK = "MuSiQuE"            # 'TwoWikiMultiHop' or 'MuSiQuE'
NUMBER_OF_QUESTIONS = 4
BATCH_SIZE = 2                  # 4 batches of 50 questions
N_TRIALS = 5
TOP_K_RESULTS = 2
CONCURRENCY = 3
SEED = 53

# LLM / embeddings (local Ollama, OpenAI-compatible endpoint)
LLM_ENDPOINT = "http://172.17.0.1:11434/v1"
LLM_API_KEY = "not-needed"
LLM_MODEL = "phi4-mini:latest"
EMBEDDING_MODEL = "bge-large:latest"

# Neo4j (reachable without Docker)
NEO4J_URL = ""
NEO4J_USER = ""
NEO4J_PW = ""

# Prompt / dataset paths (relative to src/)
ANSWERING_PROMPT = "../prompts/answering.txt"
REASONING_PROMPT_COT = "../prompts/reasoning_cot.txt"
TEMPLATE_RE_LOC = "../prompts/re.txt"
TEMPLATE_NER_LOC = "../prompts/ner.txt"

# Fixed SA pipeline params (not tuned here)
ACTIVATION_THRESHOLD = 0.5
K_HOP = 3

# =====================================================================
# Helpers
# =====================================================================
def _dataset_path_for(benchmark: str) -> str:
    if benchmark == "MuSiQuE":
        return "../datasets/musique_ans_v1.0_dev.jsonl"
    if benchmark == "TwoWikiMultiHop":
        return "../datasets/2wikimultihop_dev.json"
    raise ValueError(f"Unknown benchmark: {benchmark}")


def _build_eval_model() -> OllamaModel:
    return OllamaModel(
        model=LLM_MODEL,
        base_url=LLM_ENDPOINT.rstrip("/").removesuffix("/v1"),
        temperature=0.0,
    )


def _build_correctness_metric(eval_model: OllamaModel) -> GEval:
    return GEval(
        name="correctness",
        criteria="Determine whether the actual output is factually correct based on the expected output.",
        evaluation_steps=[
            "Check whether the facts in 'actual output' contradicts any facts in 'expected output'",
            "Do not concentrate on the style, grammar, or formatting of the answer. Answers are considered correct as long as they convey the same factual information as the expected output.",
            "If the 'actual output' does not contradict the 'expected output' but does not convey information that is present in the 'expected output', it is considered completely incorrect. (0\\% correct)",
        ],
        evaluation_params=[
            SingleTurnParams.INPUT,
            SingleTurnParams.ACTUAL_OUTPUT,
            SingleTurnParams.EXPECTED_OUTPUT,
        ],
        model=eval_model,
    )


def _batched(items: List[Any], size: int) -> List[List[Any]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def _safe_answer(a):
    if isinstance(a, Exception):
        return {"answer": "", "error": repr(a)}
    return a


def _build_pipeline(
    normalization_parameter: float,
    pruning_threshold: float,
    shared_k: int,
) -> SaPipelineCot:
    return SaPipelineCot(
        name="sa_cot_tuning",
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
        reasoning_steps=0,
        retrieve_k=shared_k,
        activating_descriptions=shared_k,
        normalization_parameter=normalization_parameter,
        activation_threshold=ACTIVATION_THRESHOLD,
        pruning_threshold=pruning_threshold,
        k_hop=K_HOP,
    )


# =====================================================================
# Performance function: 4-fold cross-validation over the 200 questions
# =====================================================================
async def performance_function(
    pipeline: SaPipelineCot,
    questions: List[str],
    golden_answers: List[str],
    eval_model: OllamaModel,
) -> float:
    """Run 4 batches of 50 questions, average correctness per batch, then
    average the batch scores (cross-validation)."""
    batches = _batched(list(zip(questions, golden_answers)), BATCH_SIZE)

    batch_scores: List[float] = []
    for batch_idx, batch in enumerate(batches):
        batch_questions = [q for (q, _) in batch]
        batch_golden = [g for (_, g) in batch]

        answers = await retrieve_generate(
            pipeline=pipeline,
            questions=batch_questions,
            max_concurrency=CONCURRENCY,
            show_progress=False,
            return_exceptions=True,
        )
        answers = [_safe_answer(a) for a in answers]
        produced = [a["answer"] for a in answers]

        results = execute_evaluation(
            batch_questions,
            produced,
            batch_golden,
            evaluator_metrics=["correctness"],
            eval_model=eval_model,
        )

        scores = [
            r["metrics"]["correctness"]["score"]
            for r in results
            if r.get("metrics", {}).get("correctness", {}).get("score") is not None
        ]
        batch_avg = sum(scores) / len(scores) if scores else 0.0
        batch_scores.append(batch_avg)
        print(f"  batch {batch_idx + 1}/{len(batches)} correctness avg: {batch_avg:.4f}")

    overall = sum(batch_scores) / len(batch_scores) if batch_scores else 0.0
    return overall


# =====================================================================
# Async setup: sample data + ingest knowledge graph
# =====================================================================
async def setup():
    print(f"Loading {NUMBER_OF_QUESTIONS} samples from '{BENCHMARK}'...")
    corpus_list, qa_pairs = await load_corpus(
        adapter_name=BENCHMARK,
        embedding_model=EMBEDDING_MODEL,
        embedding_endpoint_url=LLM_ENDPOINT,
        embedding_api_key=LLM_API_KEY,
        dataset_path=_dataset_path_for(BENCHMARK),
        limit=NUMBER_OF_QUESTIONS,
    )
    questions = [item["question"] for item in qa_pairs]
    golden_answers = [item["answer"] for item in qa_pairs]
    print(f"Loaded {len(questions)} questions, {len(corpus_list)} corpus docs.")

    print("Ingesting knowledge graph...")
    await run_ingestion(
        corpus_list,
        neo4j_url=NEO4J_URL,
        neo4j_user=NEO4J_USER,
        neo4j_password=NEO4J_PW,
        ner_model=LLM_MODEL,
        re_model=LLM_MODEL,
        llm_endpoint_url=LLM_ENDPOINT,
        llm_api_key=LLM_API_KEY,
        template_re_loc=TEMPLATE_RE_LOC,
        template_ner_loc=TEMPLATE_NER_LOC,
        embedding_model=EMBEDDING_MODEL,
        concurrency=CONCURRENCY,
    )
    print("Ingestion complete.")
    return questions, golden_answers


# =====================================================================
# Per-trial async run (fresh pipeline + driver within its own event loop)
# =====================================================================
async def trial_run(
    normalization_parameter: float,
    pruning_threshold: float,
    shared_k: int,
    questions: List[str],
    golden_answers: List[str],
    eval_model: OllamaModel,
) -> float:
    pipeline = _build_pipeline(normalization_parameter, pruning_threshold, shared_k)
    try:
        return await performance_function(pipeline, questions, golden_answers, eval_model)
    finally:
        await pipeline.driver.close()


# =====================================================================
# Optuna objective (sync; launches its own event loop per trial)
# =====================================================================
def make_objective(questions, golden_answers, eval_model):
    def objective(trial: optuna.Trial) -> float:
        normalization_parameter = trial.suggest_float(
            "normalization_parameter", 0.3, 0.5, step=0.05
        )
        pruning_threshold = trial.suggest_float(
            "pruning_threshold", 0.3, 0.5, step=0.05
        )
        # retrieve_k and activating_descriptions share the same value
        shared_k = trial.suggest_int(
            "retrieve_k_activating_descriptions", 2, 4, step=1
        )

        print(
            f"[trial {trial.number}] "
            f"normalization={normalization_parameter:.2f} "
            f"pruning={pruning_threshold:.2f} "
            f"retrieve_k=activating_descriptions={shared_k}"
        )
        score = asyncio.run(
            trial_run(
                normalization_parameter,
                pruning_threshold,
                shared_k,
                questions,
                golden_answers,
                eval_model,
            )
        )
        print(f"[trial {trial.number}] mean correctness: {score:.4f}")
        return score

    return objective


# =====================================================================
# Main
# =====================================================================
def main():
    start_time = time.perf_counter()

    # 1-2) Sample 200 questions + ingest knowledge graph
    questions, golden_answers = asyncio.run(setup())

    # 3) Evaluation model (reused across trials)
    eval_model = _build_eval_model()

    # 4-6) Optuna hyperparameter tuning (20 trials, maximize correctness)
    objective = make_objective(questions, golden_answers, eval_model)

    study = optuna.create_study(direction="maximize", study_name="sa_cot_tuning")
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

    # 7) Display results + top-5 params
    print("\n==================== STUDY SUMMARY ====================")
    print(f"Number of trials: {len(study.trials)}")
    print(f"Best score: {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")

    completed = [
        t for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
    ]
    ranked = sorted(
        completed,
        key=lambda t: t.value if t.value is not None else 0.0,
        reverse=True,
    )
    top = ranked[:TOP_K_RESULTS]

    print(f"\n==================== TOP {TOP_K_RESULTS} TRIALS ====================")
    for rank, t in enumerate(top, start=1):
        print(
            f"{rank}. trial {t.number} | score={t.value:.4f} | "
            f"normalization={t.params['normalization_parameter']:.2f} "
            f"pruning={t.params['pruning_threshold']:.2f} "
            f"retrieve_k=activating_descriptions="
            f"{t.params['retrieve_k_activating_descriptions']}"
        )

    duration = time.perf_counter() - start_time
    print(f"\nScript finished in {duration:.4f} seconds")

    # Save tuning results to a text file in the same format as printed above
    import os

    summary_lines = [
        "==================== STUDY SUMMARY ====================",
        f"Number of trials: {len(study.trials)}",
        f"Best score: {study.best_value:.4f}",
        f"Best params: {study.best_params}",
        "",
        f"==================== TOP {TOP_K_RESULTS} TRIALS ====================",
    ]
    for rank, t in enumerate(top, start=1):
        summary_lines.append(
            f"{rank}. trial {t.number} | score={t.value:.4f} | "
            f"normalization={t.params['normalization_parameter']:.2f} "
            f"pruning={t.params['pruning_threshold']:.2f} "
            f"retrieve_k=activating_descriptions="
            f"{t.params['retrieve_k_activating_descriptions']}"
        )

    out_path = "../results/tunning/params.txt"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))


if __name__ == "__main__":
    main()
