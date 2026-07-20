from typing import Any, Dict, List, Sequence

from .metrics import METRIC_REGISTRY
from .test_case import TestCase


async def execute_evaluation(
    questions: Sequence[str],
    answers: Sequence[str],
    golden_answers: Sequence[str],
    evaluator_metrics: Sequence[str],
) -> List[Dict[str, Any]]:
    """Evaluate a batch of produced answers against golden answers.

    All three input sequences must have the same length and are aligned
    positionally: `answers[i]` is the system's output for `questions[i]`,
    judged against `golden_answers[i]`.

    `evaluator_metrics` is a list of registry keys, e.g. ["EM", "f1"].
    Returns one dict per item, in input order:
        {"question": ..., "answer": ..., "golden_answer": ...,
         "metrics": {metric_name: {"score": float, "reason": str}}}
    """
    n = len(questions)
    if not (len(answers) == len(golden_answers) == n):
        raise ValueError(
            "questions, answers, and golden_answers must have the same length "
            f"(got {n}, {len(answers)}, {len(golden_answers)})"
        )

    for metric in evaluator_metrics:
        if metric not in METRIC_REGISTRY:
            raise ValueError(f"Unsupported metric: {metric}")

    # One reusable instance per metric — matches the original DeepEvalAdapter,
    # which created metrics once in __init__ and reused them across all answers.
    metric_instances = {name: METRIC_REGISTRY[name]() for name in evaluator_metrics}

    results: List[Dict[str, Any]] = []
    for q, a, gold in zip(questions, answers, golden_answers):
        test_case = TestCase(input=q, actual_output=a, expected_output=gold)

        metric_results: Dict[str, Dict[str, Any]] = {}
        for name in evaluator_metrics:
            metric = metric_instances[name]
            metric.measure(test_case)
            metric_results[name] = {
                "score": metric.score,
                "reason": metric.reason,
            }

        results.append({
            "question": q,
            "answer": a,
            "golden_answer": gold,
            "metrics": metric_results,
        })

    return results