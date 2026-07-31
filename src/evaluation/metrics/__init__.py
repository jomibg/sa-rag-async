from evaluation.metrics.exact_match import ExactMatchMetric
from evaluation.metrics.f1 import F1ScoreMetric
from deepeval.metrics import GEval


def _make_correctness(**kwargs):
    return GEval(name="correctness", **kwargs)

METRIC_REGISTRY = {
    "EM": ExactMatchMetric,
    "f1": F1ScoreMetric,
    "correctness": _make_correctness,
}