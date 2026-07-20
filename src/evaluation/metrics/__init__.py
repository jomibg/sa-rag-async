from evaluation.metrics.exact_match import ExactMatchMetric
from evaluation.metrics.f1 import F1ScoreMetric

METRIC_REGISTRY = {
    "EM": ExactMatchMetric,
    "f1": F1ScoreMetric,
}