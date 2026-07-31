from collections import Counter
import re

from deepeval.test_case import LLMTestCase


class F1ScoreMetric:
    """Token-level F1 evaluation metric."""

    def __init__(self):
        """Initialize the metric with no score or reason."""
        self.score = None
        self.reason = None

    def measure(self, test_case: LLMTestCase) -> float:
        """Compute token-level F1 score between actual and expected output.

        Returns:
            The F1 score in [0.0, 1.0].
        """
        actual = (test_case.actual_output or "").lower()
        expected = (test_case.expected_output or "").lower()

        actual_tokens = [re.sub(r"\W+", "", token.strip()) for token in actual.split()]
        expected_tokens = [re.sub(r"\W+", "", token.strip()) for token in expected.split()]

        if not actual_tokens and not expected_tokens:
            self.score = 1.0
            self.reason = "Both actual and expected are empty"
            return self.score

        actual_counts = Counter(actual_tokens)
        expected_counts = Counter(expected_tokens)

        tp = sum(min(actual_counts[word], expected_counts[word]) for word in actual_counts)
        fp = sum(actual_counts[word] for word in actual_counts) - tp
        fn = sum(expected_counts[word] for word in expected_counts) - tp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        self.score = f1
        self.reason = f"F1: {f1:.2f} (Precision: {precision:.2f}, Recall: {recall:.2f})"
        return self.score