from evaluation.test_case import TestCase


class ExactMatchMetric:
    """Exact-match evaluation metric."""

    def __init__(self):
        """Initialize the metric with no score or reason."""
        self.score = None
        self.reason = None

    def measure(self, test_case: TestCase) -> float:
        """Compute exact-match score between actual and expected output.

        Returns:
            1.0 if outputs match exactly, else 0.0.
        """
        actual = test_case.actual_output.strip().lower() if test_case.actual_output else ""
        expected = test_case.expected_output.strip().lower() if test_case.expected_output else ""
        self.score = 1.0 if actual == expected else 0.0
        self.reason = "Exact match" if self.score == 1.0 else "Not an exact match"
        return self.score