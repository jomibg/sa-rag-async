from dataclasses import dataclass


@dataclass
class TestCase:
    """Minimal test case mirroring DeepEval's LLMTestCase API.

    Attributes:
        input: The original question/prompt.
        actual_output: The system's produced answer.
        expected_output: The golden reference answer.
    """
    input: str
    actual_output: str
    expected_output: str