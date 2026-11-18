import json
from pydantic import BaseModel, Field

from llm_engineering.config import TEST_FILE


class EvalCase(BaseModel):
    """A test question with expected keywords and reference answer."""

    question: str = Field(description="The question to ask the RAG system")
    keywords: list[str] = Field(description="Keywords that must appear in retrieved context")
    reference_answer: str = Field(description="The reference answer for this question")
    category: str = Field(description="Question category (e.g., direct_fact, spanning, temporal)")


def load_tests(path=TEST_FILE) -> list[EvalCase]:
    """Load test questions from a JSONL file."""
    tests = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:                      # skip blank lines
                tests.append(EvalCase(**json.loads(line)))
    return tests
