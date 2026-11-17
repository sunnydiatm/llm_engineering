from pathlib import Path
from types import SimpleNamespace
import pytest
from llm_engineering.config import DB_NAME

pytestmark = pytest.mark.skipif(
    not Path(DB_NAME).exists(), reason="vector DB not built; run ingest first"
)

def test_answer_question_uses_retrieved_context(monkeypatch):
    from llm_engineering.rag import answer

    class FakeLLM:
        def invoke(self, messages):
            self.messages = messages
            return SimpleNamespace(content="fake answer")

    fake = FakeLLM()
    monkeypatch.setattr(answer, "llm", fake)

    text, docs = answer.answer_question("What products does Insurellm offer?")

    assert text == "fake answer"
    assert len(docs) > 0
    assert "Insurellm" in fake.messages[0].content   # context went into the system prompt