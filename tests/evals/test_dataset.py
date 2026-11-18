
import json
import pytest
from pydantic import ValidationError

from llm_engineering.evals.dataset import EvalCase, load_tests


# ---------- EvalCase model tests ----------

def test_eval_case_valid_data():
    case = EvalCase(
        question="What is insurance?",
        keywords=["insurance", "policy"],
        reference_answer="Insurance provides financial protection.",
        category="direct_fact",
    )

    assert case.question == "What is insurance?"
    assert case.keywords == ["insurance", "policy"]
    assert case.reference_answer == "Insurance provides financial protection."
    assert case.category == "direct_fact"


def test_eval_case_missing_required_field():
    with pytest.raises(ValidationError):
        EvalCase(
            question="What is insurance?",
            keywords=["insurance"],
            reference_answer="Insurance provides protection.",
            # category is missing
        )


def test_eval_case_wrong_field_type():
    with pytest.raises(ValidationError):
        EvalCase(
            question="What is insurance?",
            keywords="insurance",  # Should be list[str]
            reference_answer="Insurance provides protection.",
            category="direct_fact",
        )


def test_eval_case_empty_keywords():
    case = EvalCase(
        question="What is insurance?",
        keywords=[],
        reference_answer="Insurance provides protection.",
        category="direct_fact",
    )

    assert case.keywords == []


# ---------- load_tests function tests ----------

def test_load_tests_single_record(tmp_path):
    file_path = tmp_path / "tests.jsonl"

    record = {
        "question": "What is insurance?",
        "keywords": ["insurance", "policy"],
        "reference_answer": "Insurance provides financial protection.",
        "category": "direct_fact",
    }

    file_path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    result = load_tests(file_path)

    assert len(result) == 1
    assert isinstance(result[0], EvalCase)
    assert result[0].question == "What is insurance?"
    assert result[0].keywords == ["insurance", "policy"]


def test_load_tests_multiple_records(tmp_path):
    file_path = tmp_path / "tests.jsonl"

    records = [
        {
            "question": "What is insurance?",
            "keywords": ["insurance"],
            "reference_answer": "Financial protection.",
            "category": "direct_fact",
        },
        {
            "question": "How does a policy work?",
            "keywords": ["policy"],
            "reference_answer": "A policy defines coverage.",
            "category": "spanning",
        },
    ]

    with file_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")

    result = load_tests(file_path)

    assert len(result) == 2
    assert result[0].question == "What is insurance?"
    assert result[1].question == "How does a policy work?"


def test_load_tests_skips_blank_lines(tmp_path):
    file_path = tmp_path / "tests.jsonl"

    record = {
        "question": "What is insurance?",
        "keywords": ["insurance"],
        "reference_answer": "Financial protection.",
        "category": "direct_fact",
    }

    file_path.write_text(
        "\n" + json.dumps(record) + "\n\n",
        encoding="utf-8",
    )

    result = load_tests(file_path)

    assert len(result) == 1


def test_load_tests_empty_file(tmp_path):
    file_path = tmp_path / "tests.jsonl"
    file_path.write_text("", encoding="utf-8")

    result = load_tests(file_path)

    assert result == []


def test_load_tests_invalid_json(tmp_path):
    file_path = tmp_path / "tests.jsonl"
    file_path.write_text("{invalid json}\n", encoding="utf-8")

    with pytest.raises(json.JSONDecodeError):
        load_tests(file_path)


def test_load_tests_invalid_eval_case(tmp_path):
    file_path = tmp_path / "tests.jsonl"

    # Missing required category field
    record = {
        "question": "What is insurance?",
        "keywords": ["insurance"],
        "reference_answer": "Financial protection.",
    }

    file_path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    with pytest.raises(ValidationError):
        load_tests(file_path)


def test_load_tests_file_not_found(tmp_path):
    file_path = tmp_path / "does_not_exist.jsonl"

    with pytest.raises(FileNotFoundError):
        load_tests(file_path)
