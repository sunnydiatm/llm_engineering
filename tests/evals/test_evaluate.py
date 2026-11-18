from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from llm_engineering.evals import evaluate


# --------------------------------------------------
# Helpers
# --------------------------------------------------

def make_doc(content):
    return SimpleNamespace(page_content=content)


def make_test(
    question="What is an insurance policy?",
    keywords=None,
    category="Insurance",
    reference_answer="An insurance policy is a contract.",
):
    return SimpleNamespace(
        question=question,
        keywords=keywords if keywords is not None else ["insurance", "policy"],
        category=category,
        reference_answer=reference_answer,
    )


# --------------------------------------------------
# Tests for Pydantic models
# --------------------------------------------------

def test_retrieval_eval_model():
    result = evaluate.RetrievalEval(
        mrr=0.8,
        ndcg=0.9,
        keywords_found=4,
        total_keywords=5,
        keyword_coverage=80.0,
    )

    assert result.mrr == 0.8
    assert result.ndcg == 0.9
    assert result.keywords_found == 4
    assert result.total_keywords == 5
    assert result.keyword_coverage == 80.0


def test_answer_eval_model():
    result = evaluate.AnswerEval(
        feedback="Good answer",
        accuracy=4.5,
        completeness=4.0,
        relevance=4.8,
    )

    assert result.feedback == "Good answer"
    assert result.accuracy == 4.5
    assert result.completeness == 4.0
    assert result.relevance == 4.8


# --------------------------------------------------
# Tests for calculate_mrr()
# --------------------------------------------------

def test_calculate_mrr_first_document():
    docs = [
        make_doc("This document discusses insurance policies"),
        make_doc("Another document"),
    ]

    result = evaluate.calculate_mrr("insurance", docs)

    assert result == 1.0


def test_calculate_mrr_second_document():
    docs = [
        make_doc("Unrelated document"),
        make_doc("Information about insurance"),
    ]

    result = evaluate.calculate_mrr("insurance", docs)

    assert result == 0.5


def test_calculate_mrr_case_insensitive():
    docs = [
        make_doc("INSURANCE is important"),
    ]

    result = evaluate.calculate_mrr("insurance", docs)

    assert result == 1.0


def test_calculate_mrr_keyword_not_found():
    docs = [
        make_doc("First document"),
        make_doc("Second document"),
    ]

    result = evaluate.calculate_mrr("insurance", docs)

    assert result == 0.0


def test_calculate_mrr_empty_documents():
    result = evaluate.calculate_mrr("insurance", [])

    assert result == 0.0


# --------------------------------------------------
# Tests for calculate_dcg()
# --------------------------------------------------

def test_calculate_dcg():
    relevances = [1, 1, 0]

    result = evaluate.calculate_dcg(relevances, 3)

    expected = 1.0 + (1.0 / 1.5849625007)

    assert result == pytest.approx(expected)


def test_calculate_dcg_respects_k():
    relevances = [1, 1, 1]

    result = evaluate.calculate_dcg(relevances, 1)

    assert result == 1.0


def test_calculate_dcg_empty_list():
    assert evaluate.calculate_dcg([], 10) == 0.0


def test_calculate_dcg_zero_k():
    assert evaluate.calculate_dcg([1, 1], 0) == 0.0


# --------------------------------------------------
# Tests for calculate_ndcg()
# --------------------------------------------------

def test_calculate_ndcg_first_document():
    docs = [
        make_doc("Insurance policy details"),
        make_doc("Other information"),
    ]

    result = evaluate.calculate_ndcg("insurance", docs)

    assert result == 1.0


def test_calculate_ndcg_second_document():
    docs = [
        make_doc("Unrelated information"),
        make_doc("Insurance policy details"),
    ]

    result = evaluate.calculate_ndcg("insurance", docs)

    expected = 1.0 / 1.5849625007

    assert result == pytest.approx(expected)


def test_calculate_ndcg_keyword_not_found():
    docs = [
        make_doc("First document"),
        make_doc("Second document"),
    ]

    result = evaluate.calculate_ndcg("insurance", docs)

    assert result == 0.0


def test_calculate_ndcg_case_insensitive():
    docs = [
        make_doc("INSURANCE policy details"),
    ]

    result = evaluate.calculate_ndcg("insurance", docs)

    assert result == 1.0


def test_calculate_ndcg_respects_k():
    docs = [
        make_doc("Unrelated document"),
        make_doc("Insurance information"),
    ]

    result = evaluate.calculate_ndcg("insurance", docs, k=1)

    assert result == 0.0


def test_calculate_ndcg_multiple_relevant_documents():
    docs = [
        make_doc("Unrelated"),
        make_doc("Insurance information"),
        make_doc("Insurance policy"),
    ]

    result = evaluate.calculate_ndcg("insurance", docs)

    expected_dcg = (1 / 1.5849625007) + 0.5
    expected_idcg = 1.0 + (1 / 1.5849625007)

    assert result == pytest.approx(expected_dcg / expected_idcg)


# --------------------------------------------------
# Tests for evaluate_retrieval()
# --------------------------------------------------

def test_evaluate_retrieval(monkeypatch):
    test = make_test(
        keywords=["insurance", "policy", "claims"]
    )

    docs = [
        make_doc("Insurance information"),
        make_doc("Policy and claims information"),
        make_doc("Other information"),
    ]

    monkeypatch.setattr(
        evaluate,
        "fetch_context",
        lambda question: docs,
    )

    result = evaluate.evaluate_retrieval(test)

    # MRR:
    # insurance = 1.0
    # policy = 0.5
    # claims = 0.5
    assert result.mrr == pytest.approx(2 / 3)

    # nDCG:
    assert 0.0 <= result.ndcg <= 1.0

    assert result.keywords_found == 3
    assert result.total_keywords == 3
    assert result.keyword_coverage == 100.0


def test_evaluate_retrieval_partial_coverage(monkeypatch):
    test = make_test(
        keywords=["insurance", "policy", "claims", "premium"]
    )

    docs = [
        make_doc("Insurance information"),
        make_doc("Policy information"),
    ]

    monkeypatch.setattr(
        evaluate,
        "fetch_context",
        lambda question: docs,
    )

    result = evaluate.evaluate_retrieval(test)

    assert result.keywords_found == 2
    assert result.total_keywords == 4
    assert result.keyword_coverage == 50.0


def test_evaluate_retrieval_no_keywords(monkeypatch):
    test = make_test(keywords=[])

    monkeypatch.setattr(
        evaluate,
        "fetch_context",
        lambda question: [],
    )

    result = evaluate.evaluate_retrieval(test)

    assert result.mrr == 0.0
    assert result.ndcg == 0.0
    assert result.keywords_found == 0
    assert result.total_keywords == 0
    assert result.keyword_coverage == 0.0


def test_evaluate_retrieval_passes_question_to_fetch_context(
    monkeypatch,
):
    test = make_test(question="What is insurance?")

    fetch_mock = Mock(return_value=[])

    monkeypatch.setattr(
        evaluate,
        "fetch_context",
        fetch_mock,
    )

    evaluate.evaluate_retrieval(test)

    fetch_mock.assert_called_once_with("What is insurance?")


# --------------------------------------------------
# Tests for evaluate_answer()
# --------------------------------------------------

def test_evaluate_answer(monkeypatch):
    test = make_test()

    generated_answer = "An insurance policy is a legal contract."
    retrieved_docs = [
        make_doc("Insurance policy information")
    ]

    monkeypatch.setattr(
        evaluate,
        "answer_question",
        lambda question: (generated_answer, retrieved_docs),
    )

    judge_result = evaluate.AnswerEval(
        feedback="Accurate and complete answer",
        accuracy=5.0,
        completeness=4.5,
        relevance=5.0,
    )

    completion_mock = Mock(
        return_value=SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=judge_result.model_dump_json()
                    )
                )
            ]
        )
    )

    monkeypatch.setattr(
        evaluate,
        "completion",
        completion_mock,
    )

    result, answer, docs = evaluate.evaluate_answer(test)

    assert result.accuracy == 5.0
    assert result.completeness == 4.5
    assert result.relevance == 5.0
    assert result.feedback == "Accurate and complete answer"

    assert answer == generated_answer
    assert docs == retrieved_docs

    completion_mock.assert_called_once()

    call_args = completion_mock.call_args.kwargs

    assert call_args["model"] == evaluate.JUDGE_MODEL
    assert call_args["response_format"] == evaluate.AnswerEval
    assert call_args["extra_body"] == {
        "reasoning_effort": "none"
    }

    # Check that the prompt contains the required context.
    messages = call_args["messages"]
    user_prompt = messages[1]["content"]

    assert test.question in user_prompt
    assert generated_answer in user_prompt
    assert test.reference_answer in user_prompt


def test_evaluate_answer_calls_answer_question(monkeypatch):
    test = make_test()

    answer_mock = Mock(
        return_value=("Generated answer", [])
    )

    judge_result = evaluate.AnswerEval(
        feedback="Good",
        accuracy=4.0,
        completeness=4.0,
        relevance=4.0,
    )

    monkeypatch.setattr(
        evaluate,
        "answer_question",
        answer_mock,
    )

    monkeypatch.setattr(
        evaluate,
        "completion",
        Mock(
            return_value=SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content=judge_result.model_dump_json()
                        )
                    )
                ]
            )
        ),
    )

    evaluate.evaluate_answer(test)

    answer_mock.assert_called_once_with(test.question)


def test_evaluate_answer_invalid_judge_response(monkeypatch):
    test = make_test()

    monkeypatch.setattr(
        evaluate,
        "answer_question",
        lambda question: ("Generated answer", []),
    )

    monkeypatch.setattr(
        evaluate,
        "completion",
        Mock(
            return_value=SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content='{"feedback": "Missing scores"}'
                        )
                    )
                ]
            )
        ),
    )

    with pytest.raises(ValueError):
        evaluate.evaluate_answer(test)


# --------------------------------------------------
# Tests for evaluate_all_retrieval()
# --------------------------------------------------

def test_evaluate_all_retrieval(monkeypatch):
    tests = [
        make_test(category="Claims"),
        make_test(category="Policies"),
    ]

    monkeypatch.setattr(
        evaluate,
        "load_tests",
        lambda: tests,
    )

    retrieval_mock = Mock(
        side_effect=[
            evaluate.RetrievalEval(
                mrr=0.8,
                ndcg=0.9,
                keywords_found=2,
                total_keywords=2,
                keyword_coverage=100.0,
            ),
            evaluate.RetrievalEval(
                mrr=0.6,
                ndcg=0.7,
                keywords_found=1,
                total_keywords=2,
                keyword_coverage=50.0,
            ),
        ]
    )

    monkeypatch.setattr(
        evaluate,
        "evaluate_retrieval",
        retrieval_mock,
    )

    results = list(evaluate.evaluate_all_retrieval())

    assert len(results) == 2

    assert results[0][0] == tests[0]
    assert results[0][2] == 0.5

    assert results[1][0] == tests[1]
    assert results[1][2] == 1.0

    assert results[0][1].mrr == 0.8
    assert results[1][1].mrr == 0.6


def test_evaluate_all_retrieval_empty_dataset(monkeypatch):
    monkeypatch.setattr(
        evaluate,
        "load_tests",
        lambda: [],
    )

    results = list(evaluate.evaluate_all_retrieval())

    assert results == []


# --------------------------------------------------
# Tests for evaluate_all_answers()
# --------------------------------------------------

def test_evaluate_all_answers(monkeypatch):
    tests = [
        make_test(category="Claims"),
        make_test(category="Policies"),
    ]

    monkeypatch.setattr(
        evaluate,
        "load_tests",
        lambda: tests,
    )

    answer_mock = Mock(
        side_effect=[
            (
                evaluate.AnswerEval(
                    feedback="Good",
                    accuracy=4.5,
                    completeness=4.0,
                    relevance=4.5,
                ),
                "Answer 1",
                [],
            ),
            (
                evaluate.AnswerEval(
                    feedback="Excellent",
                    accuracy=5.0,
                    completeness=5.0,
                    relevance=5.0,
                ),
                "Answer 2",
                [],
            ),
        ]
    )

    monkeypatch.setattr(
        evaluate,
        "evaluate_answer",
        answer_mock,
    )

    results = list(evaluate.evaluate_all_answers())

    assert len(results) == 2

    assert results[0][0] == tests[0]
    assert results[0][2] == 0.5

    assert results[1][0] == tests[1]
    assert results[1][2] == 1.0

    assert results[0][1].accuracy == 4.5
    assert results[1][1].accuracy == 5.0


def test_evaluate_all_answers_empty_dataset(monkeypatch):
    monkeypatch.setattr(
        evaluate,
        "load_tests",
        lambda: [],
    )

    results = list(evaluate.evaluate_all_answers())

    assert results == []


# --------------------------------------------------
# Tests for run_cli_evaluation()
# --------------------------------------------------

def test_run_cli_evaluation(monkeypatch, capsys):
    test = make_test(
        question="What is insurance?",
        keywords=["insurance", "policy"],
        category="Claims",
        reference_answer="Insurance is a contract.",
    )

    monkeypatch.setattr(
        evaluate,
        "load_tests",
        lambda: [test],
    )

    retrieval_result = evaluate.RetrievalEval(
        mrr=0.8,
        ndcg=0.9,
        keywords_found=2,
        total_keywords=2,
        keyword_coverage=100.0,
    )

    answer_result = evaluate.AnswerEval(
        feedback="Good answer",
        accuracy=4.5,
        completeness=4.0,
        relevance=5.0,
    )

    monkeypatch.setattr(
        evaluate,
        "evaluate_retrieval",
        lambda test: retrieval_result,
    )

    monkeypatch.setattr(
        evaluate,
        "evaluate_answer",
        lambda test: (answer_result, "Generated answer", []),
    )

    evaluate.run_cli_evaluation(0)

    output = capsys.readouterr().out

    assert "Question: What is insurance?" in output
    assert "Category: Claims" in output
    assert "MRR: 0.8000" in output
    assert "nDCG: 0.9000" in output
    assert "Keyword Coverage: 100.0%" in output
    assert "Generated answer" in output
    assert "Good answer" in output
    assert "Accuracy: 4.50/5" in output


def test_run_cli_evaluation_invalid_number(monkeypatch, capsys):
    monkeypatch.setattr(
        evaluate,
        "load_tests",
        lambda: [make_test()],
    )

    with pytest.raises(SystemExit) as exc:
        evaluate.run_cli_evaluation(5)

    assert exc.value.code == 1

    output = capsys.readouterr().out

    assert "test_row_number must be between 0 and 0" in output


def test_run_cli_evaluation_negative_number(monkeypatch):
    monkeypatch.setattr(
        evaluate,
        "load_tests",
        lambda: [make_test()],
    )

    with pytest.raises(SystemExit):
        evaluate.run_cli_evaluation(-1)


# --------------------------------------------------
# Tests for main()
# --------------------------------------------------

def test_main_missing_argument(monkeypatch, capsys):
    monkeypatch.setattr(
        "sys.argv",
        ["evaluate.py"],
    )

    with pytest.raises(SystemExit):
        evaluate.main()

    output = capsys.readouterr().out

    assert "Usage: uv run evaluate.py" in output


def test_main_invalid_argument(monkeypatch, capsys):
    monkeypatch.setattr(
        "sys.argv",
        ["evaluate.py", "abc"],
    )

    with pytest.raises(SystemExit):
        evaluate.main()

    output = capsys.readouterr().out

    assert "test_row_number must be an integer" in output


def test_main_valid_argument(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        ["evaluate.py", "0"],
    )

    run_mock = Mock()

    monkeypatch.setattr(
        evaluate,
        "run_cli_evaluation",
        run_mock,
    )

    evaluate.main()

    run_mock.assert_called_once_with(0)