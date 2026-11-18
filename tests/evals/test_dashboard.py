
from types import SimpleNamespace

import pandas as pd
import pytest

from llm_engineering.evals import dashboard


# --------------------------------------------------
# Tests for get_color()
# --------------------------------------------------

@pytest.mark.parametrize(
    "value, metric_type, expected",
    [
        (0.95, "mrr", "green"),
        (0.90, "mrr", "green"),
        (0.80, "mrr", "orange"),
        (0.75, "mrr", "orange"),
        (0.50, "mrr", "red"),

        (0.95, "ndcg", "green"),
        (0.80, "ndcg", "orange"),
        (0.50, "ndcg", "red"),

        (95.0, "coverage", "green"),
        (80.0, "coverage", "orange"),
        (50.0, "coverage", "red"),

        (4.8, "accuracy", "green"),
        (4.2, "accuracy", "orange"),
        (3.5, "accuracy", "red"),

        (4.8, "completeness", "green"),
        (4.2, "relevance", "orange"),
        (3.0, "relevance", "red"),

        (0.9, "unknown", "black"),
    ],
)
def test_get_color(value, metric_type, expected):
    assert dashboard.get_color(value, metric_type) == expected


# --------------------------------------------------
# Tests for format_metric_html()
# --------------------------------------------------

def test_format_metric_html_percentage():
    result = dashboard.format_metric_html(
        label="Keyword Coverage",
        value=92.5,
        metric_type="coverage",
        is_percentage=True,
    )

    assert "Keyword Coverage" in result
    assert "92.5%" in result
    assert "green" in result


def test_format_metric_html_score():
    result = dashboard.format_metric_html(
        label="Accuracy",
        value=4.75,
        metric_type="accuracy",
        score_format=True,
    )

    assert "Accuracy" in result
    assert "4.75/5" in result
    assert "green" in result


def test_format_metric_html_default_format():
    result = dashboard.format_metric_html(
        label="MRR",
        value=0.87654,
        metric_type="mrr",
    )

    assert "MRR" in result
    assert "0.8765" in result
    assert "orange" in result


# --------------------------------------------------
# Tests for run_retrieval_evaluation()
# --------------------------------------------------

def test_run_retrieval_evaluation(monkeypatch):
    test_data = [
        (
            SimpleNamespace(category="Claims"),
            SimpleNamespace(
                mrr=0.8,
                ndcg=0.9,
                keyword_coverage=80.0,
            ),
            0.5,
        ),
        (
            SimpleNamespace(category="Claims"),
            SimpleNamespace(
                mrr=1.0,
                ndcg=0.8,
                keyword_coverage=90.0,
            ),
            1.0,
        ),
        (
            SimpleNamespace(category="Policies"),
            SimpleNamespace(
                mrr=0.6,
                ndcg=0.7,
                keyword_coverage=70.0,
            ),
            1.0,
        ),
    ]

    monkeypatch.setattr(
        dashboard,
        "evaluate_all_retrieval",
        lambda: iter(test_data),
    )

    progress_calls = []

    def mock_progress(value, desc=None):
        progress_calls.append((value, desc))

    html, df = dashboard.run_retrieval_evaluation(
        progress=mock_progress
    )

    # Average metrics
    assert "0.8000" in html
    assert "0.8000" in html
    assert "80.0%" in html

    # Completion message
    assert "Evaluation Complete: 3 tests" in html

    # Category averages
    assert isinstance(df, pd.DataFrame)

    claims = df[df["Category"] == "Claims"].iloc[0]
    policies = df[df["Category"] == "Policies"].iloc[0]

    assert claims["Average MRR"] == pytest.approx(0.9)
    assert policies["Average MRR"] == pytest.approx(0.6)

    # Progress updates
    assert len(progress_calls) == 3
    assert progress_calls[-1][0] == 1.0


# --------------------------------------------------
# Tests for run_answer_evaluation()
# --------------------------------------------------

def test_run_answer_evaluation(monkeypatch):
    test_data = [
        (
            SimpleNamespace(category="Claims"),
            SimpleNamespace(
                accuracy=4.5,
                completeness=4.0,
                relevance=4.8,
            ),
            0.5,
        ),
        (
            SimpleNamespace(category="Claims"),
            SimpleNamespace(
                accuracy=5.0,
                completeness=4.5,
                relevance=4.2,
            ),
            1.0,
        ),
        (
            SimpleNamespace(category="Policies"),
            SimpleNamespace(
                accuracy=3.5,
                completeness=3.0,
                relevance=3.8,
            ),
            1.0,
        ),
    ]

    monkeypatch.setattr(
        dashboard,
        "evaluate_all_answers",
        lambda: iter(test_data),
    )

    progress_calls = []

    def mock_progress(value, desc=None):
        progress_calls.append((value, desc))

    html, df = dashboard.run_answer_evaluation(
        progress=mock_progress
    )

    # Average metrics
    assert "4.33/5" in html
    assert "3.83/5" in html
    assert "4.27/5" in html

    # Completion message
    assert "Evaluation Complete: 3 tests" in html

    # Category averages
    assert isinstance(df, pd.DataFrame)

    claims = df[df["Category"] == "Claims"].iloc[0]
    policies = df[df["Category"] == "Policies"].iloc[0]

    assert claims["Average Accuracy"] == pytest.approx(4.75)
    assert policies["Average Accuracy"] == pytest.approx(3.5)

    # Progress updates
    assert len(progress_calls) == 3
    assert progress_calls[-1][0] == 1.0


# --------------------------------------------------
# Tests for empty evaluation results
# --------------------------------------------------

@pytest.mark.parametrize(
    "evaluation_function, evaluator_name",
    [
        (dashboard.run_retrieval_evaluation, "evaluate_all_retrieval"),
        (dashboard.run_answer_evaluation, "evaluate_all_answers"),
    ],
)
def test_empty_evaluation_results(
    monkeypatch,
    evaluation_function,
    evaluator_name,
):
    monkeypatch.setattr(
        dashboard,
        evaluator_name,
        lambda: iter([]),
    )

    with pytest.raises(ZeroDivisionError):
        evaluation_function(progress=lambda *args, **kwargs: None)
