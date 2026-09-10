"""Tests for IAA, leakage CLI helpers, and enhanced escalation metrics."""

from __future__ import annotations

from src.evaluation.iaa import compute_binary_agreement, compute_categorical_agreement
from src.evaluation.metrics import compute_escalation_metrics
from src.evaluation.validation_enhancement import (
    format_leakage_text,
    run_leakage_check,
    compute_second_annotator_iaa,
)


def test_cohen_kappa_perfect_agreement():
    labels = ["a", "b", "a", "b"]
    out = compute_categorical_agreement(labels, labels, labels=["a", "b"])
    assert out["exact_agreement"] == 1.0
    assert out["cohen_kappa"] == 1.0


def test_cohen_kappa_binary():
    a = [True, True, False, False]
    b = [True, False, False, False]
    out = compute_binary_agreement(a, b)
    assert out["n"] == 4
    assert out["n_disagreements"] == 1
    assert 0.0 <= out["cohen_kappa"] <= 1.0


def test_escalation_metrics_include_f1_and_cm():
    true = [True, True, False, False]
    pred = [True, False, True, False]
    m = compute_escalation_metrics(true, pred)
    assert m.tp == 1 and m.fn == 1 and m.fp == 1 and m.tn == 1
    assert m.escalation_f1 is not None
    assert m.escalation_accuracy == 0.5
    assert m.confusion_matrix == [[1, 1], [1, 1]]
    assert m.unsafe_auto_handle_rate == 0.5


def test_second_annotator_iaa_runs():
    report = compute_second_annotator_iaa()
    assert report["n_subset"] == 50
    assert "cohen_kappa" in report["intent"]
    assert "cohen_kappa" in report["escalation"]
    assert 0.0 <= report["intent"]["exact_agreement"] <= 1.0


def test_leakage_check_format():
    result = run_leakage_check()
    text = format_leakage_text(result)
    assert "Golden-set leakage check" in text
    assert "Training overlap:" in text
    assert "Status:" in text
    # With local artifacts present, expect PASS
    if result["artifacts_present"]["train_labeled"] and result["artifacts_present"]["retrieval_corpus"]:
        assert result["status"] == "PASS"
