"""Intent metrics and evaluation interfaces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)


@dataclass
class IntentMetrics:
    accuracy: float
    macro_f1: float
    weighted_f1: float
    per_class: dict[str, dict[str, float]]
    confusion_matrix: list[list[int]]
    labels: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "accuracy": self.accuracy,
            "macro_f1": self.macro_f1,
            "weighted_f1": self.weighted_f1,
            "per_class": self.per_class,
            "confusion_matrix": self.confusion_matrix,
            "labels": self.labels,
        }


def compute_intent_metrics(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    labels: Sequence[str] | None = None,
) -> IntentMetrics:
    y_true_l = [str(x) for x in y_true]
    y_pred_l = [str(x) for x in y_pred]
    if labels is None:
        labels = sorted(set(y_true_l) | set(y_pred_l))
    else:
        labels = [str(x) for x in labels]

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true_l,
        y_pred_l,
        labels=labels,
        average=None,
        zero_division=0,
    )
    per_class = {}
    for i, label in enumerate(labels):
        per_class[label] = {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }

    cm = confusion_matrix(y_true_l, y_pred_l, labels=labels)
    return IntentMetrics(
        accuracy=float(accuracy_score(y_true_l, y_pred_l)),
        macro_f1=float(f1_score(y_true_l, y_pred_l, average="macro", zero_division=0)),
        weighted_f1=float(
            f1_score(y_true_l, y_pred_l, average="weighted", zero_division=0)
        ),
        per_class=per_class,
        confusion_matrix=cm.astype(int).tolist(),
        labels=list(labels),
    )


def intent_classification_report(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    labels: Sequence[str] | None = None,
) -> str:
    return classification_report(
        y_true, y_pred, labels=labels, zero_division=0, digits=3
    )


@dataclass
class EscalationMetrics:
    """Decision metrics. Safety dominates coverage."""

    escalation_precision: float | None
    escalation_recall: float | None
    escalation_f1: float | None
    escalation_accuracy: float | None
    unsafe_auto_handle_rate: float | None
    auto_handle_coverage: float | None
    tp: int
    fp: int
    fn: int
    tn: int
    confusion_matrix: list[list[int]]
    notes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "escalation_precision": self.escalation_precision,
            "escalation_recall": self.escalation_recall,
            "escalation_f1": self.escalation_f1,
            "escalation_accuracy": self.escalation_accuracy,
            "unsafe_auto_handle_rate": self.unsafe_auto_handle_rate,
            "auto_handle_coverage": self.auto_handle_coverage,
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "tn": self.tn,
            "confusion_matrix": self.confusion_matrix,
            "confusion_matrix_labels": ["auto_handle", "escalate"],
            "notes": list(self.notes),
        }


def compute_escalation_metrics(
    y_true_escalate: Sequence[bool],
    y_pred_escalate: Sequence[bool],
) -> EscalationMetrics:
    """Compute escalation decision metrics.

    Definitions:
    - escalation_precision: among predicted escalations, fraction that should escalate
    - escalation_recall: among true escalations, fraction predicted
    - escalation_f1: harmonic mean of precision and recall
    - escalation_accuracy: overall escalate/auto agreement
    - unsafe_auto_handle_rate: among cases that should escalate, fraction auto-handled
      (false negatives for escalation) — primary safety metric
    - auto_handle_coverage: fraction predicted auto-handle
    Confusion matrix rows/cols: [auto_handle=False escalate, escalate=True escalate]
    stored as [[TN, FP], [FN, TP]] with True=escalate.
    """
    true = np.asarray([bool(x) for x in y_true_escalate])
    pred = np.asarray([bool(x) for x in y_pred_escalate])
    if len(true) == 0:
        return EscalationMetrics(
            None, None, None, None, None, None, 0, 0, 0, 0, [[0, 0], [0, 0]], ("empty input",)
        )

    pred_pos = int(pred.sum())
    true_pos = int(true.sum())
    tp = int(((pred) & (true)).sum())
    fp = int(((pred) & (~true)).sum())
    fn = int(((~pred) & (true)).sum())
    tn = int(((~pred) & (~true)).sum())

    precision = float(tp / pred_pos) if pred_pos else None
    recall = float(tp / true_pos) if true_pos else None
    if precision is not None and recall is not None and (precision + recall) > 0:
        f1 = float(2 * precision * recall / (precision + recall))
    else:
        f1 = None
    accuracy = float((tp + tn) / len(true)) if len(true) else None
    unsafe = float(fn / true_pos) if true_pos else 0.0
    coverage = float((~pred).mean())
    # sklearn-style labels [False, True] => [[tn, fp], [fn, tp]]
    cm = [[tn, fp], [fn, tp]]

    return EscalationMetrics(
        escalation_precision=precision,
        escalation_recall=recall,
        escalation_f1=f1,
        escalation_accuracy=accuracy,
        unsafe_auto_handle_rate=unsafe,
        auto_handle_coverage=coverage,
        tp=tp,
        fp=fp,
        fn=fn,
        tn=tn,
        confusion_matrix=cm,
        notes=(
            "Do not optimize auto_handle_coverage alone.",
            "unsafe_auto_handle_rate is the primary safety metric (escalation false negatives).",
            "Intent classification and escalation are separate decision heads.",
        ),
    )


# Reply-quality dimensions (human / LLM-judge). Interfaces only in Phase 0.
REPLY_QUALITY_DIMENSIONS = (
    "correctness",
    "groundedness",
    "helpfulness_actionability",
    "brand_style_fit",
    "unsupported_claims",  # lower is better; scored separately
)
