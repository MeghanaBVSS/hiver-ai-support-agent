"""Shared evaluation harness interfaces for baselines and future systems."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from src.evaluation.metrics import (
    EscalationMetrics,
    IntentMetrics,
    compute_escalation_metrics,
    compute_intent_metrics,
)


@dataclass
class AgentPrediction:
    intent: str | None
    intent_confidence: float | None
    reply: str | None
    escalate: bool
    escalate_reason: str
    retrieved_evidence: list[dict[str, Any]] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExampleRecord:
    example_id: str
    customer_message: str
    gold_intent: str | None = None
    gold_escalate: bool | None = None
    conversation_context: list[str] = field(default_factory=list)


class SupportAgent(Protocol):
    """Common interface so baselines and future LLM agents share one harness."""

    name: str

    def predict(self, example: ExampleRecord) -> AgentPrediction:
        ...


@dataclass
class HarnessResult:
    intent_metrics: IntentMetrics | None
    escalation_metrics: EscalationMetrics | None
    predictions: list[AgentPrediction]
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "intent_metrics": None if self.intent_metrics is None else self.intent_metrics.to_dict(),
            "escalation_metrics": None
            if self.escalation_metrics is None
            else self.escalation_metrics.to_dict(),
            "n_predictions": len(self.predictions),
            "notes": self.notes,
        }


def evaluate_agent(
    agent: SupportAgent,
    examples: list[ExampleRecord],
) -> HarnessResult:
    preds: list[AgentPrediction] = []
    y_true_intent: list[str] = []
    y_pred_intent: list[str] = []
    y_true_esc: list[bool] = []
    y_pred_esc: list[bool] = []
    notes: list[str] = []

    for ex in examples:
        pred = agent.predict(ex)
        preds.append(pred)
        if ex.gold_intent is not None and pred.intent is not None:
            y_true_intent.append(ex.gold_intent)
            y_pred_intent.append(pred.intent)
        if ex.gold_escalate is not None:
            y_true_esc.append(bool(ex.gold_escalate))
            y_pred_esc.append(bool(pred.escalate))

    intent_metrics = (
        compute_intent_metrics(y_true_intent, y_pred_intent) if y_true_intent else None
    )
    esc_metrics = (
        compute_escalation_metrics(y_true_esc, y_pred_esc) if y_true_esc else None
    )
    if intent_metrics is None:
        notes.append("Intent metrics skipped: missing gold intents.")
    if esc_metrics is None:
        notes.append("Escalation metrics skipped: missing gold escalate labels.")
    return HarnessResult(intent_metrics, esc_metrics, preds, notes)
