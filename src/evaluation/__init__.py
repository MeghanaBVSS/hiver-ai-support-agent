"""Evaluation package."""

from src.evaluation.harness import AgentPrediction, ExampleRecord, evaluate_agent
from src.evaluation.leakage import (
    check_conversation_split_leakage,
    check_golden_contamination,
    check_retrieval_index_contamination,
)
from src.evaluation.metrics import compute_escalation_metrics, compute_intent_metrics
from src.evaluation.splits import conversation_level_split, temporal_split

__all__ = [
    "AgentPrediction",
    "ExampleRecord",
    "evaluate_agent",
    "check_conversation_split_leakage",
    "check_golden_contamination",
    "check_retrieval_index_contamination",
    "compute_escalation_metrics",
    "compute_intent_metrics",
    "conversation_level_split",
    "temporal_split",
]
