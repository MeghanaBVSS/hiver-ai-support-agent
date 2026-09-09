"""Reply generation baselines.

Phase 0: retrieve most similar historical customer message and copy its
associated support response. No LLM drafting yet.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.evaluation.harness import AgentPrediction, ExampleRecord
from src.policy.escalation import EscalationPolicy, EscalationSignals
from src.retrieval.tfidf_baseline import TfidfResponseRetriever


@dataclass
class RetrieveAndCopyReplier:
    """Baseline reply generator: TF-IDF nearest-neighbor response copy."""

    retriever: TfidfResponseRetriever
    policy: EscalationPolicy | None = None
    name: str = "baseline_retrieve_and_copy"

    def predict(self, example: ExampleRecord) -> AgentPrediction:
        evidence = self.retriever.retrieve(example.customer_message)
        top = evidence[0] if evidence else None
        reply = top.support_response if top else None
        sim = top.score if top else 0.0

        policy = self.policy or EscalationPolicy()
        decision = policy.decide(
            EscalationSignals(
                intent_confidence=None,
                retrieval_similarity=sim,
                customer_message=example.customer_message,
                has_conflicting_evidence=_conflicting(evidence),
                unsupported_action=False,
            )
        )
        return AgentPrediction(
            intent=None,
            intent_confidence=None,
            reply=reply,
            escalate=decision.escalate,
            escalate_reason=decision.reason,
            retrieved_evidence=[e.to_dict() for e in evidence],
            meta={"baseline": self.name, "top_similarity": sim},
        )


def _conflicting(evidence: list) -> bool:
    if len(evidence) < 2:
        return False
    # Crude conflict signal: top responses differ a lot in length ratio / first tokens.
    a = (evidence[0].support_response or "").strip().lower()
    b = (evidence[1].support_response or "").strip().lower()
    if not a or not b:
        return True
    return a.split()[:5] != b.split()[:5] and abs(len(a) - len(b)) > 80
