"""Conservative escalation policy.

Safety dominates coverage. Every decision returns an explicit reason.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


_ACCOUNT_SPECIFIC = re.compile(
    r"\b(account|password|login|username|email|phone|ssn|social security)\b",
    re.IGNORECASE,
)
_BILLING = re.compile(
    r"\b(refund|charge|charged|billing|invoice|payment|cancel(?:lation)?|subscription)\b",
    re.IGNORECASE,
)
_SECURITY = re.compile(
    r"\b(hacked|unauthorized|fraud|stolen|privacy|data breach|leak)\b",
    re.IGNORECASE,
)


@dataclass
class EscalationSignals:
    intent_confidence: float | None = None
    retrieval_similarity: float | None = None
    customer_message: str = ""
    ambiguous: bool = False
    unsupported_action: bool = False
    has_conflicting_evidence: bool = False
    missing_required_info: bool = False


@dataclass
class EscalationDecision:
    escalate: bool
    reason: str
    signals_fired: list[str] = field(default_factory=list)
    auto_handle_eligible: bool = False

    def to_dict(self) -> dict:
        return {
            "escalate": self.escalate,
            "reason": self.reason,
            "signals_fired": self.signals_fired,
            "auto_handle_eligible": self.auto_handle_eligible,
        }


@dataclass
class EscalationPolicy:
    """Rule-based conservative policy for Phase 0/1.

    Auto-handle only when multiple safe signals are present and no risk signal fires.
    """

    confidence_threshold: float = 0.55
    min_retrieval_similarity: float = 0.15
    prefer_safety_over_coverage: bool = True

    def decide(self, signals: EscalationSignals) -> EscalationDecision:
        fired: list[str] = []
        text = signals.customer_message or ""

        if signals.intent_confidence is not None and signals.intent_confidence < self.confidence_threshold:
            fired.append("low_intent_confidence")
        if (
            signals.retrieval_similarity is None
            or signals.retrieval_similarity < self.min_retrieval_similarity
        ):
            fired.append("low_retrieval_evidence_quality")
        if _ACCOUNT_SPECIFIC.search(text):
            fired.append("account_specific_request")
        if _BILLING.search(text):
            fired.append("billing_or_refund_dispute")
        if _SECURITY.search(text):
            fired.append("security_or_privacy_issue")
        if signals.ambiguous:
            fired.append("ambiguous_request")
        if signals.unsupported_action:
            fired.append("unsupported_action")
        if signals.has_conflicting_evidence:
            fired.append("conflicting_historical_evidence")
        if signals.missing_required_info:
            fired.append("missing_required_information")

        if fired:
            return EscalationDecision(
                escalate=True,
                reason="Escalate due to: " + ", ".join(fired),
                signals_fired=fired,
                auto_handle_eligible=False,
            )

        # Auto-handle only when we have clear evidence and no risk signals.
        auto_ok = (
            signals.retrieval_similarity is not None
            and signals.retrieval_similarity >= max(self.min_retrieval_similarity, 0.35)
            and (
                signals.intent_confidence is None
                or signals.intent_confidence >= self.confidence_threshold
            )
        )
        if auto_ok:
            return EscalationDecision(
                escalate=False,
                reason=(
                    "Auto-handle: clear request with strong historical evidence and "
                    "no account/billing/security risk signals."
                ),
                signals_fired=[],
                auto_handle_eligible=True,
            )

        # Conservative default.
        return EscalationDecision(
            escalate=True,
            reason=(
                "Escalate by default: insufficient positive auto-handle signals "
                "(prefer safety over coverage)."
            ),
            signals_fired=["insufficient_auto_handle_confidence"],
            auto_handle_eligible=False,
        )
