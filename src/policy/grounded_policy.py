"""Phase-2 escalation policy: deterministic final authority."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any


_ACCOUNT = re.compile(
    r"\b(account|password|log\s?in|sign\s?in|username|activation code|__email__)\b",
    re.I,
)
_BILLING = re.compile(
    r"\b(refund|charge[d]?|billing|invoice|payment|cancel(?:lation)?|subscription|money back)\b",
    re.I,
)
_SECURITY = re.compile(
    r"\b(hacked|unauthorized|fraud|stolen|privacy|data breach|leak)\b",
    re.I,
)
_UNSUPPORTED = re.compile(
    r"\b(sue|lawyer|attorney|police|ssn|social security)\b",
    re.I,
)


@dataclass
class PolicyThresholds:
    intent_confidence_min: float = 0.45
    top_similarity_min: float = 0.22
    evidence_strength_min: float = 0.35
    auto_handle_similarity_min: float = 0.35
    auto_handle_evidence_min: float = 0.45


@dataclass
class PolicyInput:
    customer_message: str
    predicted_intent: str | None
    intent_confidence: float | None
    top_similarity: float | None
    evidence_strength: float | None
    evidence_intent_agreement: float | None = None
    has_conflicting_evidence: bool = False
    n_hits: int = 0


@dataclass
class PolicyResult:
    should_escalate: bool
    escalation_reason: str
    policy_signals: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class GroundedEscalationPolicy:
    """Deterministic policy. LLM must not override this decision."""

    def __init__(self, thresholds: PolicyThresholds | None = None):
        self.thresholds = thresholds or PolicyThresholds()

    def decide(self, inp: PolicyInput) -> PolicyResult:
        t = self.thresholds
        text = inp.customer_message or ""
        signals: dict[str, Any] = {
            "intent_confidence": inp.intent_confidence,
            "top_similarity": inp.top_similarity,
            "evidence_strength": inp.evidence_strength,
            "evidence_intent_agreement": inp.evidence_intent_agreement,
            "n_hits": inp.n_hits,
            "predicted_intent": inp.predicted_intent,
        }
        reasons: list[str] = []

        if inp.intent_confidence is not None and inp.intent_confidence < t.intent_confidence_min:
            reasons.append("low_intent_confidence")
        if inp.top_similarity is None or inp.top_similarity < t.top_similarity_min:
            reasons.append("weak_evidence")
        if inp.evidence_strength is None or inp.evidence_strength < t.evidence_strength_min:
            if "weak_evidence" not in reasons:
                reasons.append("weak_evidence")
        if inp.has_conflicting_evidence or (
            inp.evidence_intent_agreement is not None and inp.evidence_intent_agreement < 0.5 and (inp.n_hits or 0) >= 2
        ):
            reasons.append("conflicting_evidence")
        if _ACCOUNT.search(text) or inp.predicted_intent == "login_account":
            reasons.append("account_specific")
        if _BILLING.search(text) or inp.predicted_intent == "billing_subscription":
            reasons.append("billing_refund")
        if _SECURITY.search(text):
            reasons.append("security_privacy")
        if inp.predicted_intent in {"other_ambiguous", "ambiguous_or_other"}:
            reasons.append("ambiguous_request")
        if _UNSUPPORTED.search(text):
            reasons.append("unsupported_operation")
            reasons.append("potentially_harmful_automation")

        signals["fired"] = reasons
        if reasons:
            # Primary reason = first high-priority risk if present
            priority = [
                "security_privacy",
                "billing_refund",
                "account_specific",
                "potentially_harmful_automation",
                "unsupported_operation",
                "ambiguous_request",
                "conflicting_evidence",
                "weak_evidence",
                "low_intent_confidence",
            ]
            primary = next((r for r in priority if r in reasons), reasons[0])
            return PolicyResult(True, primary, signals)

        # Positive auto-handle gate
        auto_ok = (
            inp.top_similarity is not None
            and inp.top_similarity >= t.auto_handle_similarity_min
            and inp.evidence_strength is not None
            and inp.evidence_strength >= t.auto_handle_evidence_min
            and (
                inp.intent_confidence is None
                or inp.intent_confidence >= t.intent_confidence_min
            )
        )
        if auto_ok:
            return PolicyResult(
                False,
                "auto_handle_strong_evidence",
                signals | {"fired": []},
            )
        return PolicyResult(
            True,
            "weak_evidence",
            signals | {"fired": ["insufficient_auto_handle_confidence"]},
        )


def tune_thresholds_on_valid(
    records: list[dict[str, Any]],
    *,
    prefer_safety: bool = True,
) -> PolicyThresholds:
    """Tune thresholds on validation proxy scores — NOT golden.

    ``records`` items need: intent_confidence, top_similarity, evidence_strength, gold_escalate.
    Selects a conservative threshold combo minimizing unsafe auto-handle rate,
    then maximizing coverage among safe options.
    """
    if not records:
        return PolicyThresholds()

    conf_grid = [0.35, 0.45, 0.55]
    sim_grid = [0.15, 0.22, 0.30]
    evid_grid = [0.30, 0.35, 0.45]
    best: tuple[float, float, PolicyThresholds] | None = None
    # score = (-unsafe, coverage)
    for c in conf_grid:
        for s in sim_grid:
            for e in evid_grid:
                th = PolicyThresholds(
                    intent_confidence_min=c,
                    top_similarity_min=s,
                    evidence_strength_min=e,
                    auto_handle_similarity_min=max(s, 0.30),
                    auto_handle_evidence_min=max(e, 0.40),
                )
                policy = GroundedEscalationPolicy(th)
                y_true = []
                y_pred = []
                for r in records:
                    dec = policy.decide(
                        PolicyInput(
                            customer_message=r.get("customer_message", ""),
                            predicted_intent=r.get("predicted_intent"),
                            intent_confidence=r.get("intent_confidence"),
                            top_similarity=r.get("top_similarity"),
                            evidence_strength=r.get("evidence_strength"),
                            evidence_intent_agreement=r.get("evidence_intent_agreement"),
                            has_conflicting_evidence=bool(r.get("has_conflicting_evidence")),
                            n_hits=int(r.get("n_hits") or 0),
                        )
                    )
                    y_true.append(bool(r["gold_escalate"]))
                    y_pred.append(bool(dec.should_escalate))
                # unsafe auto-handle = true escalate predicted false
                unsafe = sum(1 for yt, yp in zip(y_true, y_pred) if yt and not yp) / max(
                    sum(1 for yt in y_true if yt), 1
                )
                coverage = sum(1 for yp in y_pred if not yp) / max(len(y_pred), 1)
                key = (-unsafe, coverage if not prefer_safety else coverage * 0.1 - unsafe)
                cand = (key[0], key[1], th)
                if best is None or (cand[0], cand[1]) > (best[0], best[1]):
                    best = cand
    assert best is not None
    return best[2]
