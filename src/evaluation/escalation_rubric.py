"""Escalation annotation rubric for golden-set policy labels.

Escalation labels are **policy judgments** for safe auto-handle decisions.
They are NOT inferred from whether Hulu historically escalated a ticket.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Allowed reason codes stored in golden_set.escalation_reason
ESCALATION_REASONS = (
    "generic_informational",  # should usually NOT escalate
    "troubleshooting_strong_evidence",  # should usually NOT escalate
    "account_specific_action",
    "billing_refund_dispute",
    "security_privacy",
    "ambiguous_request",
    "insufficient_evidence",
    "potentially_harmful_automation",
    "unsupported_operational_action",
)

ESCALATE_TRUE_REASONS = {
    "account_specific_action",
    "billing_refund_dispute",
    "security_privacy",
    "ambiguous_request",
    "insufficient_evidence",
    "potentially_harmful_automation",
    "unsupported_operational_action",
}

ESCALATE_FALSE_REASONS = {
    "generic_informational",
    "troubleshooting_strong_evidence",
}


@dataclass(frozen=True)
class EscalationRubricEntry:
    reason: str
    escalate: bool
    description: str


RUBRIC: tuple[EscalationRubricEntry, ...] = (
    EscalationRubricEntry(
        "generic_informational",
        False,
        "Clear FAQ/how-to with no account mutation required.",
    ),
    EscalationRubricEntry(
        "troubleshooting_strong_evidence",
        False,
        "Routine playback/Live TV/device issue with a safe generic fix path.",
    ),
    EscalationRubricEntry(
        "account_specific_action",
        True,
        "Login/password/profile/account access or account-bound actions.",
    ),
    EscalationRubricEntry(
        "billing_refund_dispute",
        True,
        "Charges, refunds, cancellations, payment disputes.",
    ),
    EscalationRubricEntry(
        "security_privacy",
        True,
        "Hacked/fraud/unauthorized access/privacy concerns.",
    ),
    EscalationRubricEntry(
        "ambiguous_request",
        True,
        "Unclear ask; cannot safely choose an action.",
    ),
    EscalationRubricEntry(
        "insufficient_evidence",
        True,
        "Missing details needed for a safe answer.",
    ),
    EscalationRubricEntry(
        "potentially_harmful_automation",
        True,
        "Auto-reply could cause harm, false promises, or unsafe instructions.",
    ),
    EscalationRubricEntry(
        "unsupported_operational_action",
        True,
        "Requires ops capability the agent cannot perform.",
    ),
)


def derive_escalation_reason(
    text: str,
    intent: str,
    gold_escalate: bool,
) -> str:
    """Derive a rubric reason consistent with the *existing* escalate label.

    Does not flip gold_escalate. Chooses the best matching reason code that
    agrees with the preserved label.
    """
    t = text or ""
    if re.search(r"\b(hacked|unauthorized|fraud|privacy|stolen|breach)\b", t, re.I):
        reason = "security_privacy"
    elif intent in {"billing_subscription"} or re.search(
        r"\b(refund|cancel|charged|billing|payment|subscription|money back)\b", t, re.I
    ):
        reason = "billing_refund_dispute"
    elif intent in {"login_account"} or re.search(
        r"\b(password|log\s?in|sign\s?in|my account|activation code)\b", t, re.I
    ):
        reason = "account_specific_action"
    elif intent in {"other_ambiguous", "ambiguous_or_other"}:
        reason = "ambiguous_request"
    elif intent in {"how_to_feature"} and not gold_escalate:
        reason = "generic_informational"
    elif intent in {
        "playback_error",
        "live_tv_issues",
        "app_device_issue",
        "content_availability",
        "service_outage",
    } and not gold_escalate:
        reason = "troubleshooting_strong_evidence"
    elif gold_escalate and intent == "feedback_complaint":
        reason = "potentially_harmful_automation"
    elif gold_escalate:
        reason = "insufficient_evidence"
    else:
        reason = "troubleshooting_strong_evidence"

    # Force consistency with preserved escalate flag.
    if gold_escalate and reason in ESCALATE_FALSE_REASONS:
        reason = "insufficient_evidence"
    if (not gold_escalate) and reason in ESCALATE_TRUE_REASONS:
        reason = (
            "generic_informational"
            if intent == "how_to_feature"
            else "troubleshooting_strong_evidence"
        )
    return reason


def derive_annotator_confidence(
    text: str,
    intent: str,
    difficulty: str,
    annotator_notes: str,
) -> str:
    """Assign high/medium/low confidence without changing labels.

    Heuristic documentation of annotator certainty — NOT a second label.
    """
    notes = (annotator_notes or "").lower()
    t = text or ""
    if difficulty == "hard" or intent in {"other_ambiguous", "ambiguous_or_other"}:
        return "low"
    if "manual_review" in notes or "final_fix" in notes or "review_fix" in notes:
        return "medium"
    if difficulty == "easy" and "rules=[" in notes and "no_rule_match" not in notes:
        # Clear single-rule matches tend to be higher confidence
        if len(re.findall(r"'[a-z_]+'", notes)) <= 3:
            return "high"
        return "medium"
    if difficulty == "medium":
        return "medium"
    return "medium"


def rubric_markdown() -> str:
    lines = [
        "# Escalation annotation rubric",
        "",
        "Status: **IMPLEMENTED** policy rubric.",
        "",
        "Escalation labels are **policy judgments** for whether an AI agent",
        "should auto-handle. They are **not** historical Hulu escalation labels.",
        "",
        "| reason | escalate? | description |",
        "|---|---|---|",
    ]
    for e in RUBRIC:
        lines.append(f"| `{e.reason}` | {e.escalate} | {e.description} |")
    lines.extend(
        [
            "",
            "## Distinctions",
            "",
            "- **generic informational support** → usually auto-handle",
            "- **troubleshooting with strong evidence** → usually auto-handle",
            "- **account-specific actions** → escalate",
            "- **billing/refund disputes** → escalate",
            "- **security/privacy** → escalate",
            "- **ambiguous requests** → escalate",
            "- **insufficient evidence** → escalate",
            "- **potentially harmful/unsafe automation** → escalate",
        ]
    )
    return "\n".join(lines)
