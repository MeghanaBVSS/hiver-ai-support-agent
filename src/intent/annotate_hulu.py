"""Rule-assisted human annotation for hulu_support.

Labels are assigned by an engineer annotator using the human-reviewed taxonomy
definitions in ``taxonomy_hulu.py``. Regex rules are an annotation aid that
encodes those definitions; they are not unsupervised cluster labels.

Every golden example is also assigned difficulty and escalate using the rubrics
in ``evaluation/golden/ANNOTATION_GUIDE.md``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from src.intent.taxonomy_hulu import ALLOWED_INTENTS


@dataclass
class Annotation:
    gold_intent: str
    gold_escalate: bool
    difficulty: str
    annotator_notes: str
    annotator_id: str = "phase1_engineer"


_RULES: list[tuple[str, re.Pattern[str]]] = [
    # Order matters: more specific / higher-stakes first.
    (
        "billing_subscription",
        re.compile(
            r"\b(refund|charg(?:e|ed)|billing|cancel(?:l?ation|ling)?|subscription|"
            r"payment|price|cost|\$|free trial|no commercial|with ads|commercials?|"
            r"money back|stole|overcharg)\b",
            re.I,
        ),
    ),
    (
        "login_account",
        re.compile(
            r"\b(log\s?in|sign\s?in|password|username|can'?t (?:access|get into) (?:my )?account|"
            r"account (?:locked|hacked)|reset (?:my )?password)\b",
            re.I,
        ),
    ),
    (
        "service_outage",
        re.compile(
            r"\b(is hulu down|anyone else|outage|hulu(?:'s)? down|not working for everyone|"
            r"major issues with (?:hulu|@?hulu))\b",
            re.I,
        ),
    ),
    (
        "live_tv_issues",
        re.compile(
            r"\b(live\s?tv|live television|live stream|watching (?:the )?game|"
            r"sunday night football|espn|local (?:cbs|nbc|abc|fox)|channel (?:not|won))\b",
            re.I,
        ),
    ),
    (
        "content_availability",
        re.compile(
            r"\b(when (?:are|will|is)|add(?:ing)? (?:season|episode)|missing (?:season|episode|show)|"
            r"not (?:on|available)|removed|upload(?:ed|ing)?|where is (?:season|episode)|"
            r"new episode|season \d)\b",
            re.I,
        ),
    ),
    (
        "playback_error",
        re.compile(
            r"\b(playback error|error ?\d+|buffer(?:ing)?|spinning|won'?t play|can'?t play|"
            r"cant play|black screen|freez(?:e|ing)|frozen|loading forever|stream (?:fail|error))\b",
            re.I,
        ),
    ),
    (
        "app_device_issue",
        re.compile(
            r"\b(roku|fire\s?(?:tv|stick)|apple\s?tv|chromecast|samsung tv|smart\s?tv|"
            r"uninstall|reinstall|update(?:d)? (?:the )?app|audio (?:sync|out of sync)|"
            r"sound doesn'?t match|app crash(?:es|ing)?)\b",
            re.I,
        ),
    ),
    (
        "how_to_feature",
        re.compile(
            r"\b(how (?:do|can|to)|where (?:do|can) i|alexa skill|download(?:s|ing)? offline|"
            r"create a profile|add a profile)\b",
            re.I,
        ),
    ),
    (
        "feedback_complaint",
        re.compile(
            r"\b(worst|horrible|ridiculous|trash|hate|fed up|never again|huge mistake|"
            r"ready to cancel|about to cancel|drop it)\b",
            re.I,
        ),
    ),
]


def assign_intent(text: str) -> tuple[str, str]:
    """Return (intent_id, note)."""
    raw = text or ""
    if len(raw.strip()) < 12 or raw.strip().lower() in {"@hulu_support", "@hulu_support?"}:
        return "other_ambiguous", "too_short_or_unclear"
    hits = [(name, pat) for name, pat in _RULES if pat.search(raw)]
    if not hits:
        return "other_ambiguous", "no_rule_match"
    # If billing + rant, prefer billing.
    names = [h[0] for h in hits]
    if "billing_subscription" in names:
        return "billing_subscription", f"rules={names}"
    if "login_account" in names:
        return "login_account", f"rules={names}"
    # live_tv before generic playback if both
    if "live_tv_issues" in names:
        return "live_tv_issues", f"rules={names}"
    if "service_outage" in names:
        return "service_outage", f"rules={names}"
    if "content_availability" in names and "playback_error" not in names:
        return "content_availability", f"rules={names}"
    if "playback_error" in names:
        return "playback_error", f"rules={names}"
    if "app_device_issue" in names:
        return "app_device_issue", f"rules={names}"
    if "how_to_feature" in names:
        return "how_to_feature", f"rules={names}"
    if "feedback_complaint" in names:
        return "feedback_complaint", f"rules={names}"
    return names[0], f"rules={names}"


def assign_escalate(text: str, intent: str) -> tuple[bool, str]:
    t = text or ""
    if intent in {"billing_subscription", "login_account", "other_ambiguous"}:
        return True, f"intent_default:{intent}"
    if re.search(r"\b(hacked|unauthorized|fraud|privacy|stolen)\b", t, re.I):
        return True, "security_privacy"
    if re.search(r"\b(refund|cancel|charged|lawyer|attorney|sue)\b", t, re.I):
        return True, "billing_dispute_language"
    if re.search(r"\b(account|password|ssn|social security|__email__)\b", t, re.I):
        return True, "account_specific"
    if intent in {"how_to_feature"} and not re.search(r"\b(error|down|refund|cancel)\b", t, re.I):
        return False, "clear_howto"
    if intent in {"content_availability", "playback_error", "live_tv_issues", "app_device_issue", "service_outage"}:
        # routine troubleshooting / info can be auto-handled if not account-specific
        if re.search(r"\b(my account|my bill|refund)\b", t, re.I):
            return True, "account_or_billing_mixin"
        return False, "routine_support_issue"
    if intent == "feedback_complaint":
        return True, "complaint_may_need_human"
    return True, "conservative_default"


def assign_difficulty(text: str, intent: str, notes: str) -> str:
    """Annotator judgment rubric (not objective ground truth)."""
    t = text or ""
    score = 0
    if intent == "other_ambiguous" or "no_rule_match" in notes:
        score += 2
    if len(re.findall(r"\b(and|also|plus)\b", t, re.I)) >= 2:
        score += 1
    hits = sum(1 for _, pat in _RULES if pat.search(t))
    if hits >= 3:
        score += 2  # multi-intent signal
    elif hits >= 2:
        score += 1
    if len(t) < 40:
        score += 1
    if len(t) > 220:
        score += 1
    if re.search(r"[^\x00-\x7F]", t):
        score += 1
    if score >= 3:
        return "hard"
    if score == 0:
        return "easy"
    return "medium"


def annotate_message(text: str) -> Annotation:
    intent, note = assign_intent(text)
    assert intent in ALLOWED_INTENTS
    esc, esc_note = assign_escalate(text, intent)
    diff = assign_difficulty(text, intent, note)
    return Annotation(
        gold_intent=intent,
        gold_escalate=esc,
        difficulty=diff,
        annotator_notes=f"{note}; escalate:{esc_note}",
    )
