"""Phase 1.5 evaluation integrity utilities."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.evaluation.escalation_rubric import (
    ESCALATION_REASONS,
    derive_annotator_confidence,
    derive_escalation_reason,
)
from src.intent.taxonomy_hulu import ALLOWED_INTENTS

# Conceptual alias — label values in files remain `other_ambiguous` (preserved).
AMBIGUOUS_OR_OTHER_ID = "other_ambiguous"
AMBIGUOUS_OR_OTHER_DEFINITION = (
    "Insufficient information to reliably assign one supported intent "
    "(unclear, multi-intent without a dominant ask, too short, or residual)."
)


def enrich_golden_metadata(golden: pd.DataFrame) -> pd.DataFrame:
    """Add escalation_reason + annotator_confidence without changing core labels."""
    out = golden.copy()
    # Preserve originals
    assert "gold_intent" in out.columns and "gold_escalate" in out.columns

    reasons = []
    confs = []
    for row in out.itertuples(index=False):
        intent = str(row.gold_intent)
        esc = bool(row.gold_escalate)
        text = str(row.customer_message)
        diff = str(getattr(row, "difficulty", "medium"))
        notes = str(getattr(row, "annotator_notes", ""))
        reasons.append(derive_escalation_reason(text, intent, esc))
        confs.append(derive_annotator_confidence(text, intent, diff, notes))

    out["escalation_reason"] = reasons
    out["annotator_confidence"] = confs
    # Validate consistency
    for reason in out["escalation_reason"]:
        if reason not in ESCALATION_REASONS:
            raise ValueError(f"Invalid escalation_reason: {reason}")
    for conf in out["annotator_confidence"]:
        if conf not in {"high", "medium", "low"}:
            raise ValueError(f"Invalid annotator_confidence: {conf}")
    return out


def validate_golden_metadata(df: pd.DataFrame) -> list[str]:
    issues: list[str] = []
    required = {
        "example_id",
        "customer_message",
        "gold_intent",
        "gold_escalate",
        "escalation_reason",
        "annotator_confidence",
        "difficulty",
        "annotator_id",
    }
    missing = required - set(df.columns)
    if missing:
        issues.append(f"missing columns: {sorted(missing)}")
    if "gold_intent" in df.columns:
        bad = set(df["gold_intent"]) - set(ALLOWED_INTENTS)
        if bad:
            issues.append(f"unknown intents: {sorted(bad)}")
    if "escalation_reason" in df.columns:
        bad = set(df["escalation_reason"]) - set(ESCALATION_REASONS)
        if bad:
            issues.append(f"unknown escalation_reason: {sorted(bad)}")
    if "annotator_confidence" in df.columns:
        bad = set(df["annotator_confidence"]) - {"high", "medium", "low"}
        if bad:
            issues.append(f"unknown confidence: {sorted(bad)}")
    if df["example_id"].duplicated().any():
        issues.append("duplicate example_id")
    return issues


def file_fingerprint(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    return {
        "path": str(path),
        "size_bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "mtime_utc": datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat(),
    }


def conversation_size_stats(sizes: list[int]) -> dict[str, Any]:
    s = pd.Series(sizes, dtype=float)
    if s.empty:
        return {}
    bins = {
        "n_eq_1": int((s == 1).sum()),
        "n_eq_2": int((s == 2).sum()),
        "n_3_to_5": int(((s >= 3) & (s <= 5)).sum()),
        "n_6_to_10": int(((s >= 6) & (s <= 10)).sum()),
        "n_gt_10": int((s > 10).sum()),
        "n_gt_20": int((s > 20).sum()),
        "n_gt_50": int((s > 50).sum()),
        "n_gt_100": int((s > 100).sum()),
    }
    return {
        "min": float(s.min()),
        "p25": float(s.quantile(0.25)),
        "median": float(s.median()),
        "p75": float(s.quantile(0.75)),
        "p90": float(s.quantile(0.90)),
        "p95": float(s.quantile(0.95)),
        "p99": float(s.quantile(0.99)),
        "max": float(s.max()),
        "mean": float(s.mean()),
        "count": int(len(s)),
        "bins": bins,
    }
