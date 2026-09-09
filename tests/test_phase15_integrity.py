"""Phase 1.5 evaluation integrity tests."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.data.brand_score import DEFAULT_WEIGHTS, compute_suitability_score
from src.evaluation.escalation_rubric import ESCALATION_REASONS, derive_escalation_reason
from src.evaluation.phase15 import (
    AMBIGUOUS_OR_OTHER_DEFINITION,
    AMBIGUOUS_OR_OTHER_ID,
    enrich_golden_metadata,
    validate_golden_metadata,
)

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT / "evaluation" / "golden" / "golden_set.csv"
SECOND = ROOT / "evaluation" / "golden" / "second_annotator.csv"
LARGEST = ROOT / "reports" / "phase1_5" / "largest_conversations.json"


@pytest.mark.skipif(not GOLDEN.exists(), reason="golden missing")
def test_golden_has_escalation_reason_and_confidence():
    df = pd.read_csv(GOLDEN)
    issues = validate_golden_metadata(df)
    assert not issues, issues
    assert set(df["escalation_reason"]).issubset(set(ESCALATION_REASONS))
    assert set(df["annotator_confidence"]).issubset({"high", "medium", "low"})


@pytest.mark.skipif(not GOLDEN.exists(), reason="golden missing")
def test_enrichment_preserves_core_labels():
    df = pd.read_csv(GOLDEN)
    # Drop metadata and re-enrich; intents/escalate must stay identical to file values.
    base = df.drop(columns=["escalation_reason", "annotator_confidence"], errors="ignore")
    # Keep a copy of core labels from current file
    intents = base["gold_intent"].tolist()
    esc = base["gold_escalate"].astype(bool).tolist()
    enriched = enrich_golden_metadata(base)
    assert enriched["gold_intent"].tolist() == intents
    assert enriched["gold_escalate"].astype(bool).tolist() == esc


def test_escalation_reason_validation_codes():
    # Consistent with escalate=True
    r = derive_escalation_reason("I need a refund now", "billing_subscription", True)
    assert r in ESCALATION_REASONS
    assert r == "billing_refund_dispute"
    # Consistent with escalate=False howto
    r2 = derive_escalation_reason("How do I create a profile?", "how_to_feature", False)
    assert r2 == "generic_informational"


def test_ambiguous_or_other_definition():
    assert AMBIGUOUS_OR_OTHER_ID == "other_ambiguous"
    assert "insufficient information" in AMBIGUOUS_OR_OTHER_DEFINITION.lower()


@pytest.mark.skipif(not SECOND.exists(), reason="second annotator pack missing")
def test_second_annotator_pack_integrity():
    """Phase 1.5 shipped blanks; Phase 2 completed second-pass human labels."""
    from src.intent.taxonomy_hulu import ALLOWED_INTENTS

    df = pd.read_csv(SECOND)
    assert 40 <= len(df) <= 60
    for col in ("gold_intent_2", "gold_escalate_2"):
        assert col in df.columns
    # must not include first annotator label columns (blind pack)
    assert "gold_intent" not in df.columns
    assert "gold_escalate" not in df.columns
    intents = df["gold_intent_2"].fillna("").astype(str).str.strip()
    assert intents.ne("").all(), "second-pass intents should be filled"
    assert set(intents).issubset(set(ALLOWED_INTENTS))
    esc = df["gold_escalate_2"].astype(str).str.lower().str.strip()
    assert esc.isin({"true", "false", "1", "0"}).all()


def test_brand_score_reproducibility():
    a = compute_suitability_score(
        inbound_count=18615,
        n_pairs=21468,
        coverage=1.1533,
        lexical=0.9857,
        outbound_n=21872,
    )
    b = compute_suitability_score(
        inbound_count=18615,
        n_pairs=21468,
        coverage=1.1533,
        lexical=0.9857,
        outbound_n=21872,
    )
    assert a["suitability_score"] == b["suitability_score"]
    # Formula sanity: saturated inbound/pairs + clipped coverage
    assert a["components"]["inbound_term"] == pytest.approx(0.25)
    assert a["components"]["pairs_term"] == pytest.approx(0.30)
    assert a["components"]["coverage_term"] == pytest.approx(0.15)
    assert DEFAULT_WEIGHTS.cap_n == 5000


@pytest.mark.skipif(not LARGEST.exists(), reason="conversation audit missing")
def test_conversation_component_audit_artifact():
    import json

    data = json.loads(LARGEST.read_text())
    assert 10 <= len(data) <= 20
    top = data[0]
    for key in ("conversation_id", "tweet_count", "unique_authors", "representative_tweets"):
        assert key in top
    assert top["tweet_count"] >= data[-1]["tweet_count"]
