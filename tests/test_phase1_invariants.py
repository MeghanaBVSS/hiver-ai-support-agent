"""Phase 1 integration tests: splits, golden isolation, retrieval isolation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.evaluation.leakage import (
    check_golden_contamination,
    check_retrieval_index_contamination,
)
from src.intent.taxonomy_hulu import ALLOWED_INTENTS
from src.preprocessing.text_normalize import normalize_tweet_text

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT / "evaluation" / "golden" / "golden_set.csv"
TRAIN = ROOT / "data" / "processed" / "hulu_train_labeled.parquet"
PAIRS = ROOT / "data" / "processed" / "hulu_support_pairs_split.parquet"


@pytest.mark.skipif(not GOLDEN.exists(), reason="golden set not built yet")
def test_golden_schema_and_intents():
    df = pd.read_csv(GOLDEN)
    required = {
        "example_id",
        "customer_message",
        "gold_intent",
        "gold_escalate",
        "difficulty",
        "annotator_notes",
    }
    assert required.issubset(df.columns)
    assert df["example_id"].is_unique
    assert set(df["gold_intent"]).issubset(set(ALLOWED_INTENTS))
    assert set(df["difficulty"]).issubset({"easy", "medium", "hard"})
    assert 150 <= len(df) <= 250


@pytest.mark.skipif(not (GOLDEN.exists() and TRAIN.exists()), reason="artifacts missing")
def test_golden_not_in_train_or_retrieval():
    golden = pd.read_csv(GOLDEN)
    train = pd.read_parquet(TRAIN)
    gids = set(golden["customer_tweet_id"].astype(str))
    tids = set(train["customer_tweet_id"].astype(str))
    report = check_golden_contamination(gids, tids, retrieval_ids=tids)
    assert report.ok, report.issues
    rreport = check_retrieval_index_contamination(tids, gids, gids)
    assert rreport.ok, rreport.issues


@pytest.mark.skipif(not PAIRS.exists(), reason="pairs split missing")
def test_no_conversation_across_splits():
    pairs = pd.read_parquet(PAIRS)
    mapping = pairs.groupby("conversation_id")["split"].nunique()
    assert (mapping == 1).all()


@pytest.mark.skipif(not PAIRS.exists(), reason="pairs split missing")
def test_no_duplicate_customer_text_across_splits():
    pairs = pd.read_parquet(PAIRS)
    pairs = pairs.copy()
    pairs["norm"] = pairs["customer_message"].map(normalize_tweet_text)
    # After Phase-1 cleanup this should be empty for non-empty norms.
    bad = 0
    for text, g in pairs.groupby("norm"):
        if text and g["split"].nunique() > 1:
            bad += 1
    # Allow tiny residual only if cleanup file differs; prefer zero.
    assert bad == 0


def test_conservative_conversation_id_helper_smoke():
    from src.preprocessing.threads import compare_reconstruction_methods
    from src.data.loader import load_tweets

    mini = load_tweets(ROOT / "tests" / "fixtures" / "mini_twcs.csv")
    cmp = compare_reconstruction_methods(mini)
    assert "undirected" in cmp and "directed" in cmp
