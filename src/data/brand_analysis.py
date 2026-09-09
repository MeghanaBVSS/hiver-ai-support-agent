"""Brand candidate analysis for selecting a support brand to model."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.loader import load_tweets
from src.preprocessing.conversations import (
    build_customer_support_pairs,
    reconstruct_conversations,
)

_NUMERIC_AUTHOR = re.compile(r"^\d+$")


def identify_brand_authors(df: pd.DataFrame) -> pd.Series:
    """Outbound authors with non-numeric IDs are treated as brand handles.

    ASSUMPTION (from Kaggle docs): company accounts appear with readable
    handles; customers are anonymized numeric IDs. We only use outbound
    (inbound==False) rows for brand discovery.
    """
    outbound = df[df["inbound"] == False]
    authors = outbound["author_id"].astype(str)
    return authors[~authors.str.fullmatch(r"\d+")].value_counts()


def analyze_brand(df: pd.DataFrame, brand: str) -> dict[str, Any]:
    """Estimate suitability metrics for one brand."""
    brand = str(brand)
    # Tweets authored by brand (outbound support)
    brand_out = df[(df["author_id"].astype(str) == brand) & (df["inbound"] == False)]
    # Inbound tweets that mention the brand handle in text OR that the brand replied to.
    mention_pattern = re.compile(rf"@{re.escape(brand)}\\b", re.IGNORECASE)
    inbound = df[df["inbound"] == True].copy()
    inbound_mentions = inbound[
        inbound["text"].astype(str).str.contains(mention_pattern, na=False)
    ]

    # Conversations and pairs scoped loosely by brand presence in thread authors / mentions.
    conversations = reconstruct_conversations(df)
    brand_convs = [
        c
        for c in conversations
        if brand in c.participant_authors
        or any(brand.lower() in (t.text or "").lower() for t in c.tweets)
    ]
    pairs = build_customer_support_pairs(df, brand=brand)

    inbound_count = int(len(inbound_mentions))
    outbound_count = int(len(brand_out))
    total_brand_related = int(
        len(
            df[
                (df["author_id"].astype(str) == brand)
                | df["text"].astype(str).str.contains(mention_pattern, na=False)
            ]
        )
    )

    # Response coverage: inbound mentions that receive at least one brand reply pair.
    covered_inbound_ids = {p.customer_tweet_id for p in pairs}
    response_coverage = (
        len(covered_inbound_ids) / inbound_count if inbound_count else 0.0
    )

    # Approximate issue diversity via unique normalized message prefixes / lexical variety.
    texts = inbound_mentions["text"].astype(str)
    unique_texts = int(texts.nunique())
    avg_len = float(texts.str.len().mean()) if len(texts) else 0.0
    lexical_diversity = (unique_texts / inbound_count) if inbound_count else 0.0

    golden_suitability = "poor"
    if 150 <= inbound_count:
        golden_suitability = "good" if inbound_count >= 250 else "borderline"
    if inbound_count >= 500 and response_coverage >= 0.3:
        golden_suitability = "strong"

    suitability_intent = _score_intent(inbound_count, lexical_diversity, unique_texts)
    suitability_retrieval = _score_retrieval(len(pairs), response_coverage)

    return {
        "brand": brand,
        "tweets_total_brand_related": total_brand_related,
        "inbound_customer_tweets_mentioning_brand": inbound_count,
        "outbound_support_tweets": outbound_count,
        "conversations_brand_related": len(brand_convs),
        "customer_support_reply_pairs": len(pairs),
        "response_coverage_approx": round(response_coverage, 4),
        "unique_inbound_texts": unique_texts,
        "avg_inbound_text_length": round(avg_len, 2),
        "lexical_diversity_unique_over_n": round(lexical_diversity, 4),
        "suitability_intent_classification": suitability_intent,
        "suitability_historical_retrieval": suitability_retrieval,
        "suitability_golden_set_150_250": golden_suitability,
        "notes": [
            "Inbound count uses @brand mentions in text; anonymized mentions may undercount.",
            "Conversation count uses reconstructed threads containing the brand.",
            "These metrics are estimates for ranking, not labels.",
        ],
    }


def _score_intent(inbound_count: int, lexical_diversity: float, unique_texts: int) -> str:
    if inbound_count < 200 or unique_texts < 100:
        return "weak"
    if inbound_count >= 1000 and lexical_diversity >= 0.5:
        return "strong"
    if inbound_count >= 400:
        return "moderate"
    return "limited"


def _score_retrieval(n_pairs: int, coverage: float) -> str:
    if n_pairs < 100:
        return "weak"
    if n_pairs >= 500 and coverage >= 0.4:
        return "strong"
    if n_pairs >= 200:
        return "moderate"
    return "limited"


def rank_brands(df: pd.DataFrame, top_n: int = 15) -> list[dict[str, Any]]:
    """Rank candidate brands by a transparent composite score."""
    brand_counts = identify_brand_authors(df)
    candidates = list(brand_counts.head(top_n).index)
    analyses = [analyze_brand(df, brand) for brand in candidates]

    def score(row: dict[str, Any]) -> float:
        # Transparent heuristic for ranking only — not a claim of optimality.
        return (
            0.35 * row["inbound_customer_tweets_mentioning_brand"]
            + 0.35 * row["customer_support_reply_pairs"]
            + 0.20 * row["conversations_brand_related"]
            + 0.10 * (row["response_coverage_approx"] * 1000)
        )

    for row in analyses:
        row["composite_score"] = round(score(row), 3)
    analyses.sort(key=lambda r: r["composite_score"], reverse=True)
    for i, row in enumerate(analyses, start=1):
        row["rank"] = i
    return analyses


def recommend_brand(ranked: list[dict[str, Any]]) -> dict[str, Any]:
    """Pick top-ranked brand and document trade-offs."""
    if not ranked:
        return {
            "status": "NOT_AVAILABLE",
            "brand": None,
            "evidence": "No brand candidates found in the provided data.",
            "trade_offs": [],
        }
    top = ranked[0]
    runners = [r["brand"] for r in ranked[1:4]]
    return {
        "status": "RECOMMENDED_FROM_DATA",
        "brand": top["brand"],
        "evidence": {
            "rank": top["rank"],
            "composite_score": top["composite_score"],
            "inbound_customer_tweets_mentioning_brand": top[
                "inbound_customer_tweets_mentioning_brand"
            ],
            "customer_support_reply_pairs": top["customer_support_reply_pairs"],
            "conversations_brand_related": top["conversations_brand_related"],
            "response_coverage_approx": top["response_coverage_approx"],
            "suitability_intent_classification": top["suitability_intent_classification"],
            "suitability_historical_retrieval": top["suitability_historical_retrieval"],
            "suitability_golden_set_150_250": top["suitability_golden_set_150_250"],
        },
        "trade_offs": [
            "Composite score weights volume and reply pairs; niche brands with cleaner "
            "issue structure may score lower but be easier to label.",
            "Mention-based inbound detection can undercount when @handles are anonymized.",
            f"Strong runners-up to revisit: {runners}",
            "SpotifyCares is only preferred if it ranks highly on these observed metrics.",
        ],
        "alternatives": runners,
    }


def run_brand_analysis(
    data_path: Path | str,
    output_path: Path | str | None = None,
    sample_size: int | None = None,
    random_seed: int = 42,
    top_n: int = 15,
) -> dict[str, Any]:
    df = load_tweets(data_path, sample_size=sample_size, random_seed=random_seed)
    ranked = rank_brands(df, top_n=top_n)
    recommendation = recommend_brand(ranked)
    result = {
        "status": "OBSERVED",
        "source_path": str(Path(data_path).resolve()),
        "n_rows_analyzed": int(len(df)),
        "ranked_candidates": ranked,
        "recommendation": recommendation,
    }
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2, default=str)
    return result
