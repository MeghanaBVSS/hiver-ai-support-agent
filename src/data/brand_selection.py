"""Phase 1 brand ranking on large TWCS files without reconstructing all conversations globally."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.loader import load_tweets
from src.data.schema import coerce_twcs_dtypes
from src.preprocessing.conversations import (
    build_customer_support_pairs,
    reconstruct_conversations,
)

_NUMERIC = re.compile(r"^\d+$")


def count_outbound_brands(path: Path | str, chunksize: int = 200_000) -> pd.Series:
    """Count outbound non-numeric author_ids (brand handles) via chunked scan."""
    counts: dict[str, int] = {}
    for chunk in pd.read_csv(path, chunksize=chunksize, low_memory=False):
        out = chunk[chunk["inbound"] == False]
        authors = out["author_id"].astype(str)
        brands = authors[~authors.str.fullmatch(r"\d+")]
        for brand, n in brands.value_counts().items():
            counts[str(brand)] = counts.get(str(brand), 0) + int(n)
    return pd.Series(counts).sort_values(ascending=False)


def extract_brand_tweets(path: Path | str, brand: str, chunksize: int = 200_000) -> pd.DataFrame:
    """Extract tweets authored by brand OR mentioning @brand OR replied-to by brand.

    Pass 1: collect brand outbound tweet_ids and their parent ids.
    Pass 2: keep rows authored by brand, mentioning brand, or matching those ids.
    """
    brand = str(brand)
    mention = re.compile(rf"@{re.escape(brand)}\b", re.IGNORECASE)
    brand_tweet_ids: set[str] = set()
    parent_ids: set[str] = set()

    for chunk in pd.read_csv(path, chunksize=chunksize, low_memory=False):
        chunk = coerce_twcs_dtypes(chunk)
        authored = chunk[chunk["author_id"].astype(str) == brand]
        for tid in authored["tweet_id"].astype(str):
            brand_tweet_ids.add(tid)
        for pid in authored["in_response_to_tweet_id"].dropna().astype(str):
            if pid not in {"None", "nan", ""}:
                parent_ids.add(pid)

    keep_ids = brand_tweet_ids | parent_ids
    parts: list[pd.DataFrame] = []
    for chunk in pd.read_csv(path, chunksize=chunksize, low_memory=False):
        chunk = coerce_twcs_dtypes(chunk)
        mask = (
            (chunk["author_id"].astype(str) == brand)
            | chunk["text"].astype(str).str.contains(mention, na=False)
            | chunk["tweet_id"].astype(str).isin(keep_ids)
        )
        # Also keep replies whose parent is a kept id (one hop expansion)
        parents = chunk["in_response_to_tweet_id"].astype(str)
        mask = mask | parents.isin(keep_ids)
        sub = chunk.loc[mask]
        if len(sub):
            parts.append(sub)
    if not parts:
        return coerce_twcs_dtypes(pd.DataFrame(columns=[
            "tweet_id", "author_id", "inbound", "created_at", "text",
            "response_tweet_id", "in_response_to_tweet_id",
        ]))
    out = pd.concat(parts, ignore_index=True)
    out = out.drop_duplicates(subset=["tweet_id"], keep="first")
    return out


def analyze_brand_frame(df: pd.DataFrame, brand: str) -> dict[str, Any]:
    brand = str(brand)
    mention = re.compile(rf"@{re.escape(brand)}\b", re.IGNORECASE)
    outbound = df[(df["author_id"].astype(str) == brand) & (df["inbound"] == False)]
    inbound = df[df["inbound"] == True]
    inbound_mentions = inbound[inbound["text"].astype(str).str.contains(mention, na=False)]
    conversations = reconstruct_conversations(df)
    brand_convs = [
        c for c in conversations
        if c.brand == brand or brand in c.participant_authors
        or any(brand.lower() in (t.text or "").lower() for t in c.tweets)
    ]
    pairs = build_customer_support_pairs(df, brand=brand, first_support_only=True)
    inbound_count = int(len(inbound_mentions))
    covered = {p.customer_tweet_id for p in pairs}
    coverage = (len(covered) / inbound_count) if inbound_count else 0.0
    lengths = [c.n_turns for c in brand_convs]
    texts = inbound_mentions["text"].astype(str)
    unique_texts = int(texts.nunique())
    lexical = (unique_texts / inbound_count) if inbound_count else 0.0
    avg_len = float(texts.str.len().mean()) if len(texts) else 0.0
    avg_turns = float(sum(lengths) / len(lengths)) if lengths else 0.0

    # Suitability: not pure volume — prefer enough pairs, diversity, golden feasibility.
    golden_ok = inbound_count >= 250 and len(pairs) >= 200
    intent_ok = unique_texts >= 150 and lexical >= 0.4
    retrieval_ok = len(pairs) >= 300 and coverage >= 0.25

    suitability_score = (
        0.25 * min(inbound_count, 5000) / 5000
        + 0.30 * min(len(pairs), 5000) / 5000
        + 0.15 * min(coverage, 1.0)
        + 0.15 * min(lexical, 1.0)
        + 0.15 * (1.0 if golden_ok else 0.0)
    )

    return {
        "brand": brand,
        "tweets_in_brand_extract": int(len(df)),
        "outbound_support_tweets": int(len(outbound)),
        "inbound_customer_tweets_mentioning_brand": inbound_count,
        "conversations_brand_related": len(brand_convs),
        "customer_support_reply_pairs": len(pairs),
        "response_coverage_approx": round(coverage, 4),
        "avg_conversation_turns": round(avg_turns, 3),
        "unique_inbound_texts": unique_texts,
        "avg_inbound_text_length": round(avg_len, 2),
        "lexical_diversity_unique_over_n": round(lexical, 4),
        "golden_set_feasible_150_250": bool(golden_ok),
        "intent_diversity_proxy_ok": bool(intent_ok),
        "retrieval_suitability_ok": bool(retrieval_ok),
        "suitability_score": round(float(suitability_score), 4),
        "notes": [
            "Suitability favors reply-pair coverage and diversity, not raw tweet volume alone.",
            "Inbound counts use @brand mentions; anonymized mentions may undercount.",
        ],
    }


def rank_brands_from_path(
    path: Path | str,
    top_n_outbound: int = 20,
    analyze_top: int = 12,
    chunksize: int = 200_000,
) -> dict[str, Any]:
    path = Path(path)
    outbound = count_outbound_brands(path, chunksize=chunksize)
    candidates = list(outbound.head(top_n_outbound).index)
    analyses: list[dict[str, Any]] = []
    # Analyze top outbound brands; stop early candidates for cost but cover analyze_top.
    for brand in candidates[:analyze_top]:
        df = extract_brand_tweets(path, brand, chunksize=chunksize)
        row = analyze_brand_frame(df, brand)
        row["outbound_rank_by_volume"] = int(list(outbound.index).index(brand) + 1)
        row["outbound_volume"] = int(outbound[brand])
        analyses.append(row)

    # Rank by suitability, not volume.
    analyses.sort(key=lambda r: (-r["suitability_score"], -r["customer_support_reply_pairs"]))
    for i, row in enumerate(analyses, start=1):
        row["rank"] = i

    recommendation = _recommend(analyses)
    return {
        "status": "OBSERVED",
        "source_path": str(path.resolve()),
        "n_outbound_brand_authors": int(len(outbound)),
        "top_outbound_by_volume": {k: int(v) for k, v in outbound.head(25).items()},
        "ranked_candidates": analyses,
        "recommendation": recommendation,
    }


def _recommend(ranked: list[dict[str, Any]]) -> dict[str, Any]:
    if not ranked:
        return {"status": "NOT_AVAILABLE", "brand": None}
    # Prefer brands that pass golden + retrieval gates; else highest suitability.
    gated = [
        r for r in ranked
        if r["golden_set_feasible_150_250"] and r["retrieval_suitability_ok"]
    ]
    top = gated[0] if gated else ranked[0]
    alts = [r for r in ranked if r["brand"] != top["brand"]][:3]
    why_rejected = []
    for alt in alts:
        reasons = []
        if alt["suitability_score"] < top["suitability_score"]:
            reasons.append("lower suitability_score")
        if not alt["golden_set_feasible_150_250"]:
            reasons.append("golden set not clearly feasible")
        if alt["outbound_volume"] > top["outbound_volume"] and alt["rank"] > top["rank"]:
            reasons.append("higher volume but weaker coverage/diversity mix")
        why_rejected.append({"brand": alt["brand"], "reasons": reasons or ["ranked lower on suitability"]})

    return {
        "status": "RECOMMENDED_FROM_DATA",
        "brand": top["brand"],
        "why_selected": {
            "suitability_score": top["suitability_score"],
            "customer_support_reply_pairs": top["customer_support_reply_pairs"],
            "inbound_customer_tweets_mentioning_brand": top["inbound_customer_tweets_mentioning_brand"],
            "response_coverage_approx": top["response_coverage_approx"],
            "lexical_diversity_unique_over_n": top["lexical_diversity_unique_over_n"],
            "golden_set_feasible_150_250": top["golden_set_feasible_150_250"],
            "outbound_rank_by_volume": top["outbound_rank_by_volume"],
            "note": "Selected on suitability (pairs/coverage/diversity/golden feasibility), not max volume.",
        },
        "top_alternatives_rejected": why_rejected,
        "trade_offs": [
            "High-volume brands (e.g. AmazonHelp) may dominate volume rankings but be broader/noisier for a 6–12 intent taxonomy.",
            "Smaller brands may be cleaner but lack enough pairs for retrieval + golden set.",
            "Mention-based inbound detection can undercount anonymized @handles.",
        ],
    }


def write_brand_selection_csv(ranked: list[dict[str, Any]], path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(ranked).to_csv(path, index=False)
    return path
