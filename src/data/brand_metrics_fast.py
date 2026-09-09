"""Lightweight brand metrics that avoid full conversation reconstruction."""

from __future__ import annotations

import re
from typing import Any

import pandas as pd


def lightweight_brand_metrics(df: pd.DataFrame, brand: str) -> dict[str, Any]:
    """Compute pair/inbound/outbound stats without connected-component reconstruction.

    A pair is counted when an outbound brand tweet replies to an inbound tweet
    present in ``df``.
    """
    brand = str(brand)
    mention = re.compile(rf"@{re.escape(brand)}\b", re.IGNORECASE)
    authored = df["author_id"].astype(str) == brand
    outbound = df[authored & (df["inbound"] == False)]
    inbound = df[df["inbound"] == True]
    inbound_mentions = inbound[inbound["text"].astype(str).str.contains(mention, na=False)]

    by_id = df.drop_duplicates(subset=["tweet_id"]).set_index(df.drop_duplicates(subset=["tweet_id"])["tweet_id"].astype(str))
    # Fix index properly
    uniq = df.drop_duplicates(subset=["tweet_id"], keep="first").copy()
    uniq["tweet_id"] = uniq["tweet_id"].astype(str)
    by_id = uniq.set_index("tweet_id", drop=False)

    pair_customer_ids: set[str] = set()
    n_pairs = 0
    for row in outbound.itertuples(index=False):
        parent = row.in_response_to_tweet_id
        if parent is None or (isinstance(parent, float) and pd.isna(parent)):
            continue
        pid = str(parent)
        if pid not in by_id.index:
            continue
        parent_row = by_id.loc[pid]
        # loc can return Series
        inbound_flag = bool(parent_row["inbound"]) if not isinstance(parent_row, pd.DataFrame) else bool(parent_row.iloc[0]["inbound"])
        if inbound_flag:
            n_pairs += 1
            pair_customer_ids.add(pid)

    inbound_count = int(len(inbound_mentions))
    coverage = (len(pair_customer_ids) / inbound_count) if inbound_count else 0.0
    texts = inbound_mentions["text"].astype(str)
    unique_texts = int(texts.nunique())
    lexical = (unique_texts / inbound_count) if inbound_count else 0.0
    avg_len = float(texts.str.len().mean()) if len(texts) else 0.0

    golden_ok = inbound_count >= 250 and len(pair_customer_ids) >= 200
    intent_ok = unique_texts >= 150 and lexical >= 0.4
    retrieval_ok = len(pair_customer_ids) >= 300 and coverage >= 0.25

    # Prefer mid-size: soft penalty above 100k outbound for taxonomy focus.
    volume_penalty = 0.0
    outbound_n = int(len(outbound))
    if outbound_n > 100_000:
        volume_penalty = 0.08
    elif outbound_n > 80_000:
        volume_penalty = 0.04

    suitability = (
        0.25 * min(inbound_count, 5000) / 5000
        + 0.30 * min(len(pair_customer_ids), 5000) / 5000
        + 0.15 * min(coverage, 1.0)
        + 0.15 * min(lexical, 1.0)
        + 0.15 * (1.0 if golden_ok else 0.0)
        - volume_penalty
    )

    return {
        "brand": brand,
        "tweets_in_brand_extract": int(len(df)),
        "outbound_support_tweets": outbound_n,
        "inbound_customer_tweets_mentioning_brand": inbound_count,
        "customer_support_reply_pairs": int(len(pair_customer_ids)),
        "customer_support_reply_links_raw": int(n_pairs),
        "response_coverage_approx": round(coverage, 4),
        "unique_inbound_texts": unique_texts,
        "avg_inbound_text_length": round(avg_len, 2),
        "lexical_diversity_unique_over_n": round(lexical, 4),
        "golden_set_feasible_150_250": bool(golden_ok),
        "intent_diversity_proxy_ok": bool(intent_ok),
        "retrieval_suitability_ok": bool(retrieval_ok),
        "suitability_score": round(float(suitability), 4),
        "volume_penalty_applied": volume_penalty,
        "notes": [
            "Lightweight metrics: no connected-component reconstruction.",
            "Pairs = unique inbound parents replied to by the brand.",
            "Mega-brand volume penalty discourages selecting purely by size.",
        ],
    }


def extract_brand_from_frame(df: pd.DataFrame, brand: str) -> pd.DataFrame:
    brand = str(brand)
    mention = re.compile(rf"@{re.escape(brand)}\b", re.IGNORECASE)
    authored = df["author_id"].astype(str) == brand
    brand_ids = set(df.loc[authored, "tweet_id"].astype(str))
    parent_ids = set(df.loc[authored, "in_response_to_tweet_id"].dropna().astype(str))
    keep = brand_ids | parent_ids
    mask = (
        authored
        | df["text"].astype(str).str.contains(mention, na=False)
        | df["tweet_id"].astype(str).isin(keep)
        | df["in_response_to_tweet_id"].astype(str).isin(keep)
    )
    return df.loc[mask].drop_duplicates(subset=["tweet_id"], keep="first").copy()
