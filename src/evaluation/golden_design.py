"""Golden-set sampling design and contamination utilities.

Phase 0 designs the methodology. Final labeled golden set is NOT built unless
manually labeled data already exists.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from src.evaluation.leakage import check_golden_contamination


@dataclass
class GoldenSamplingPlan:
    """Documented methodology for a 150–250 example golden set."""

    target_size_min: int = 150
    target_size_max: int = 250
    strata: dict[str, str] = field(
        default_factory=lambda: {
            "intents": "Approximate equal allocation across proposed intents; "
            "oversample rare intents if needed for minimum support.",
            "common_vs_uncommon": "Include both high-frequency issue templates and long-tail cases.",
            "ambiguous": "Reserve ~10–15% for multi-intent / unclear asks.",
            "difficult": "Include noisy text, heavy slang, and partial context.",
            "escalation_worthy": "Reserve ~20% for billing/security/account-specific cases.",
            "message_length": "Stratify short / medium / long customer messages.",
            "multi_intent": "Include examples where two intents co-occur when present.",
        }
    )
    labeling_guidelines: list[str] = field(
        default_factory=lambda: [
            "Labels are assigned by humans; clustering output is proposal-only.",
            "Each example gets: intent_id (or multi-label list), escalate (bool), "
            "escalate_reason, notes, difficulty (easy/medium/hard).",
            "Ambiguous cases may use intent_id='ambiguous' plus free-text note.",
            "Golden examples must be isolated from train/valid/retrieval/prompt pools "
            "before any model fitting or index build.",
            "Record annotator id and timestamp for later agreement analysis.",
        ]
    )
    isolation_rules: list[str] = field(
        default_factory=lambda: [
            "Remove golden conversation_ids from train/valid/test modeling sets used for fitting.",
            "Exclude golden customer_tweet_ids from retrieval index evidence.",
            "Do not use golden texts as few-shot prompt exemplars.",
            "Run check_golden_contamination() in CI before evaluation.",
        ]
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def propose_stratified_sample_ids(
    frame: pd.DataFrame,
    intent_col: str,
    id_col: str,
    target_size: int = 200,
    random_seed: int = 42,
    escalate_col: str | None = None,
) -> list[str]:
    """Propose IDs for labeling using available strata columns.

    This does not create labels. It only selects candidates.
    """
    if target_size <= 0:
        return []
    df = frame.copy()
    if intent_col not in df.columns or id_col not in df.columns:
        raise KeyError("intent_col and id_col must exist")

    rng = np.random.default_rng(random_seed)
    intents = df[intent_col].astype(str).unique().tolist()
    per_intent = max(1, target_size // max(1, len(intents)))
    chosen: list[str] = []

    for intent in sorted(intents):
        subset = df[df[intent_col].astype(str) == intent]
        ids = subset[id_col].astype(str).tolist()
        rng.shuffle(ids)
        chosen.extend(ids[:per_intent])

    # Optionally boost escalation-worthy if column present.
    if escalate_col and escalate_col in df.columns:
        esc_ids = (
            df[df[escalate_col] == True][id_col].astype(str).tolist()  # noqa: E712
        )
        rng.shuffle(esc_ids)
        for i in esc_ids[: max(1, target_size // 5)]:
            if i not in chosen:
                chosen.append(i)

    # Fill remainder randomly.
    remaining = [i for i in df[id_col].astype(str).tolist() if i not in chosen]
    rng.shuffle(remaining)
    while len(chosen) < min(target_size, len(df)) and remaining:
        chosen.append(remaining.pop())

    return chosen[:target_size]


def write_golden_design_artifact(path: Path, plan: GoldenSamplingPlan | None = None) -> Path:
    plan = plan or GoldenSamplingPlan()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(plan.to_dict(), fh, indent=2)
    return path


def contamination_check_utility(
    golden_ids: Iterable[str],
    train_ids: Iterable[str],
    retrieval_ids: Iterable[str],
    valid_ids: Iterable[str] | None = None,
    prompt_example_ids: Iterable[str] | None = None,
) -> dict[str, Any]:
    report = check_golden_contamination(
        golden_ids=golden_ids,
        train_ids=train_ids,
        valid_ids=valid_ids,
        retrieval_ids=retrieval_ids,
        prompt_example_ids=prompt_example_ids,
    )
    return report.to_dict()
