"""Train/validation/test splitting strategies.

Default: conversation-level split to prevent cross-split leakage.
Temporal split is also available and documented with trade-offs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

import numpy as np
import pandas as pd

SplitName = Literal["train", "valid", "test"]


@dataclass(frozen=True)
class SplitResult:
    train_ids: tuple[str, ...]
    valid_ids: tuple[str, ...]
    test_ids: tuple[str, ...]
    strategy: str

    def assignment(self) -> dict[str, str]:
        out = {i: "train" for i in self.train_ids}
        out.update({i: "valid" for i in self.valid_ids})
        out.update({i: "test" for i in self.test_ids})
        return out

    def to_dict(self) -> dict:
        return {
            "strategy": self.strategy,
            "n_train": len(self.train_ids),
            "n_valid": len(self.valid_ids),
            "n_test": len(self.test_ids),
            "train_ids": list(self.train_ids),
            "valid_ids": list(self.valid_ids),
            "test_ids": list(self.test_ids),
        }


def conversation_level_split(
    conversation_ids: Iterable[str],
    train_ratio: float = 0.7,
    valid_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42,
) -> SplitResult:
    """Split unique conversation IDs without overlap."""
    ids = sorted({str(x) for x in conversation_ids})
    if not ids:
        return SplitResult((), (), (), strategy="conversation")

    total = train_ratio + valid_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError("ratios must sum to 1.0")

    rng = np.random.default_rng(random_seed)
    shuffled = ids.copy()
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * train_ratio)
    n_valid = int(n * valid_ratio)
    # Remainder goes to test to ensure all IDs assigned.
    train = tuple(shuffled[:n_train])
    valid = tuple(shuffled[n_train : n_train + n_valid])
    test = tuple(shuffled[n_train + n_valid :])
    return SplitResult(train, valid, test, strategy="conversation")


def temporal_split(
    conversation_id_to_time: dict[str, pd.Timestamp | str | None],
    train_ratio: float = 0.7,
    valid_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> SplitResult:
    """Split conversations by earliest timestamp (sorted ascending).

    Trade-offs vs conversation random split:
    - Pros: closer to production (train on past, test on future); reduces
      temporal leakage from evolving canned replies / product changes.
    - Cons: distribution shift can dominate metrics; rare intents may be
      unevenly allocated; requires reliable timestamps.
    """
    total = train_ratio + valid_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError("ratios must sum to 1.0")

    items: list[tuple[str, pd.Timestamp]] = []
    for cid, ts in conversation_id_to_time.items():
        parsed = pd.to_datetime(ts, errors="coerce", utc=True)
        if pd.isna(parsed):
            # Put unparseable at the end deterministically by id.
            parsed = pd.Timestamp.max.tz_localize("UTC")
        items.append((str(cid), parsed))
    items.sort(key=lambda x: (x[1], x[0]))
    ids = [cid for cid, _ in items]
    n = len(ids)
    n_train = int(n * train_ratio)
    n_valid = int(n * valid_ratio)
    train = tuple(ids[:n_train])
    valid = tuple(ids[n_train : n_train + n_valid])
    test = tuple(ids[n_train + n_valid :])
    return SplitResult(train, valid, test, strategy="temporal")


def assign_split_column(
    df: pd.DataFrame,
    id_column: str,
    split: SplitResult,
) -> pd.DataFrame:
    """Add a 'split' column based on conversation (or other) IDs."""
    out = df.copy()
    mapping = split.assignment()
    out["split"] = out[id_column].astype(str).map(mapping)
    if out["split"].isna().any():
        missing = out.loc[out["split"].isna(), id_column].astype(str).unique()[:10]
        raise ValueError(f"IDs missing from split assignment (sample): {list(missing)}")
    return out
