"""Dataset loading utilities."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.schema import coerce_twcs_dtypes, validate_schema


def load_tweets(
    path: Path | str,
    sample_size: int | None = None,
    random_seed: int = 42,
    coerce: bool = True,
    validate: bool = True,
) -> pd.DataFrame:
    """Load a TWCS-style CSV from disk.

    Parameters
    ----------
    path:
        Path to sample.csv or twcs.csv.
    sample_size:
        If set, randomly sample this many rows after load (for development).
        This is NOT a substitute for conversation-aware splitting.
    random_seed:
        Seed used when sample_size is set.
    coerce:
        Apply dtype coercion helpers.
    validate:
        Raise if required schema columns are missing.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path, low_memory=False)
    result = validate_schema(df, strict=True)
    if validate and not result.ok:
        raise ValueError(
            "Schema validation failed. Missing columns: "
            f"{result.missing_columns}. Observed: {result.observed_columns}"
        )

    if coerce:
        df = coerce_twcs_dtypes(df)

    if sample_size is not None:
        if sample_size < 0:
            raise ValueError("sample_size must be >= 0")
        if sample_size < len(df):
            df = df.sample(n=sample_size, random_state=random_seed).reset_index(drop=True)

    return df


def describe_file(path: Path | str) -> dict:
    """Return basic filesystem metadata for a dataset file."""
    path = Path(path)
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "filename": path.name,
        "size_bytes": stat.st_size,
        "size_mb": round(stat.st_size / (1024 * 1024), 3),
        "exists": True,
    }
