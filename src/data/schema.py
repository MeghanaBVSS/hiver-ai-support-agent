"""Dataset schema definitions and validation for TWCS-style CSVs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd

# Columns documented by the Kaggle "Customer Support on Twitter" dataset.
# We still validate against the actual file at load time and do not assume extras.
EXPECTED_COLUMNS: tuple[str, ...] = (
    "tweet_id",
    "author_id",
    "inbound",
    "created_at",
    "text",
    "response_tweet_id",
    "in_response_to_tweet_id",
)

REQUIRED_COLUMNS: tuple[str, ...] = EXPECTED_COLUMNS


@dataclass(frozen=True)
class SchemaValidationResult:
    ok: bool
    missing_columns: tuple[str, ...]
    unexpected_columns: tuple[str, ...]
    observed_columns: tuple[str, ...]
    dtypes: dict[str, str]
    notes: tuple[str, ...]

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "missing_columns": list(self.missing_columns),
            "unexpected_columns": list(self.unexpected_columns),
            "observed_columns": list(self.observed_columns),
            "dtypes": self.dtypes,
            "notes": list(self.notes),
        }


def validate_schema(df: pd.DataFrame, strict: bool = True) -> SchemaValidationResult:
    """Validate that a dataframe has the expected TWCS columns.

    Parameters
    ----------
    df:
        Loaded tweet dataframe.
    strict:
        If True, missing required columns make ok=False.
        Extra columns are reported but do not fail validation.
    """
    observed = tuple(str(c) for c in df.columns)
    missing = tuple(c for c in REQUIRED_COLUMNS if c not in df.columns)
    unexpected = tuple(c for c in observed if c not in EXPECTED_COLUMNS)
    notes: list[str] = []

    if missing:
        notes.append(f"Missing required columns: {missing}")
    if unexpected:
        notes.append(f"Unexpected extra columns present: {unexpected}")

    dtypes = {col: str(df[col].dtype) for col in observed}
    ok = (not missing) if strict else True
    return SchemaValidationResult(
        ok=ok,
        missing_columns=missing,
        unexpected_columns=unexpected,
        observed_columns=observed,
        dtypes=dtypes,
        notes=tuple(notes),
    )


def coerce_twcs_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Best-effort dtype coercion after schema validation.

    Assumptions (documented):
    - tweet_id is treated as string to avoid float precision issues when IDs
      arrive as floats due to missing values in related columns.
    - inbound may arrive as bool, int, or string; normalized to nullable boolean.
    - response_tweet_id may be comma-separated and is kept as string.
    - in_response_to_tweet_id may be float/NaN; stored as nullable string.
    """
    out = df.copy()
    if "tweet_id" in out.columns:
        out["tweet_id"] = out["tweet_id"].map(_id_to_str)
    if "author_id" in out.columns:
        out["author_id"] = out["author_id"].astype(str)
    if "inbound" in out.columns:
        out["inbound"] = out["inbound"].map(_to_bool)
    if "text" in out.columns:
        out["text"] = out["text"].astype("string")
    if "response_tweet_id" in out.columns:
        out["response_tweet_id"] = out["response_tweet_id"].map(_optional_str)
    if "in_response_to_tweet_id" in out.columns:
        out["in_response_to_tweet_id"] = out["in_response_to_tweet_id"].map(_id_to_str)
    if "created_at" in out.columns:
        # Keep original string; parse elsewhere when needed.
        out["created_at"] = out["created_at"].astype("string")
    return out


def _id_to_str(value: object) -> str | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, str):
        text = value.strip()
        if text == "" or text.lower() == "nan":
            return None
        # Handle "123.0" float-string artifacts.
        if text.endswith(".0") and text[:-2].isdigit():
            return text[:-2]
        return text
    if isinstance(value, (int,)):
        return str(value)
    if isinstance(value, float):
        if pd.isna(value):
            return None
        return str(int(value)) if value.is_integer() else str(value)
    return str(value)


def _optional_str(value: object) -> str | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    if text == "" or text.lower() == "nan":
        return None
    return text


def _to_bool(value: object) -> bool | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "1", "t", "yes"}:
        return True
    if text in {"false", "0", "f", "no"}:
        return False
    return None


def parse_response_ids(value: object) -> list[str]:
    """Parse comma-separated response_tweet_id values into a list of IDs."""
    text = _optional_str(value)
    if text is None:
        return []
    parts: list[str] = []
    for part in text.split(","):
        cleaned = _id_to_str(part.strip())
        if cleaned is not None:
            parts.append(cleaned)
    return parts


def filter_inbound(df: pd.DataFrame, inbound: bool = True) -> pd.DataFrame:
    """Return rows matching inbound flag after boolean coercion."""
    if "inbound" not in df.columns:
        raise KeyError("Column 'inbound' is required to filter inbound tweets")
    mask = df["inbound"].map(_to_bool) == inbound
    return df.loc[mask].copy()


def assert_columns(df: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"DataFrame missing required columns: {missing}")
