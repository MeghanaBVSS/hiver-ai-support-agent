"""Reproducible dataset audit.

Produces machine-readable JSON summarizing schema, quality, and structure.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.loader import describe_file, load_tweets
from src.data.schema import parse_response_ids, validate_schema


def _safe_nunique(series: pd.Series) -> int:
    return int(series.nunique(dropna=True))


def audit_dataframe(df: pd.DataFrame, source_path: Path | None = None) -> dict[str, Any]:
    """Audit an in-memory TWCS dataframe."""
    schema = validate_schema(df, strict=True)
    report: dict[str, Any] = {
        "status": "OBSERVED",
        "source_path": str(source_path.resolve()) if source_path else None,
        "file": describe_file(source_path) if source_path and source_path.exists() else None,
        "row_count": int(len(df)),
        "column_count": int(df.shape[1]),
        "columns": list(df.columns),
        "schema": schema.to_dict(),
        "dtypes": {c: str(df[c].dtype) for c in df.columns},
        "missing_values": {c: int(df[c].isna().sum()) for c in df.columns},
        "missing_rate": {
            c: float(df[c].isna().mean()) for c in df.columns
        },
    }

    # Duplicate analysis
    dup_rows = int(df.duplicated().sum())
    report["duplicate_rows"] = dup_rows
    if "tweet_id" in df.columns:
        tweet_id_dupes = int(df["tweet_id"].duplicated().sum())
        report["duplicate_tweet_ids"] = tweet_id_dupes
        report["unique_tweet_ids"] = _safe_nunique(df["tweet_id"])
    else:
        report["duplicate_tweet_ids"] = None
        report["unique_tweet_ids"] = None

    # Inbound / outbound
    if "inbound" in df.columns:
        inbound_counts = df["inbound"].value_counts(dropna=False).to_dict()
        report["inbound_distribution"] = {
            str(k): int(v) for k, v in inbound_counts.items()
        }
    else:
        report["inbound_distribution"] = None

    # Authors / brands
    if "author_id" in df.columns:
        report["unique_authors"] = _safe_nunique(df["author_id"])
        # Brand hypothesis: outbound authors with non-numeric handles are brands.
        outbound = df[df["inbound"] == False] if "inbound" in df.columns else df.iloc[0:0]
        brand_authors = outbound["author_id"].astype(str)
        # Keep authors that are not purely numeric anonymized IDs.
        brand_mask = ~brand_authors.str.fullmatch(r"\d+")
        brands = brand_authors[brand_mask]
        report["unique_brand_authors_outbound_non_numeric"] = int(brands.nunique())
        top_brands = brands.value_counts().head(25)
        report["top_outbound_brands"] = {str(k): int(v) for k, v in top_brands.items()}
    else:
        report["unique_authors"] = None
        report["unique_brand_authors_outbound_non_numeric"] = None
        report["top_outbound_brands"] = {}

    # Timestamps
    if "created_at" in df.columns:
        parsed = pd.to_datetime(
            df["created_at"], errors="coerce", utc=True, format="mixed"
        )
        report["timestamps"] = {
            "parseable": int(parsed.notna().sum()),
            "unparseable": int(parsed.isna().sum()),
            "min": parsed.min().isoformat() if parsed.notna().any() else None,
            "max": parsed.max().isoformat() if parsed.notna().any() else None,
        }
    else:
        report["timestamps"] = None

    # Response relationships
    if "response_tweet_id" in df.columns:
        has_response = df["response_tweet_id"].notna() & (df["response_tweet_id"].astype(str) != "")
        report["rows_with_response_tweet_id"] = int(has_response.sum())
    else:
        report["rows_with_response_tweet_id"] = None

    if "in_response_to_tweet_id" in df.columns:
        has_parent = df["in_response_to_tweet_id"].notna()
        report["rows_with_in_response_to_tweet_id"] = int(has_parent.sum())
        # Coverage: parent IDs that exist in this dataframe.
        parent_ids = set(df.loc[has_parent, "in_response_to_tweet_id"].astype(str))
        tweet_ids = set(df["tweet_id"].astype(str)) if "tweet_id" in df.columns else set()
        missing_parents = parent_ids - tweet_ids
        report["unique_parent_refs"] = len(parent_ids)
        report["missing_parent_refs_in_file"] = len(missing_parents)
        report["parent_coverage_in_file"] = (
            1.0 - (len(missing_parents) / len(parent_ids)) if parent_ids else None
        )
    else:
        report["rows_with_in_response_to_tweet_id"] = None

    # Quick response_id parse sanity
    if "response_tweet_id" in df.columns:
        multi = 0
        for value in df["response_tweet_id"].dropna().head(5000):
            if len(parse_response_ids(value)) > 1:
                multi += 1
        report["multi_response_id_rows_in_first_5k_non_null"] = multi

    report["quality_issues"] = _quality_issues(report)
    return report


def _quality_issues(report: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    if report.get("duplicate_rows", 0):
        issues.append(f"Found {report['duplicate_rows']} fully duplicate rows.")
    if report.get("duplicate_tweet_ids", 0):
        issues.append(f"Found {report['duplicate_tweet_ids']} duplicate tweet_id values.")
    missing = report.get("missing_values") or {}
    for col, count in missing.items():
        if count:
            issues.append(f"Column '{col}' has {count} missing values.")
    ts = report.get("timestamps") or {}
    if ts.get("unparseable"):
        issues.append(f"{ts['unparseable']} timestamps failed to parse.")
    if report.get("missing_parent_refs_in_file"):
        issues.append(
            f"{report['missing_parent_refs_in_file']} parent tweet refs are missing "
            "from this file (orphans / incomplete extract)."
        )
    if not report.get("schema", {}).get("ok", False):
        issues.append("Schema validation failed.")
    return issues


def run_audit(
    data_path: Path | str,
    output_path: Path | str | None = None,
    sample_size: int | None = None,
    random_seed: int = 42,
) -> dict[str, Any]:
    """Load data, audit, optionally write JSON."""
    data_path = Path(data_path)
    df = load_tweets(data_path, sample_size=sample_size, random_seed=random_seed)
    report = audit_dataframe(df, source_path=data_path)
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, default=str)
    return report
