"""Leakage-safe retrieval corpus for hulu_support (train only)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from src.evaluation.leakage import check_retrieval_index_contamination
from src.preprocessing.text_normalize import normalize_tweet_text


@dataclass
class RetrievalCase:
    case_id: str
    conversation_id: str
    customer_message: str
    support_response: str
    intent: str | None = None
    timestamp: str | None = None
    prior_customer_context: list[str] = field(default_factory=list)
    customer_tweet_id: str | None = None
    support_tweet_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


FORBIDDEN_SPLITS = {"test", "golden"}


def build_train_retrieval_corpus(
    train_labeled_path: Path | str,
    *,
    golden_path: Path | str | None = None,
    pairs_split_path: Path | str | None = None,
) -> list[RetrievalCase]:
    """Build retrievable cases from TRAIN labeled examples only.

    Explicitly excludes golden/test conversations and tweet IDs.
    """
    train = pd.read_parquet(train_labeled_path)
    gold_tids: set[str] = set()
    gold_cids: set[str] = set()
    test_tids: set[str] = set()
    test_cids: set[str] = set()

    if golden_path and Path(golden_path).exists():
        golden = pd.read_csv(golden_path)
        gold_tids = set(golden["customer_tweet_id"].astype(str))
        if "conversation_id" in golden.columns:
            gold_cids = set(golden["conversation_id"].astype(str))

    if pairs_split_path and Path(pairs_split_path).exists():
        pairs = pd.read_parquet(pairs_split_path)
        test = pairs[pairs["split"] == "test"]
        test_tids = set(test["customer_tweet_id"].astype(str))
        test_cids = set(test["conversation_id"].astype(str))

    blocked_tids = gold_tids | test_tids
    blocked_cids = gold_cids | test_cids

    cases: list[RetrievalCase] = []
    for row in train.itertuples(index=False):
        cid = str(row.conversation_id)
        tid = str(row.customer_tweet_id)
        if tid in blocked_tids or cid in blocked_cids:
            continue
        ctx = getattr(row, "conversation_context", None)
        if isinstance(ctx, str):
            try:
                import ast

                ctx = ast.literal_eval(ctx)
            except Exception:
                ctx = []
        if ctx is None or (isinstance(ctx, float) and pd.isna(ctx)):
            ctx = []
        cases.append(
            RetrievalCase(
                case_id=f"case_{tid}",
                conversation_id=cid,
                customer_message=str(row.customer_message),
                support_response=str(row.support_response),
                intent=str(row.gold_intent) if hasattr(row, "gold_intent") else None,
                timestamp=str(getattr(row, "customer_timestamp", None) or getattr(row, "customer_created_at", "") or ""),
                prior_customer_context=list(ctx) if isinstance(ctx, list) else [],
                customer_tweet_id=tid,
                support_tweet_id=str(getattr(row, "support_tweet_id", "") or ""),
            )
        )
    return cases


def assert_corpus_not_contaminated(
    cases: Iterable[RetrievalCase],
    *,
    golden_ids: Iterable[str],
    test_ids: Iterable[str],
    golden_conversation_ids: Iterable[str] | None = None,
) -> dict[str, Any]:
    case_list = list(cases)
    retrieval_ids = {c.customer_tweet_id or c.case_id for c in case_list}
    report = check_retrieval_index_contamination(
        retrieval_evidence_ids=retrieval_ids,
        test_ids=test_ids,
        golden_ids=golden_ids,
    )
    issues = list(report.issues)
    if golden_conversation_ids is not None:
        gold_c = set(map(str, golden_conversation_ids))
        overlap_c = {c.conversation_id for c in case_list} & gold_c
        if overlap_c:
            issues.append(f"Golden conversation overlap: {len(overlap_c)}")
    return {
        "ok": len(issues) == 0,
        "issues": issues,
        "n_cases": len(case_list),
        "details": report.details,
    }


def save_corpus(cases: list[RetrievalCase], path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [c.to_dict() for c in cases]
    if path.suffix == ".jsonl":
        with path.open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row) + "\n")
    else:
        pd.DataFrame(rows).to_parquet(path, index=False)
    return path


def load_corpus(path: Path | str) -> list[RetrievalCase]:
    path = Path(path)
    if path.suffix == ".jsonl":
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        rows = pd.read_parquet(path).to_dict(orient="records")
    out: list[RetrievalCase] = []
    for r in rows:
        out.append(
            RetrievalCase(
                case_id=str(r["case_id"]),
                conversation_id=str(r["conversation_id"]),
                customer_message=str(r["customer_message"]),
                support_response=str(r["support_response"]),
                intent=None if r.get("intent") in (None, "nan") else str(r.get("intent")),
                timestamp=None if r.get("timestamp") in (None, "nan") else str(r.get("timestamp")),
                prior_customer_context=list(r.get("prior_customer_context") or []),
                customer_tweet_id=None if r.get("customer_tweet_id") in (None, "nan") else str(r.get("customer_tweet_id")),
                support_tweet_id=None if r.get("support_tweet_id") in (None, "nan") else str(r.get("support_tweet_id")),
            )
        )
    return out


def near_duplicate_key(text: str) -> str:
    return normalize_tweet_text(text)
