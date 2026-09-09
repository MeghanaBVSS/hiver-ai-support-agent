"""Leakage detection and contamination checks.

These utilities enforce evaluation hygiene:
- no conversation overlap across splits
- no golden examples in train/valid/retrieval
- no test examples in retrieval evidence
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class LeakageReport:
    ok: bool
    issues: tuple[str, ...]
    details: dict

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "issues": list(self.issues),
            "details": self.details,
        }


def _as_set(values: Iterable[str] | None) -> set[str]:
    if values is None:
        return set()
    return {str(v) for v in values}


def check_disjoint_sets(
    named_sets: dict[str, Iterable[str]],
) -> LeakageReport:
    """Ensure all provided ID sets are pairwise disjoint."""
    issues: list[str] = []
    details: dict = {}
    names = list(named_sets.keys())
    materialized = {k: _as_set(v) for k, v in named_sets.items()}
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            overlap = materialized[a] & materialized[b]
            details[f"overlap:{a}|{b}"] = sorted(overlap)[:50]
            details[f"overlap_count:{a}|{b}"] = len(overlap)
            if overlap:
                issues.append(
                    f"Overlap between '{a}' and '{b}': {len(overlap)} ids"
                )
    return LeakageReport(ok=not issues, issues=tuple(issues), details=details)


def check_conversation_split_leakage(
    train_conversation_ids: Iterable[str],
    valid_conversation_ids: Iterable[str],
    test_conversation_ids: Iterable[str],
) -> LeakageReport:
    return check_disjoint_sets(
        {
            "train": train_conversation_ids,
            "valid": valid_conversation_ids,
            "test": test_conversation_ids,
        }
    )


def check_golden_contamination(
    golden_ids: Iterable[str],
    train_ids: Iterable[str],
    valid_ids: Iterable[str] | None = None,
    retrieval_ids: Iterable[str] | None = None,
    prompt_example_ids: Iterable[str] | None = None,
) -> LeakageReport:
    """Golden evaluation IDs must not appear in train/retrieval/prompt pools."""
    named = {
        "golden": golden_ids,
        "train": train_ids,
    }
    if valid_ids is not None:
        named["valid"] = valid_ids
    if retrieval_ids is not None:
        named["retrieval"] = retrieval_ids
    if prompt_example_ids is not None:
        named["prompt_examples"] = prompt_example_ids

    base = check_disjoint_sets(named)
    # Disjointness among non-golden sets is nice-to-have but not required here.
    # Re-check only golden vs others.
    issues: list[str] = []
    details: dict = {}
    golden = _as_set(golden_ids)
    for name, values in named.items():
        if name == "golden":
            continue
        overlap = golden & _as_set(values)
        details[f"golden_in_{name}"] = sorted(overlap)[:50]
        details[f"golden_in_{name}_count"] = len(overlap)
        if overlap:
            issues.append(f"Golden contamination in '{name}': {len(overlap)} ids")
    return LeakageReport(ok=not issues, issues=tuple(issues), details=details)


def check_retrieval_index_contamination(
    retrieval_evidence_ids: Iterable[str],
    test_ids: Iterable[str],
    golden_ids: Iterable[str] | None = None,
) -> LeakageReport:
    """Retrieval corpus must exclude test and golden examples."""
    named = {
        "retrieval": retrieval_evidence_ids,
        "test": test_ids,
    }
    if golden_ids is not None:
        named["golden"] = golden_ids
    report = check_disjoint_sets(named)
    # Focus messaging on retrieval contamination.
    issues = [i for i in report.issues if "retrieval" in i]
    return LeakageReport(ok=not issues, issues=tuple(issues), details=report.details)


def find_duplicate_texts(
    ids: list[str],
    texts: list[str],
) -> dict[str, list[str]]:
    """Map normalized exact-duplicate text -> list of ids."""
    buckets: dict[str, list[str]] = {}
    for i, text in zip(ids, texts):
        key = (text or "").strip().lower()
        buckets.setdefault(key, []).append(str(i))
    return {k: v for k, v in buckets.items() if len(v) > 1 and k != ""}
