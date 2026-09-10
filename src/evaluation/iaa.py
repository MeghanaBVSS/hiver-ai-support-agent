"""Inter-annotator agreement for the second-annotator subset."""

from __future__ import annotations

from collections import Counter
from typing import Any, Sequence

import numpy as np
from sklearn.metrics import cohen_kappa_score, confusion_matrix


def _as_str(values: Sequence[Any]) -> list[str]:
    return [str(v) for v in values]


def _as_bool(values: Sequence[Any]) -> list[bool]:
    out: list[bool] = []
    for v in values:
        if isinstance(v, bool):
            out.append(v)
        else:
            out.append(str(v).strip().lower() in {"1", "true", "t", "yes", "y"})
    return out


def compute_categorical_agreement(
    a: Sequence[Any],
    b: Sequence[Any],
    *,
    labels: Sequence[str] | None = None,
) -> dict[str, Any]:
    a_l = _as_str(a)
    b_l = _as_str(b)
    if len(a_l) != len(b_l):
        raise ValueError("Label sequences must have equal length")
    n = len(a_l)
    exact = float(np.mean([x == y for x, y in zip(a_l, b_l)])) if n else 0.0
    label_list = list(labels) if labels is not None else sorted(set(a_l) | set(b_l))
    kappa = float(cohen_kappa_score(a_l, b_l, labels=label_list)) if n else float("nan")
    cm = confusion_matrix(a_l, b_l, labels=label_list).astype(int).tolist()
    disagreements = [
        {"index": i, "annotator_1": x, "annotator_2": y}
        for i, (x, y) in enumerate(zip(a_l, b_l))
        if x != y
    ]
    # Which primary labels disagree most (from annotator_1 side)
    by_a1 = Counter(d["annotator_1"] for d in disagreements)
    by_pair = Counter((d["annotator_1"], d["annotator_2"]) for d in disagreements)
    return {
        "n": n,
        "exact_agreement": exact,
        "cohen_kappa": kappa,
        "n_disagreements": len(disagreements),
        "labels": label_list,
        "confusion_matrix": cm,
        "disagreement_count_by_annotator1_label": dict(by_a1.most_common()),
        "top_disagreement_pairs": [
            {"annotator_1": a, "annotator_2": b, "count": c}
            for (a, b), c in by_pair.most_common(15)
        ],
    }


def compute_binary_agreement(a: Sequence[Any], b: Sequence[Any]) -> dict[str, Any]:
    a_b = _as_bool(a)
    b_b = _as_bool(b)
    n = len(a_b)
    exact = float(np.mean([x == y for x, y in zip(a_b, b_b)])) if n else 0.0
    kappa = float(cohen_kappa_score(a_b, b_b)) if n else float("nan")
    cm = confusion_matrix(a_b, b_b, labels=[False, True]).astype(int).tolist()
    disagreements = [
        {"index": i, "annotator_1": x, "annotator_2": y}
        for i, (x, y) in enumerate(zip(a_b, b_b))
        if x != y
    ]
    return {
        "n": n,
        "exact_agreement": exact,
        "cohen_kappa": kappa,
        "n_disagreements": len(disagreements),
        "labels": [False, True],
        "confusion_matrix": cm,
        "confusion_matrix_note": "rows=annotator_1 [False,True], cols=annotator_2 [False,True]",
        "disagreements": disagreements,
    }
