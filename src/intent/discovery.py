"""Intent discovery via clustering (proposal only, not ground truth)."""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

from src.preprocessing.text_normalize import normalize_tweet_text


@dataclass
class IntentProposal:
    """PROPOSAL artifact — not ground-truth labels."""

    intent_id: str
    intent_name: str
    description: str
    representative_examples: list[str]
    possible_confusions: list[str] = field(default_factory=list)
    notes: str = ""
    n_examples_in_cluster: int = 0
    status: str = "PROPOSAL_NOT_GROUND_TRUTH"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _top_terms(vectorizer: TfidfVectorizer, matrix, row_indices: list[int], k: int = 8) -> list[str]:
    if not row_indices:
        return []
    sub = matrix[row_indices]
    mean = np.asarray(sub.mean(axis=0)).ravel()
    terms = np.array(vectorizer.get_feature_names_out())
    top_idx = mean.argsort()[::-1][:k]
    return [str(t) for t in terms[top_idx] if mean[top_idx[0]] >= 0]


def _name_from_terms(terms: list[str], cluster_id: int) -> str:
    if not terms:
        return f"cluster_{cluster_id}"
    cleaned = [re.sub(r"[^a-z0-9]+", "_", t.lower()).strip("_") for t in terms[:3]]
    cleaned = [c for c in cleaned if c]
    return "_".join(cleaned[:3]) or f"cluster_{cluster_id}"


def discover_intent_proposals(
    texts: list[str],
    n_clusters: int = 10,
    random_seed: int = 42,
    max_examples_per_intent: int = 5,
    min_df: int = 2,
) -> list[IntentProposal]:
    """Cluster normalized inbound messages into a proposed taxonomy.

    IMPORTANT: Output is a PROPOSAL for human review, not labeled ground truth.
    Phase 0 uses TF-IDF + KMeans for reproducibility with light dependencies.
    Embedding-based clustering can replace this later via EMBEDDING_MODEL.
    """
    if not texts:
        return []

    normalized = [normalize_tweet_text(t) for t in texts]
    # Drop empties but keep alignment via indices.
    keep = [i for i, t in enumerate(normalized) if t]
    if len(keep) < max(2, n_clusters):
        # Fall back to fewer clusters when sample is tiny.
        n_clusters = max(1, min(n_clusters, len(keep)))

    docs = [normalized[i] for i in keep]
    if not docs:
        return []

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=min_df if len(docs) >= 20 else 1,
        max_df=0.95,
        stop_words="english",
    )
    matrix = vectorizer.fit_transform(docs)
    n_clusters = max(1, min(n_clusters, len(docs)))
    model = KMeans(n_clusters=n_clusters, random_state=random_seed, n_init=10)
    labels = model.fit_predict(matrix)

    proposals: list[IntentProposal] = []
    for cid in range(n_clusters):
        idxs = [i for i, lab in enumerate(labels) if lab == cid]
        terms = _top_terms(vectorizer, matrix, idxs)
        examples = [texts[keep[i]] for i in idxs[:max_examples_per_intent]]
        name = _name_from_terms(terms, cid)
        proposals.append(
            IntentProposal(
                intent_id=f"intent_{cid:02d}",
                intent_name=name,
                description=(
                    "Auto-proposed from TF-IDF KMeans top terms: "
                    + ", ".join(terms[:8])
                ),
                representative_examples=examples,
                possible_confusions=[],
                notes=(
                    "PROPOSAL ONLY. Requires human renaming, merging/splitting, "
                    "and boundary documentation before labeling a golden set."
                ),
                n_examples_in_cluster=len(idxs),
            )
        )

    # Heuristic confusion notes: overlapping top terms.
    term_sets = {
        p.intent_id: set(re.findall(r"[a-z0-9_]+", p.description.lower()))
        for p in proposals
    }
    for p in proposals:
        overlaps = []
        for other in proposals:
            if other.intent_id == p.intent_id:
                continue
            shared = term_sets[p.intent_id] & term_sets[other.intent_id]
            shared -= {"auto", "proposed", "from", "tf", "idf", "kmeans", "top", "terms"}
            if len(shared) >= 2:
                overlaps.append(other.intent_name)
        p.possible_confusions = overlaps[:5]

    proposals.sort(key=lambda p: (-p.n_examples_in_cluster, p.intent_id))
    return proposals


def run_intent_discovery(
    inbound_texts: list[str],
    output_path: Path | str | None = None,
    n_clusters: int = 10,
    random_seed: int = 42,
    sample_size: int | None = 2000,
) -> dict[str, Any]:
    texts = list(inbound_texts)
    if sample_size is not None and len(texts) > sample_size:
        rng = np.random.default_rng(random_seed)
        idx = rng.choice(len(texts), size=sample_size, replace=False)
        texts = [texts[i] for i in sorted(idx)]

    proposals = discover_intent_proposals(
        texts, n_clusters=n_clusters, random_seed=random_seed
    )
    result = {
        "status": "PROPOSAL_NOT_GROUND_TRUTH",
        "methodology": [
            "Normalize inbound customer texts (URLs/whitespace).",
            "Embed with TF-IDF unigrams+bigrams (Phase 0; embeddings planned).",
            "Cluster with KMeans.",
            "Name clusters from top TF-IDF terms.",
            "Inspect representative examples; human must finalize taxonomy.",
        ],
        "n_texts_used": len(texts),
        "n_intents_proposed": len(proposals),
        "proposed_intents": [p.to_dict() for p in proposals],
        "ambiguous_multi_intent_notes": [
            "Messages requesting both account action and troubleshooting are multi-intent.",
            "Complaints that also ask for refunds may span billing + service_quality.",
            "Human labeling should allow secondary_intent and ambiguous flags.",
        ],
    }
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2)
    return result


def propose_from_brand_frame(
    df: pd.DataFrame,
    brand: str,
    n_clusters: int = 10,
    random_seed: int = 42,
    sample_size: int | None = 2000,
    output_path: Path | str | None = None,
) -> dict[str, Any]:
    pattern = re.compile(rf"@{re.escape(brand)}\b", re.IGNORECASE)
    inbound = df[df["inbound"] == True]
    texts = (
        inbound.loc[
            inbound["text"].astype(str).str.contains(pattern, na=False), "text"
        ]
        .astype(str)
        .tolist()
    )
    result = run_intent_discovery(
        texts,
        output_path=output_path,
        n_clusters=n_clusters,
        random_seed=random_seed,
        sample_size=sample_size,
    )
    result["brand"] = brand
    result["label"] = "INITIAL_INTENT_TAXONOMY_PROPOSAL"
    return result
