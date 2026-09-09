"""Semantic retriever with diversity controls and evidence scoring."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from src.preprocessing.text_normalize import normalize_tweet_text
from src.retrieval.corpus import RetrievalCase, near_duplicate_key
from src.retrieval.embeddings import EmbeddingBackend, create_embedding_backend


@dataclass
class RetrievedHit:
    case_id: str
    similarity: float
    customer_message: str
    support_response: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EvidenceScore:
    """Explicit configurable evidence-strength formula."""

    score: float
    top_similarity: float
    top1_top2_gap: float
    intent_agreement: float
    n_above_threshold: int
    reply_consistency: float
    details: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EvidenceScoreConfig:
    min_similarity: float = 0.20
    w_top_sim: float = 0.35
    w_gap: float = 0.15
    w_intent_agree: float = 0.20
    w_n_above: float = 0.15
    w_reply_consistency: float = 0.15
    weak_threshold: float = 0.35


class SemanticRetriever:
    def __init__(
        self,
        backend: EmbeddingBackend | None = None,
        top_k: int = 5,
        min_similarity: float = 0.15,
        diversity: bool = True,
    ):
        self.backend = backend or create_embedding_backend("auto")
        self.top_k = top_k
        self.min_similarity = min_similarity
        self.diversity = diversity
        self.cases: list[RetrievalCase] = []
        self._matrix: np.ndarray | None = None

    def fit(self, cases: list[RetrievalCase]) -> "SemanticRetriever":
        if not cases:
            raise ValueError("Need at least one retrieval case")
        # Deduplicate near-identical customer messages, keep first
        seen: set[str] = set()
        unique: list[RetrievalCase] = []
        for c in cases:
            key = near_duplicate_key(c.customer_message)
            if key in seen:
                continue
            seen.add(key)
            unique.append(c)
        self.cases = unique
        texts = [c.customer_message for c in self.cases]
        self.backend.fit(texts)
        self._matrix = self.backend.encode(texts)
        return self

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievedHit]:
        if self._matrix is None:
            raise RuntimeError("Retriever not fitted")
        k = top_k or self.top_k
        q = self.backend.encode([query])
        sims = cosine_similarity(q, self._matrix).ravel()
        order = np.argsort(-sims)
        hits: list[RetrievedHit] = []
        used_convs: set[str] = set()
        used_norms: set[str] = set()
        for idx in order:
            score = float(sims[idx])
            if score < self.min_similarity and hits:
                break
            case = self.cases[idx]
            if self.diversity:
                if case.conversation_id in used_convs:
                    continue
                nkey = near_duplicate_key(case.customer_message)
                if nkey in used_norms:
                    continue
                used_convs.add(case.conversation_id)
                used_norms.add(nkey)
            hits.append(
                RetrievedHit(
                    case_id=case.case_id,
                    similarity=score,
                    customer_message=case.customer_message,
                    support_response=case.support_response,
                    metadata={
                        "conversation_id": case.conversation_id,
                        "intent": case.intent,
                        "timestamp": case.timestamp,
                        "customer_tweet_id": case.customer_tweet_id,
                        "support_tweet_id": case.support_tweet_id,
                        "prior_customer_context": case.prior_customer_context,
                    },
                )
            )
            if len(hits) >= k:
                break
        return hits

    def save(self, directory: Path | str) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "backend_name": self.backend.name,
                "backend": self.backend,
                "cases": self.cases,
                "matrix": self._matrix,
                "top_k": self.top_k,
                "min_similarity": self.min_similarity,
                "diversity": self.diversity,
            },
            directory / "retriever.joblib",
        )
        meta = {
            "backend": self.backend.name,
            "n_cases": len(self.cases),
            "top_k": self.top_k,
            "min_similarity": self.min_similarity,
            "diversity": self.diversity,
            "embedding_model": getattr(self.backend, "model_name", self.backend.name),
        }
        (directory / "meta.json").write_text(json.dumps(meta, indent=2))
        return directory

    @classmethod
    def load(cls, directory: Path | str) -> "SemanticRetriever":
        directory = Path(directory)
        payload = joblib.load(directory / "retriever.joblib")
        obj = cls(
            backend=payload["backend"],
            top_k=payload["top_k"],
            min_similarity=payload["min_similarity"],
            diversity=payload.get("diversity", True),
        )
        obj.cases = payload["cases"]
        obj._matrix = payload["matrix"]
        return obj


def compute_evidence_strength(
    hits: list[RetrievedHit],
    config: EvidenceScoreConfig | None = None,
) -> EvidenceScore:
    cfg = config or EvidenceScoreConfig()
    if not hits:
        return EvidenceScore(
            score=0.0,
            top_similarity=0.0,
            top1_top2_gap=0.0,
            intent_agreement=0.0,
            n_above_threshold=0,
            reply_consistency=0.0,
            details={"empty": True},
        )
    top = hits[0].similarity
    gap = top - hits[1].similarity if len(hits) > 1 else top
    intents = [h.metadata.get("intent") for h in hits if h.metadata.get("intent")]
    if intents:
        from collections import Counter

        agree = Counter(intents).most_common(1)[0][1] / len(intents)
    else:
        agree = 0.0
    n_above = sum(1 for h in hits if h.similarity >= cfg.min_similarity)
    # reply consistency: Jaccard of token sets between top replies
    replies = [normalize_tweet_text(h.support_response).split() for h in hits[:3]]
    if len(replies) >= 2:
        sets = [set(r) for r in replies if r]
        if len(sets) >= 2:
            inter = set.intersection(*sets)
            union = set.union(*sets)
            reply_consistency = len(inter) / len(union) if union else 0.0
        else:
            reply_consistency = 0.0
    else:
        reply_consistency = 1.0 if hits else 0.0

    score = (
        cfg.w_top_sim * min(max(top, 0.0), 1.0)
        + cfg.w_gap * min(max(gap, 0.0), 1.0)
        + cfg.w_intent_agree * agree
        + cfg.w_n_above * min(n_above / max(len(hits), 1), 1.0)
        + cfg.w_reply_consistency * reply_consistency
    )
    return EvidenceScore(
        score=float(score),
        top_similarity=float(top),
        top1_top2_gap=float(gap),
        intent_agreement=float(agree),
        n_above_threshold=int(n_above),
        reply_consistency=float(reply_consistency),
        details={"weak": score < cfg.weak_threshold, "config": cfg.__dict__},
    )
