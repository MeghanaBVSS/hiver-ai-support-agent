"""TF-IDF retrieval baseline for historical response grounding."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.preprocessing.text_normalize import normalize_tweet_text


@dataclass
class RetrievedEvidence:
    customer_message: str
    support_response: str
    score: float
    source_id: str | None = None
    conversation_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "customer_message": self.customer_message,
            "support_response": self.support_response,
            "score": self.score,
            "source_id": self.source_id,
            "conversation_id": self.conversation_id,
        }


class TfidfResponseRetriever:
    """Retrieve nearest historical customer messages via TF-IDF cosine similarity."""

    def __init__(self, top_k: int = 5, min_similarity: float = 0.15):
        self.top_k = top_k
        self.min_similarity = min_similarity
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=1,
            stop_words="english",
        )
        self._matrix = None
        self._responses: list[str] = []
        self._customers: list[str] = []
        self._source_ids: list[str | None] = []
        self._conversation_ids: list[str | None] = []

    def fit(
        self,
        customer_messages: list[str],
        support_responses: list[str],
        source_ids: list[str | None] | None = None,
        conversation_ids: list[str | None] | None = None,
    ) -> "TfidfResponseRetriever":
        if len(customer_messages) != len(support_responses):
            raise ValueError("customer_messages and support_responses length mismatch")
        if not customer_messages:
            raise ValueError("Need at least one historical pair to build retrieval index")
        self._customers = customer_messages
        self._responses = support_responses
        self._source_ids = source_ids or [None] * len(customer_messages)
        self._conversation_ids = conversation_ids or [None] * len(customer_messages)
        corpus = [normalize_tweet_text(t) for t in customer_messages]
        self._matrix = self.vectorizer.fit_transform(corpus)
        return self

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievedEvidence]:
        if self._matrix is None:
            raise RuntimeError("Retriever is not fitted")
        k = top_k or self.top_k
        q = self.vectorizer.transform([normalize_tweet_text(query)])
        sims = cosine_similarity(q, self._matrix).ravel()
        order = np.argsort(-sims)
        out: list[RetrievedEvidence] = []
        for idx in order[:k]:
            score = float(sims[idx])
            if score < self.min_similarity:
                continue
            out.append(
                RetrievedEvidence(
                    customer_message=self._customers[idx],
                    support_response=self._responses[idx],
                    score=score,
                    source_id=self._source_ids[idx],
                    conversation_id=self._conversation_ids[idx],
                )
            )
        return out

    @property
    def size(self) -> int:
        return len(self._customers)
