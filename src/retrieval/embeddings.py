"""Embedding backend abstraction for retrieval."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from src.preprocessing.text_normalize import normalize_tweet_text


class EmbeddingBackend(ABC):
    name: str

    @abstractmethod
    def fit(self, texts: Sequence[str]) -> "EmbeddingBackend":
        ...

    @abstractmethod
    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Return L2-normalized dense vectors shape (n, d)."""


class TfidfSvdEmbedding(EmbeddingBackend):
    """Local deterministic embedding: TF-IDF → TruncatedSVD.

    Used as default/fallback so CI and demos run without downloading models.
    """

    name = "tfidf_svd"

    def __init__(self, n_components: int = 256, random_seed: int = 42):
        self.n_components = n_components
        self.random_seed = random_seed
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95,
            stop_words="english",
        )
        self.svd = TruncatedSVD(n_components=n_components, random_state=random_seed)
        self._fitted = False

    def fit(self, texts: Sequence[str]) -> "TfidfSvdEmbedding":
        corpus = [normalize_tweet_text(t) for t in texts]
        matrix = self.vectorizer.fit_transform(corpus)
        n_comp = min(self.n_components, max(2, matrix.shape[1] - 1), matrix.shape[0] - 1)
        if n_comp < 2:
            n_comp = 2
        self.svd = TruncatedSVD(n_components=n_comp, random_state=self.random_seed)
        self.svd.fit(matrix)
        self._fitted = True
        return self

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("TfidfSvdEmbedding is not fitted")
        corpus = [normalize_tweet_text(t) for t in texts]
        matrix = self.vectorizer.transform(corpus)
        dense = self.svd.transform(matrix)
        return normalize(dense)


class SentenceTransformerEmbedding(EmbeddingBackend):
    """Optional sentence-transformers backend."""

    name = "sentence_transformers"

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None

    def fit(self, texts: Sequence[str]) -> "SentenceTransformerEmbedding":
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(self.model_name)
        return self

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if self._model is None:
            # lazy load
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        vectors = self._model.encode(
            list(texts),
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)


def create_embedding_backend(
    kind: str = "auto",
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    random_seed: int = 42,
) -> EmbeddingBackend:
    """Factory. ``auto`` prefers sentence-transformers if importable."""
    if kind in {"tfidf_svd", "deterministic", "local"}:
        return TfidfSvdEmbedding(random_seed=random_seed)
    if kind in {"sentence_transformers", "st", "semantic"}:
        return SentenceTransformerEmbedding(model_name=model_name)
    if kind == "auto":
        try:
            import sentence_transformers  # noqa: F401

            return SentenceTransformerEmbedding(model_name=model_name)
        except Exception:
            return TfidfSvdEmbedding(random_seed=random_seed)
    raise ValueError(f"Unknown embedding backend kind: {kind}")
