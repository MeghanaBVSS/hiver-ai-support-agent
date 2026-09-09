"""Intent classification baselines sharing the evaluation harness interface."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.evaluation.harness import AgentPrediction, ExampleRecord
from src.preprocessing.text_normalize import normalize_tweet_text


@dataclass
class MajorityIntentBaseline:
    """Baseline 0: always predict the majority training intent."""

    name: str = "baseline0_majority_intent"
    majority_label: str = "unknown"

    def fit(self, texts: list[str], labels: list[str]) -> "MajorityIntentBaseline":
        if not labels:
            self.majority_label = "unknown"
            return self
        self.majority_label = Counter(labels).most_common(1)[0][0]
        return self

    def predict(self, example: ExampleRecord) -> AgentPrediction:
        return AgentPrediction(
            intent=self.majority_label,
            intent_confidence=1.0,
            reply=None,
            escalate=True,
            escalate_reason="Baseline 0 does not draft replies; escalate by default.",
            meta={"baseline": self.name},
        )


@dataclass
class TfidfLogRegIntentBaseline:
    """Baseline 1: TF-IDF + Logistic Regression intent classifier."""

    name: str = "baseline1_tfidf_logreg"
    pipeline: Pipeline | None = None
    labels_: list[str] | None = None

    def fit(self, texts: list[str], labels: list[str]) -> "TfidfLogRegIntentBaseline":
        if not texts or not labels:
            raise ValueError("Need texts and labels to fit Baseline 1")
        normed = [normalize_tweet_text(t) for t in texts]
        self.labels_ = sorted(set(labels))
        self.pipeline = Pipeline(
            steps=[
                (
                    "tfidf",
                    TfidfVectorizer(
                        ngram_range=(1, 2),
                        min_df=1,
                        max_df=0.95,
                        stop_words="english",
                    ),
                ),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=1000,
                        random_state=42,
                        class_weight="balanced",
                    ),
                ),
            ]
        )
        self.pipeline.fit(normed, labels)
        return self

    def predict(self, example: ExampleRecord) -> AgentPrediction:
        if self.pipeline is None:
            raise RuntimeError("Baseline 1 is not fitted")
        text = normalize_tweet_text(example.customer_message)
        proba = self.pipeline.predict_proba([text])[0]
        classes = list(self.pipeline.classes_)
        idx = int(np.argmax(proba))
        intent = str(classes[idx])
        conf = float(proba[idx])
        escalate = conf < 0.55
        reason = (
            f"Low intent confidence ({conf:.3f} < 0.55)."
            if escalate
            else f"Intent confidence {conf:.3f} above threshold; reply not handled by this baseline."
        )
        return AgentPrediction(
            intent=intent,
            intent_confidence=conf,
            reply=None,
            escalate=escalate,
            escalate_reason=reason,
            meta={"baseline": self.name},
        )
