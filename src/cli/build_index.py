"""Build leakage-safe retrieval index and fit classifier artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.config import ROOT, load_config
from src.intent.baselines import TfidfLogRegIntentBaseline
from src.policy.grounded_policy import PolicyThresholds, tune_thresholds_on_valid
from src.retrieval.corpus import (
    assert_corpus_not_contaminated,
    build_train_retrieval_corpus,
    save_corpus,
)
from src.retrieval.embeddings import create_embedding_backend
from src.retrieval.semantic import SemanticRetriever, compute_evidence_strength
from src.agent.grounded_agent import GroundedSupportAgent
from src.policy.grounded_policy import GroundedEscalationPolicy
import joblib


def build_artifacts(
    *,
    embedding_kind: str = "tfidf_svd",
    out_dir: Path | None = None,
) -> dict:
    cfg = load_config()
    cfg.ensure_dirs()
    out_dir = out_dir or (ROOT / "data" / "processed" / "agent_artifacts")
    out_dir.mkdir(parents=True, exist_ok=True)

    train_path = ROOT / "data" / "processed" / "hulu_train_labeled.parquet"
    valid_path = ROOT / "data" / "processed" / "hulu_valid_labeled.parquet"
    golden_path = ROOT / "evaluation" / "golden" / "golden_set.csv"
    pairs_path = ROOT / "data" / "processed" / "hulu_support_pairs_split.parquet"

    cases = build_train_retrieval_corpus(
        train_path,
        golden_path=golden_path,
        pairs_split_path=pairs_path,
    )
    golden = pd.read_csv(golden_path)
    pairs = pd.read_parquet(pairs_path)
    test = pairs[pairs["split"] == "test"]
    contamination = assert_corpus_not_contaminated(
        cases,
        golden_ids=golden["customer_tweet_id"].astype(str),
        test_ids=test["customer_tweet_id"].astype(str),
        golden_conversation_ids=golden["conversation_id"].astype(str),
    )
    if not contamination["ok"]:
        raise RuntimeError(f"Retrieval corpus contaminated: {contamination}")

    corpus_path = save_corpus(cases, out_dir / "retrieval_corpus.parquet")
    backend = create_embedding_backend(embedding_kind, random_seed=cfg.random_seed)
    retriever = SemanticRetriever(
        backend=backend,
        top_k=cfg.retrieval_top_k,
        min_similarity=cfg.min_retrieval_similarity,
        diversity=True,
    ).fit(cases)
    retriever.save(out_dir / "retriever")

    train = pd.read_parquet(train_path)
    clf = TfidfLogRegIntentBaseline().fit(
        train["customer_message"].astype(str).tolist(),
        train["gold_intent"].astype(str).tolist(),
    )
    joblib.dump(clf, out_dir / "intent_classifier.joblib")

    # Tune thresholds on validation (NOT golden)
    valid = pd.read_parquet(valid_path)
    records = []
    # Use a temporary agent-less loop
    for row in valid.itertuples():
        hits = retriever.retrieve(str(row.customer_message))
        strength = compute_evidence_strength(hits)
        # intent confidence from classifier
        from src.evaluation.harness import ExampleRecord

        pred = clf.predict(ExampleRecord("v", str(row.customer_message)))
        records.append(
            {
                "customer_message": str(row.customer_message),
                "predicted_intent": pred.intent,
                "intent_confidence": pred.intent_confidence,
                "top_similarity": hits[0].similarity if hits else 0.0,
                "evidence_strength": strength.score,
                "evidence_intent_agreement": strength.intent_agreement,
                "has_conflicting_evidence": strength.intent_agreement < 0.5 and len(hits) >= 2,
                "n_hits": len(hits),
                "gold_escalate": bool(row.gold_escalate),
            }
        )
    thresholds = tune_thresholds_on_valid(records, prefer_safety=True)
    (out_dir / "policy_thresholds.json").write_text(
        json.dumps(
            {
                "thresholds": thresholds.__dict__,
                "rationale": (
                    "Chosen on validation labels to minimize unsafe auto-handle rate "
                    "with secondary coverage consideration. Golden was NOT used."
                ),
                "embedding_backend": backend.name,
                "n_retrieval_cases": len(cases),
                "contamination": contamination,
            },
            indent=2,
        )
    )
    return {
        "out_dir": str(out_dir),
        "n_cases": len(cases),
        "embedding": backend.name,
        "contamination_ok": contamination["ok"],
        "thresholds": thresholds.__dict__,
        "corpus_path": str(corpus_path),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build Phase-2 agent artifacts")
    parser.add_argument(
        "--embedding",
        default="tfidf_svd",
        help="tfidf_svd | sentence_transformers | auto",
    )
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args(argv)
    result = build_artifacts(
        embedding_kind=args.embedding,
        out_dir=Path(args.out_dir) if args.out_dir else None,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
