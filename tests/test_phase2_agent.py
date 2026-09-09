"""Phase-2 unit tests: retrieval, policy, generation, contamination, agent."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.agent.grounded_agent import GroundedSupportAgent
from src.evaluation.harness import ExampleRecord
from src.generation.providers import (
    DeterministicCopyGenerator,
    GenerationRequest,
    GenerationResult,
    OpenAICompatibleGenerator,
    validate_generation,
)
from src.intent.baselines import TfidfLogRegIntentBaseline
from src.policy.grounded_policy import (
    GroundedEscalationPolicy,
    PolicyInput,
    PolicyThresholds,
    tune_thresholds_on_valid,
)
from src.retrieval.corpus import (
    RetrievalCase,
    assert_corpus_not_contaminated,
    build_train_retrieval_corpus,
    near_duplicate_key,
)
from src.retrieval.embeddings import TfidfSvdEmbedding, create_embedding_backend
from src.retrieval.semantic import (
    EvidenceScoreConfig,
    RetrievedHit,
    SemanticRetriever,
    compute_evidence_strength,
)


@pytest.fixture
def tiny_cases() -> list[RetrievalCase]:
    return [
        RetrievalCase(
            case_id="case_1",
            conversation_id="c1",
            customer_message="Hulu keeps buffering during live TV",
            support_response="Sorry about the buffering — try restarting the app and check your connection.",
            intent="live_tv_issues",
            customer_tweet_id="1",
            support_tweet_id="11",
        ),
        RetrievalCase(
            case_id="case_2",
            conversation_id="c2",
            customer_message="Stream freezes every few minutes on Firestick",
            support_response="Please power-cycle your Firestick and update the Hulu app.",
            intent="app_device_issue",
            customer_tweet_id="2",
            support_tweet_id="22",
        ),
        RetrievalCase(
            case_id="case_3",
            conversation_id="c3",
            customer_message="I was charged twice this month for Hulu",
            support_response="We can review billing — please DM your account email so we can look into the charges.",
            intent="billing_subscription",
            customer_tweet_id="3",
            support_tweet_id="33",
        ),
        RetrievalCase(
            case_id="case_4",
            conversation_id="c4",
            customer_message="Cannot log into my Hulu account password not working",
            support_response="Try resetting your password via the login screen; if still stuck, DM us.",
            intent="login_account",
            customer_tweet_id="4",
            support_tweet_id="44",
        ),
        RetrievalCase(
            case_id="case_5",
            conversation_id="c1",  # same conversation as case_1 — diversity should prefer others
            customer_message="Hulu keeps buffering during live TV again",
            support_response="Sorry — still looking into live TV buffering.",
            intent="live_tv_issues",
            customer_tweet_id="5",
            support_tweet_id="55",
        ),
    ]


def test_near_duplicate_key_stable():
    a = near_duplicate_key("Hello @Hulu World")
    b = near_duplicate_key("hello @hulu world")
    assert a == b
    # Exact same normalized text maps to same key (used for dedupe)
    assert near_duplicate_key("Same Text") == near_duplicate_key("same text")


def test_contamination_detection(tiny_cases):
    report = assert_corpus_not_contaminated(
        tiny_cases,
        golden_ids=["1"],
        test_ids=["99"],
        golden_conversation_ids=["c9"],
    )
    assert report["ok"] is False
    assert any("Golden" in i or "golden" in i.lower() or "overlap" in i.lower() for i in report["issues"]) or report[
        "details"
    ]

    clean = assert_corpus_not_contaminated(
        tiny_cases,
        golden_ids=["999"],
        test_ids=["888"],
        golden_conversation_ids=["cx"],
    )
    assert clean["ok"] is True


def test_build_corpus_excludes_golden(tmp_path: Path):
    train = pd.DataFrame(
        [
            {
                "conversation_id": "train_c",
                "customer_tweet_id": "100",
                "customer_message": "buffering issue",
                "support_response": "try restart",
                "support_tweet_id": "101",
                "gold_intent": "playback_error",
                "conversation_context": [],
            },
            {
                "conversation_id": "gold_c",
                "customer_tweet_id": "200",
                "customer_message": "should be excluded",
                "support_response": "secret gold reply",
                "support_tweet_id": "201",
                "gold_intent": "playback_error",
                "conversation_context": [],
            },
        ]
    )
    train_path = tmp_path / "train.parquet"
    train.to_parquet(train_path, index=False)
    golden = pd.DataFrame(
        [{"customer_tweet_id": "200", "conversation_id": "gold_c", "example_id": "g1"}]
    )
    gold_path = tmp_path / "golden.csv"
    golden.to_csv(gold_path, index=False)
    pairs = pd.DataFrame(
        [
            {"split": "train", "customer_tweet_id": "100", "conversation_id": "train_c"},
            {"split": "test", "customer_tweet_id": "300", "conversation_id": "test_c"},
        ]
    )
    pairs_path = tmp_path / "pairs.parquet"
    pairs.to_parquet(pairs_path, index=False)

    cases = build_train_retrieval_corpus(
        train_path, golden_path=gold_path, pairs_split_path=pairs_path
    )
    ids = {c.customer_tweet_id for c in cases}
    assert "200" not in ids
    assert "100" in ids
    assert all("secret gold" not in c.support_response for c in cases)


def test_embedding_and_retrieval_ordering(tiny_cases):
    backend = TfidfSvdEmbedding(n_components=16, random_seed=42)
    retriever = SemanticRetriever(backend=backend, top_k=3, min_similarity=0.0, diversity=True)
    retriever.fit(tiny_cases)
    hits = retriever.retrieve("live TV buffering on Hulu")
    assert hits
    sims = [h.similarity for h in hits]
    assert sims == sorted(sims, reverse=True)
    # diversity: at most one hit per conversation
    convs = [h.metadata["conversation_id"] for h in hits]
    assert len(convs) == len(set(convs))


def test_evidence_strength_empty_and_strong():
    empty = compute_evidence_strength([])
    assert empty.score == 0.0
    hits = [
        RetrievedHit("a", 0.9, "m1", "r1", {"intent": "playback_error"}),
        RetrievedHit("b", 0.85, "m2", "r1 similar reply words here", {"intent": "playback_error"}),
        RetrievedHit("c", 0.8, "m3", "r1 similar reply words again", {"intent": "playback_error"}),
    ]
    strong = compute_evidence_strength(hits, EvidenceScoreConfig(min_similarity=0.2))
    assert strong.score > empty.score
    assert strong.intent_agreement == 1.0


def test_escalation_policy_signals():
    policy = GroundedEscalationPolicy(
        PolicyThresholds(
            intent_confidence_min=0.45,
            top_similarity_min=0.22,
            evidence_strength_min=0.35,
            auto_handle_similarity_min=0.35,
            auto_handle_evidence_min=0.45,
        )
    )
    # billing must escalate
    bill = policy.decide(
        PolicyInput(
            customer_message="I want a refund for a double charge",
            predicted_intent="billing_subscription",
            intent_confidence=0.99,
            top_similarity=0.95,
            evidence_strength=0.9,
            evidence_intent_agreement=1.0,
            n_hits=3,
        )
    )
    assert bill.should_escalate is True
    assert bill.escalation_reason == "billing_refund"

    # strong non-sensitive auto-handle
    ok = policy.decide(
        PolicyInput(
            customer_message="Video keeps buffering on my TV app",
            predicted_intent="playback_error",
            intent_confidence=0.8,
            top_similarity=0.5,
            evidence_strength=0.55,
            evidence_intent_agreement=1.0,
            n_hits=3,
        )
    )
    assert ok.should_escalate is False

    # low confidence
    low = policy.decide(
        PolicyInput(
            customer_message="something weird",
            predicted_intent="how_to_feature",
            intent_confidence=0.2,
            top_similarity=0.5,
            evidence_strength=0.55,
            n_hits=2,
        )
    )
    assert low.should_escalate is True
    assert "low_intent_confidence" in low.policy_signals["fired"]


def test_tune_thresholds_not_empty():
    records = [
        {
            "customer_message": "buffering",
            "predicted_intent": "playback_error",
            "intent_confidence": 0.7,
            "top_similarity": 0.4,
            "evidence_strength": 0.5,
            "evidence_intent_agreement": 1.0,
            "n_hits": 3,
            "gold_escalate": False,
        },
        {
            "customer_message": "refund please",
            "predicted_intent": "billing_subscription",
            "intent_confidence": 0.9,
            "top_similarity": 0.6,
            "evidence_strength": 0.6,
            "n_hits": 2,
            "gold_escalate": True,
        },
    ]
    th = tune_thresholds_on_valid(records)
    assert isinstance(th, PolicyThresholds)


def test_no_llm_fallback_generator():
    gen = DeterministicCopyGenerator()
    result = gen.generate_reply(
        GenerationRequest(
            customer_message="buffering",
            conversation_context=[],
            predicted_intent="playback_error",
            evidence=[
                {
                    "case_id": "case_1",
                    "similarity": 0.8,
                    "customer_message": "buffer",
                    "support_response": "Try restarting the app.",
                }
            ],
        )
    )
    assert result.reply == "Try restarting the app."
    assert result.provider == "local"


def test_validate_generation_flags_hallucinations():
    result = GenerationResult(
        reply="We refunded $9.99 within 2 hours and reset your password.",
        grounding_summary="x",
        model="test",
    )
    evidence = [{"customer_message": "app crash", "support_response": "try reinstall"}]
    out = validate_generation(result, evidence)
    assert "hallucinated_price" in out["flags"]
    assert out["force_escalate"] is True


def test_structured_llm_response_mocked():
    import sys

    gen = OpenAICompatibleGenerator(api_key_env="OPENAI_API_KEY")
    gen.api_key = "fake-key"
    fake_json = json.dumps(
        {
            "reply": "Sorry about buffering — try restarting.",
            "grounding_summary": "Based on historical restart advice.",
            "unsupported_claims": [],
            "recommended_escalation": False,
        }
    )
    mock_choice = MagicMock()
    mock_choice.message.content = fake_json
    mock_completion = MagicMock()
    mock_completion.choices = [mock_choice]
    fake_openai = MagicMock()
    fake_openai.OpenAI.return_value.chat.completions.create.return_value = mock_completion
    with patch.dict(sys.modules, {"openai": fake_openai}):
        out = gen.generate_reply(
            GenerationRequest(
                customer_message="buffering",
                conversation_context=[],
                predicted_intent="playback_error",
                evidence=[{"case_id": "1", "similarity": 0.7, "support_response": "restart"}],
            )
        )
    assert out.reply.startswith("Sorry")
    assert out.recommended_escalation is False


def test_grounded_agent_deterministic(tiny_cases):
    texts = [c.customer_message for c in tiny_cases]
    labels = [c.intent or "other_ambiguous" for c in tiny_cases]
    clf = TfidfLogRegIntentBaseline().fit(texts, labels)
    backend = create_embedding_backend("tfidf_svd")
    retriever = SemanticRetriever(backend=backend, top_k=3, min_similarity=0.0).fit(tiny_cases)
    agent = GroundedSupportAgent(
        classifier=clf,
        retriever=retriever,
        policy=GroundedEscalationPolicy(),
        generator=DeterministicCopyGenerator(),
        mode="deterministic",
    )
    # billing → escalate
    r = agent.run("I was charged twice and need a refund")
    assert r.should_escalate is True
    assert r.escalation_reason in {"billing_refund", "account_specific", "weak_evidence"}
    assert isinstance(r.to_dict(), dict)
    # playback-ish may escalate or auto depending on thresholds; still structured
    r2 = agent.run("Hulu keeps buffering during live TV")
    assert r2.intent is not None
    assert "evidence" in r2.to_dict()


def test_retriever_save_load(tmp_path: Path, tiny_cases):
    backend = TfidfSvdEmbedding(n_components=8)
    retriever = SemanticRetriever(backend=backend, top_k=2).fit(tiny_cases)
    path = retriever.save(tmp_path / "ret")
    loaded = SemanticRetriever.load(path)
    a = retriever.retrieve("buffering live")
    b = loaded.retrieve("buffering live")
    assert [h.case_id for h in a] == [h.case_id for h in b]


def test_create_embedding_auto_falls_back():
    be = create_embedding_backend("tfidf_svd")
    assert be.name == "tfidf_svd"
