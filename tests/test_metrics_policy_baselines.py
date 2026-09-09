"""Metrics, escalation policy, baselines, and audit smoke tests."""

from src.data.audit import audit_dataframe
from src.data.brand_analysis import rank_brands, recommend_brand
from src.evaluation.harness import ExampleRecord, evaluate_agent
from src.evaluation.metrics import compute_escalation_metrics, compute_intent_metrics
from src.generation.retrieve_and_copy import RetrieveAndCopyReplier
from src.intent.baselines import MajorityIntentBaseline, TfidfLogRegIntentBaseline
from src.intent.discovery import discover_intent_proposals
from src.policy.escalation import EscalationPolicy, EscalationSignals
from src.retrieval.tfidf_baseline import TfidfResponseRetriever


def test_intent_metrics():
    metrics = compute_intent_metrics(
        ["a", "b", "a", "c"],
        ["a", "b", "b", "c"],
        labels=["a", "b", "c"],
    )
    assert 0.0 <= metrics.accuracy <= 1.0
    assert "a" in metrics.per_class
    assert len(metrics.confusion_matrix) == 3


def test_escalation_metrics_safety():
    # One unsafe auto-handle (true escalate, predicted false)
    metrics = compute_escalation_metrics(
        y_true_escalate=[True, True, False, False],
        y_pred_escalate=[True, False, False, False],
    )
    assert metrics.unsafe_auto_handle_rate == 0.5
    assert metrics.auto_handle_coverage == 0.75


def test_escalation_policy_conservative():
    policy = EscalationPolicy()
    billing = policy.decide(
        EscalationSignals(
            intent_confidence=0.99,
            retrieval_similarity=0.9,
            customer_message="I need a refund please",
        )
    )
    assert billing.escalate is True
    assert "billing" in billing.reason.lower() or "refund" in ",".join(billing.signals_fired)

    security = policy.decide(
        EscalationSignals(
            intent_confidence=0.99,
            retrieval_similarity=0.9,
            customer_message="I think my account is hacked",
        )
    )
    assert security.escalate is True

    safe = policy.decide(
        EscalationSignals(
            intent_confidence=0.9,
            retrieval_similarity=0.8,
            customer_message="How do I update the app?",
        )
    )
    assert safe.escalate is False
    assert safe.auto_handle_eligible is True


def test_majority_and_tfidf_baselines():
    texts = [
        "login failed again",
        "cannot sign in",
        "want a refund",
        "refund my charge",
        "app crashed",
    ]
    labels = ["login", "login", "billing", "billing", "crash"]
    b0 = MajorityIntentBaseline().fit(texts, labels)
    pred0 = b0.predict(ExampleRecord("1", "anything"))
    assert pred0.intent in {"login", "billing"}

    b1 = TfidfLogRegIntentBaseline().fit(texts, labels)
    pred1 = b1.predict(ExampleRecord("2", "I need a refund now"))
    assert pred1.intent is not None
    assert pred1.intent_confidence is not None


def test_retrieve_and_copy_baseline():
    retriever = TfidfResponseRetriever(top_k=2, min_similarity=0.0).fit(
        customer_messages=["app crashes on launch", "need refund for charge"],
        support_responses=["Please update the app.", "We escalated billing."],
        source_ids=["1", "2"],
        conversation_ids=["c1", "c2"],
    )
    agent = RetrieveAndCopyReplier(retriever=retriever)
    pred = agent.predict(ExampleRecord("x", "my app is crashing"))
    assert pred.reply is not None
    assert pred.retrieved_evidence


def test_harness_runs():
    texts = ["login issue", "refund please", "login broken"]
    labels = ["login", "billing", "login"]
    agent = MajorityIntentBaseline().fit(texts, labels)
    examples = [
        ExampleRecord("1", "login issue", gold_intent="login", gold_escalate=False),
        ExampleRecord("2", "refund", gold_intent="billing", gold_escalate=True),
    ]
    result = evaluate_agent(agent, examples)
    assert result.intent_metrics is not None
    assert result.escalation_metrics is not None


def test_audit_and_brand_smoke(mini_df):
    report = audit_dataframe(mini_df)
    assert report["row_count"] == len(mini_df)
    assert "quality_issues" in report
    ranked = rank_brands(mini_df, top_n=5)
    assert ranked
    rec = recommend_brand(ranked)
    assert rec["brand"] in {"AcmeHelp", "BetaCare"}


def test_intent_discovery_proposal(mini_df):
    texts = mini_df.loc[mini_df["inbound"] == True, "text"].astype(str).tolist()
    proposals = discover_intent_proposals(texts, n_clusters=3, random_seed=42)
    assert proposals
    assert all(p.status == "PROPOSAL_NOT_GROUND_TRUTH" for p in proposals)
