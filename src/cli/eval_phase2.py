"""Phase-2 evaluation: retrieval quality, agent modes, matrix, failures, human pack."""

from __future__ import annotations

import argparse
import ast
import json
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.agent.grounded_agent import GroundedSupportAgent
from src.config import ROOT
from src.evaluation.harness import ExampleRecord, evaluate_agent
from src.evaluation.judge import agreement_metrics_spec, build_judge_prompt, JudgeInput
from src.generation.providers import DeterministicCopyGenerator
from src.generation.retrieve_and_copy import RetrieveAndCopyReplier
from src.intent.baselines import MajorityIntentBaseline, TfidfLogRegIntentBaseline
from src.policy.grounded_policy import GroundedEscalationPolicy, PolicyThresholds
from src.retrieval.semantic import SemanticRetriever
from src.retrieval.tfidf_baseline import TfidfResponseRetriever


def _parse_context(value) -> list[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            out = ast.literal_eval(value)
            return list(out) if isinstance(out, list) else []
        except Exception:
            return []
    return []


def load_golden_examples() -> list[ExampleRecord]:
    golden = pd.read_csv(ROOT / "evaluation" / "golden" / "golden_set.csv")
    examples = []
    for r in golden.itertuples():
        examples.append(
            ExampleRecord(
                example_id=str(r.example_id),
                customer_message=str(r.customer_message),
                gold_intent=str(r.gold_intent),
                gold_escalate=bool(r.gold_escalate),
                conversation_context=_parse_context(
                    getattr(r, "conversation_context", None)
                ),
            )
        )
    return examples


def evaluate_retrieval(retriever: SemanticRetriever, examples: list[ExampleRecord]) -> dict:
    top1_agree = []
    top3_agree = []
    sims = []
    below = 0
    repeated = 0
    threshold = retriever.min_similarity
    for ex in examples:
        hits = retriever.retrieve(ex.customer_message)
        if not hits:
            top1_agree.append(False)
            top3_agree.append(False)
            sims.append(0.0)
            below += 1
            continue
        sims.append(hits[0].similarity)
        if hits[0].similarity < threshold:
            below += 1
        intents = [h.metadata.get("intent") for h in hits]
        top1_agree.append(bool(intents) and intents[0] == ex.gold_intent)
        top3_agree.append(ex.gold_intent in intents[:3])
        convs = [h.metadata.get("conversation_id") for h in hits]
        if len(convs) != len(set(convs)):
            repeated += 1
    arr = np.asarray(sims, dtype=float)
    return {
        "status": "OBSERVED",
        "n": len(examples),
        "top1_intent_agreement": float(np.mean(top1_agree)),
        "top3_intent_agreement": float(np.mean(top3_agree)),
        "similarity_mean": float(arr.mean()),
        "similarity_median": float(np.median(arr)),
        "similarity_p25": float(np.percentile(arr, 25)),
        "similarity_p75": float(np.percentile(arr, 75)),
        "pct_below_evidence_threshold": float(below / max(len(examples), 1)),
        "pct_duplicate_conversation_in_hits": float(repeated / max(len(examples), 1)),
        "note": "Neighbor intent agreement is a retrieval diagnostic, not reply correctness.",
    }


class NeighborIntentWrapper:
    """Attach top-neighbor intent for retrieval-only systems in the matrix."""

    def __init__(self, inner, intent_lookup: dict[str, str], name: str):
        self.inner = inner
        self.intent_lookup = intent_lookup
        self.name = name

    def predict(self, example: ExampleRecord):
        pred = self.inner.predict(example)
        if pred.intent is None and pred.retrieved_evidence:
            top = pred.retrieved_evidence[0]
            sid = str(top.get("source_id") or top.get("case_id") or "")
            # try train tweet id keys
            intent = self.intent_lookup.get(sid) or self.intent_lookup.get(
                str(top.get("conversation_id") or "")
            )
            if intent is None:
                # semantic evidence metadata path
                meta_intent = None
                if "metadata" in top and isinstance(top["metadata"], dict):
                    meta_intent = top["metadata"].get("intent")
                intent = meta_intent or top.get("intent")
            pred.intent = intent
            pred.intent_confidence = float(top.get("similarity") or top.get("score") or 0.0)
        return pred


def run_phase2_eval(mode_llm: bool = False) -> dict:
    art = ROOT / "data" / "processed" / "agent_artifacts"
    out = ROOT / "reports" / "phase2"
    out.mkdir(parents=True, exist_ok=True)
    results_dir = ROOT / "evaluation" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    examples = load_golden_examples()
    train = pd.read_parquet(ROOT / "data" / "processed" / "hulu_train_labeled.parquet")
    clf = joblib.load(art / "intent_classifier.joblib")
    retriever = SemanticRetriever.load(art / "retriever")
    th = PolicyThresholds(**json.loads((art / "policy_thresholds.json").read_text())["thresholds"])

    retrieval_report = evaluate_retrieval(retriever, examples)
    (out / "retrieval_quality.json").write_text(json.dumps(retrieval_report, indent=2))

    majority = MajorityIntentBaseline().fit(
        train.customer_message.tolist(), train.gold_intent.tolist()
    )
    intent_by_tid = {
        str(r.customer_tweet_id): str(r.gold_intent) for r in train.itertuples()
    }

    tfidf_retriever = TfidfResponseRetriever(top_k=5, min_similarity=0.0).fit(
        train.customer_message.tolist(),
        train.support_response.tolist(),
        source_ids=train.customer_tweet_id.astype(str).tolist(),
        conversation_ids=train.conversation_id.astype(str).tolist(),
    )
    tfidf_copy = NeighborIntentWrapper(
        RetrieveAndCopyReplier(retriever=tfidf_retriever),
        intent_by_tid,
        name="tfidf_retrieval",
    )

    det_agent = GroundedSupportAgent(
        classifier=clf,
        retriever=retriever,
        policy=GroundedEscalationPolicy(th),
        generator=DeterministicCopyGenerator(),
        mode="deterministic",
    )

    agents: dict = {
        "majority": majority,
        "tfidf_logreg": clf,
        "tfidf_retrieval": tfidf_copy,
        "semantic_retrieval_no_llm": det_agent,
    }

    llm_status = "NOT_RUN_NO_KEY"
    if mode_llm:
        try:
            from src.generation.providers import create_generator, OpenAICompatibleGenerator

            gen = create_generator("llm")
            if isinstance(gen, OpenAICompatibleGenerator) and not gen.api_key:
                llm_status = "UNAVAILABLE: no OPENAI_API_KEY"
            else:
                llm_agent = GroundedSupportAgent(
                    classifier=clf,
                    retriever=retriever,
                    policy=GroundedEscalationPolicy(th),
                    generator=gen,
                    mode="llm",
                )
                agents["semantic_llm"] = llm_agent
                llm_status = "ATTEMPTED"
        except Exception as exc:
            llm_status = f"UNAVAILABLE: {exc}"
    else:
        # Placeholder row for matrix completeness
        pass

    matrix: dict = {
        "status": "OBSERVED",
        "brand": "hulu_support",
        "llm_status": llm_status,
        "systems": {},
        "reply_quality_note": "Human/LLM-judge reply dimensions are NOT YET MEASURED until human_ratings.csv is labeled.",
    }
    all_preds: dict[str, list] = {}

    for name, agent in agents.items():
        harness = evaluate_agent(agent, examples)
        entry = {
            "intent": harness.intent_metrics.to_dict() if harness.intent_metrics else "NOT_APPLICABLE",
            "escalation": harness.escalation_metrics.to_dict()
            if harness.escalation_metrics
            else "NOT_APPLICABLE",
            "n_predictions": len(harness.predictions),
            "reply_quality": {
                "correctness": "NOT YET MEASURED",
                "groundedness": "NOT YET MEASURED",
                "helpfulness": "NOT YET MEASURED",
                "actionability": "NOT YET MEASURED",
                "brand_style_fit": "NOT YET MEASURED",
                "note": "Requires human ratings / validated judge agreement.",
            },
        }
        matrix["systems"][name] = entry
        all_preds[name] = harness.predictions

    if "semantic_llm" not in matrix["systems"]:
        matrix["systems"]["semantic_llm"] = {
            "intent": "NOT YET MEASURED",
            "escalation": "NOT YET MEASURED",
            "reply_quality": {
                "correctness": "NOT YET MEASURED",
                "groundedness": "NOT YET MEASURED",
                "helpfulness": "NOT YET MEASURED",
                "actionability": "NOT YET MEASURED",
                "brand_style_fit": "NOT YET MEASURED",
            },
            "note": llm_status,
        }

    # Coverage / unsafe for semantic agent
    if "semantic_retrieval_no_llm" in matrix["systems"]:
        esc = matrix["systems"]["semantic_retrieval_no_llm"].get("escalation")
        if isinstance(esc, dict):
            matrix["systems"]["semantic_retrieval_no_llm"]["decision_summary"] = {
                "escalation_precision": esc.get("escalation_precision"),
                "escalation_recall": esc.get("escalation_recall"),
                "unsafe_auto_handle_rate": esc.get("unsafe_auto_handle_rate"),
                "coverage": esc.get("auto_handle_coverage"),
            }

    failures = []
    for ex, pred in zip(examples, all_preds["semantic_retrieval_no_llm"]):
        reasons = []
        top_sim = pred.retrieved_evidence[0]["similarity"] if pred.retrieved_evidence else 0.0
        if top_sim < 0.22:
            reasons.append("low_retrieval_similarity")
        if ex.gold_intent and pred.intent and ex.gold_intent != pred.intent:
            reasons.append("intent_confusion")
        if ex.gold_escalate is not None and bool(ex.gold_escalate) != bool(pred.escalate):
            reasons.append("escalation_disagreement")
        strength = pred.meta.get("evidence_strength", 0.0)
        if strength is not None and strength < 0.35:
            reasons.append("weak_evidence")
        if ex.gold_intent == "other_ambiguous":
            reasons.append("ambiguous_messages")
        if len(ex.customer_message) < 40:
            reasons.append("short_messages")
        if len(ex.customer_message) > 220:
            reasons.append("long_messages")
        if len(ex.conversation_context) >= 2:
            reasons.append("multi_turn_context")
        # rare intents from train distribution
        rare = {"service_outage", "how_to_feature"}
        if ex.gold_intent in rare:
            reasons.append("rare_intents")
        if reasons:
            failures.append(
                {
                    "example_id": ex.example_id,
                    "reasons": reasons,
                    "gold_intent": ex.gold_intent,
                    "pred_intent": pred.intent,
                    "gold_escalate": ex.gold_escalate,
                    "pred_escalate": pred.escalate,
                    "top_similarity": top_sim,
                    "customer_message": ex.customer_message[:240],
                }
            )

    by_reason = Counter(r for f in failures for r in f["reasons"])
    failure_report = {
        "status": "OBSERVED",
        "n_examples_with_flags": len(failures),
        "reason_counts": dict(by_reason.most_common()),
        "examples": failures[:80],
    }
    (out / "failure_analysis.json").write_text(json.dumps(failure_report, indent=2))
    (results_dir / "phase2_evaluation_matrix.json").write_text(json.dumps(matrix, indent=2))
    (out / "evaluation_matrix.json").write_text(json.dumps(matrix, indent=2))

    # Markdown comparison table (no invented values)
    lines = [
        "# Phase 2 evaluation matrix",
        "",
        f"Brand: `hulu_support` | LLM status: `{llm_status}`",
        "",
        "| System | Intent acc | Macro F1 | Weighted F1 | Esc precision | Esc recall | Unsafe auto-handle | Coverage | Reply quality |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for name, entry in matrix["systems"].items():
        intent = entry.get("intent")
        esc = entry.get("escalation")
        if isinstance(intent, dict):
            ia, mf, wf = intent.get("accuracy"), intent.get("macro_f1"), intent.get("weighted_f1")
            ia_s, mf_s, wf_s = f"{ia:.3f}", f"{mf:.3f}", f"{wf:.3f}"
        else:
            ia_s = mf_s = wf_s = str(intent) if intent else "NOT YET MEASURED"
        if isinstance(esc, dict):
            ep = esc.get("escalation_precision")
            er = esc.get("escalation_recall")
            uns = esc.get("unsafe_auto_handle_rate")
            cov = esc.get("auto_handle_coverage")
            ep_s = "NA" if ep is None else f"{ep:.3f}"
            er_s = "NA" if er is None else f"{er:.3f}"
            uns_s = "NA" if uns is None else f"{uns:.3f}"
            cov_s = "NA" if cov is None else f"{cov:.3f}"
        else:
            ep_s = er_s = uns_s = cov_s = "NOT YET MEASURED"
        lines.append(
            f"| {name} | {ia_s} | {mf_s} | {wf_s} | {ep_s} | {er_s} | {uns_s} | {cov_s} | NOT YET MEASURED |"
        )
    lines.append("")
    lines.append("Retrieval quality (semantic, golden):")
    lines.append("```json")
    lines.append(json.dumps(retrieval_report, indent=2))
    lines.append("```")
    (out / "evaluation_matrix.md").write_text("\n".join(lines))

    # Human rating package (~50)
    golden = pd.read_csv(ROOT / "evaluation" / "golden" / "golden_set.csv")
    idxs: list[int] = []
    for intent in sorted(golden.gold_intent.unique()):
        sub = golden[golden.gold_intent == intent]
        idxs.extend(sub.sample(n=min(3, len(sub)), random_state=42).index.tolist())
    remain = golden.drop(index=list(dict.fromkeys(idxs)))
    need = 50 - len(dict.fromkeys(idxs))
    if need > 0 and len(remain):
        idxs.extend(remain.sample(n=min(need, len(remain)), random_state=0).index.tolist())
    idxs = list(dict.fromkeys(idxs))[:50]
    human_dir = ROOT / "evaluation" / "reply_human"
    human_dir.mkdir(parents=True, exist_ok=True)

    id_to_tfidf = {e.example_id: p for e, p in zip(examples, all_preds["tfidf_retrieval"])}
    id_to_sem = {
        e.example_id: p for e, p in zip(examples, all_preds["semantic_retrieval_no_llm"])
    }

    rows = []
    for i in idxs:
        r = golden.loc[i]
        eid = str(r.example_id)
        rows.append(
            {
                "example_id": eid,
                "customer_message": r.customer_message,
                "baseline_reply": (id_to_tfidf[eid].reply if eid in id_to_tfidf else "") or "",
                "semantic_reply": (id_to_sem[eid].reply if eid in id_to_sem else "") or "",
                "llm_reply": "",
                "human_correctness": "",
                "human_groundedness": "",
                "human_helpfulness": "",
                "human_escalation": "",
                "human_comments": "",
            }
        )
    human_df = pd.DataFrame(rows)
    human_df.to_csv(human_dir / "human_ratings.csv", index=False)
    (human_dir / "INSTRUCTIONS.md").write_text(
        """# Human reply rating instructions

Rate replies **without** looking at model confidence scores or LLM-judge scores.

For each `example_id`, you will see:
- `baseline_reply` (TF-IDF retrieve-and-copy)
- `semantic_reply` (semantic retrieval + deterministic copy / escalation template)
- `llm_reply` (may be blank if LLM eval was not run)

Score 1–5 (integers) for:
- `human_correctness`
- `human_groundedness`
- `human_helpfulness`

Set `human_escalation` to `true` or `false` for whether a human specialist should handle the case.

Optional free-text: `human_comments`.

Leave fields blank until rated. Do not invent scores.
"""
    )

    judge_spec = {
        "status": "IMPLEMENTED",
        "class": "OpenAICompatibleJudge",
        "rubric": "src.evaluation.judge.JUDGE_RUBRIC",
        "prompt_builder": "src.evaluation.judge.build_judge_prompt",
        "agreement_status": "NOT YET MEASURED",
        "agreement_spec": agreement_metrics_spec(),
        "note": "Human ratings blank; do not compute judge-human agreement yet.",
        "sample_prompt_prefix": build_judge_prompt(
            JudgeInput(
                customer_message="example",
                generated_reply="example reply",
                predicted_intent="playback_error",
                escalation_decision=False,
                retrieved_evidence=[],
            )
        )[:400],
    }
    (out / "llm_judge_spec.json").write_text(json.dumps(judge_spec, indent=2))

    # Agreement tool stub result
    from src.evaluation.judge import compute_judge_human_agreement

    agreement = compute_judge_human_agreement(human_df.to_dict(orient="records"), [])
    (out / "judge_human_agreement.json").write_text(json.dumps(agreement, indent=2))

    thresholds_doc = json.loads((art / "policy_thresholds.json").read_text())
    summary = {
        "retrieval": retrieval_report,
        "matrix_path": str(results_dir / "phase2_evaluation_matrix.json"),
        "failures_path": str(out / "failure_analysis.json"),
        "human_ratings": str(human_dir / "human_ratings.csv"),
        "n_human_pack": len(human_df),
        "llm_status": llm_status,
        "thresholds": thresholds_doc,
        "index_size": len(retriever.cases),
        "embedding": json.loads((art / "retriever" / "meta.json").read_text()),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm", action="store_true", help="Attempt LLM-enhanced evaluation if key present")
    args = parser.parse_args(argv)
    print(json.dumps(run_phase2_eval(mode_llm=args.llm), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
