"""Enhanced evaluation analyses: IAA, leakage printout, slices, FN audit, rule-aid."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.config import ROOT
from src.evaluation.iaa import compute_binary_agreement, compute_categorical_agreement
from src.evaluation.leakage import check_golden_contamination
from src.evaluation.metrics import compute_escalation_metrics, compute_intent_metrics
from src.intent.annotate_hulu import assign_escalate, assign_intent
from src.intent.taxonomy_hulu import ALLOWED_INTENTS


FN_CATEGORIES = (
    "account_specific_action",
    "billing_refund",
    "security_privacy",
    "ambiguous",
    "unsafe_automation",
    "unsupported_operation",
    "other",
)


def _bool_series(s: pd.Series) -> pd.Series:
    return s.map(lambda v: str(v).strip().lower() in {"1", "true", "t", "yes", "y"} if not isinstance(v, bool) else v)


def load_golden() -> pd.DataFrame:
    return pd.read_csv(ROOT / "evaluation" / "golden" / "golden_set.csv")


def load_second_annotator() -> pd.DataFrame:
    return pd.read_csv(ROOT / "evaluation" / "golden" / "second_annotator.csv")


def compute_second_annotator_iaa(golden: pd.DataFrame | None = None, second: pd.DataFrame | None = None) -> dict[str, Any]:
    golden = load_golden() if golden is None else golden
    second = load_second_annotator() if second is None else second
    merged = second.merge(
        golden[["example_id", "gold_intent", "gold_escalate"]],
        on="example_id",
        how="inner",
    )
    if merged.empty:
        raise ValueError("No overlapping example_ids between golden and second_annotator")

    intent = compute_categorical_agreement(
        merged["gold_intent"],
        merged["gold_intent_2"],
        labels=list(ALLOWED_INTENTS),
    )
    # Attach example_ids to intent disagreements
    intent_disagreements = []
    for i, (a, b) in enumerate(zip(merged["gold_intent"], merged["gold_intent_2"])):
        if str(a) != str(b):
            intent_disagreements.append(
                {
                    "example_id": str(merged.iloc[i]["example_id"]),
                    "gold_intent": str(a),
                    "second_annotator_intent": str(b),
                    "customer_message": str(merged.iloc[i]["customer_message"])[:180],
                }
            )
    intent["disagreements"] = intent_disagreements

    esc = compute_binary_agreement(merged["gold_escalate"], merged["gold_escalate_2"])
    esc_disagreements = []
    for i, (a, b) in enumerate(
        zip(_bool_series(merged["gold_escalate"]), _bool_series(merged["gold_escalate_2"]))
    ):
        if a != b:
            esc_disagreements.append(
                {
                    "example_id": str(merged.iloc[i]["example_id"]),
                    "gold_escalate": bool(a),
                    "second_annotator_escalate": bool(b),
                    "customer_message": str(merged.iloc[i]["customer_message"])[:180],
                }
            )
    esc["disagreements"] = esc_disagreements

    return {
        "status": "OBSERVED",
        "terminology": "human-annotated golden evaluation set vs blind second-annotator subset",
        "annotator_1_id": "phase1_engineer",
        "annotator_2_id": "annotator_2",
        "n_subset": int(len(merged)),
        "protocol": (
            "Second annotator pack provides message+context only; labels filled without "
            "first-annotator labels, model predictions, or rule-aid suggestions in the pack."
        ),
        "intent": intent,
        "escalation": esc,
        "interpretation": (
            "Independently validated on a subset using inter-annotator agreement. "
            "Cohen's kappa accounts for chance agreement; exact % alone overstates reliability."
        ),
    }


def run_leakage_check() -> dict[str, Any]:
    golden = load_golden()
    golden_ids = set(golden["customer_tweet_id"].astype(str))
    golden_example_ids = set(golden["example_id"].astype(str))

    train_path = ROOT / "data" / "processed" / "hulu_train_labeled.parquet"
    train_ids: set[str] = set()
    if train_path.exists():
        train = pd.read_parquet(train_path)
        id_col = "customer_tweet_id" if "customer_tweet_id" in train.columns else None
        if id_col:
            train_ids = set(train[id_col].astype(str))

    retrieval_ids: set[str] = set()
    corpus_path = ROOT / "data" / "processed" / "agent_artifacts" / "retrieval_corpus.parquet"
    if corpus_path.exists():
        corpus = pd.read_parquet(corpus_path)
        for col in ("customer_tweet_id", "case_id", "example_id"):
            if col in corpus.columns:
                retrieval_ids |= set(corpus[col].astype(str))

    # Prompt exemplars: this project does not inject golden IDs into prompts.
    prompt_path = ROOT / "configs" / "prompt_examples.json"
    prompt_ids: set[str] = set()
    if prompt_path.exists():
        data = json.loads(prompt_path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            prompt_ids = {str(x.get("example_id") or x.get("id")) for x in data if isinstance(x, dict)}
        elif isinstance(data, dict):
            prompt_ids = {str(x) for x in data.get("example_ids", [])}

    report = check_golden_contamination(
        golden_ids=golden_ids,
        train_ids=train_ids,
        retrieval_ids=retrieval_ids,
        prompt_example_ids=prompt_ids,
    )

    train_overlap = sorted(golden_ids & train_ids)
    retrieval_overlap = sorted(golden_ids & retrieval_ids)
    # Also block example_id collisions if any retrieval uses gold_* ids
    retrieval_example_overlap = sorted(golden_example_ids & retrieval_ids)
    prompt_overlap = sorted(golden_ids & prompt_ids) + sorted(golden_example_ids & prompt_ids)

    status = "PASS" if (
        len(train_overlap) == 0
        and len(retrieval_overlap) == 0
        and len(retrieval_example_overlap) == 0
        and len(prompt_overlap) == 0
        and report.ok
    ) else "FAIL"

    return {
        "status": status,
        "training_overlap": len(train_overlap),
        "retrieval_overlap": len(retrieval_overlap) + len(retrieval_example_overlap),
        "prompt_example_overlap": len(prompt_overlap),
        "training_overlap_ids_sample": train_overlap[:20],
        "retrieval_overlap_ids_sample": (retrieval_overlap + retrieval_example_overlap)[:20],
        "prompt_overlap_ids_sample": prompt_overlap[:20],
        "n_golden_tweet_ids": len(golden_ids),
        "n_train_ids": len(train_ids),
        "n_retrieval_ids": len(retrieval_ids),
        "n_prompt_ids": len(prompt_ids),
        "artifacts_present": {
            "train_labeled": train_path.exists(),
            "retrieval_corpus": corpus_path.exists(),
            "prompt_examples": prompt_path.exists(),
        },
        "library_report": report.to_dict(),
        "note": (
            "Golden is drawn from held-out test conversations by design; "
            "test∩golden overlap is expected and not a leakage failure. "
            "Leakage fails only if golden IDs appear in train fitting, retrieval index, or prompt exemplars."
        ),
    }


def format_leakage_text(result: dict[str, Any]) -> str:
    return (
        "Golden-set leakage check\n"
        "------------------------\n"
        f"Training overlap:       {result['training_overlap']}\n"
        f"Retrieval overlap:      {result['retrieval_overlap']}\n"
        f"Prompt-example overlap: {result['prompt_example_overlap']}\n"
        f"Status: {result['status']}\n"
    )


def categorize_escalation_fn(message: str, gold_intent: str, escalation_reason: str | None) -> str:
    text = (message or "").lower()
    reason = (escalation_reason or "").lower()
    if gold_intent == "login_account" or "account" in reason or "password" in text:
        return "account_specific_action"
    if gold_intent == "billing_subscription" or "billing" in reason or "refund" in text:
        return "billing_refund"
    if "security" in reason or any(w in text for w in ("hacked", "fraud", "privacy", "stolen")):
        return "security_privacy"
    if gold_intent == "other_ambiguous" or "ambiguous" in reason:
        return "ambiguous"
    if any(w in text for w in ("sue", "lawyer", "ssn", "social security")):
        return "unsupported_operation"
    if "unsafe" in reason or "harmful" in reason:
        return "unsafe_automation"
    return "other"


def confidence_band_from_prob(p: float | None) -> str:
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "unknown"
    if p >= 0.75:
        return "high"
    if p >= 0.55:
        return "medium"
    return "low"


def analyze_predictions(
    golden: pd.DataFrame,
    pred_intent: list[str],
    pred_escalate: list[bool],
    intent_confidence: list[float | None] | None = None,
) -> dict[str, Any]:
    y_true_i = golden["gold_intent"].astype(str).tolist()
    y_true_e = _bool_series(golden["gold_escalate"]).tolist()
    intent_m = compute_intent_metrics(y_true_i, pred_intent, labels=list(ALLOWED_INTENTS))
    esc_m = compute_escalation_metrics(y_true_e, pred_escalate)

    # Difficulty slices
    by_diff: dict[str, Any] = {}
    for diff in ["easy", "medium", "hard"]:
        mask = (golden["difficulty"].astype(str) == diff).to_numpy()
        if mask.sum() == 0:
            continue
        idxs = np.where(mask)[0]
        by_diff[diff] = {
            "n": int(mask.sum()),
            "intent": compute_intent_metrics(
                [y_true_i[i] for i in idxs],
                [pred_intent[i] for i in idxs],
                labels=list(ALLOWED_INTENTS),
            ).to_dict(),
            "escalation": compute_escalation_metrics(
                [y_true_e[i] for i in idxs],
                [pred_escalate[i] for i in idxs],
            ).to_dict(),
        }

    # Annotator-confidence slices (label metadata, not model conf)
    by_ann_conf: dict[str, Any] = {}
    if "annotator_confidence" in golden.columns:
        for band in ["high", "medium", "low"]:
            mask = (golden["annotator_confidence"].astype(str) == band).to_numpy()
            if mask.sum() == 0:
                continue
            idxs = np.where(mask)[0]
            by_ann_conf[band] = {
                "n": int(mask.sum()),
                "intent_accuracy": float(
                    np.mean([y_true_i[i] == pred_intent[i] for i in idxs])
                ),
                "escalation_accuracy": float(
                    np.mean([y_true_e[i] == pred_escalate[i] for i in idxs])
                ),
            }

    # Model confidence calibration
    conf_calib: dict[str, Any] = {}
    if intent_confidence is not None:
        bands = [confidence_band_from_prob(p) for p in intent_confidence]
        for band in ["high", "medium", "low", "unknown"]:
            idxs = [i for i, b in enumerate(bands) if b == band]
            if not idxs:
                continue
            conf_calib[band] = {
                "n": len(idxs),
                "intent_accuracy": float(np.mean([y_true_i[i] == pred_intent[i] for i in idxs])),
            }

    # other_ambiguous analysis
    pred_amb_idx = [i for i, p in enumerate(pred_intent) if p == "other_ambiguous"]
    gold_amb_idx = [i for i, t in enumerate(y_true_i) if t == "other_ambiguous"]
    false_ambiguous = []
    for i in pred_amb_idx:
        if y_true_i[i] != "other_ambiguous":
            false_ambiguous.append(
                {
                    "example_id": str(golden.iloc[i]["example_id"]),
                    "gold_intent": y_true_i[i],
                    "pred_intent": pred_intent[i],
                    "customer_message": str(golden.iloc[i]["customer_message"])[:200],
                    "manual_review": "Predicted other_ambiguous but gold assigned a concrete intent — likely dump/under-specified model behavior.",
                }
            )
    ambiguous = {
        "gold_other_ambiguous_rate": float(len(gold_amb_idx) / max(len(golden), 1)),
        "pred_other_ambiguous_rate": float(len(pred_amb_idx) / max(len(golden), 1)),
        "n_pred_other_ambiguous": len(pred_amb_idx),
        "n_false_ambiguous": len(false_ambiguous),
        "false_ambiguous_examples": false_ambiguous[:15],
        "true_ambiguous_correct": int(
            sum(1 for i in gold_amb_idx if pred_intent[i] == "other_ambiguous")
        ),
    }

    # Escalation FN audit (dangerous errors)
    fn_rows = []
    for i, (yt, yp) in enumerate(zip(y_true_e, pred_escalate)):
        if yt and not yp:
            reason = golden.iloc[i]["escalation_reason"] if "escalation_reason" in golden.columns else ""
            cat = categorize_escalation_fn(
                str(golden.iloc[i]["customer_message"]),
                y_true_i[i],
                str(reason) if reason is not None else "",
            )
            fn_rows.append(
                {
                    "example_id": str(golden.iloc[i]["example_id"]),
                    "gold_intent": y_true_i[i],
                    "pred_intent": pred_intent[i],
                    "category": cat,
                    "gold_escalation_reason": str(reason),
                    "customer_message": str(golden.iloc[i]["customer_message"])[:220],
                }
            )
    fn_by_cat = Counter(r["category"] for r in fn_rows)

    # Error analysis on mistakes (intent wrong OR escalation wrong)
    error_cats = Counter()
    examples_by_cat: dict[str, list[dict]] = defaultdict(list)
    for i in range(len(golden)):
        intent_err = y_true_i[i] != pred_intent[i]
        esc_err = y_true_e[i] != pred_escalate[i]
        if not intent_err and not esc_err:
            continue
        msg = str(golden.iloc[i]["customer_message"])
        cats: list[str] = []
        if intent_err:
            pair = {y_true_i[i], pred_intent[i]}
            if pair == {"playback_error", "live_tv_issues"} or pair == {"playback_error", "app_device_issue"}:
                cats.append("intent_boundary_confusion")
            elif y_true_i[i] == "other_ambiguous" or pred_intent[i] == "other_ambiguous":
                cats.append("ambiguous_customer_wording")
            elif len(msg) > 220 or msg.lower().count(" and ") + msg.lower().count(" also ") >= 2:
                cats.append("multiple_intents")
            elif any(d in msg.lower() for d in ("roku", "fire tv", "apple tv", "android", "ios", "ps4")):
                cats.append("device_context_confusion")
            else:
                cats.append("intent_boundary_confusion")
        if esc_err:
            cats.append("escalation_policy_ambiguity")
        for c in cats:
            error_cats[c] += 1
            if len(examples_by_cat[c]) < 3:
                examples_by_cat[c].append(
                    {
                        "example_id": str(golden.iloc[i]["example_id"]),
                        "gold_intent": y_true_i[i],
                        "pred_intent": pred_intent[i],
                        "gold_escalate": bool(y_true_e[i]),
                        "pred_escalate": bool(pred_escalate[i]),
                        "customer_message": msg[:180],
                    }
                )

    n_mistakes = sum(
        1
        for i in range(len(golden))
        if y_true_i[i] != pred_intent[i] or y_true_e[i] != pred_escalate[i]
    )
    error_analysis = {
        "n_examples_with_any_error": n_mistakes,
        "category_counts": dict(error_cats),
        "category_pct_of_errors": {
            k: round(100.0 * v / max(sum(error_cats.values()), 1), 1)
            for k, v in error_cats.most_common()
        },
        "examples": dict(examples_by_cat),
        "note": "Categories co-occur; percentages are over category tags, not mutually exclusive rows.",
    }

    return {
        "intent": intent_m.to_dict(),
        "escalation": esc_m.to_dict(),
        "by_difficulty": by_diff,
        "by_annotator_confidence": by_ann_conf,
        "model_confidence_calibration": conf_calib,
        "other_ambiguous": ambiguous,
        "escalation_false_negatives": {
            "n": len(fn_rows),
            "by_category": dict(fn_by_cat),
            "examples": fn_rows,
        },
        "error_analysis": error_analysis,
    }


def evaluate_rule_aid_stack(golden: pd.DataFrame) -> dict[str, Any]:
    """Compare rule-aid alone vs LR baseline vs final semantic+policy when available."""
    messages = golden["customer_message"].astype(str).tolist()
    y_true_i = golden["gold_intent"].astype(str).tolist()
    y_true_e = _bool_series(golden["gold_escalate"]).tolist()

    # Rule-aid alone
    rule_intents = []
    rule_esc = []
    for msg in messages:
        intent, _ = assign_intent(msg)
        esc, _ = assign_escalate(msg, intent)
        rule_intents.append(intent)
        rule_esc.append(esc)

    systems: dict[str, Any] = {
        "rule_aid_only": {
            "intent": compute_intent_metrics(y_true_i, rule_intents, labels=list(ALLOWED_INTENTS)).to_dict(),
            "escalation": compute_escalation_metrics(y_true_e, rule_esc).to_dict(),
            "note": "Regex/priority rules encoding taxonomy — annotation aid evaluated as a standalone system.",
        }
    }

    clf_path = ROOT / "data" / "processed" / "agent_artifacts" / "intent_classifier.joblib"
    if clf_path.exists():
        from src.evaluation.harness import ExampleRecord

        clf = joblib.load(clf_path)
        lr_intents = []
        lr_conf: list[float | None] = []
        for msg in messages:
            ex = ExampleRecord(example_id="tmp", customer_message=msg)
            pred = clf.predict(ex)
            lr_intents.append(str(pred.intent))
            lr_conf.append(float(pred.intent_confidence) if pred.intent_confidence is not None else None)
        thr = 0.55
        lr_esc = []
        for intent, conf, msg in zip(lr_intents, lr_conf, messages):
            if intent in {"billing_subscription", "login_account", "other_ambiguous"}:
                lr_esc.append(True)
            elif conf is not None and conf < thr:
                lr_esc.append(True)
            else:
                esc, _ = assign_escalate(msg, intent)
                lr_esc.append(esc)
        systems["tfidf_logreg_plus_simple_policy"] = {
            "intent": compute_intent_metrics(y_true_i, lr_intents, labels=list(ALLOWED_INTENTS)).to_dict(),
            "escalation": compute_escalation_metrics(y_true_e, lr_esc).to_dict(),
            "note": "TF-IDF+LR intent with simple high-stakes/confidence escalate (no semantic retrieval).",
        }
        hybrid_intents = []
        hybrid_esc = []
        for msg, li, conf in zip(messages, lr_intents, lr_conf):
            ri, _ = assign_intent(msg)
            intent = ri if (conf is not None and conf < 0.45) else li
            esc, _ = assign_escalate(msg, intent)
            hybrid_intents.append(intent)
            hybrid_esc.append(esc)
        systems["logreg_plus_rule_aid"] = {
            "intent": compute_intent_metrics(y_true_i, hybrid_intents, labels=list(ALLOWED_INTENTS)).to_dict(),
            "escalation": compute_escalation_metrics(y_true_e, hybrid_esc).to_dict(),
            "note": "LR intent with rule-aid fallback on low confidence; rule escalate.",
        }

    return {
        "status": "OBSERVED",
        "systems": systems,
        "comparison_note": (
            "Rule-aid is an annotation/taxonomy encoding aid. Final production decisioning uses "
            "semantic retrieval + GroundedEscalationPolicy (see final system block in validation report)."
        ),
        "lr_available": clf_path.exists(),
    }
