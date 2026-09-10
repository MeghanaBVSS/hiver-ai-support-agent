"""CLI: run IAA, leakage, and enhanced evaluation validation reports."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

import joblib
import pandas as pd

from src.agent.grounded_agent import GroundedSupportAgent
from src.config import ROOT
from src.evaluation.harness import ExampleRecord
from src.evaluation.validation_enhancement import (
    analyze_predictions,
    compute_second_annotator_iaa,
    evaluate_rule_aid_stack,
    format_leakage_text,
    load_golden,
    run_leakage_check,
)
from src.generation.providers import DeterministicCopyGenerator
from src.policy.grounded_policy import GroundedEscalationPolicy, PolicyThresholds
from src.retrieval.semantic import SemanticRetriever


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


def _load_final_agent() -> GroundedSupportAgent | None:
    art = ROOT / "data" / "processed" / "agent_artifacts"
    clf_path = art / "intent_classifier.joblib"
    ret_dir = art / "retriever"
    thr_path = art / "policy_thresholds.json"
    if not clf_path.exists() or not (ret_dir / "retriever.joblib").exists():
        return None
    clf = joblib.load(clf_path)
    retriever = SemanticRetriever.load(ret_dir)
    th = PolicyThresholds()
    if thr_path.exists():
        raw = json.loads(thr_path.read_text(encoding="utf-8"))
        vals = raw.get("thresholds", raw)
        th = PolicyThresholds(**{k: vals[k] for k in PolicyThresholds.__dataclass_fields__ if k in vals})
    return GroundedSupportAgent(
        classifier=clf,
        retriever=retriever,
        policy=GroundedEscalationPolicy(th),
        generator=DeterministicCopyGenerator(),
        mode="deterministic",
    )


def _write_md_report(out_dir: Path, payload: dict) -> None:
    iaa = payload["second_annotator_iaa"]
    leak = payload["leakage"]
    final = payload.get("final_system")
    stack = payload["rule_aid_stack"]
    lines = [
        "# Validation enhancement report",
        "",
        "Terminology: **human-annotated golden evaluation set** (taxonomy-guided). "
        "Independently validated on a subset using inter-annotator agreement.",
        "",
        "## 1. Second-annotator inter-annotator agreement",
        "",
        f"- Subset N = **{iaa['n_subset']}**",
        f"- Intent exact agreement = **{iaa['intent']['exact_agreement']:.3f}**",
        f"- Intent Cohen's κ = **{iaa['intent']['cohen_kappa']:.3f}**",
        f"- Escalation exact agreement = **{iaa['escalation']['exact_agreement']:.3f}**",
        f"- Escalation Cohen's κ = **{iaa['escalation']['cohen_kappa']:.3f}**",
        f"- Intent disagreements = **{iaa['intent']['n_disagreements']}**",
        f"- Escalation disagreements = **{iaa['escalation']['n_disagreements']}**",
        "",
        "Top intents causing disagreement (annotator_1 label):",
        "",
    ]
    for label, count in list(iaa["intent"]["disagreement_count_by_annotator1_label"].items())[:8]:
        lines.append(f"- `{label}`: {count}")
    lines += ["", "## 2. Golden-set leakage check", "", "```", format_leakage_text(leak).rstrip(), "```", ""]

    lines += [
        "",
        "## 3. Rule-aid vs baseline stack",
        "",
        "| System | Intent macro-F1 | Escalation F1 | Unsafe auto-handle |",
        "|---|---:|---:|---:|",
    ]
    for name, block in stack["systems"].items():
        i = block["intent"]["macro_f1"]
        e = block["escalation"].get("escalation_f1")
        u = block["escalation"].get("unsafe_auto_handle_rate")
        ef = f"{e:.3f}" if e is not None else "n/a"
        uf = f"{u:.3f}" if u is not None else "n/a"
        lines.append(f"| {name} | {i:.3f} | {ef} | {uf} |")
    lines += [
        "",
        "> **Important:** High rule-aid ↔ golden agreement is partly **circular** (golden labeling used rule-aid). "
        "Use it to show rule consistency with the annotation process, not as proof that rules beat the learned model. "
        "The fair ML comparison is TF-IDF+LR / final semantic+policy vs the human-annotated evaluation set.",
        "",
    ]

    if final:
        lines += [
            "",
            "## 4. Final system (semantic retrieval + policy)",
            "",
            "### Intent classification",
            f"- Accuracy: **{final['intent']['accuracy']:.3f}**",
            f"- Macro F1: **{final['intent']['macro_f1']:.3f}**",
            f"- Weighted F1: **{final['intent']['weighted_f1']:.3f}**",
            "",
            "### Escalation (separate decision)",
            f"- Accuracy: **{final['escalation']['escalation_accuracy']:.3f}**",
            f"- Precision: **{final['escalation']['escalation_precision']:.3f}**",
            f"- Recall: **{final['escalation']['escalation_recall']:.3f}**",
            f"- F1: **{final['escalation']['escalation_f1']:.3f}**",
            f"- Unsafe auto-handle (FN rate): **{final['escalation']['unsafe_auto_handle_rate']:.3f}**",
            f"- Confusion [[TN,FP],[FN,TP]]: `{final['escalation']['confusion_matrix']}`",
            "",
            "### By difficulty",
            "",
            "| Difficulty | N | Intent macro-F1 | Escalation F1 |",
            "|---|---:|---:|---:|",
        ]
        for diff, block in final["by_difficulty"].items():
            ef = block["escalation"].get("escalation_f1")
            ef_s = f"{ef:.3f}" if ef is not None else "n/a"
            lines.append(
                f"| {diff} | {block['n']} | {block['intent']['macro_f1']:.3f} | {ef_s} |"
            )
        lines += [
            "",
            "### other_ambiguous",
            f"- Gold rate: **{final['other_ambiguous']['gold_other_ambiguous_rate']:.3f}**",
            f"- Pred rate: **{final['other_ambiguous']['pred_other_ambiguous_rate']:.3f}**",
            f"- False ambiguous: **{final['other_ambiguous']['n_false_ambiguous']}**",
            "",
            "### Model confidence → intent accuracy",
            "",
            "| Confidence | # examples | Intent accuracy |",
            "|---|---:|---:|",
        ]
        for band, block in final["model_confidence_calibration"].items():
            lines.append(f"| {band} | {block['n']} | {block['intent_accuracy']:.3f} |")
        lines += [
            "",
            "### Escalation false negatives (dangerous errors)",
            f"- N = **{final['escalation_false_negatives']['n']}**",
            "",
        ]
        for cat, n in final["escalation_false_negatives"]["by_category"].items():
            lines.append(f"- `{cat}`: {n}")
        lines += ["", "### Error analysis (tagged mistakes)", ""]
        for cat, pct in final["error_analysis"]["category_pct_of_errors"].items():
            lines.append(f"1. **{cat.replace('_', ' ')}** — {pct}%")
            for ex in final["error_analysis"]["examples"].get(cat, [])[:2]:
                lines.append(
                    f"   - `{ex['example_id']}`: gold={ex['gold_intent']}/{ex['gold_escalate']} "
                    f"pred={ex['pred_intent']}/{ex['pred_escalate']}"
                )

    lines += [
        "",
        "## 5. Decision policy (explicit)",
        "",
        "```",
        "IF account-specific action     → ESCALATE",
        "IF billing/refund dispute      → ESCALATE",
        "IF security/privacy            → ESCALATE",
        "IF ambiguous / weak evidence   → ESCALATE",
        "IF unsafe / unsupported op     → ESCALATE",
        "ELSE IF strong evidence gates  → auto-handle (copy historical reply)",
        "ELSE                           → ESCALATE (conservative default)",
        "```",
        "",
        "Intent classification ≠ escalation decision. Policy is final authority.",
        "",
        "## 6. Architecture",
        "",
        "```",
        "Customer Message",
        "       │",
        "       ▼",
        "Preprocessing / normalize",
        "       │",
        "       ▼",
        "Intent Classification (TF-IDF+LR)",
        "       │",
        "       ├──────────────────┐",
        "       ▼                  ▼",
        "Semantic Retrieval   Escalation Policy",
        "       │                  │",
        "       └────────┬─────────┘",
        "                ▼",
        "         Final Prediction",
        "        ┌───────┴────────┐",
        "        ▼                ▼",
        "     Intent          Escalate?",
        "        │",
        "        ▼",
        "    Confidence",
        "```",
        "",
        "## 7. Limitations",
        "",
        "- Golden labels are taxonomy-guided human annotations; IAA measured on a 50-example subset.",
        "- Rule-aid may introduce confirmation bias during initial labeling.",
        "- Evaluation set is relatively small (N=199).",
        "- Taxonomy boundaries contain subjective cases (playback vs live_tv).",
        "- Performance may not generalize to unseen customer language or current Hulu policy.",
        "- Historical Hulu tweet replies are evidence, not escalation ground truth.",
        "- Semantic+LLM golden reply quality remains NOT MEASURED without API key.",
        "",
    ]
    (out_dir / "VALIDATION_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validation enhancement suite")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "reports" / "validation",
    )
    args = parser.parse_args()
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    golden = load_golden()
    iaa = compute_second_annotator_iaa()
    (out / "second_annotator_iaa.json").write_text(json.dumps(iaa, indent=2), encoding="utf-8")
    # Keep phase2 agreement file in sync with kappa
    phase2 = {
        "status": "OBSERVED",
        "annotator_id": "annotator_2",
        "n_labeled": iaa["n_subset"],
        "n_unlabeled": 0,
        "unlabeled_ids": [],
        "intent_exact_agreement_vs_gold": iaa["intent"]["exact_agreement"],
        "escalate_exact_agreement_vs_gold": iaa["escalation"]["exact_agreement"],
        "intent_cohen_kappa": iaa["intent"]["cohen_kappa"],
        "escalate_cohen_kappa": iaa["escalation"]["cohen_kappa"],
        "intent_disagreements": iaa["intent"]["n_disagreements"],
        "escalate_disagreements": iaa["escalation"]["n_disagreements"],
        "disagreement_count_by_annotator1_label": iaa["intent"][
            "disagreement_count_by_annotator1_label"
        ],
    }
    (ROOT / "reports" / "phase2" / "second_annotator_agreement.json").write_text(
        json.dumps(phase2, indent=2), encoding="utf-8"
    )

    leak = run_leakage_check()
    (out / "leakage_check.json").write_text(json.dumps(leak, indent=2), encoding="utf-8")
    (out / "leakage_check.txt").write_text(format_leakage_text(leak), encoding="utf-8")
    print(format_leakage_text(leak))

    stack = evaluate_rule_aid_stack(golden)
    (out / "rule_aid_stack.json").write_text(json.dumps(stack, indent=2), encoding="utf-8")

    final_block = None
    agent = _load_final_agent()
    if agent is not None:
        pred_intent = []
        pred_esc = []
        confs = []
        for r in golden.itertuples():
            ctx = _parse_context(getattr(r, "conversation_context", None))
            resp = agent.run(str(r.customer_message), context=ctx)
            pred_intent.append(str(resp.intent))
            pred_esc.append(bool(resp.should_escalate))
            confs.append(resp.intent_confidence)
        final_block = analyze_predictions(golden, pred_intent, pred_esc, confs)
        (out / "final_system_metrics.json").write_text(
            json.dumps(final_block, indent=2), encoding="utf-8"
        )

    payload = {
        "second_annotator_iaa": iaa,
        "leakage": leak,
        "rule_aid_stack": stack,
        "final_system": final_block,
    }
    (out / "validation_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_md_report(out, payload)
    print(f"Wrote validation reports under {out}")


if __name__ == "__main__":
    main()
