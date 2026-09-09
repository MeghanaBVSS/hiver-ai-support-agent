"""Targeted re-judge of Phase-3B disagreement subset with rubric v2.

Does NOT overwrite v1 scores.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from src.cli.run_agent import load_agent
from src.config import ROOT
from src.evaluation.judge import JudgeInput, OllamaJudge


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen2.5:3b")
    parser.add_argument("--retries", type=int, default=3)
    args = parser.parse_args(argv)

    rh = ROOT / "evaluation" / "reply_human"
    candidates = json.loads((ROOT / "reports" / "phase3" / "rejudge_candidates.json").read_text())
    ids = candidates["example_ids"]
    human = pd.read_csv(rh / "human_ratings.csv")
    v1 = pd.read_csv(rh / "llm_judge_scores_v1.csv")
    by_h = {str(r.example_id): r for r in human.itertuples()}
    by_v1 = {str(r.example_id): r for r in v1.itertuples()}

    agent = load_agent("deterministic")
    judge = OllamaJudge(model=args.model, rubric_version="v2")
    rows = []
    compare = []
    for i, eid in enumerate(ids, start=1):
        r = by_h[eid]
        pred = agent.run(str(r.customer_message))
        payload = JudgeInput(
            customer_message=str(r.customer_message),
            generated_reply=str(r.semantic_reply),
            predicted_intent=pred.intent,
            escalation_decision=bool(pred.should_escalate),
            escalation_reason=pred.escalation_reason,
            retrieved_evidence=pred.evidence,
        )
        scores = None
        last_err = None
        for attempt in range(1, args.retries + 1):
            try:
                scores = judge.score(payload)
                break
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                print(f"[{i}/{len(ids)}] {eid} retry {attempt}: {exc}", flush=True)
                time.sleep(2 * attempt)
        if scores is None:
            raise RuntimeError(f"failed {eid}: {last_err}")
        row = {
            "example_id": eid,
            "judge_correctness": scores.correctness,
            "judge_groundedness": scores.groundedness,
            "judge_helpfulness": scores.helpfulness,
            "judge_actionability": scores.actionability,
            "judge_brand_style": scores.brand_style,
            "judge_escalation_appropriateness": scores.escalation_appropriateness,
            "judge_rationale": scores.rationale,
            "judge_model": scores.model_name,
            "rubric_version": "v2",
        }
        rows.append(row)
        old_g = int(by_v1[eid].judge_groundedness)
        compare.append(
            {
                "example_id": eid,
                "human_groundedness": int(r.human_groundedness),
                "v1_judge_groundedness": old_g,
                "v2_judge_groundedness": scores.groundedness,
                "delta_v2_minus_v1": scores.groundedness - old_g,
                "abs_err_v1": abs(old_g - int(r.human_groundedness)),
                "abs_err_v2": abs(scores.groundedness - int(r.human_groundedness)),
                "reason_for_rerun": "top groundedness disagreement; test rubric v2 escalation rule",
            }
        )
        print(
            f"[{i}/{len(ids)}] {eid} g v1={old_g} -> v2={scores.groundedness} human={int(r.human_groundedness)}",
            flush=True,
        )

    out_csv = rh / "llm_judge_scores_v2.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    summary = {
        "n": len(rows),
        "example_ids": ids,
        "model": args.model,
        "rubric": "v2",
        "mean_abs_err_v1": float(sum(c["abs_err_v1"] for c in compare) / len(compare)),
        "mean_abs_err_v2": float(sum(c["abs_err_v2"] for c in compare) / len(compare)),
        "mean_v2_minus_v1_groundedness": float(sum(c["delta_v2_minus_v1"] for c in compare) / len(compare)),
        "within_1_v1": float(sum(c["abs_err_v1"] <= 1 for c in compare) / len(compare)),
        "within_1_v2": float(sum(c["abs_err_v2"] <= 1 for c in compare) / len(compare)),
        "comparisons": compare,
    }
    (ROOT / "reports" / "phase3" / "targeted_rejudge_v2.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: summary[k] for k in summary if k != "comparisons"}, indent=2))
    print("wrote", out_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
