"""Run LLM-as-judge on the human reply pack and compute agreement."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from src.cli.run_agent import load_agent
from src.config import ROOT
from src.evaluation.judge import (
    JudgeInput,
    OllamaJudge,
    compute_judge_human_agreement,
    create_judge,
)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["ollama", "openai"], default="ollama")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--model", default=None)
    parser.add_argument("--retries", type=int, default=3)
    args = parser.parse_args(argv)

    human_path = ROOT / "evaluation" / "reply_human" / "human_ratings.csv"
    out_dir = ROOT / "evaluation" / "reply_human"
    judge_path = out_dir / "llm_judge_scores.csv"
    human = pd.read_csv(human_path).head(args.limit)

    done: dict[str, dict] = {}
    if judge_path.exists():
        prev = pd.read_csv(judge_path)
        for r in prev.to_dict(orient="records"):
            done[str(r["example_id"])] = r
        print(f"resuming with {len(done)} cached scores", flush=True)

    agent = load_agent("deterministic")
    if args.mode == "ollama":
        judge = OllamaJudge(model=args.model) if args.model else OllamaJudge()
    else:
        judge = create_judge("openai")

    rows = []
    for i, r in enumerate(human.itertuples(), start=1):
        eid = str(r.example_id)
        if eid in done:
            rows.append(done[eid])
            print(f"[{i}/{len(human)}] {eid} cached", flush=True)
            continue

        msg = str(r.customer_message)
        pred = agent.run(msg)
        payload = JudgeInput(
            customer_message=msg,
            generated_reply=str(r.semantic_reply),
            predicted_intent=pred.intent,
            escalation_decision=bool(pred.should_escalate),
            escalation_reason=pred.escalation_reason,
            retrieved_evidence=pred.evidence,
        )
        last_err = None
        scores = None
        for attempt in range(1, args.retries + 1):
            try:
                scores = judge.score(payload)
                break
            except Exception as exc:  # noqa: BLE001 — retry transient Ollama drops
                last_err = exc
                print(f"[{i}/{len(human)}] {eid} retry {attempt}/{args.retries}: {exc}", flush=True)
                time.sleep(2 * attempt)
        if scores is None:
            raise RuntimeError(f"Judge failed for {eid}: {last_err}")

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
        }
        rows.append(row)
        pd.DataFrame(rows).to_csv(judge_path, index=False)
        print(f"[{i}/{len(human)}] {eid} ok", flush=True)

    agreement = compute_judge_human_agreement(
        human.to_dict(orient="records"),
        rows,
        dimensions=("correctness", "groundedness", "helpfulness", "actionability", "brand_style"),
    )
    agreement["judge_model"] = rows[0]["judge_model"] if rows else None
    agreement["n_scored"] = len(rows)
    agreement_path = ROOT / "reports" / "phase2" / "judge_human_agreement.json"
    agreement_path.write_text(json.dumps(agreement, indent=2))
    print(json.dumps(agreement, indent=2))
    print("wrote", judge_path)
    print("wrote", agreement_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
