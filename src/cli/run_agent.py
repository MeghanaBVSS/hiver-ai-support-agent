"""CLI to run the grounded support agent on a message."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib  # noqa: F401 used below

from src.agent.grounded_agent import GroundedSupportAgent
from src.config import ROOT
from src.generation.providers import DeterministicCopyGenerator, create_generator
from src.policy.grounded_policy import GroundedEscalationPolicy, PolicyThresholds
from src.retrieval.semantic import SemanticRetriever


def load_agent(mode: str = "deterministic") -> GroundedSupportAgent:
    art = ROOT / "data" / "processed" / "agent_artifacts"
    if not (art / "retriever" / "retriever.joblib").exists():
        raise FileNotFoundError(
            f"Missing artifacts at {art}. Run: python -m src.cli.build_index"
        )
    clf = joblib.load(art / "intent_classifier.joblib")
    retriever = SemanticRetriever.load(art / "retriever")
    th_path = art / "policy_thresholds.json"
    if th_path.exists():
        th = PolicyThresholds(**json.loads(th_path.read_text())["thresholds"])
    else:
        th = PolicyThresholds()
    if mode == "llm":
        try:
            generator = create_generator("llm")
        except Exception:
            generator = DeterministicCopyGenerator()
            mode = "deterministic"
    else:
        generator = DeterministicCopyGenerator()
    return GroundedSupportAgent(
        classifier=clf,
        retriever=retriever,
        policy=GroundedEscalationPolicy(th),
        generator=generator,
        mode="deterministic" if mode != "llm" else "llm",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run grounded Hulu support agent")
    parser.add_argument("--message", required=True)
    parser.add_argument("--mode", choices=["deterministic", "llm"], default="deterministic")
    args = parser.parse_args(argv)
    agent = load_agent(args.mode)
    result = agent.run(args.message)
    print(json.dumps(result.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
