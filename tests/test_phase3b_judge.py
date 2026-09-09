"""Phase 3B tests: judge parsing, agreement, rubric versions, esc aggregation."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.evaluation.judge import (
    JUDGE_RUBRIC_V1,
    JUDGE_RUBRIC_V2,
    JudgeInput,
    JudgeScores,
    build_judge_prompt,
    compute_judge_human_agreement,
)

ROOT = Path(__file__).resolve().parents[1]
RH = ROOT / "evaluation" / "reply_human"
P3 = ROOT / "reports" / "phase3"


def test_judge_score_files_aligned():
    human = pd.read_csv(RH / "human_ratings.csv")
    judge = pd.read_csv(RH / "llm_judge_scores.csv")
    assert len(human) == 50
    assert len(judge) == 50
    assert set(human.example_id.astype(str)) == set(judge.example_id.astype(str))
    score_cols = [
        "human_correctness",
        "human_groundedness",
        "human_helpfulness",
        "human_actionability",
        "human_brand_style",
    ]
    for c in score_cols:
        assert human[c].isna().sum() == 0
        assert human[c].between(1, 5).all()
    for c in [
        "judge_correctness",
        "judge_groundedness",
        "judge_helpfulness",
        "judge_actionability",
        "judge_brand_style",
        "judge_escalation_appropriateness",
    ]:
        assert judge[c].isna().sum() == 0
        assert judge[c].between(1, 5).all()


def test_v1_scores_preserved():
    v1 = RH / "llm_judge_scores_v1.csv"
    assert v1.exists()
    a = pd.read_csv(RH / "llm_judge_scores.csv")
    b = pd.read_csv(v1)
    assert len(a) == len(b) == 50
    assert list(a.example_id) == list(b.example_id)
    assert list(a.judge_groundedness) == list(b.judge_groundedness)


def test_rubric_versioning_files_and_prompts():
    assert (RH / "judge_rubric_v1.md").exists()
    assert (RH / "judge_rubric_v2.md").exists()
    assert "CRITICAL RULE" in JUDGE_RUBRIC_V2
    assert "CRITICAL RULE" not in JUDGE_RUBRIC_V1
    p1 = build_judge_prompt(
        JudgeInput("hi", "bye", "playback_error", True, "weak_evidence"),
        rubric_version="v1",
    )
    p2 = build_judge_prompt(
        JudgeInput("hi", "bye", "playback_error", True, "weak_evidence"),
        rubric_version="v2",
    )
    assert "Rubric version: v1" in p1
    assert "Rubric version: v2" in p2
    assert "CRITICAL RULE" in p2
    assert "CRITICAL RULE" not in p1


def test_agreement_calculation_smoke():
    human = [{"example_id": "a", "human_correctness": 4, "human_groundedness": 5}]
    judge = [{"example_id": "a", "judge_correctness": 4, "judge_groundedness": 1}]
    out = compute_judge_human_agreement(
        human, judge, dimensions=("correctness", "groundedness")
    )
    assert out["status"] == "OBSERVED"
    assert out["dimensions"]["correctness"]["exact_agreement"] == 1.0
    assert out["dimensions"]["groundedness"]["agreement_within_1"] == 0.0


def test_escalation_specific_aggregation_artifact():
    path = P3 / "escalation_groundedness_split.json"
    assert path.exists()
    data = json.loads(path.read_text())
    assert data["escalated"]["within_1_agreement"] < data["non_escalated"]["within_1_agreement"]
    assert data["non_escalated"]["within_1_agreement"] == pytest.approx(1.0)


def test_hypothesis_artifact_present():
    path = P3 / "escalation_template_hypothesis.json"
    data = json.loads(path.read_text())
    assert data["n_broad"] >= 1
    assert 0 <= data["pct_of_disagreements_broad"] <= 1


def test_judge_scores_dataclass_serializable():
    s = JudgeScores(3, 4, 3, 3, 4, 2, "ok", model_name="test")
    d = s.to_dict()
    assert d["groundedness"] == 4
    json.dumps(d)
