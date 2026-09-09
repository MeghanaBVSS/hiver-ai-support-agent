"""LLM-as-judge for reply quality (no gold labels in the prompt)."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol


JUDGE_DIMENSIONS = (
    "correctness",
    "groundedness",
    "helpfulness",
    "actionability",
    "brand_style",
    "escalation_appropriateness",
)

JUDGE_RUBRIC_V1 = """
Fixed rubric (1–5 ordinal scales):
1 = clearly wrong / ungrounded / unhelpful / unsafe auto-handle or wrong escalate
2 = major issues
3 = mixed / partial
4 = mostly good with minor issues
5 = excellent for the dimension

correctness: factual alignment with customer issue + evidence (no invented policy)
groundedness: claims supported by retrieved evidence / customer context
helpfulness: useful next step or clear explanation
actionability: customer knows what to do next (or why escalate)
brand_style: polite, concise, brand-appropriate support tone
escalation_appropriateness: escalate when evidence/risk weak; avoid unnecessary escalate when strong
""".strip()

JUDGE_RUBRIC_V2 = """
Fixed rubric (1–5 ordinal scales):
1 = clearly wrong / invents unsupported facts / unsafe auto-handle
2 = major issues
3 = mixed / partial
4 = mostly good with minor issues
5 = excellent for the dimension

correctness: factual alignment with the customer issue; no invented policy/pricing/account state
groundedness:
  - Direct / evidence-backed answers: claims must be supported by retrieved evidence or customer text
  - Safe escalation / insufficient-evidence responses: score HIGH if the reply abstains from unsupported claims and states escalation/uncertainty clearly
  - CRITICAL RULE: Do NOT penalize groundedness for failing to provide unsupported information when the correct behavior is escalation
  - Account-specific / billing / security: escalating without fabricating account actions is grounded
  - Unsupported claims (prices, refunds, timelines, password resets not in evidence): score LOW
helpfulness: useful next step or clear explanation (escalation can be helpful if it is the right next step)
actionability: customer knows what to do next OR why a specialist will take over
brand_style: polite, concise, brand-appropriate support tone
escalation_appropriateness: escalate when evidence/risk weak; avoid unnecessary escalate when strong evidence supports a safe reply
""".strip()

# Default for new calls = v2; v1 preserved for historical 50/50 run
JUDGE_RUBRIC = JUDGE_RUBRIC_V2


@dataclass
class JudgeInput:
    customer_message: str
    generated_reply: str
    predicted_intent: str | None
    escalation_decision: bool
    escalation_reason: str | None = None
    conversation_context: list[str] = field(default_factory=list)
    retrieved_evidence: list[dict[str, Any]] = field(default_factory=list)
    # Explicitly excluded: gold_intent, gold_reply, gold_escalate


@dataclass
class JudgeScores:
    correctness: int
    groundedness: int
    helpfulness: int
    actionability: int
    brand_style: int
    escalation_appropriateness: int
    rationale: str
    model_name: str | None = None
    raw_response: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class HumanScores:
    example_id: str
    correctness: int
    groundedness: int
    helpfulness: int
    actionability: int | None = None
    brand_style: int | None = None
    escalation_appropriateness: int | None = None
    annotator_id: str = "human"


class LLMJudge(Protocol):
    def score(self, payload: JudgeInput) -> JudgeScores:
        """Score a single example without gold labels."""


class NullJudge:
    """Placeholder judge used until an LLM backend is configured."""

    def score(self, payload: JudgeInput) -> JudgeScores:
        raise NotImplementedError(
            "LLM judge backend requires OPENAI_API_KEY / LLM_MODEL. "
            "Use OpenAICompatibleJudge when configured."
        )


def build_judge_prompt(payload: JudgeInput, rubric_version: str = "v2") -> str:
    """Construct a judge prompt that excludes gold labels."""
    rubric = JUDGE_RUBRIC_V2 if rubric_version == "v2" else JUDGE_RUBRIC_V1
    evidence_lines = []
    for i, ev in enumerate(payload.retrieved_evidence, start=1):
        evidence_lines.append(
            f"[{i}] case={ev.get('case_id', 'n/a')} sim={ev.get('similarity', ev.get('score', 'n/a'))} "
            f"customer={ev.get('customer_message', '')!r} "
            f"support={ev.get('support_response', '')!r}"
        )
    evidence_block = "\n".join(evidence_lines) if evidence_lines else "(none)"
    context_block = (
        "\n".join(payload.conversation_context) if payload.conversation_context else "(none)"
    )

    return f"""
You are evaluating a customer-support agent reply for Hulu. Score each dimension from 1 (poor) to 5 (excellent).
Do NOT receive or invent gold labels. Do not assume facts that are not in the customer message or retrieved evidence.

Rubric version: {rubric_version}

{rubric}

Customer message:
{payload.customer_message}

Conversation context:
{context_block}

Predicted intent: {payload.predicted_intent}
Escalation decision: {payload.escalation_decision}
Escalation reason: {payload.escalation_reason}

Generated reply:
{payload.generated_reply}

Retrieved historical evidence:
{evidence_block}

Return JSON with integer keys:
correctness, groundedness, helpfulness, actionability, brand_style, escalation_appropriateness, rationale
""".strip()


class OpenAICompatibleJudge:
    """LLM judge via OpenAI-compatible API. Never receives gold labels."""

    def __init__(
        self,
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 500,
        timeout: float = 60.0,
        rubric_version: str = "v2",
    ):
        self.model = model or os.environ.get("LLM_JUDGE_MODEL") or os.environ.get("LLM_MODEL") or "gpt-4o-mini"
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.api_key = os.environ.get("OPENAI_API_KEY")
        self.base_url = os.environ.get("OPENAI_BASE_URL")
        self.rubric_version = rubric_version

    def score(self, payload: JudgeInput) -> JudgeScores:
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY required for LLM judge")
        from openai import OpenAI

        client = OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=self.timeout)
        prompt = build_judge_prompt(payload, rubric_version=self.rubric_version)
        completion = client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": "You are a careful support-reply evaluator. Return JSON only. Never invent gold labels.",
                },
                {"role": "user", "content": prompt},
            ],
        )
        raw = completion.choices[0].message.content or "{}"
        data = json.loads(raw)

        def _clamp(key: str) -> int:
            val = int(data.get(key, 1))
            return max(1, min(5, val))

        return JudgeScores(
            correctness=_clamp("correctness"),
            groundedness=_clamp("groundedness"),
            helpfulness=_clamp("helpfulness"),
            actionability=_clamp("actionability"),
            brand_style=_clamp("brand_style"),
            escalation_appropriateness=_clamp("escalation_appropriateness"),
            rationale=str(data.get("rationale") or ""),
            model_name=self.model,
            raw_response=raw,
        )


class OllamaJudge:
    """LLM judge via local Ollama HTTP API. Never receives gold labels."""

    def __init__(
        self,
        model: str | None = None,
        temperature: float = 0.0,
        timeout: float = 180.0,
        base_url: str | None = None,
        rubric_version: str = "v2",
    ):
        self.model = model or os.environ.get("OLLAMA_JUDGE_MODEL") or os.environ.get("OLLAMA_MODEL") or "qwen2.5-coder:14b"
        self.temperature = temperature
        self.timeout = timeout or float(os.environ.get("OLLAMA_TIMEOUT", "300"))
        self.base_url = (base_url or os.environ.get("OLLAMA_HOST") or "http://127.0.0.1:11434").rstrip("/")
        self.rubric_version = rubric_version

    def score(self, payload: JudgeInput) -> JudgeScores:
        import urllib.request

        prompt = build_judge_prompt(payload, rubric_version=self.rubric_version)
        body = {
            "model": self.model,
            "stream": False,
            "format": "json",
            "options": {"temperature": self.temperature},
            "messages": [
                {
                    "role": "system",
                    "content": "You are a careful support-reply evaluator. Return JSON only. Never invent gold labels.",
                },
                {"role": "user", "content": prompt},
            ],
        }
        req = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            out = json.loads(resp.read().decode("utf-8"))
        raw = (out.get("message") or {}).get("content") or "{}"
        data = json.loads(raw)

        def _clamp(key: str) -> int:
            val = int(data.get(key, 1))
            return max(1, min(5, val))

        return JudgeScores(
            correctness=_clamp("correctness"),
            groundedness=_clamp("groundedness"),
            helpfulness=_clamp("helpfulness"),
            actionability=_clamp("actionability"),
            brand_style=_clamp("brand_style"),
            escalation_appropriateness=_clamp("escalation_appropriateness"),
            rationale=str(data.get("rationale") or ""),
            model_name=f"ollama:{self.model}",
            raw_response=raw,
        )


def create_judge(mode: str = "null") -> LLMJudge:
    if mode in {"null", "none"}:
        return NullJudge()
    if mode in {"llm", "openai"}:
        return OpenAICompatibleJudge()
    if mode in {"ollama", "local"}:
        return OllamaJudge()
    raise ValueError(f"Unknown judge mode: {mode}")


def agreement_metrics_spec() -> dict[str, Any]:
    """Document planned agreement metrics. Do not compute without human labels."""
    return {
        "status": "NOT_YET_COMPUTED",
        "requires": "human annotations on ~50 examples",
        "metrics": [
            "exact_agreement",
            "agreement_within_one",
            "spearman_correlation",
            "cohens_kappa",
            "weighted_kappa",
        ],
        "note": "Do not compute until real human annotations exist.",
    }


def compute_judge_human_agreement(
    human_rows: list[dict[str, Any]],
    judge_rows: list[dict[str, Any]],
    dimensions: tuple[str, ...] = ("correctness", "groundedness", "helpfulness"),
) -> dict[str, Any]:
    """Compare judge vs human ordinal scores. Call only when human labels exist."""
    by_id_h = {str(r["example_id"]): r for r in human_rows if r.get("example_id")}
    by_id_j = {str(r["example_id"]): r for r in judge_rows if r.get("example_id")}
    common = sorted(set(by_id_h) & set(by_id_j))
    if not common:
        return {
            "status": "NOT YET MEASURED",
            "reason": "No overlapping labeled examples",
            "spec": agreement_metrics_spec(),
        }

    # Require at least one human dimension filled
    labeled = []
    for eid in common:
        h = by_id_h[eid]
        if any(str(h.get(f"human_{d}", "")).strip() not in {"", "nan", "None"} for d in dimensions):
            labeled.append(eid)
    if not labeled:
        return {
            "status": "NOT YET MEASURED",
            "reason": "Human rating fields are blank",
            "n_pack": len(common),
            "spec": agreement_metrics_spec(),
        }

    out: dict[str, Any] = {"status": "OBSERVED", "n": len(labeled), "dimensions": {}}
    disagreements: list[dict[str, Any]] = []
    for dim in dimensions:
        h_scores = []
        j_scores = []
        for eid in labeled:
            hv = by_id_h[eid].get(f"human_{dim}", by_id_h[eid].get(dim))
            jv = by_id_j[eid].get(f"judge_{dim}", by_id_j[eid].get(dim))
            if hv in (None, "", "nan") or jv in (None, "", "nan"):
                continue
            try:
                hi, ji = int(hv), int(jv)
            except (TypeError, ValueError):
                continue
            h_scores.append(hi)
            j_scores.append(ji)
            if abs(hi - ji) >= 2:
                disagreements.append(
                    {
                        "example_id": eid,
                        "dimension": dim,
                        "human": hi,
                        "judge": ji,
                        "delta": ji - hi,
                    }
                )
        if not h_scores:
            out["dimensions"][dim] = "NOT YET MEASURED"
            continue
        exact = sum(1 for a, b in zip(h_scores, j_scores) if a == b) / len(h_scores)
        within1 = sum(1 for a, b in zip(h_scores, j_scores) if abs(a - b) <= 1) / len(h_scores)
        # Spearman via rank correlation
        import numpy as np
        from scipy.stats import spearmanr

        corr = float(spearmanr(h_scores, j_scores).correlation) if len(set(h_scores)) > 1 else None
        # Weighted kappa if sklearn available
        try:
            from sklearn.metrics import cohen_kappa_score

            kappa = float(cohen_kappa_score(h_scores, j_scores, weights="quadratic"))
        except Exception:
            kappa = None
        bias = float(np.mean(np.asarray(j_scores) - np.asarray(h_scores)))
        out["dimensions"][dim] = {
            "n": len(h_scores),
            "exact_agreement": exact,
            "agreement_within_1": within1,
            "spearman": corr,
            "weighted_kappa": kappa,
            "mean_judge_minus_human": bias,
        }
    out["disagreements"] = disagreements[:50]
    out["note"] = "Disagreements with |delta|>=2 are surfaced; do not hide them."
    return out
