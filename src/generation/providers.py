"""LLM provider abstraction + deterministic fallback generation."""

from __future__ import annotations

import json
import os
import re
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class GenerationRequest:
    customer_message: str
    conversation_context: list[str]
    predicted_intent: str | None
    evidence: list[dict[str, Any]]
    should_escalate_hint: bool | None = None


@dataclass
class GenerationResult:
    reply: str
    grounding_summary: str
    unsupported_claims: list[str] = field(default_factory=list)
    recommended_escalation: bool = False
    model: str = "deterministic"
    provider: str = "local"
    raw: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ReplyGenerator(ABC):
    @abstractmethod
    def generate_reply(self, request: GenerationRequest) -> GenerationResult:
        ...


class DeterministicCopyGenerator(ReplyGenerator):
    """No-API fallback: use top historical support response."""

    def generate_reply(self, request: GenerationRequest) -> GenerationResult:
        if not request.evidence:
            return GenerationResult(
                reply=(
                    "Thanks for reaching out. I don't have enough similar historical "
                    "cases to draft a safe answer, so I'm escalating this to a specialist."
                ),
                grounding_summary="No retrieval evidence available.",
                unsupported_claims=[],
                recommended_escalation=True,
                model="deterministic_copy",
                provider="local",
            )
        top = request.evidence[0]
        reply = str(top.get("support_response") or "").strip()
        summary = (
            f"Copied historical support reply from case {top.get('case_id')} "
            f"(similarity={top.get('similarity')})."
        )
        return GenerationResult(
            reply=reply,
            grounding_summary=summary,
            unsupported_claims=[],
            recommended_escalation=False,
            model="deterministic_copy",
            provider="local",
        )


SYSTEM_PROMPT = """You are a customer-support drafting assistant for Hulu.

Your job is to draft a helpful response using ONLY the supplied customer context and retrieved historical support evidence.

Do not invent:
- policies
- pricing
- refunds
- account information
- features
- guarantees
- timelines
- operational actions

If the evidence is insufficient or contradictory, produce a short escalation recommendation rather than fabricating an answer.

Retrieved historical examples are evidence of how support handled similar issues, not guaranteed current policy.

Return structured JSON with keys:
reply, grounding_summary, unsupported_claims (array of strings), recommended_escalation (boolean).
"""


class OpenAICompatibleGenerator(ReplyGenerator):
    """Optional OpenAI-compatible chat completion generator."""

    def __init__(
        self,
        model: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 400,
        timeout: float = 60.0,
        api_key_env: str = "OPENAI_API_KEY",
        base_url: str | None = None,
    ):
        self.model = model or os.environ.get("LLM_MODEL") or "gpt-4o-mini"
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.api_key = os.environ.get(api_key_env) or os.environ.get("OPENAI_API_KEY")
        self.base_url = base_url or os.environ.get("OPENAI_BASE_URL")

    def generate_reply(self, request: GenerationRequest) -> GenerationResult:
        if not self.api_key:
            raise RuntimeError("No API key configured for LLM generator")
        from openai import OpenAI

        client = OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=self.timeout)
        user_payload = {
            "customer_message": request.customer_message,
            "conversation_context": request.conversation_context,
            "predicted_intent": request.predicted_intent,
            "retrieved_evidence": request.evidence,
        }
        completion = client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(user_payload)},
            ],
        )
        raw = completion.choices[0].message.content or "{}"
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return GenerationResult(
                reply="",
                grounding_summary="Model returned non-JSON output.",
                unsupported_claims=["non_json_output"],
                recommended_escalation=True,
                model=self.model,
                provider="openai_compatible",
                raw=raw,
            )
        return GenerationResult(
            reply=str(data.get("reply") or ""),
            grounding_summary=str(data.get("grounding_summary") or ""),
            unsupported_claims=[str(x) for x in (data.get("unsupported_claims") or [])],
            recommended_escalation=bool(data.get("recommended_escalation")),
            model=self.model,
            provider="openai_compatible",
            raw=raw,
        )


def create_generator(mode: str = "deterministic") -> ReplyGenerator:
    if mode in {"deterministic", "no_llm", "local"}:
        return DeterministicCopyGenerator()
    if mode in {"llm", "openai"}:
        return OpenAICompatibleGenerator()
    raise ValueError(f"Unknown generator mode: {mode}")


_PRICE = re.compile(r"\$\s?\d+(?:\.\d+)?|\b\d+\s?(?:dollars|usd)\b", re.I)
_TIMELINE = re.compile(
    r"\b(?:within|in)\s+\d+\s+(?:minutes?|hours?|days?|weeks?)\b|\bby tomorrow\b|\btoday\b",
    re.I,
)
_ACCOUNT_ACTION = re.compile(
    r"\b(?:i (?:have )?reset|we (?:have )?refunded|credited your|changed your password|closed your account)\b",
    re.I,
)


def validate_generation(
    result: GenerationResult,
    evidence: list[dict[str, Any]],
) -> dict[str, Any]:
    """Lightweight post-generation checks. Imperfect by design."""
    flags: list[str] = []
    reply = (result.reply or "").strip()
    if not reply:
        flags.append("empty_response")
    evidence_text = " ".join(
        f"{e.get('customer_message','')} {e.get('support_response','')}" for e in evidence
    ).lower()
    if _PRICE.search(reply) and not _PRICE.search(evidence_text):
        flags.append("hallucinated_price")
    if _TIMELINE.search(reply) and not _TIMELINE.search(evidence_text):
        flags.append("hallucinated_timeline")
    if _ACCOUNT_ACTION.search(reply):
        flags.append("hallucinated_account_action")
    # unsupported claims already from model
    flags.extend([f"model_claim:{c}" for c in result.unsupported_claims])
    escalate = bool(flags) or result.recommended_escalation or not evidence
    return {
        "ok": len(flags) == 0 and bool(reply),
        "flags": flags,
        "force_escalate": escalate and bool(flags),
    }
