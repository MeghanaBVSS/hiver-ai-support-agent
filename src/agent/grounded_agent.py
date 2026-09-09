"""Grounded support agent for hulu_support."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from src.evaluation.harness import AgentPrediction, ExampleRecord
from src.generation.providers import (
    DeterministicCopyGenerator,
    GenerationRequest,
    ReplyGenerator,
    create_generator,
    validate_generation,
)
from src.intent.baselines import TfidfLogRegIntentBaseline
from src.policy.grounded_policy import (
    GroundedEscalationPolicy,
    PolicyInput,
    PolicyThresholds,
)
from src.retrieval.semantic import (
    EvidenceScoreConfig,
    SemanticRetriever,
    compute_evidence_strength,
)


Mode = Literal["deterministic", "llm"]


@dataclass
class AgentResponse:
    message: str
    intent: str | None
    intent_confidence: float | None
    evidence: list[dict[str, Any]]
    evidence_strength: float
    draft_reply: str
    should_escalate: bool
    escalation_reason: str
    policy_signals: dict[str, Any] = field(default_factory=dict)
    generation_model: str = "deterministic"
    grounding_summary: str = ""
    validation_flags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class GroundedSupportAgent:
    """Intent → retrieve → policy → (optional) grounded generation."""

    name = "grounded_hulu_agent"

    def __init__(
        self,
        classifier: TfidfLogRegIntentBaseline,
        retriever: SemanticRetriever,
        policy: GroundedEscalationPolicy | None = None,
        generator: ReplyGenerator | None = None,
        mode: Mode = "deterministic",
        evidence_config: EvidenceScoreConfig | None = None,
    ):
        self.classifier = classifier
        self.retriever = retriever
        self.policy = policy or GroundedEscalationPolicy()
        self.mode = mode
        self.generator = generator or (
            create_generator("llm") if mode == "llm" else DeterministicCopyGenerator()
        )
        self.evidence_config = evidence_config or EvidenceScoreConfig()

    def run(self, message: str, context: list[str] | None = None) -> AgentResponse:
        example = ExampleRecord(
            example_id="adhoc",
            customer_message=message,
            conversation_context=context or [],
        )
        return self._run_example(example)

    def predict(self, example: ExampleRecord) -> AgentPrediction:
        result = self._run_example(example)
        return AgentPrediction(
            intent=result.intent,
            intent_confidence=result.intent_confidence,
            reply=result.draft_reply,
            escalate=result.should_escalate,
            escalate_reason=result.escalation_reason,
            retrieved_evidence=result.evidence,
            meta={
                "evidence_strength": result.evidence_strength,
                "policy_signals": result.policy_signals,
                "generation_model": result.generation_model,
                "validation_flags": result.validation_flags,
                "grounding_summary": result.grounding_summary,
            },
        )

    def _run_example(self, example: ExampleRecord) -> AgentResponse:
        clf = self.classifier.predict(example)
        hits = self.retriever.retrieve(example.customer_message)
        evidence = [h.to_dict() for h in hits]
        strength = compute_evidence_strength(hits, self.evidence_config)
        conflicting = strength.intent_agreement < 0.5 and len(hits) >= 2
        # crude reply conflict: top replies start differently
        if len(hits) >= 2:
            a = (hits[0].support_response or "").split()[:6]
            b = (hits[1].support_response or "").split()[:6]
            if a != b and abs(len(hits[0].support_response) - len(hits[1].support_response)) > 80:
                conflicting = True

        decision = self.policy.decide(
            PolicyInput(
                customer_message=example.customer_message,
                predicted_intent=clf.intent,
                intent_confidence=clf.intent_confidence,
                top_similarity=hits[0].similarity if hits else None,
                evidence_strength=strength.score,
                evidence_intent_agreement=strength.intent_agreement,
                has_conflicting_evidence=conflicting,
                n_hits=len(hits),
            )
        )

        draft = ""
        grounding = ""
        model = "none"
        flags: list[str] = []

        if decision.should_escalate:
            draft = (
                "I'm escalating this to a Hulu specialist because: "
                f"{decision.escalation_reason}. "
                "A human agent can take a closer look with your account context."
            )
            grounding = "Escalation path — no unverified operational answer drafted."
            model = "escalation_template"
        else:
            # Generate grounded draft (deterministic copy or LLM)
            gen = self.generator.generate_reply(
                GenerationRequest(
                    customer_message=example.customer_message,
                    conversation_context=example.conversation_context,
                    predicted_intent=clf.intent,
                    evidence=evidence,
                )
            )
            validation = validate_generation(gen, evidence)
            flags = list(validation["flags"])
            draft = gen.reply
            grounding = gen.grounding_summary
            model = gen.model
            if validation["force_escalate"]:
                decision.should_escalate = True
                decision.escalation_reason = "potentially_harmful_automation"
                decision.policy_signals = {
                    **decision.policy_signals,
                    "validation_flags": flags,
                    "fired": list(decision.policy_signals.get("fired") or [])
                    + ["generation_validation_failed"],
                }
                draft = (
                    "I'm escalating this because the drafted answer may include "
                    "unsupported claims or insufficient grounding."
                )
                grounding = "Forced escalation after generation validation flags."

        return AgentResponse(
            message=example.customer_message,
            intent=clf.intent,
            intent_confidence=clf.intent_confidence,
            evidence=evidence,
            evidence_strength=strength.score,
            draft_reply=draft,
            should_escalate=decision.should_escalate,
            escalation_reason=decision.escalation_reason,
            policy_signals=decision.policy_signals,
            generation_model=model,
            grounding_summary=grounding,
            validation_flags=flags,
        )
