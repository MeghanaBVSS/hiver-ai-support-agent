# Judge rubric v1 (original)

Used for the full 50/50 evaluation with `ollama:qwen2.5:3b`.

Scores are integers 1–5.

```
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
```

## Known ambiguity (discovered in Phase 3B)

`groundedness` did not explicitly say how to score a **safe escalation template** that invents no policies/prices/actions when evidence is insufficient.

Result: the local judge often scored escalate replies as poorly grounded (mean judge groundedness ≈1.87 on escalated cases) while humans scored them 5 (no unsupported claims).
