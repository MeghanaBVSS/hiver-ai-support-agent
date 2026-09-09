# Judge calibration (Phase 3B)

## Status

| Item | Value |
|---|---|
| Judge model | `ollama:qwen2.5:3b` |
| Pack size | **50/50 complete** (v1 rubric) |
| Runtime | ~**48 minutes** for 50 examples (~1 min/example) |
| Original scores | `evaluation/reply_human/llm_judge_scores_v1.csv` (also `llm_judge_scores.csv`) |
| Human reference | `evaluation/reply_human/human_ratings.csv` (`annotator_1`) |

## Why an LLM judge

Automated ordinal scoring of reply quality at scale, with a fixed rubric and **no gold labels in the prompt**. The judge is an **evaluator**, not ground truth. Human ratings are the reference for calibration.

## Rubric

- **v1** (original 50-run): `evaluation/reply_human/judge_rubric_v1.md`
- **v2** (calibrated): `evaluation/reply_human/judge_rubric_v2.md`

v2 adds an explicit rule: **do not penalize groundedness for omitting unsupported details when escalation is the correct behavior.**

## Observed agreement (v1 vs human, N=50)

| Dimension | Exact | Within ±1 | MAE | Weighted κ | Bias (J−H) |
|---|---:|---:|---:|---:|---:|
| correctness | 0.36 | 0.72 | — | 0.15 | −0.66 |
| groundedness | 0.06 | **0.12** | — | −0.03 | **−2.86** |
| helpfulness | 0.32 | **0.82** | — | 0.21 | +0.84 |
| actionability | 0.38 | **0.88** | — | 0.10 | +0.14 |
| brand_style | 0.06 | 0.56 | — | 0.01 | +1.38 |

Full group splits: `reports/phase3/agreement_by_group.json`.

## Groundedness root cause (evidence-backed)

| Group | n | Human g mean | Judge g mean | Within ±1 |
|---|---:|---:|---:|---:|
| Non-escalated | 5 | 3.6 | 3.2 | **1.00** |
| Escalated | 45 | 5.0 | 1.87 | **0.02** |
| All | 50 | 4.86 | 2.0 | 0.12 |

Hypothesis: *“judge treats safe escalation as insufficiently grounded.”*

- Broad match (escalated ∧ human≥4 ∧ judge≤2): **31** cases  
- Share of |diff|>1 disagreements: **~70%**  
- **Supported by data** (not assumed a priori)

Top-15 disagreements: `reports/phase3/top15_groundedness_disagreements.json`.

## Reliability verdict

- **Strongest:** actionability, helpfulness (within±1)
- **Weakest:** groundedness (v1)
- **Systematic bias:** yes on groundedness for escalated templates (judge under-scores)

## Is qwen2.5:3b adequate?

| Aspect | Assessment |
|---|---|
| Helpfulness / actionability screening | Usable with caution |
| Groundedness (v1 rubric) | **Not adequate** for headline metrics |
| Runtime | Acceptable for N=50; too slow for frequent full golden runs |
| Headline metric | **Do not use judge as final headline** — use human ratings + deterministic safety metrics |

## Targeted re-judge (v2)

Re-run **only** the 15 largest groundedness disagreements with rubric v2 (`qwen2.5:3b`).  
Results: `evaluation/reply_human/llm_judge_scores_v2.csv`, `reports/phase3/targeted_rejudge_v2.json`.  
Original v1 scores are never overwritten.

| Metric on 15-subset | v1 | v2 |
|---|---:|---:|
| Mean |diff| to human | 4.00 | **3.07** |
| Within ±1 | 0.00 | **0.13** |
| Mean (v2−v1) groundedness | — | **+0.93** |

**Interpretation:** Rubric v2 moves scores in the right direction on average but does **not** fully repair groundedness agreement on this 3B model. Headline recommendation unchanged: do not use judge groundedness as product GT.

## What human validation would improve

1. Blind re-rate groundedness on escalated templates with the v2 definition.
2. Separate “grounded abstention” from “grounded answer” in the human form.
3. Larger N and a second annotator for IAA on reply dimensions.

## Non-negotiable

> **LLM judge = evaluator. Human ratings = reference. Judge must not label the test/golden set as ground truth.**
