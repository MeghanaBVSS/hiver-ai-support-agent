# Headline metric

## Chosen metric

**Unsafe auto-handle rate** of the frozen semantic retrieval + deterministic policy agent on the golden set.

**Value (MEASURED, exact):** **6 / 87 = 0.068965… ≈ 0.069**

## Exact definition

Among golden examples with `gold_escalate = true` (N⁺ = **87** of 199):

\[
\text{unsafe auto-handle rate} = \frac{\mathrm{FN}}{N^{+}} = \frac{\#\{\text{gold escalate} \land \text{predicted auto-handle}\}}{\#\{\text{gold escalate}\}}
\]

Independent recomputation (frozen agent):

| Cell | Count |
|---|---:|
| TP (escalate∧gold escalate) | 81 |
| FP (escalate∧gold auto) | 85 |
| FN (auto∧gold escalate) | **6** |
| TN (auto∧gold auto) | 27 |
| Gold escalate positives | **87** |
| Predicted auto-handle | 33 |

- **Unsafe auto-handle** = 6/87 ≈ **0.069**
- **Auto-handle coverage** = 33/199 ≈ **0.166**
- **Escalation recall** = 81/87 ≈ **0.931**
- **Escalation precision** = 81/(81+85) ≈ **0.488**

## Critical wording

**Do not say** “6.9% of all messages are unsafe.”  
It is **6.9% of gold-escalate-positive cases** that were incorrectly auto-handled.

At the same operating point, only **16.6% of all golden messages** are auto-handled.

## Why chosen

Primary trust risk for a support agent is auto-handling cases that should escalate. Judge groundedness is disqualified (systematic escalation bias). Intent accuracy is shared with TF-IDF+LR.

## What it does not measure

Reply helpfulness; over-escalation UX cost; current Hulu policy truth; LLM draft quality.

## Source

`reports/final/metric_audit_recompute.json` and `evaluation/results/phase2_evaluation_matrix.json`
