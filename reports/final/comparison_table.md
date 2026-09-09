# Canonical final comparison table

Sources: `evaluation/results/phase2_evaluation_matrix.json`, `reports/phase2/human_reply_summary.json`, human_ratings.csv.

| SYSTEM | Intent Acc | Macro F1 | W-F1 | Intent src | Reply C | Reply G | Reply H | Reply Act | Reply Style | Reply src | Coverage | Esc rate | Unsafe↓ | Esc P | Esc R | Decision src |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---|
| Majority | 0.126 | 0.022 | 0.028 | MEASURED golden199 | — | — | — | — | — | NOT MEASURED | 0.000 | 1.000 | 0.000 | 0.437 | 1.000 | MEASURED |
| TF-IDF+LR | 0.693 | 0.668 | 0.675 | MEASURED | — | — | — | — | — | NOT MEASURED | 0.317 | 0.683 | 0.253 | 0.478 | 0.747 | MEASURED |
| TF-IDF retrieval | 0.352* | 0.362* | 0.360* | MEASURED* neighbor | 3.14 | 3.18 | 3.06 | NOT MEASURED | NOT MEASURED | HUMAN N=50 | 0.176 | 0.824 | 0.149 | 0.451 | 0.851 | MEASURED |
| Semantic retrieval+policy | 0.693 | 0.668 | 0.675 | MEASURED (same LR) | 2.92 | 4.86 | 2.58 | 2.58 | 3.02 | HUMAN N=50 | 0.166 | 0.834 | **0.069** | 0.488 | 0.931 | MEASURED |
| Semantic+LLM | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED |

\* Intent for TF-IDF retrieval = neighbor-intent diagnostic (AUTOMATED), not a trained classifier.

**Architecture note (Semantic retrieval+policy):** intent accuracy matches TF-IDF+LR because the **intent classifier is the same TF-IDF+LogReg**. Semantic retrieval changes **evidence retrieval**; the grounded policy changes **escalation / auto-handle**. Do not treat equal intent metrics as a retrieval win.

**LLM judge scores:** AUTOMATED evaluator — **not** used as reply headline (groundedness within±1=0.12).
