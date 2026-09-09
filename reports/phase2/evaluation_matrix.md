# Phase 2 evaluation matrix

Brand: `hulu_support` | LLM status: `NOT_RUN_NO_KEY`

| System | Intent acc | Macro F1 | Weighted F1 | Esc precision | Esc recall | Unsafe auto-handle | Coverage | Reply C/G/H |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| majority | 0.126 | 0.022 | 0.028 | 0.437 | 1.000 | 0.000 | 0.000 | NOT YET MEASURED |
| tfidf_logreg | 0.693 | 0.668 | 0.675 | 0.478 | 0.747 | 0.253 | 0.317 | NOT YET MEASURED |
| tfidf_retrieval | 0.352 | 0.362 | 0.360 | 0.451 | 0.851 | 0.149 | 0.176 | 3.14/3.18/3.06 (human) |
| semantic_retrieval_no_llm | 0.693 | 0.668 | 0.675 | 0.488 | 0.931 | 0.069 | 0.166 | 2.92/4.86/2.58 (human) |
| semantic_llm | NOT YET MEASURED | NOT YET MEASURED | NOT YET MEASURED | NOT YET MEASURED | NOT YET MEASURED | NOT YET MEASURED | NOT YET MEASURED | NOT YET MEASURED |

## Human evaluation (annotator_1 / annotator_2)

- Pack size: 50 reply ratings (`annotator_1`)
- Semantic escalate rate: 90%
- Human would-escalate: 54%
- Escalation agreement (semantic vs human): 60%
- Second-annotator (`annotator_2`) intent agreement vs gold: 62%
- Second-annotator escalate agreement vs gold: 72%

Full write-up: `reports/phase2/HUMAN_ANNOTATION.md`
