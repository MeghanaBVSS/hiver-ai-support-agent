# Final evaluation table (Phase 3B)

| SYSTEM | INTENT MACRO-F1 | RETRIEVAL QUALITY | REPLY CORRECTNESS | REPLY GROUNDEDNESS | REPLY HELPFULNESS | ESCALATION SAFETY (unsafe auto-handle ↓) | AUTO-HANDLE COVERAGE |
|---|---:|---:|---:|---:|---:|---:|---:|
| Majority | 0.022 | N/A | N/A | N/A | N/A | 0.000 | 0.000 |
| TF-IDF+LR | 0.668 | N/A | N/A | N/A | N/A | 0.253 | 0.317 |
| TF-IDF retrieve-and-copy | 0.362* | TF-IDF NN | 3.14 (human) | 3.18 (human) | 3.06 (human) | 0.149 | 0.176 |
| Semantic + policy (no LLM) | 0.668 | top1 intent agree 0.43† | 2.92 (human) | 4.86 (human) | 2.58 (human) | **0.069** | 0.166 |
| Semantic + LLM draft | NOT YET MEASURED | NOT YET MEASURED | NOT YET MEASURED | NOT YET MEASURED | NOT YET MEASURED | NOT YET MEASURED | NOT YET MEASURED |
| LLM judge v1 as metric | N/A | N/A | not for headline | **not for headline** (esc bias) | screening only | N/A | N/A |

Notes:
- \* neighbor-intent diagnostic for retrieval systems
- † retrieval diagnostic, not reply correctness
- Human reply scores: N=50 `annotator_1` on semantic vs baseline replies
- Escalation safety = `unsafe_auto_handle_rate` on golden N=199
