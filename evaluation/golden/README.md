# Golden evaluation set

This directory holds the human-labeled golden set (150–250 examples) once labeled.

Phase 0 status: **DESIGN ONLY** — see `reports/phase0/golden_sampling_plan.json`.

Rules:
- Do not mix golden examples into train/valid fitting sets.
- Do not add golden examples to the retrieval index.
- Do not use golden texts as few-shot prompt exemplars.
- Run contamination checks before evaluation.
