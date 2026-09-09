# Experiment freeze

**Status:** FROZEN for submission

| Component | Frozen value |
|---|---|
| Brand | `hulu_support` |
| Intent taxonomy | 10 intents in `src/intent/taxonomy_hulu.py` |
| Golden set | `evaluation/golden/golden_set.csv` N=**199** |
| Splits | conversation-level 70/15/15 in `hulu_support_pairs_split.parquet` |
| Retrieval corpus | train-only, 4000 cases, `tfidf_svd` |
| Escalation policy | `GroundedEscalationPolicy` |
| Thresholds | `data/processed/agent_artifacts/policy_thresholds.json` (validation-tuned) |
| Eval config | golden N=199; human reply N=50; judge v1 N=50 |

Do **not** change these to chase metrics. Bug fixes require re-measurement and documentation.
