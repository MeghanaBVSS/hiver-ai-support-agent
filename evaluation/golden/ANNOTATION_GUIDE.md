# Golden-set annotation guide (hulu_support)

## Goal

Label ~150–250 evaluation examples for intent + escalation + difficulty + confidence.

File: `evaluation/golden/golden_set.csv`

## Labeling status (IMPORTANT)

The golden set is a **taxonomy-guided human-annotated evaluation set** (N=199), labeled by `phase1_engineer`.

It must **not** be described as “ground truth,” “99% accurate ground truth,” or “fully validated gold labels.”

**Independent validation (subset):** a second annotator pack (`evaluation/golden/second_annotator.csv`, N=50) was labeled without first-annotator labels, model predictions, or rule-aid suggestions in the pack. Measured agreement:

| Task | Exact agreement | Cohen's κ |
|---|---:|---:|
| Intent | 0.62 | 0.575 |
| Escalation | 0.72 | 0.435 |

See `reports/validation/second_annotator_iaa.json` and `reports/validation/VALIDATION_REPORT.md`.

Actual process:

1. Human-reviewed taxonomy definitions
2. Taxonomy-guided annotation
3. Rule-aid involvement (regex/priority rules encoding the taxonomy)
4. Full review/fix pass
5. Phase 1.5 metadata enrichment (`escalation_reason`, `annotator_confidence`) **without changing** preserved `gold_intent` / `gold_escalate`
6. Second-annotator subset IAA (intent + escalation + Cohen's κ)

Risks include **confirmation bias** toward rule-aid suggestions and residual single-annotator idiosyncrasy on the full N=199 (IAA covers 50 examples only).

## Columns

| column | meaning |
|---|---|
| example_id | stable id |
| customer_message | inbound text |
| conversation_context | prior customer texts only (no future support replies) |
| gold_intent | one of the allowed intents |
| gold_escalate | true/false (**policy judgment**, not historical Hulu escalation) |
| escalation_reason | rubric reason code (see `ESCALATION_RUBRIC.md`) |
| annotator_confidence | `high` / `medium` / `low` |
| difficulty | easy / medium / hard (annotator judgment) |
| annotator_notes | free text |
| annotator_id | who labeled |
