# Golden-set annotation guide (hulu_support)

## Goal

Label ~150–250 evaluation examples for intent + escalation + difficulty + confidence.

File: `evaluation/golden/golden_set.csv`

## Labeling status (IMPORTANT)

Status: **KNOWN LIMITATION** / **NOT YET VALIDATED** by an independent second annotator.

The current golden set was labeled by a **single annotator** (`phase1_engineer`).

It must **not** be described as independently validated human ground truth.

Actual process:

1. Human-reviewed taxonomy definitions
2. Taxonomy-guided annotation
3. Rule-aid involvement (regex/priority rules encoding the taxonomy)
4. Full review/fix pass
5. Phase 1.5 metadata enrichment (`escalation_reason`, `annotator_confidence`) **without changing** preserved `gold_intent` / `gold_escalate`

Risks include **confirmation bias** toward rule-aid suggestions and single-annotator idiosyncrasy.

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

## Allowed intents

See `reports/phase1/intent_taxonomy.json`.

### `other_ambiguous` (= conceptual `ambiguous_or_other`)

Definition: **insufficient information to reliably assign one supported intent.**

- Qualifies: too short, unclear, multi-intent without a dominant ask, residual
- Does **not** qualify: a clear supported intent with minor noise
- Must **not** become a dumping ground — prefer a supported intent when one dominates

## Escalation

See `evaluation/golden/ESCALATION_RUBRIC.md`.

Escalation labels are **policy judgments** for safe auto-handle decisions.
Do **not** infer escalation solely from whether Hulu historically escalated.

### WHEN TO ESCALATE

- account-specific actions
- billing/refund disputes
- security/privacy
- ambiguous requests
- insufficient evidence
- potentially harmful/unsafe automation
- unsupported operational actions

### WHEN NOT TO ESCALATE

- generic informational support
- troubleshooting with strong evidence / safe generic guidance

## Difficulty rubric (annotator judgment — not objective truth)

- **easy:** single clear intent, little ambiguity
- **medium:** some device/context mix or mild ambiguity
- **hard:** multi-intent, sarcasm, very short, or conflicting cues

## Confidence rubric

- **high:** clear single intent, little boundary doubt
- **medium:** some ambiguity or rule/review involvement
- **low:** hard/ambiguous/other cases

## Isolation rules

Golden examples must not appear in train fitting data, retrieval index, or prompt exemplars.

## Second annotator

See `evaluation/golden/SECOND_ANNOTATOR_INSTRUCTIONS.md` and `second_annotator.csv`.
Do not show first-annotator labels or model outputs to the second annotator.
