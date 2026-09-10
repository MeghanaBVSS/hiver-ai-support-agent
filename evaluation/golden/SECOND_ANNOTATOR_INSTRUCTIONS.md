# Second-annotator instructions (independent labeling)

## Purpose

Provide an independent label set on ~50 examples for inter-annotator agreement
(intent agreement, escalation agreement, **Cohen's kappa**).

## Status

Labels in `evaluation/golden/second_annotator.csv` are **complete** (50/50).
Agreement + kappa are computed by:

```bash
python -m src.cli.validate_evaluation
# → reports/validation/second_annotator_iaa.json
```

## What the second annotator sees

In `evaluation/golden/second_annotator.csv`:

- `example_id`
- `customer_message`
- `conversation_context` (prior customer messages only)
- `gold_intent_2` / `gold_escalate_2` / `annotator2_notes` (filled by annotator_2)

## What must NOT be used while labeling

- first annotator labels (`gold_intent`, `gold_escalate`, …)
- model predictions
- retrieval results
- baseline predictions
- rule-aid suggestions

## References (definitions only)

- Intent definitions: `reports/phase1/intent_taxonomy.json`
- Escalation rubric: `evaluation/golden/ESCALATION_RUBRIC.md`
- Full guide: `evaluation/golden/ANNOTATION_GUIDE.md`

## How to label

1. Read the customer message and context.
2. Assign exactly one `gold_intent_2` from the allowed intent ids.
3. Assign `gold_escalate_2` as `true` or `false` using the escalation rubric (policy judgment).
4. Optionally add notes.

Use `other_ambiguous` only when there is **insufficient information to reliably assign one supported intent**.

## Observed agreement (do not invent)

See `reports/validation/VALIDATION_REPORT.md` for exact agreement, Cohen's κ, and disagreement breakdowns.
