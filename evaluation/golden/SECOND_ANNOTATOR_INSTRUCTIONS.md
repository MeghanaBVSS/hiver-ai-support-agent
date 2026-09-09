# Second-annotator instructions (independent labeling)

## Purpose

Provide an independent label set on ~50 examples for later agreement analysis
(intent agreement, escalation agreement, Cohen's kappa).

**Do NOT calculate agreement until these labels exist.**

## What you will see

In `evaluation/golden/second_annotator.csv`:

- `example_id`
- `customer_message`
- `conversation_context` (prior customer messages only)
- blank `gold_intent_2`
- blank `gold_escalate_2`
- blank `annotator2_notes`

## What you must NOT see / use

- first annotator labels (`gold_intent`, `gold_escalate`, …)
- model predictions
- retrieval results
- baseline predictions

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

## After labeling

Save the CSV. A later harness pass can compute agreement metrics.
Do not fill labels with model outputs.
