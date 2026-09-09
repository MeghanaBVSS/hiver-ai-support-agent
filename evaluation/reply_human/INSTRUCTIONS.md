# Human reply rating instructions

## Annotators

- `annotator_1` — reply quality ratings on this pack
- `annotator_2` — second-pass intent/escalate on `evaluation/golden/second_annotator.csv`

## How to rate

Rate replies without looking at model confidence scores or LLM-judge scores.

For each `example_id`:

| Column | Meaning |
|---|---|
| `baseline_reply` | TF-IDF retrieve-and-copy |
| `semantic_reply` | Phase-2 semantic agent (deterministic copy or escalate template) |
| `llm_reply` | Blank unless LLM eval was run |
| `human_correctness` | 1–5 for **semantic** reply correctness |
| `human_groundedness` | 1–5 for **semantic** reply groundedness |
| `human_helpfulness` | 1–5 for **semantic** reply helpfulness |
| `baseline_*` | Same scales for baseline reply |
| `human_escalation` | `true`/`false` — should a specialist handle? |
| `human_comments` | Short rationale |

Scale: 1=poor … 5=excellent.
