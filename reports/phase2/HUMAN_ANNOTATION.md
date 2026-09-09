# Human evaluation work (Phase 2)

## Annotators

| Role | ID | Task |
|---|---|---|
| Reply quality rater | `annotator_1` | Scored 50 golden replies (baseline + semantic) |
| Second intent/escalate rater | `annotator_2` | Labeled 50-example second-annotator pack |

## What was labeled

1. `evaluation/reply_human/human_ratings.csv` — 50 examples
   - `human_*` scores rate the **semantic / Phase-2 agent** reply
   - `baseline_*` scores rate TF-IDF retrieve-and-copy for comparison
   - `human_escalation` = whether a specialist should handle the case
2. `evaluation/golden/second_annotator.csv` — second-pass intent/escalate for 50 examples

## Reply quality means (1–5)

| System | Correctness | Groundedness | Helpfulness |
|---|---:|---:|---:|
| TF-IDF retrieve-and-copy (baseline) | 3.14 | 3.18 | 3.06 |
| Semantic agent (deterministic) | 2.92 | 4.86 | 2.58 |

- Semantic escalate-template rate: **90.00%**
- Human would-escalate rate: **54.00%**
- Escalation decision agreement (semantic vs human): **60.00%**

## Interpretation

- Semantic agent scores **high groundedness** because escalation templates invent no policy.
- Semantic **correctness/helpfulness** are lower when the system over-escalates easy content/playback FAQs.
- Baseline sometimes wins on helpfulness (good historical copy) but can be off-topic (low correctness).

## Sample annotations (first 12)

| example_id | base C/G/H | sem C/G/H | human_esc | comment |
|---|---|---|---|---|
| gold_714588 | 3/3/3 | 2/5/2 | false | UI rollback request; baseline empathizes but no how-to; escalate template over-safe |
| gold_817311 | 3/3/3 | 2/3/2 | false | Customer already fixed it; both replies oddly reopen troubleshooting |
| gold_1681450 | 4/4/4 | 3/5/3 | true | Broken Fire TV + failed contact channels; human follow-up warranted |
| gold_1224206 | 4/4/4 | 3/5/3 | true | Ad-tier complaint/pricing; escalate correct; baseline informative |
| gold_2181625 | 1/2/1 | 3/5/3 | true | Repeat ads; baseline off-topic; escalate ok |
| gold_310432 | 4/4/4 | 4/5/3 | true | Charge confusion; both push to specialist/chat |
| gold_1494174 | 4/4/3 | 2/5/2 | false | Missing show; baseline acknowledges; semantic over-escalates |
| gold_815865 | 5/5/5 | 2/5/2 | false | Episode timing FAQ; baseline excellent; semantic over-escalates |
| gold_2214147 | 5/5/4 | 2/5/2 | false | Rick&Morty request; baseline good; semantic over-escalates |
| gold_507602 | 4/4/4 | 4/5/3 | true | Angry follow-up needing human contact; escalate appropriate |
| gold_870898 | 3/3/2 | 3/5/3 | true | Churn threat; escalate ok |
| gold_1587983 | 3/3/3 | 3/3/3 | true | UI hate; escalate reasonable for retention |

## Second-annotator agreement vs original gold

- n labeled: 50
- intent exact agreement: **62%**
- escalate exact agreement: **72%**

## Files

- `evaluation/reply_human/human_ratings.csv`
- `evaluation/reply_human/INSTRUCTIONS.md`
- `evaluation/golden/second_annotator.csv`
- `reports/phase2/human_reply_summary.json`
- `reports/phase2/second_annotator_agreement.json`
