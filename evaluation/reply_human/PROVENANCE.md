# Provenance — reply human ratings

**File:** `evaluation/reply_human/human_ratings.csv`  
**Do not change score values.** This document only clarifies how fields were produced.

## How the pack was built

1. Phase-2 eval sampled **50** golden examples (intent-stratified + fill to 50).
2. For each example, system replies were generated:
   - `baseline_reply` = TF-IDF retrieve-and-copy output
   - `semantic_reply` = frozen semantic+policy deterministic agent output
   - `llm_reply` = left blank (LLM eval not run in freeze)
3. Quality fields were filled by **`annotator_id=annotator_1`** with `annotation_type=human_reply_eval`.
4. Extra dimensions `human_actionability` / `human_brand_style` were added in a second rating pass (same annotator id).

## Field mapping

| Columns | Rates which reply |
|---|---|
| `human_correctness`, `human_groundedness`, `human_helpfulness`, `human_actionability`, `human_brand_style` | **semantic_reply** |
| `baseline_correctness`, `baseline_groundedness`, `baseline_helpfulness` | **baseline_reply** |
| `human_escalation` | Whether a specialist **should** handle (annotator judgment) |

## Terminology (use consistently)

- **Human reply ratings** / **annotator_1 ratings** — preferred
- Not “proxy”, not “automated scores”
- LLM judge scores are separate (`llm_judge_scores*.csv`) and are **evaluator outputs**, not human ratings

## Related

- Instructions: `evaluation/reply_human/INSTRUCTIONS.md`
- Summary means: `reports/phase2/human_reply_summary.json`
- Second-pass intent/escalate labels: `evaluation/golden/second_annotator.csv` (`annotator_2`) — also human-labeled fields; **not** independent multi-person IAA study
