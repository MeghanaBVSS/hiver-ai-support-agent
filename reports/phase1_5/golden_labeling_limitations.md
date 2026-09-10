# Golden set labeling limitations

The Phase 1 golden set (N=199) is a **taxonomy-guided human-annotated evaluation set**, labeled primarily by `phase1_engineer`.

Do **not** call it independently validated ground truth for all 199 rows.

**Subset IAA (OBSERVED):** `annotator_2` labeled N=50 examples blind to first labels/model/rule suggestions.

| Task | Exact agreement | Cohen's κ |
|---|---:|---:|
| Intent | 0.62 | 0.575 |
| Escalation | 0.72 | 0.435 |

Most intent disagreements involve `other_ambiguous` and boundary classes (`how_to_feature`, billing/content/feedback). Full artifact: `reports/validation/second_annotator_iaa.json`.

Process used:
1. Human-reviewed taxonomy definitions
2. Taxonomy-guided annotation
3. Rule aid involvement (regex/priority rules encoding the taxonomy)
4. Full review/fix pass on golden rows
5. Metadata enrichment in Phase 1.5 (`escalation_reason`, `annotator_confidence`) **without changing** `gold_intent` / `gold_escalate`
6. Second-annotator IAA on a 50-example subset

Remaining risks:
- confirmation bias toward rule-aid suggestions on the primary label set
- IAA covers only 50/199 examples
- boundary errors (live vs playback; how-to vs billing)
- historical Hulu behavior is not treated as escalation truth

`annotator_confidence` (high/medium/low) documents certainty; it is not a second label.
