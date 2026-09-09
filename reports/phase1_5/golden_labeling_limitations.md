# Golden set labeling limitations

Status: **KNOWN LIMITATION** / **NOT YET VALIDATED** by a second annotator.

The Phase 1 golden set (N=199) was labeled by a **single annotator** (`phase1_engineer`).

It is **not** independently validated human ground truth.

Process actually used:
1. Human-reviewed taxonomy definitions
2. Taxonomy-guided annotation
3. Rule aid involvement (regex/priority rules encoding the taxonomy)
4. Full review/fix pass on golden rows
5. Metadata enrichment in Phase 1.5 (`escalation_reason`, `annotator_confidence`) **without changing** `gold_intent` / `gold_escalate`

Risks:
- confirmation bias toward rule-aid suggestions
- single-annotator idiosyncrasy
- boundary errors (live vs playback; how-to vs billing)

`annotator_confidence` (high/medium/low) documents certainty; it is not a second label.
