# Baseline interpretation (Phase 1.5)

Status: **OBSERVED** metrics from clean re-run (see baseline_execution_manifest.json).

## What ~69.3% accuracy means
On the stratified golden set (N=199), TF-IDF+LogReg assigns the correct intent about 69.3% of the time.
This is **not** production accuracy: the golden set is intentionally stratified across intents, so class priors differ from raw Hulu traffic.

## Why ~66.8% macro F1 matters
Macro F1 averages per-class F1 equally. It penalizes ignoring rare/weak classes even if overall accuracy looks acceptable.
Accuracy can look healthier than macro F1 when some classes are easy and others fail.

## Why the majority baseline matters
Majority accuracy ≈ 12.6% with macro F1 ≈ 2.2%.
Because golden is stratified, always predicting the train majority class is weak — this is the floor any real model must beat.

## Class imbalance effects
Train labels are imbalanced (especially `other_ambiguous` residual mass from rule-aided train annotation).
`class_weight='balanced'` mitigates but does not eliminate mismatch between train prevalence and stratified golden.
Accuracy alone can overstate usefulness under imbalance/stratification; inspect per-class F1.

## Weak intents
- `how_to_feature`: low F1 — FAQ phrasing overlaps billing/device/live language; sparse train support.
- `service_outage`: low F1 — rare in train, overlaps playback/live “not working” language.
These are **KNOWN LIMITATIONS**, not reasons to relabel golden to chase metrics.

## Reply retrieval similarity ≠ reply correctness
Cosine neighborhood shows lexical proximity only. It does not prove correct action, groundedness, or brand-safe wording.
