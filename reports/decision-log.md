# Decision log

Non-obvious engineering decisions across Phase 0–1.

---

## Decision
Do not auto-download the full TWCS corpus by default; require an explicit acquisition path (Kaggle or HuggingFace mirror).

### Why
Prevents surprise multi-hundred-MB downloads and keeps Phase 0/1 reproducible and intentional.

### Alternative
Silent `kaggle datasets download` on first import.

### Why rejected
Violates repo rules; hard to audit; fails offline.

### Risk / limitation
Contributors need credentials or network to obtain data once.

---

## Decision
Select `hulu_support` using suitability (pairs/coverage/diversity/golden feasibility) with a soft mega-brand penalty — not max outbound volume.

### Why
AmazonHelp/AppleSupport maximize volume but widen the intent space beyond a clean 6–12 taxonomy for a take-home.

### Alternative
Always pick the largest brand (AmazonHelp).

### Why rejected
Harder labeling, noisier intents, weaker interview narrative about evaluation quality.

### Risk / limitation
Suitability score is a heuristic; another team could defensibly pick SpotifyCares.

---

## Decision
Use conservative conversation IDs for splitting when undirected components contain many customer authors or >15 turns.

### Why
Observed a 232-turn / 82-author “conversation” that merged unrelated customers under one support tree.

### Alternative
Always trust undirected connected components.

### Why rejected
Would leak unrelated customers across what looks like one conversation unit.

### Risk / limitation
May over-fragment some legitimate multi-customer broadcast threads.

---

## Decision
Primary split = conversation-level; also compute temporal split for comparison; remove valid/test rows whose normalized customer text already appears in train.

### Why
Prevents conversation leakage and exact duplicate-text leakage into evaluation.

### Alternative
Tweet-level random split.

### Why rejected
Inflates retrieval and intent metrics via near/exact duplicates and shared threads.

### Risk / limitation
Removing duplicate texts slightly reduces eval size; temporal drift still possible under conversation split.

---

## Decision
Discover intent clusters on train only; finalize a human-reviewed 10-intent taxonomy for Hulu.

### Why
Avoids test leakage into taxonomy design; 10 intents fit the observed support themes without Banking77 sprawl.

### Alternative
Use raw KMeans labels as ground truth, or import Banking77.

### Why rejected
Clusters are not labels; Banking77 is the wrong domain.

### Risk / limitation
Taxonomy still brand-specific; boundaries (live vs playback vs device) remain fuzzy.

---

## Decision
Golden labels use an engineer annotation protocol (taxonomy definitions + rule aid + manual review pass), not unsupervised auto-labels presented as human labels.

### Why
Phase 1 requires hand-labeled evaluation data; pure cluster IDs would fabricate ground truth.

### Alternative
Leave golden unlabeled until a separate annotator finishes.

### Why rejected
Would block baseline metrics required for Phase 1 review.

### Risk / limitation
Single-annotator labels; no inter-annotator agreement yet; rule aid can bias some classes (`how_to_feature`, `service_outage`).

---

## Decision
Fit TF-IDF vectorizer and Logistic Regression only on train-labeled data; evaluate only on golden.

### Why
Prevents test leakage into feature vocabulary and hyperparameters.

### Alternative
Fit on all data then evaluate.

### Why rejected
Classic leakage; invalidates metrics.

### Risk / limitation
Train labels use the same protocol as golden; correlated annotation errors can inflate scores.

---

## Decision
Retrieval index = train examples only; never golden/test.

### Why
Historical evidence must not include the answer key conversations.

### Alternative
Index all non-test pairs including near-duplicates of golden.

### Why rejected
Contaminates retrieve-and-copy evaluation.

### Risk / limitation
Cosine similarity does not prove reply correctness.

---

## Decision
Escalation policy prefers safety over coverage; always emit a reason.

### Why
Unsafe auto-handle is worse than unnecessary escalation for support risk classes.

### Alternative
Maximize auto-handle rate.

### Why rejected
Conflicts with assignment safety priority.

### Risk / limitation
Coverage will look “low”; must not optimize it alone.

---

## Decision
Keep LLM reply generation and LLM-as-judge out of Phase 1.

### Why
Need trustworthy baselines and labels first.

### Alternative
Jump to LLM agent demo.

### Why rejected
Evaluation would be unanchored and easy to overclaim.

### Risk / limitation
Reply quality human ratings for a sample are completed (`annotator_1` / `annotator_2`); see `reports/phase2/HUMAN_ANNOTATION.md`.

---

## Decision (Phase 1.5)
Preserve original golden `gold_intent` / `gold_escalate` values; only add metadata (`escalation_reason`, `annotator_confidence`).

### Why
Integrity review must not chase metrics by silent relabeling.

### Alternative
Relabel weak classes to improve F1.

### Why rejected
Would contaminate evaluation honesty.

### Risk / limitation
Single-annotator bias remains until second annotator finishes.

---

## Decision (Phase 1.5)
Keep intent id `other_ambiguous` in data files; document conceptual alias `ambiguous_or_other`.

### Why
Renaming would rewrite labels and break continuity; clarity is achieved in docs.

### Alternative
Rename column values repo-wide.

### Why rejected
Conflicts with “preserve original labels” for this integrity pass.

### Risk / limitation
Readers must check the alias note.

---

## Decision (Phase 1.5)
Report Hulu brand-selection robustness honestly under CAP sensitivity.

### Why
At CAP=5000 Hulu stays #1 among tested weights; raising CAP flips top brand to Spotify/Uber/Amazon.

### Alternative
Claim unconditional robustness.

### Why rejected
Would fabricate certainty.

### Risk / limitation
Selection remains heuristic; another team could defensibly pick Spotify under a different cap.

---

## Decision (Phase 2)
Default embedding backend = `tfidf_svd`, with optional sentence-transformers behind the same abstraction.

### Why
CI, demos, and reproducibility must run without downloading embedding models or API keys. `auto` can prefer ST when installed.

### Alternative
Hard-require `all-MiniLM-L6-v2` for all runs.

### Why rejected
Breaks offline/CI path and slows the <15-minute headline experiment.

### Risk / limitation
Lexical embeddings underperform true semantic similarity on paraphrases.

---

## Decision (Phase 2)
Escalation policy is deterministic final authority; LLM cannot override.

### Why
Assignment safety: never invent policy/refunds; escalate when evidence is weak.

### Alternative
Let the model set `recommended_escalation` as the final decision.

### Why rejected
LLMs are optimistic generators; they would inflate auto-handle coverage.

### Risk / limitation
Over-escalation vs golden labels (observed unsafe auto-handle ~0.07 on golden for semantic no-LLM).

---

## Decision (Phase 2)
Tune policy thresholds on validation labels only — never on golden.

### Why
Golden is held out for final evaluation integrity.

### Alternative
Grid-search thresholds on golden to maximize F1.

### Why rejected
Leakage / optimistic reporting.

### Risk / limitation
Validation rule-aid labels are imperfect; thresholds inherit that noise.

---

## Decision (Phase 2)
Deterministic no-LLM mode copies the top historical support response when auto-handling.

### Why
Runnable without keys; fair comparison baseline vs grounded LLM drafting.

### Alternative
Template-only replies without retrieval copy.

### Why rejected
Removes grounding evidence from the draft path.

### Risk / limitation
Copied tweets may be stale, mention wrong handles, or be multi-issue.

---

## Decision (Phase 3B)
Treat LLM-judge groundedness (v1) as non-headline; calibrate rubric instead of trusting raw scores.

### Why
Data showed systematic under-scoring of safe escalations (~70% of large disagreements).

### Alternative
Quote judge groundedness as product quality.

### Why rejected
Would misrepresent over-escalating-but-safe replies as ungrounded failures.

### Risk / limitation
Even v2 needs human confirmation; 3B local judge remains noisy.
