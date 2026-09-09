# Final Report — Hiver AI Support Agent (`hulu_support`)

**Freeze:** brand, taxonomy, golden N=199, splits, retrieval corpus, policy/thresholds.  
**Headline metric:** unsafe auto-handle rate = **6/87 ≈ 0.069** at coverage **33/199 ≈ 0.166** (semantic + policy, golden).  
*Meaning: among the 87 gold-escalate cases, 6 were incorrectly auto-handled — **not** 6.9% of all messages.*

---

## 1. Problem framing

Build a trustworthy AI support agent on real Twitter brand conversations: classify intent, draft a reply grounded in historical resolutions, and decide auto-handle vs escalate with a reason. Proof beats demos.

## 2. What good means for Hulu

Safety-first automation: invent no policy/prices/refunds/account actions; escalate when evidence is weak, conflicting, or account/billing/security-sensitive. Optimize **unsafe auto-handle rate**, not maximum coverage.

## 3. Scope and non-goals

**In scope:** `hulu_support` TWCS subsample, 10 intents, retrieve-and-ground drafts, deterministic policy, evaluation.  
**Not built:** live account tools, judge-as-GT labeling, mega-brand scope, Banking77 as primary data, cloud LLM golden eval (no key in freeze).

## 4. Data

Kaggle TWCS via HF mirror. Brand extract → pairs → conversation split. Train labeled 4000 / valid 800 / golden 199 from held-out test conversations.

## 5. Leakage prevention

Conversation-level splits; exact duplicate-text purge; retrieval index **train-only**; golden/test IDs blocked; contamination checks at index build.

## 6. Intent taxonomy

10 human-reviewed intents (playback, live_tv, app/device, login, billing, content, outage, how_to, feedback, other_ambiguous). Discovered on train; reviewed.

## 7. Baselines

1) Majority intent 2) TF-IDF+LogReg 3) TF-IDF retrieve-and-copy. Semantic agent reuses LR for intent + semantic retrieve + policy.

## 8. Agent architecture

`message → TF-IDF+LR intent → semantic retrieve → evidence strength → deterministic policy → escalate template OR historical copy (LLM optional)`

## 9. Retrieval

`tfidf_svd` embeddings; 4000 train cases; diversity (near-dupe + distinct conversations). Golden neighbor intent agree top1 **0.432** / top3 **0.568** / mean sim **0.638** (diagnostic ≠ correctness).

## 10. Grounded generation

No-LLM: copy top support reply. LLM: OpenAI-compatible JSON drafts from evidence only. Validators flag empty/price/timeline/account-action hallucinations → force escalate.

## 11. Escalation policy

Final authority. Signals: confidence, similarity, evidence strength/agreement, account/billing/security/ambiguous/unsupported. Thresholds validation-tuned (not golden).

## 12. Evaluation methodology

Golden N=199 intent+escalation. Human reply pack N=50 (`annotator_1`). Second-pass intent/escalate N=50 (`annotator_2`, not independent IAA). LLM judge `qwen2.5:3b` evaluator-only.

## 13. Results

| System | Acc | Macro-F1 | W-F1 | Unsafe↓ | Coverage | Reply C/G/H (human N=50) |
|---|---:|---:|---:|---:|---:|---|
| Majority | 0.126 | 0.022 | 0.028 | 0.000 | 0.000 | N/A |
| TF-IDF+LR | 0.693 | 0.668 | 0.675 | 0.253 | 0.317 | N/A |
| TF-IDF retrieval | 0.352* | 0.362* | 0.360* | 0.149 | 0.176 | 3.14/3.18/3.06 |
| Semantic+policy | 0.693 | 0.668 | 0.675 | **0.069** | 0.166 | 2.92/4.86/2.58 |
| Semantic+LLM | NOT MEASURED | | | | | |

\*neighbor-intent diagnostic. Reply = HUMAN VALIDATED. Semantic actionability/style means 2.58 / 3.02.

## 14. Top 5 failure modes

1. **Escalation disagreement** — 91/199 (45.7%) = **85 over-escalate (FP)** + **6 under-escalate (FN)**. Ex: `gold_2911030`, `gold_2721090`. Cause: safety-first gates. Mitigation: intent-specific thresholds. *(Flags co-occur with other modes.)*
2. **Intent confusion** — 61/199 (30.7%). Ex: `gold_1372321` playback↔live_tv. Cause: lexical overlap. Mitigation: hierarchical intents.
3. **Rare intents** — 34/199 (17.1%). Ex: `gold_45990` how_to→billing. Cause: low support + price lexicon. Mitigation: more labels / rules.
4. **Ambiguous messages** — 25/199 (12.6%). Ex: thin complaints → other_ambiguous. Cause: insufficient info. Mitigation: clarify prompts (product).
5. **Long multi-issue tweets** — 7/199 (3.5%). Cause: multi-intent. Mitigation: split/detect multi-intent.

## 15. What is misleading about the headline number?

**Headline unsafe auto-handle = 6/87 ≈ 0.069 looks strong.** Caveats (quantitative):

- Denominator is **gold-escalate positives (87)**, not all 199 messages.
- Golden N=199, **single primary annotator** + rule aid; `annotator_2` agreement only **62% intent / 72% escalate** — label ceiling ≠ 100%.
- Low unsafe rate coexists with **coverage only 33/199 ≈ 16.6%** and high escalate rate — safety via refusal; escalation disagreement = **91** (85 over + 6 under).
- Human **helpfulness 2.58 < retrieve-and-copy 3.06** — safer ≠ more helpful.
- Intent **0.693** is the shared LR classifier, not a retrieval win; semantic retrieval changes **evidence + policy**, not the intent model.
- Neighbor intent agree **0.43 ≠ reply correctness**.
- Judge groundedness within±1 **0.12** (v1); must not back the headline.
- Historical ~2017 Twitter ≠ current Hulu policy; temporal drift unmeasured under conversation split.
- Brand suitability heuristic; CAP sensitivity can flip top brand.
- `other_ambiguous` / rule-aid noise in train labels.
- Semantic+LLM reply quality **NOT MEASURED** in freeze.

## 16. Limitations

Single-annotator gold; lexical embeddings; uncalibrated confidence; over-escalation; judge weak on groundedness; no cloud LLM eval; Twitter public text ≠ authenticated support.

## 17. What I would do with one more week

Intent-specific policy; ST embeddings; LLM draft eval with key; calibrate confidence; expand IAA; near-dupe leakage scan; PII/tooling sketch for production.
