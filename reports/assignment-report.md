# Hiver SDE Intern Take-Home — Assignment Report

**Brand:** `hulu_support`  
**System:** Grounded support agent (intent → retrieve → escalate policy → draft)  
**Eval set:** Golden N=199 (human-labelled) + human reply pack N=50  

---

## 1. Problem framing — what “good” means

For Hulu Twitter support, a trustworthy agent must:

1. **Route** messages into a small, data-derived intent set (10 intents).
2. **Ground** drafts in historical Hulu replies — not invent prices, refunds, or account actions.
3. **Abstain** (escalate) when evidence is weak, conflicting, account/billing/security-sensitive, or ambiguous — with an explicit reason.

**Good ≠ maximum automation.** Primary safety metric is **unsafe auto-handle rate** (should-escalate cases that were auto-handled). Coverage is secondary.

### What we chose not to build

- End-to-end account tools (password reset, refunds, plan changes)
- Fine-tuned generative LLM as the source of truth
- Mega-brand scope (Amazon/Apple) that explodes the intent space
- Banking77 as primary data (optional only; Hulu TWCS is the product surface)

---

## 2. System (short)

```
message → TF-IDF+LogReg intent
       → semantic retrieve (train-only; tfidf_svd embeddings; diversity)
       → evidence strength score
       → deterministic escalation policy (final authority)
       → if auto-handle: copy top historical reply (or LLM draft if keyed)
         else: escalate template + reason
```

Artifacts: `data/processed/agent_artifacts/`. Demo: `python -m src.cli.run_agent --mode deterministic --message "..."`.

---

## 3. Results vs baselines (golden N=199)

| System | Intent acc | Macro F1 | Unsafe auto-handle ↓ | Auto-handle coverage |
|---|---:|---:|---:|---:|
| **Trivial:** majority intent | 0.126 | 0.022 | 0.000 | 0.000 |
| **Simple:** TF-IDF + LogReg | 0.693 | 0.668 | 0.253 | 0.317 |
| TF-IDF retrieve-and-copy | 0.352* | 0.362* | 0.149 | 0.176 |
| **Ours:** semantic retrieve + policy (no LLM) | 0.693 | 0.668 | **0.069** | 0.166 |
| Semantic + LLM draft | NOT YET MEASURED (no cloud API key in this run) | | | |

\*Neighbor-intent accuracy for retrieval systems (diagnostic), not a trained classifier.

**Human reply quality (N=50, annotator_1)** — semantic vs TF-IDF copy:

| | Correctness | Groundedness | Helpfulness |
|---|---:|---:|---:|
| TF-IDF retrieve-and-copy | 3.14 | 3.18 | 3.06 |
| Semantic + policy | 2.92 | **4.86** | 2.58 |

Semantic is safer/more grounded but over-escalates (~90% of the 50-pack), cutting helpfulness.

**LLM-as-judge vs human (N=50, model `ollama:qwen2.5:3b`; no gold labels in judge prompt):**

| Dimension | Exact agree | Within ±1 | Weighted κ | Judge−human bias |
|---|---:|---:|---:|---:|
| correctness | 0.36 | 0.72 | 0.15 | −0.66 |
| groundedness | 0.06 | 0.12 | −0.03 | −2.86 |
| helpfulness | 0.32 | 0.82 | 0.21 | +0.84 |
| actionability | 0.38 | 0.88 | 0.10 | +0.14 |
| brand_style | 0.06 | 0.56 | 0.01 | +1.38 |

**Read:** helpfulness/actionability are usable within-1; groundedness agreement is poor — the local judge under-scores escalate templates that humans rated highly grounded. Do not treat judge scores as a substitute for human labels on groundedness. Full dump: `reports/phase2/judge_human_agreement.json`.

---

## 4. Failure analysis — top 5 modes

### 1. Escalation disagreement (n=91 = 85 over-escalate FP + 6 under-escalate FN)
**Example:** `gold_2911030` — multi-device buffering; gold says auto-handle; policy escalates.  
**Hypothesis:** Safety-first thresholds + billing/account keyword nets catch too many technical cases; historical copy would have been adequate.

### 2. Intent confusion between playback vs live_tv vs ambiguous (n≈61)
**Example:** `gold_1372321` — “Hulu TV live constantly buffering” gold=`playback_error`, pred=`live_tv_issues`.  
**Hypothesis:** Overlapping vocabulary; taxonomy boundaries are real but fuzzy in short tweets.

### 3. Rare intents underperform (`how_to_feature`, `service_outage`)
**Example:** `gold_45990` — concurrent streams on basic package; gold=`how_to_feature`, pred=`billing_subscription` → billing escalate.  
**Hypothesis:** Low train support + lexical overlap with plan/pricing language.

### 4. Ambiguous / under-specified messages
**Example:** short complaints without device/error context → `other_ambiguous` → escalate.  
**Hypothesis:** Correct safety behavior, but inflates escalation rate vs gold labels that sometimes allow auto-handle.

### 5. Off-topic retrieve-and-copy when not escalating
**Example:** baseline on `gold_284187` (cannot login) retrieved a content/ghost-subs reply.  
**Hypothesis:** Lexical retrieval without strong intent gating; diversity helps but does not fix wrong neighbors.

Export: `reports/phase2/failure_analysis.json`.

---

## 5. What is misleading about my headline number?

**Headline often quoted:** TF-IDF+LR / semantic intent accuracy **≈0.69** on golden.

**Why that number misleads:**

1. **Golden labels are single-annotator** (with rule aid). Second annotator agreement is only ~62% intent / ~72% escalate — the ceiling is not 100%.
2. **Accuracy hides rare-class pain** (`how_to_feature`, `service_outage`); macro-F1 (~0.67) is the fairer summary.
3. **Intent accuracy ≠ reply quality.** Semantic intent matches TF-IDF+LR because it *uses* that classifier; the product risk is reply + escalation.
4. **Unsafe auto-handle 6/87 ≈ 0.069 looks strong** partly because the policy **over-escalates** (coverage only ~17%; escalation disagreement 91 = 85 FP + 6 FN). Safety improves by refusing work, not by answering better. The 0.069 figure is **not** “6.9% of all messages.”
5. **Human helpfulness (2.58)** on the 50-pack is *worse* than retrieve-and-copy (3.06) for the same reason: escalate templates are grounded but not helpful.
6. **Historical tweets ≠ current Hulu policy.** Groundedness to 2017–era replies can still be operationally wrong today.
7. **No cloud LLM draft eval** in the headline path — do not treat deterministic copy as “LLM agent quality.”

---

## 6. What I’d do with one more week

1. Calibrate intent confidence; retune escalation on validation for fewer over-escalations without raising unsafe auto-handle.
2. Intent-specific policies (playback auto-handle freer; billing/account always escalate).
3. Rebuild retrieval with sentence-transformers; re-measure neighbor diagnostics.
4. Run cloud LLM grounded drafts on the 50-pack + golden subsample; compare to copy baseline.
5. Expand IAA beyond 50; adjudicate disagreements into a cleaner gold.
6. Near-duplicate / paraphrase leakage scan beyond exact text.
7. Production sketch: PII redaction, tool permissions, audit log of evidence IDs.

---

## 7. Human evaluation work completed

| Work | Annotator | Artifact |
|---|---|---|
| Golden intent + escalate (N=199) | primary human | `evaluation/golden/golden_set.csv` |
| Reply quality ratings (N=50) | `annotator_1` | `evaluation/reply_human/human_ratings.csv` |
| Second-pass intent/escalate (N=50) | `annotator_2` | `evaluation/golden/second_annotator.csv` |
| LLM-judge agreement | harness + local LLM | `reports/phase2/judge_human_agreement.json` |

---

## 8. Reproduce headline results (<15 min after data present)

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
python -m src.cli.build_index --embedding tfidf_svd   # ~30s
python -m src.cli.eval_phase2                         # golden matrix
python -m src.cli.run_agent --mode deterministic \
  --message "Hulu keeps buffering when I try to watch live TV"
make test
```

Optional judge agreement (needs Ollama or OpenAI):  
`python -m src.cli.run_judge --mode ollama`

Decision log: `reports/decision-log.md` (condensed list below also in `reports/decision-log-short.md`).
