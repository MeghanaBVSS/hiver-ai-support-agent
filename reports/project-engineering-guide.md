# Hiver AI Support Agent — Project Engineering Guide

**Phase covered:** Phase 0 → 2 (grounded `hulu_support` agent)  
**PDF regeneration:** `make docs` or `python -m src.cli.generate_docs`  
**Status tags:** **IMPLEMENTED** | **PLANNED** | **OBSERVED** | **ASSUMED** | **NOT YET MEASURED**

---

## 1. Project overview

Build an AI customer-support agent on Kaggle TWCS for locked brand **`hulu_support`**.

Given a customer message the system must:

1. Classify intent (reviewed 10-intent taxonomy)
2. Retrieve historically similar support cases (train-only)
3. Draft a reply grounded in evidence (or escalate)
4. Decide auto-handle vs escalate with an explicit reason
5. Expose evidence + confidence

**Critical rule:** The LLM is **not** the source of truth. Historical support evidence is. Weak/conflicting evidence → **ESCALATE**.

---

## 2. Architecture (Phase 2)

```
Customer message
       │
       ▼
┌──────────────────┐
│ TF-IDF + LogReg  │  predicted_intent, intent_confidence (uncalibrated)
└────────┬─────────┘
         ▼
┌──────────────────┐
│ Semantic retrieve│  train-only corpus; diversity; top_k hits
│ (tfidf_svd emb.) │
└────────┬─────────┘
         ▼
┌──────────────────┐
│ Evidence strength│  top_sim, gap, intent agree, n_above, reply consistency
└────────┬─────────┘
         ▼
┌──────────────────┐
│ Escalation policy│  DETERMINISTIC FINAL AUTHORITY (LLM cannot override)
└────────┬─────────┘
         │
    escalate?──yes──► escalation template + reason + evidence
         │
         no
         ▼
┌──────────────────┐
│ Reply generator  │  deterministic copy  OR  grounded LLM JSON draft
└────────┬─────────┘
         ▼
┌──────────────────┐
│ Generation valid.│  empty / price / timeline / account-action flags → force escalate
└────────┬─────────┘
         ▼
   AgentResponse JSON
```

**Components:** `src/agent/grounded_agent.py`, `src/retrieval/*`, `src/policy/grounded_policy.py`, `src/generation/providers.py`

---

## 3. Data flow & reproducibility path

```bash
# 1 install
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

# 2 configure
cp .env.example .env   # set DATA_PATH if needed; brand already hulu_support

# 3 build retrieval index (train only; ~30s on labeled train)
python -m src.cli.build_index --embedding tfidf_svd

# 4 evaluate (golden N=199)
python -m src.cli.eval_phase2

# 5 deterministic demo (no API key)
python -m src.cli.run_agent --mode deterministic \
  --message "Hulu keeps buffering when I try to watch live TV"

# 6 optional LLM
# export OPENAI_API_KEY=... ; python -m src.cli.run_agent --mode llm --message "..."
# python -m src.cli.eval_phase2 --llm

# 7 docs
make test && make docs
```

---

## 4. Retrieval

| Item | Value |
|---|---|
| Brand | `hulu_support` (unchanged) |
| Corpus | TRAIN labeled only (4000 cases) |
| Embedding default | `tfidf_svd` (TF-IDF → TruncatedSVD, L2) |
| Optional | `sentence_transformers` / `all-MiniLM-L6-v2` |
| Diversity | near-dupe dedupe + prefer distinct conversations |
| Contamination | golden/test IDs blocked; check at build |

**Trade-off:** diversity may skip a slightly nearer duplicate from the same conversation in favor of independent evidence.

**OBSERVED retrieval quality (golden):** index size **4000**; top-1 neighbor intent agree **0.432**; top-3 **0.568**; mean similarity **0.638**. These are **diagnostics**, not reply correctness.

**Architecture note:** `semantic_retrieval_no_llm` reuses the **same TF-IDF+LR intent classifier** as `tfidf_logreg`. Semantic retrieval changes evidence; policy changes escalation. Equal intent metrics are expected.

---

## 5. Escalation policy

**Inputs:** intent confidence, top similarity, evidence strength, evidence agreement, account/billing/security/unsupported/ambiguity signals.

**Reasons:** `low_intent_confidence`, `weak_evidence`, `conflicting_evidence`, `account_specific`, `billing_refund`, `security_privacy`, `ambiguous_request`, `unsupported_operation`, `potentially_harmful_automation`.

**Thresholds (validation-tuned, NOT golden):**

- intent_confidence_min: **0.55**
- top_similarity_min: **0.15**
- evidence_strength_min: **0.30**
- auto_handle_similarity_min: **0.30**
- auto_handle_evidence_min: **0.40**

Artifact: `data/processed/agent_artifacts/policy_thresholds.json`

---

## 6. Generation

- **Deterministic / no-LLM:** copy top historical support response (or escalation template).
- **LLM:** OpenAI-compatible provider; structured JSON; grounded prompt forbids inventing policy/prices/refunds.
- **Validation:** empty, hallucinated price/timeline/account action → force escalate.

---

## 7. Evaluation matrix (OBSERVED)

Golden N=199. Reply C/G/H from human ratings pack (n=50, `annotator_1`) where noted.

| System | Intent acc | Macro F1 | Unsafe auto-handle | Coverage | Reply C/G/H |
|---|---:|---:|---:|---:|---|
| majority | 0.126 | 0.022 | 0.000 | 0.000 | NOT YET MEASURED |
| tfidf_logreg | 0.693 | 0.668 | 0.253 | 0.317 | NOT YET MEASURED |
| tfidf_retrieval | 0.352 | 0.362 | 0.149 | 0.176 | **3.14 / 3.18 / 3.06 (human)** |
| semantic_retrieval_no_llm | 0.693 | 0.668 | **0.069** (6/87) | 0.166 (33/199) | **2.92 / 4.86 / 2.58 (human)** |
| semantic_llm | NOT MEASURED | | | | NOT MEASURED |

Full JSON: `evaluation/results/phase2_evaluation_matrix.json`

---

## 8. HUMAN EVALUATION COMPLETED — READ THIS

### Annotators

| Field | Value |
|---|---|
| Reply rater | `annotator_1` |
| Second intent/escalate rater | `annotator_2` |
| Pack size | 50 examples each |

### Work item A — Reply human pack (50 examples)

**File:** `evaluation/reply_human/human_ratings.csv`

**Fields filled:**

- `human_correctness` / `human_groundedness` / `human_helpfulness` → scores of **semantic Phase-2 agent** reply (1–5)
- `human_escalation` → whether a specialist should handle (`true`/`false`)
- `human_comments` → short rationale
- `baseline_correctness` / `baseline_groundedness` / `baseline_helpfulness` → same scales for TF-IDF retrieve-and-copy
- `annotator_id` = `annotator_1`

**Means (OBSERVED):**

| System | Correctness | Groundedness | Helpfulness |
|---|---:|---:|---:|
| TF-IDF retrieve-and-copy | 3.14 | 3.18 | 3.06 |
| Semantic deterministic agent | 2.92 | **4.86** | 2.58 |

**Escalation:**

- Semantic used escalate-template on **90%** of the 50-pack
- Human would escalate **54%**
- Decision agreement semantic vs human: **60%**

**Interpretation:** Semantic is very grounded (templates invent no policy) but over-escalates easy content/playback FAQs, hurting correctness/helpfulness vs a good historical copy.

### Work item B — Second annotator pack (50 examples)

**File:** `evaluation/golden/second_annotator.csv`

Filled by `annotator_2`: `gold_intent_2`, `gold_escalate_2`, `annotator2_notes` for all 50 rows.

**Agreement vs original gold:**

- Intent exact agreement: **62%**
- Escalate exact agreement: **72%**

### Work item C — Write-ups

- `reports/phase2/HUMAN_ANNOTATION.md`
- `reports/phase2/human_reply_summary.json`
- `reports/phase2/second_annotator_agreement.json`

LLM-as-judge ↔ human agreement: **NOT YET MEASURED** (requires API judge run on the same pack).

---

## 9. Failure modes (OBSERVED)

Top flags on golden semantic agent (`reports/phase2/failure_analysis.json`):

1. Escalation disagreement (often over-escalate)
2. Intent confusion
3. Rare intents (`service_outage`, `how_to_feature`)
4. Ambiguous messages

---

## 10. Tests & CLI

- **Tests:** 51 passed (includes Phase-2 contamination, ordering, policy, no-LLM, mocked LLM)
- **Demo:** `python -m src.cli.run_agent --mode deterministic --message "..."`

Example observed demo:

- Intent `live_tv_issues` conf≈0.91
- Evidence strength≈0.53
- Auto-handle with copied historical buffering reply

---

## 11. Known limitations

- `tfidf_svd` is lexical, not deep semantic
- Classifier confidence uncalibrated
- Generation validators incomplete
- Historical tweets ≠ current Hulu policy
- LLM path not measured without API key
- Judge↔human agreement not yet measured

---

## 12. Interview questions (short)

1. Why isn’t the LLM the source of truth?
2. Why tune thresholds on validation not golden?
3. Why is semantic groundedness high but helpfulness low in the human ratings?
4. How do you prove retrieval isn’t contaminated by golden?

Interview Q&A lives outside the repo for local study: `/home/mani/Music/interview-prep.md`.

---

## HOW WE VALIDATED THE EVALUATOR (Phase 3B)

### Methodology
- 50 human-rated reply examples (`annotator_1`)
- LLM judge: `ollama:qwen2.5:3b`, fixed rubric, **no gold labels**
- Runtime ≈ 48 minutes for 50 examples

### Agreement (v1)
- Helpfulness within±1: **0.82**
- Actionability within±1: **0.88**
- Groundedness within±1: **0.12**

### Groundedness problem
Concentrated in **escalated** replies (within±1 ≈0.02) vs non-escalated (1.00). Judge mean groundedness on escalations ≈1.87 vs human 5.0.

### Escalation analysis
Hypothesis that the judge treats safe escalation as ungrounded covers ~70% of large disagreements — **supported**.

### Limitations
Judge ≠ ground truth. Do not label golden/test with the judge. Prefer human ratings + unsafe auto-handle for headlines.

### Recommended next step
Adopt rubric v2; confirm with targeted human re-rate of escalated groundedness; optional larger N.

See `reports/phase3/judge_calibration.md` and `reports/final-report.md`.

---

## HOW TO EXPLAIN THIS PROJECT IN AN INTERVIEW

### 60 seconds
Hulu Twitter support agent: classify → retrieve train-only history → policy escalate/auto-handle. Headline unsafe auto-handle **6.9%** at 16.6% coverage. Proof via golden 199 + human ratings; LLM judge is evaluator only.

### 3 minutes
Cover brand choice, leakage-safe splits, TF-IDF+LR vs retrieve-and-copy vs semantic agent, safety-first policy, over-escalation trade-off, judge groundedness failure on escalate templates.

### Demo command
`python -m src.cli.run_agent --mode deterministic --message "Hulu keeps buffering when I try to watch live TV"`

### If asked “what’s misleading?”
Unsafe 6.9% looks great partly because we refuse most cases; helpfulness is lower than retrieve-and-copy; labels are single-annotator; historical tweets ≠ current policy.

### Repo map
`src/agent` pipeline · `src/retrieval` · `src/policy` · `evaluation/golden` · `reports/final-report.md`
