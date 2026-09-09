# Hiver AI Support Agent — Project Manual

Trustworthy AI customer-support agent for **`hulu_support`** on the Kaggle Twitter Customer Support (TWCS) corpus.

**Repository:** https://github.com/MeghanaBVSS/hiver-ai-support-agent

**Frozen headline:** unsafe auto-handle rate **6/87 ≈ 0.069** at coverage **33/199 ≈ 0.166** on golden N=199 (semantic retrieval + deterministic policy).  
This is the fraction of **gold-escalate-positive** cases that were incorrectly auto-handled — **not** 6.9% of all messages.

---

## 1. What this project is

Given a customer tweet to Hulu support, the system:

1. Predicts **intent** (10-class taxonomy)
2. **Retrieves** similar historical customer cases from a train-only index
3. Scores **evidence strength**
4. Applies a **deterministic escalation policy** (final authority)
5. Either **escalates** with a reason template or **copies** the best historical support reply  
   (optional LLM draft behind a provider abstraction; golden LLM eval was **not measured** in the freeze)

Proof comes from evaluation artifacts (golden set, baselines, human reply ratings, judge calibration)—not from demos alone.

---

## 2. Architecture

```
customer message
  → TF-IDF + Logistic Regression (intent + confidence)
  → semantic retrieve (train-only tfidf_svd index, size 4000, with diversity)
  → evidence strength
  → deterministic escalation policy (final authority)
  → escalate template  OR  copy top historical support reply
     (optional: LLM draft via OpenAI-compatible provider)
```

Important design fact: **semantic + policy reuses the same TF-IDF+LR intent classifier** as the LR baseline. Equal intent accuracy (0.693) is expected. Semantic retrieval changes evidence; policy changes escalation/auto-handle.

| Component | Location |
|---|---|
| Agent orchestration | `src/agent/grounded_agent.py` |
| Intent baselines / LR | `src/intent/` |
| Retrieval + embeddings | `src/retrieval/` |
| Escalation policy | `src/policy/grounded_policy.py` |
| Generation / LLM providers | `src/generation/providers.py` |
| Metrics / leakage | `src/evaluation/` |
| CLIs | `src/cli/` |

---

## 3. Repository layout

```
hiver-ai-support-agent/
├── README.md                 ← this manual
├── Makefile                  ← install, test, index, eval, docs
├── requirements.txt
├── configs/default.yaml      ← selected_brand: hulu_support (locked)
├── src/                      ← application code
├── tests/                    ← unit tests (58)
├── evaluation/
│   ├── golden/               ← golden_set.csv (N=199), second_annotator.csv
│   ├── reply_human/          ← human_ratings.csv + PROVENANCE.md
│   └── results/              ← phase2_evaluation_matrix.json, baselines
├── reports/
│   ├── final-report.md       ← ≤6-page submission report
│   ├── project-engineering-guide.pdf
│   ├── decision-log-short.md
│   ├── SUBMISSION_PACKAGE.md
│   └── final/                ← headline, comparison, failures, metric audit
├── data/
│   ├── raw/                  ← twcs.csv locally (gitignored, ~493MB)
│   ├── interim/              ← caches (gitignored)
│   └── processed/            ← labeled splits + agent_artifacts (mostly gitignored)
└── notebooks/
```

Large / secret paths that must **not** be committed: `data/raw/twcs.csv`, interim/processed dumps, `*.joblib`, `.env`, `.venv`, `__pycache__`.

---

## 4. Environment setup

```bash
cd hiver-ai-support-agent
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Or: `make install`.

`.env` is gitignored. Optional keys:

| Variable | Purpose |
|---|---|
| `DATA_PATH` | Absolute path to `twcs.csv` or `sample.csv` |
| `OPENAI_API_KEY` | Optional LLM agent / cloud judge |
| `LLM_MODEL` | Override default chat model |
| `KAGGLE_USERNAME` / `KAGGLE_KEY` | Optional Kaggle download |

Deterministic agent and frozen evaluation **do not** require API keys.

---

## 5. Data

### What you need

| File | Role |
|---|---|
| `data/raw/twcs.csv` | Full TWCS (~2.8M rows). Required to rebuild brand extracts / labeled splits from scratch |
| `data/raw/sample.csv` | Tiny sample for dry-runs |
| `tests/fixtures/mini_twcs.csv` | Synthetic fixture for unit tests |

If `twcs.csv` is already present locally (it is on this machine), you can rebuild indexes and re-run eval without re-downloading.

### Acquire TWCS

```bash
make acquire-instructions
# or full HF mirror:
python -m src.cli.main acquire --hf-twcs
# or Kaggle sample only:
make data-sample
```

Details: `data/raw/README.md`.

### Frozen processed sizes

| Split | Size |
|---|---|
| Train labeled | 4000 |
| Valid labeled | 800 |
| Golden | 199 |
| Retrieval index | 4000 (train-only) |

Brand lock: `configs/default.yaml` → `selected_brand: hulu_support`.

---

## 6. Day-to-day commands

All commands assume the venv is activated and cwd is the repo root.

### Build / load retrieval index

```bash
python -m src.cli.build_index --embedding tfidf_svd
# same as:
make build-index
```

Writes under `data/processed/agent_artifacts/` (classifier, retriever joblib, corpus, `policy_thresholds.json`). Joblibs are gitignored; rebuild locally after clone.

### Deterministic demo

```bash
python -m src.cli.run_agent --mode deterministic \
  --message "Hulu keeps buffering when I try to watch live TV"

# or:
make run-agent
# custom message:
make run-agent MSG='My bill looks wrong after the free trial'
```

Optional LLM mode (needs `OPENAI_API_KEY`):

```bash
python -m src.cli.run_agent --mode llm --message "..."
```

### Phase-2 evaluation matrix (golden)

```bash
python -m src.cli.eval_phase2
# or:
make eval-phase2
```

Primary JSON: `evaluation/results/phase2_evaluation_matrix.json`.

### Tests

```bash
make test
# or:
python -m pytest -q
```

Expected: **58 passed** (2 benign sklearn warnings possible in judge agreement smoke).

### Regenerate engineering PDF

```bash
make docs
# writes reports/project-engineering-guide.pdf
```

### Optional local LLM judge (Ollama)

```bash
# requires Ollama + pulled model, e.g. qwen2.5:3b
python -m src.cli.run_judge --mode ollama
python -m src.cli.rejudge_subset --mode ollama   # targeted rubric v2
```

---

## 7. Makefile targets

| Target | Action |
|---|---|
| `make install` | Create `.venv` and install `requirements.txt` |
| `make build-index` | Train-only `tfidf_svd` retrieval index |
| `make eval-phase2` | Golden evaluation matrix |
| `make run-agent` | Deterministic demo |
| `make test` | Pytest |
| `make docs` | PDF from living markdown |
| `make acquire-instructions` | Print dataset acquisition steps |
| `make data-sample` | Download Kaggle sample.csv |
| `make phase0` | Phase-0 pipeline (needs data) |
| `make audit` / `make brands` | Dataset audit / brand ranking |

---

## 8. Configuration and policy

| Item | Path / value |
|---|---|
| Brand | `hulu_support` (do not change for this submission) |
| Freeze note | `reports/final/EXPERIMENT_FREEZE.md` |
| Policy thresholds | `data/processed/agent_artifacts/policy_thresholds.json` |
| Embedding default | `tfidf_svd` |
| Random seed | `42` (see `.env.example`) |

Validation-tuned thresholds (golden not used for tuning):

- `intent_confidence_min`: **0.55**
- `top_similarity_min`: **0.15**
- `evidence_strength_min`: **0.30**
- `auto_handle_similarity_min`: **0.30**
- `auto_handle_evidence_min`: **0.40**

---

## 9. Evaluation results (frozen)

### System comparison (golden N=199)

| System | Intent Acc | Macro-F1 | Unsafe↓ | Coverage | Reply C/G/H (human N=50) |
|---|---:|---:|---:|---:|---|
| Majority | 0.126 | 0.022 | 0.000 | 0.000 | — |
| TF-IDF+LR | 0.693 | 0.668 | 0.253 | 0.317 | — |
| TF-IDF retrieval | 0.352* | 0.362* | 0.149 | 0.176 | 3.14 / 3.18 / 3.06 |
| Semantic + policy | 0.693 | 0.668 | **0.069** | 0.166 | 2.92 / 4.86 / 2.58 |
| Semantic + LLM | NOT MEASURED | | | | NOT MEASURED |

\* Neighbor-intent diagnostic for retrieve-and-copy, not a trained classifier.

Exact escalation cells (semantic + policy): TP=81, FP=85, FN=6, TN=27; gold escalate⁺=87.

### Retrieval diagnostics (not reply correctness)

Index **4000**; top-1 intent agree **0.432**; top-3 **0.568**; mean similarity **0.638**.

### Failure modes (flags co-occur)

1. Escalation disagreement **91** (85 over-escalate + 6 under)  
2. Intent confusion **61**  
3. Rare intents **34**  
4. Ambiguous **25**  
5. Long messages **7**

### Human ratings provenance

See `evaluation/reply_human/PROVENANCE.md`. `annotator_1` rated semantic vs baseline replies on 50 examples. Scores must not be rewritten casually.

---

## 10. Key documents

| Document | Path |
|---|---|
| Final report | `reports/final-report.md` |
| Headline definition | `reports/final/headline_metric.md` |
| Comparison table | `reports/final/comparison_table.md` |
| Failure modes | `reports/final/top5_failure_modes.md` |
| Metric recompute | `reports/final/metric_audit_recompute.json` |
| Decision log (short) | `reports/decision-log-short.md` |
| Decision log (full) | `reports/decision-log.md` |
| Judge calibration | `reports/phase3/judge_calibration.md` |
| Engineering notes | `reports/engineering-notes.md` |
| Engineering PDF | `reports/project-engineering-guide.pdf` |
| Submission package | `reports/SUBMISSION_PACKAGE.md` |

---

## 11. Fresh clone checklist

1. `git clone https://github.com/MeghanaBVSS/hiver-ai-support-agent.git`
2. `cd hiver-ai-support-agent && make install && source .venv/bin/activate`
3. Place `twcs.csv` in `data/raw/` (or set `DATA_PATH`) if you need to rebuild processed data from scratch
4. If processed labels + artifacts are missing: run the data/phase pipelines documented in `reports/engineering-notes.md`, then `make build-index`
5. `make test`
6. `make run-agent`
7. `make eval-phase2` (optional recompute)
8. `make docs`

Committed evaluation CSVs/JSON under `evaluation/` and reports under `reports/` are enough to **read** the frozen results without re-running the full data pipeline.

---

## 12. Known limitations

- Safety is achieved partly by **over-escalation** (coverage only ~16.6%)
- Default embeddings are **lexical** (`tfidf_svd`); paraphrase recall is limited
- Golden labels: single primary annotator; second-pass agreement 62% intent / 72% escalate
- LLM judge ≠ ground truth (weak groundedness agreement)
- Historical ~2017 tweets ≠ current Hulu policy
- Semantic+LLM golden reply quality **not measured** in freeze
- Near-duplicate leakage beyond exact text not fully scanned

---

## 13. Performance (observed locally)

| Measurement | Value |
|---|---|
| Deterministic e2e median | ~27 ms |
| Retrieval-dominated | yes |
| Index build (`tfidf_svd`, 4000) | ~27 s (prior observation) |
| Judge v1 (50 examples, qwen2.5:3b) | ~48 min |
| Unit tests | 58 passed |

Details: `reports/final/performance.json`.
