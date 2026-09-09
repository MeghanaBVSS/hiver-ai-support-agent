# Engineering Notes (living document)

> Audience: a technically competent interviewer/reviewer who did not write this code.
> Update after every meaningful implementation step.
> Status tags used below: **IMPLEMENTED** | **PLANNED** | **OBSERVED** | **ASSUMED** | **NOT YET IMPLEMENTED**

---

## Phase 0 — Audit, data foundation, architecture

### 1. What was changed

Scaffolded a runnable Phase 0 repository from an empty GitHub repo (`README.md` only) into a structured ML/SDE project with:

- configuration (`configs/default.yaml`, `src/config.py`, `.env.example`)
- dataset acquisition interface (no auto full-download)
- schema validation + loader
- reproducible audit + brand ranking CLIs
- conversation reconstruction utilities
- leakage-aware conversation/temporal splits
- intent discovery proposal pipeline (TF-IDF + KMeans)
- baselines (majority; TF-IDF+LogReg; retrieve-and-copy replies)
- escalation policy with explicit reasons
- evaluation harness + metrics interfaces
- LLM-judge interface (no backend yet)
- golden-set sampling design (no labels yet)
- tests, Makefile, PDF doc generator

### 2. Why it was changed

The assignment prioritizes evaluation and proof over a flashy demo. Jumping to an LLM agent before understanding schema, brand suitability, conversation structure, and leakage would create non-reproducible, non-interviewable work.

### 3. How it works (end-to-end Phase 0)

1. User places `sample.csv` / `twcs.csv` under `data/raw/` or sets `DATA_PATH`.
2. `python -m src.cli.main audit` writes `reports/phase0/data_audit.json`.
3. `... brands` ranks outbound brand handles and recommends one from evidence.
4. `... conversations` rebuilds threads and customer→support pairs.
5. `... intents` clusters inbound texts into a **proposal** taxonomy.
6. `... golden-design` writes the labeling methodology artifact.
7. `make test` validates core invariants on a synthetic fixture.
8. `make docs` regenerates the interview PDF from Markdown sources.

### 4. Important implementation details

#### Dataset acquisition (**IMPLEMENTED** interface; full corpus **NOT OBSERVED** locally)

- Environment had **no** local Kaggle dump and **no** `~/.kaggle/kaggle.json`.
- Per project rules, we did **not** auto-download the multi-million-row file.
- `src/data/acquire.py` documents manual download and optional explicit CLI download.
- Unit tests use `tests/fixtures/mini_twcs.csv` (**synthetic**, schema-compatible).
- Any JSON under `reports/phase0/` generated against that fixture is tagged `SYNTHETIC_TEST_FIXTURE` and must not be presented as Kaggle findings.

#### Schema (**ASSUMED** from Kaggle docs, **validated** at load)

Expected columns:
`tweet_id, author_id, inbound, created_at, text, response_tweet_id, in_response_to_tweet_id`

IDs are coerced to strings to avoid float artifacts. `response_tweet_id` may be comma-separated.

#### Brand identification (**ASSUMPTION**)

Outbound (`inbound == False`) authors whose `author_id` is not purely numeric are treated as brand handles. This follows Kaggle documentation that customers are anonymized numeric IDs.

#### Conversation reconstruction (**IMPLEMENTED**)

- Build undirected connected components from `in_response_to_tweet_id` edges and in-file `response_tweet_id` edges.
- Missing parents → partial trees with `has_missing_parent=True`.
- Customer→support pairs: outbound tweet whose parent is inbound.
- `first_support_only=True` keeps earliest support reply per customer tweet.

#### Leakage strategy (**IMPLEMENTED** utilities)

- Default split: **conversation-level** random split (seeded).
- Alternative: **temporal** split by earliest timestamp.
- Contamination checkers ensure golden/test IDs never enter train or retrieval index.

Why not tweet-level random split? The same conversation (and near-duplicate canned replies) would leak across train/test and inflate retrieval + intent metrics.

#### Intent discovery (**IMPLEMENTED** proposal only)

Phase 0 uses TF-IDF + KMeans (light deps, deterministic seed). Output is explicitly `PROPOSAL_NOT_GROUND_TRUTH`. Embedding models are configured but **NOT YET IMPLEMENTED** for clustering.

#### Baselines (**IMPLEMENTED** interfaces/classes; not evaluated on real golden labels yet)

- Baseline 0: majority intent
- Baseline 1: TF-IDF + Logistic Regression
- Reply baseline: TF-IDF cosine retrieve → copy historical support response

#### Escalation (**IMPLEMENTED** conservative rules)

Escalate on low confidence/evidence, account/billing/security language, ambiguity, conflicting evidence, missing info. Auto-handle only with strong positive evidence and no risk signals. Always returns a reason.

#### LLM judge (**PLANNED** / interface **IMPLEMENTED**)

`src/evaluation/judge.py` defines inputs/outputs and prompt builder. Gold labels are excluded from judge input by design. Agreement metrics are specified but **not computed** (no human annotations yet).

### 5. Commands used

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
make test
DATA_PATH=tests/fixtures/mini_twcs.csv make phase0   # smoke only
make docs
python -m src.cli.main acquire --instructions
```

### 6. Files created/modified (Phase 0)

- `src/**` — library + CLI
- `configs/default.yaml`, `.env.example`, `requirements.txt`, `Makefile`, `pytest.ini`
- `tests/**` including synthetic fixture
- `reports/engineering-notes.md`, `reports/project-engineering-guide.md`
- `README.md` updated
- `data/`, `evaluation/`, `notebooks/` structure

### 7. Alternatives considered

| Decision | Alternative | Why not chosen now |
|---|---|---|
| TF-IDF clustering | sentence-transformers embeddings | Heavier deps; Phase 0 prioritizes foundation + reproducibility |
| Conversation split | Tweet-level random split | Causes conversation/retrieval leakage |
| Auto-download full TWCS | Explicit acquisition | Repo rule + avoid surprise multi-hundred-MB download |
| Hardcode SpotifyCares | Evidence-based ranking | Assignment says Spotify is only a hypothesis |
| Build full LLM agent | Baselines + interfaces first | Evaluation quality before system complexity |

### 8. Why the selected approach was chosen

It creates an interviewable, testable spine: data → conversations → leakage-safe splits → intent proposal → baselines → metrics/policy. Every claim can be traced to code or marked as not yet observed.

### 9. Risks / limitations

- Mention-based inbound brand detection undercounts when `@Brand` was anonymized to numeric IDs.
- TF-IDF clustering can create incoherent “intents”; human review is mandatory.
- Conservative escalation will reduce auto-handle coverage (intentional).
- Timestamp parsing of Twitter date strings can be slow/format-ambiguous at full scale.

### 10. What to explain in an interview

- How conversations are reconstructed and which edge cases exist
- Why conversation-level splits matter for retrieval systems
- How golden-set contamination would silently invalidate the take-home
- Why baselines exist before LLMs
- Why escalation reasons are first-class outputs
- What is implemented vs merely designed

---

## Phase 1 — Real data foundation (completed)

See also `reports/engineering-notes-phase1.md` for the detailed Phase 1 log.

### Summary of what changed

1. **Acquired real TWCS** (`data/raw/twcs.csv`, 2,811,774 rows) via explicit HuggingFace mirror workflow.
2. **Selected `hulu_support`** from evidence (`evaluation/brand_selection.csv`), not max volume / not hardcoded Spotify.
3. **Validated reconstruction**; introduced conservative conversation IDs after observing 232-turn multi-customer mergers.
4. **Built brand pairs dataset** with inference-safe context (no future support text).
5. **Conversation-level splits** + duplicate-text cleanup; temporal split compared.
6. **Human-reviewed 10-intent taxonomy** for Hulu; train-only discovery used as hints only.
7. **Golden set N=199** labeled under engineer protocol + review; isolated from train/retrieval.
8. **Baselines evaluated on golden** with real metrics saved under `evaluation/results/`.
9. **Decision log** + docs/PDF updated.

### Observed baseline headline numbers (golden, N=199)

- Majority accuracy ≈ **0.126**, macro-F1 ≈ **0.022**
- TF-IDF+LogReg accuracy ≈ **0.693**, macro-F1 ≈ **0.668**, weighted-F1 ≈ **0.675**
- Retrieve-and-copy mean top-1 cosine ≈ **0.310**; neighbor same-intent rate ≈ **0.352** (retrieval diagnostic only — not reply correctness)

### Phase 2 next step

Build the grounded reply + escalation agent (still evaluation-first): retrieval improvements, optional LLM drafting with evidence constraints, human reply ratings, calibrated judge — without contaminating golden/retrieval.

---

## Phase 1.5 — Evaluation Integrity Review

Status tags: **IMPLEMENTED** | **OBSERVED** | **KNOWN LIMITATION** | **NOT YET VALIDATED**

### What changed

1. **Golden labeling honesty (KNOWN LIMITATION):** docs now state single-annotator, taxonomy-guided, rule-aid, review/fix; not independently validated ground truth. Added `annotator_confidence` and `escalation_reason` **without changing** `gold_intent`/`gold_escalate`.
2. **Escalation rubric (IMPLEMENTED):** policy judgments with explicit reason codes; not historical Hulu escalation.
3. **`other_ambiguous` audit (OBSERVED):** documented as conceptual `ambiguous_or_other` = insufficient information for one supported intent; not renamed in data (labels preserved). Frequencies + confusion reported.
4. **Brand score audit (OBSERVED):** exact formula, weights, cap normalization, mega-brand penalty, sensitivity. Hulu robust under original CAP=5000; **not** robust if CAP raised (Spotify/Uber/Amazon can lead).
5. **Conversation size audit (OBSERVED):** percentiles + largest components; multi-customer merges confirmed; conservative split IDs remain the modeling choice.
6. **Leakage re-check (OBSERVED):** all integrity checks **PASS**.
7. **Baselines re-run (OBSERVED):** clean invocation + execution manifest; metrics reproduced (not hardcoded).
8. **Second-annotator package (IMPLEMENTED / NOT YET VALIDATED):** 50 blank-label examples; agreement deferred.

### Artifacts

- `reports/phase1_5/*`
- `evaluation/golden/ESCALATION_RUBRIC.md`
- `evaluation/golden/second_annotator.csv`
- `evaluation/results/intent_baseline_results.json` (reproduced)
- `reports/phase1_5/baseline_execution_manifest.json`

---

## Phase 2 — Grounded support agent (`hulu_support`)

Status tags: **IMPLEMENTED** | **OBSERVED** | **NOT YET MEASURED**

### 1. What was changed

Built the end-to-end grounded agent behind existing abstractions:

- Train-only retrieval corpus + contamination checks (`src/retrieval/corpus.py`)
- Embedding backend abstraction: `tfidf_svd` (default/CI) + optional `sentence_transformers` (`src/retrieval/embeddings.py`)
- Semantic retriever with diversity + explicit evidence-strength formula (`src/retrieval/semantic.py`)
- Deterministic escalation policy as final authority (`src/policy/grounded_policy.py`); thresholds tuned on **validation**, not golden
- Provider-abstracted reply generation + post-generation validation (`src/generation/providers.py`)
- `GroundedSupportAgent` (`src/agent/grounded_agent.py`) modes: `deterministic` | `llm`
- CLIs: `build_index`, `run_agent`, `eval_phase2`
- Evaluation matrix, retrieval quality, failure analysis, ~50 human reply pack, LLM-judge implementation
- Tests: `tests/test_phase2_agent.py` (mocked LLM)

### 2. Why

The LLM is not the source of truth. Historical support evidence grounds drafts; weak/conflicting evidence → escalate. Safety and groundedness beat automation coverage.

### 3. How (data flow)

```
customer message
  → TF-IDF+LogReg intent (+ confidence, uncalibrated)
  → semantic retrieve(top_k) from TRAIN corpus only
  → evidence_strength(top_sim, gap, intent agreement, n_above, reply consistency)
  → GroundedEscalationPolicy (deterministic; LLM cannot override)
  → if escalate: template + reason
    else: DeterministicCopyGenerator OR OpenAI-compatible grounded JSON draft
         → validate_generation (prices/timelines/account actions/empty)
         → force escalate on validation flags
  → AgentResponse JSON (intent, evidence, draft, escalate, signals)
```

### 4. Retrieval

- **Embedding (OBSERVED default):** `tfidf_svd` (TF-IDF → TruncatedSVD, L2-normalized). Optional ST model documented but not required for CI/demo.
- **Index size (OBSERVED):** 4000 train cases after contamination filter; diversity dedupes near-identical messages and prefers distinct conversations at query time.
- **Never indexed:** golden/test conversations and tweet IDs.
- **Trade-off:** nearest-neighbor relevance vs diversity — diversity may skip a slightly higher-scoring near-duplicate from the same thread in favor of independent cases.

### 5. Escalation thresholds (OBSERVED, validation-tuned)

Saved in `data/processed/agent_artifacts/policy_thresholds.json`:

- intent_confidence_min: 0.55
- top_similarity_min: 0.15
- evidence_strength_min: 0.30
- auto_handle_similarity_min: 0.30
- auto_handle_evidence_min: 0.40

Rationale: minimize unsafe auto-handle on validation labels; secondary coverage. Golden unused for tuning.

### 6. Evaluation (OBSERVED on golden N=199)

See `reports/phase2/evaluation_matrix.md`. Headline:

| System | Intent acc | Macro F1 | Unsafe auto-handle | Coverage |
|---|---:|---:|---:|---:|
| majority | 0.126 | 0.022 | 0.000 | 0.000 |
| tfidf_logreg | 0.693 | 0.668 | 0.253 | 0.317 |
| semantic_retrieval_no_llm | 0.693 | 0.668 | 0.069 | 0.166 |
| semantic_llm | NOT YET MEASURED | | | |

Reply quality (correctness/groundedness/helpfulness/…): **NOT YET MEASURED** (needs human ratings).

Retrieval diagnostics (not reply correctness): index **4000**; top-1 neighbor intent agree **0.432**; top-3 **0.568**; mean similarity **0.638**.

### 7. Human eval / judge

- Pack: `evaluation/reply_human/human_ratings.csv` (**filled** by `annotator_1` — see `reports/phase2/HUMAN_ANNOTATION.md`)
- Second annotator: `evaluation/golden/second_annotator.csv` (**filled** by `annotator_2`)
- Judge: `OpenAICompatibleJudge` **IMPLEMENTED**; judge↔human agreement **NOT YET MEASURED** (no API judge run on pack)

### 8. Limitations

- TF-IDF SVD is lexical; synonymy/paraphrase weaker than true semantic embeddings.
- Classifier confidence is not calibrated.
- Rule-based generation validation is incomplete by design.
- Historical tweets ≠ current Hulu policy.
- Escalation disagreements with golden often = over-escalate (safety-first), not under-escalate.

### 9. Commands

```bash
python -m src.cli.build_index --embedding tfidf_svd
python -m src.cli.eval_phase2
python -m src.cli.run_agent --mode deterministic --message "..."
# optional: export OPENAI_API_KEY=... ; --mode llm / eval_phase2 --llm
make test && make docs
```

---

## Phase 3B — LLM judge calibration

**OBSERVED:** 50/50 scores with `ollama:qwen2.5:3b` (~48 min). Helpfulness/actionability within±1 high; groundedness 0.12.

**Root cause (supported):** judge under-scores safe escalation templates. Escalated within±1 groundedness ≈0.02 vs non-escalated 1.00.

**Action:** rubric v2 with explicit “do not penalize abstention/escalation on groundedness”; targeted re-judge of top-15 disagreements; v1 scores preserved.

**Rule:** LLM judge = evaluator only — never golden/test ground truth.

Artifacts: `reports/phase3/*`, `evaluation/reply_human/judge_rubric_v{1,2}.md`, `llm_judge_scores_v1.csv`.
