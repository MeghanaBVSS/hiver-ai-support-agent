# Engineering Notes — Phase 1 append

> Continues the living document from Phase 0.
> Status tags: **IMPLEMENTED** | **PLANNED** | **OBSERVED** | **ASSUMED** | **NOT YET IMPLEMENTED**

---

## Phase 1 — Real data, brand selection, taxonomy, golden set, baselines

### Data acquisition (**IMPLEMENTED** + **OBSERVED**)

- Canonical source: Kaggle TWCS.
- Local acquisition used HuggingFace mirror `SunidhiSriram/twcs` → `data/raw/twcs.csv` (**OBSERVED** 516,508,641 bytes, **2,811,774** rows).
- Also stored `data/raw/sample.csv` (17,357 bytes).
- Cached `data/interim/twcs.parquet` for fast reloads.
- CLI: `acquire --sample-mirror`, `acquire --hf-twcs` (explicit only).

### Full-corpus audit (**OBSERVED**)

- Rows: 2,811,774
- Inbound true: 1,537,843 / false: 1,277,931
- Columns validated: tweet_id, author_id, inbound, created_at, text, response_tweet_id, in_response_to_tweet_id
- Outbound non-numeric brand authors: 108

### Brand selection (**OBSERVED**)

- Artifact: `evaluation/brand_selection.csv`, `reports/phase1/brand_selection.json`
- Method: lightweight suitability (pairs, coverage, diversity, golden feasibility) + soft mega-brand penalty — **not** max volume.
- **Selected: `hulu_support`**
  - outbound 21,872; inbound @mentions 18,615; unique reply pairs 21,468; suitability 0.9978
- Top rejected: `sprintcare` (0.9965), `SpotifyCares` (0.9964) — higher/similar volume but lower suitability mix / diversity.
- Config: `configs/default.yaml` → `selected_brand: hulu_support`

### Conversation reconstruction validation (**OBSERVED**)

- Brand extract: 47,496 tweets; 15,082 conversations; 21,468 first-support pairs.
- Undirected components produced a **232-turn / 82-author** mega-thread (multi-customer support tree).
- Directed reconstruction matched undirected stats on this extract (same connectivity via parent links).
- **Decision:** use **conservative conversation IDs** for splits: if component has >3 customer authors or >15 turns, isolate by `customer_tweet_id`.
- Inference context: prior customer messages only; never future support text.

### Splits (**IMPLEMENTED** / **OBSERVED**)

- Primary: conversation-level (70/15/15), seed 42.
- Temporal split computed for comparison.
- Removed valid/test rows whose normalized customer text already appears in train.
- Leakage checks: conversation overlap OK; golden ∉ train/retrieval OK.

### Intent taxonomy (**HUMAN_REVIEWED_PROPOSAL** + **OBSERVED** counts)

- Train-only TF-IDF/KMeans discovery used as hints only.
- Final 10 intents defined in `src/intent/taxonomy_hulu.py` after inspecting real Hulu messages.
- Artifact: `reports/phase1/intent_taxonomy.json` + examples JSON.

### Golden set (**OBSERVED**)

- N=199 (stratified from test-split conversations).
- Label protocol: engineer taxonomy definitions + rule aid + manual review fixes.
- Annotator id: `phase1_engineer`
- Guide: `evaluation/golden/ANNOTATION_GUIDE.md`
- File: `evaluation/golden/golden_set.csv`
- Escalate True/False: 87 / 112
- Difficulty easy/medium/hard: 94 / 96 / 9

**Limitation:** single annotator; rule aid can bias rare classes; not multi-rater IAA.

### Baselines (**OBSERVED** on golden)

- Majority: accuracy ≈ 0.126, macro-F1 ≈ 0.022
- TF-IDF + LogReg (fit train-only): accuracy ≈ 0.693, macro-F1 ≈ 0.668, weighted-F1 ≈ 0.675
- Reply retrieve-and-copy (train index only): mean top-1 cosine ≈ 0.310; neighbor same-intent rate ≈ 0.352
- **Limitation:** cosine / neighbor-intent are proxies — not proof of reply correctness.
- Results: `evaluation/results/intent_baseline_results.json`, `reply_retrieval_baseline_results.json`

### Problems / fixes

1. O(n²) context building in pair construction — fixed with per-conversation inbound index.
2. Circular import `data.__init__` ↔ `conversations` — slimmed package exports.
3. Mega-thread mergers — conservative conversation IDs.
4. Brand deep-analysis timeout on Amazon-scale reconstruct — switched to lightweight metrics for ranking.

### Not yet implemented

- Multi-annotator golden labels / IAA
- Embedding-based intent discovery
- LLM drafting + validated LLM judge
- Large human reply-quality rating sample (template ready)

### Commands

```bash
python -m src.cli.main acquire --instructions
# data already at data/raw/twcs.csv in this environment
make test
make docs
```
