# Submission package

**Status:** Local artifacts are prepared. The assignment has **not** been submitted from this workspace (no Notion form, no automated publish).

**Repository:** https://github.com/MeghanaBVSS/hiver-ai-support-agent

---

## Artifact map

| Item | Location |
|---|---|
| Repository URL | https://github.com/MeghanaBVSS/hiver-ai-support-agent |
| Final report | `reports/final-report.md` |
| Engineering PDF | `reports/project-engineering-guide.pdf` |
| Project manual (README) | `README.md` |
| Golden set | `evaluation/golden/golden_set.csv` |
| Evaluation results | `evaluation/results/` (esp. `phase2_evaluation_matrix.json`) |
| Decision log (short) | `reports/decision-log-short.md` |
| Decision log (full) | `reports/decision-log.md` |
| Headline metric | `reports/final/headline_metric.md` |
| Comparison table | `reports/final/comparison_table.md` |
| Human reply ratings | `evaluation/reply_human/human_ratings.csv` (+ `PROVENANCE.md`) |
| Checklist | `reports/SUBMISSION_CHECKLIST.md` |

---

## Pre-submit checklist

- [x] Metrics reconciled vs `reports/final/metric_audit_recompute.json`
- [x] PDF regenerable via `make docs` → `reports/project-engineering-guide.pdf`
- [x] Secrets kept out (`.env` ignored; no API keys in tree)
- [x] Large files ignored: `data/raw/twcs.csv`, `*.joblib`, interim parquet
- [ ] Commit and push the untracked project tree when ready
- [ ] Open the Hiver Notion form and paste the GitHub URL + report/PDF links
- [ ] Submit the form

---

## Manual submission steps

1. Open the repository root.
2. Optional verify: `make test` and `make docs`.
3. `git status` — stage source, tests, configs, `evaluation/`, `reports/`, lightweight `policy_thresholds.json` / `retriever/meta.json`. Exclude raw TWCS, joblibs, `.env`, venv, caches.
4. Commit and `git push -u origin HEAD` when ready.
5. Open the Hiver Notion submission form.
6. Paste: https://github.com/MeghanaBVSS/hiver-ai-support-agent plus paths/links to `reports/final-report.md` and `reports/project-engineering-guide.pdf` as the form requires.
7. Submit the form.

---

## Git state note

Almost the entire tree may still be untracked except `README.md` until committed. `.gitignore` correctly excludes raw TWCS, dumps, joblibs, `.env`, and `.venv`.
