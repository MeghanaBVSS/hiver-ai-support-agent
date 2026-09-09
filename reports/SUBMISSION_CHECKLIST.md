# Submission checklist

- [x] runnable repository
- [x] README complete (reviewer quickstart)
- [x] golden set 150–250 (N=199)
- [x] evaluation harness
- [x] baseline comparison (≥2)
- [x] reply evaluation (human N=50)
- [x] escalation evaluation (golden)
- [x] LLM judge rubric (v1 + v2)
- [x] judge analysis (Phase 3B)
- [x] five failure modes
- [x] misleading headline section
- [x] decision log (12)
- [x] six-page report (`reports/final-report.md`)
- [x] interview PDF (`make docs`)
- [x] no secrets in repo (`.env` gitignored)
- [x] no accidental huge files guidance (raw TWCS gitignored)
- [x] tests pass
- [x] deterministic demo works

## Reviewer commands

```bash
python -m src.cli.build_index --embedding tfidf_svd
python -m src.cli.eval_phase2
python -m src.cli.run_agent --mode deterministic --message "Hulu keeps buffering when I try to watch live TV"
make test
make docs
```

## Submit

Form: https://intelligent-bar-256.notion.site/39492cbf0da2800682cfc78a600a745f?pvs=105  
Include public/private repo link + `reports/final-report.md` / PDF.
