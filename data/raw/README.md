# Raw dataset directory

Place Kaggle files here. Do **not** commit large CSVs.

## Expected files

- `sample.csv` — small official Kaggle sample (~17 KB). Preferred for Phase 0 dry-runs.
- `twcs.csv` — full corpus (~2.8M rows). Required for trustworthy brand ranking and intent discovery on real scale.

## How to acquire

See `python -m src.cli.main acquire --instructions` or `make acquire-instructions`.

Quick start with Kaggle CLI (sample only):

```bash
# requires ~/.kaggle/kaggle.json
make data-sample
```

Or set an explicit path:

```bash
export DATA_PATH=/absolute/path/to/sample.csv
```

## Status

As of Phase 0 scaffolding: **no production Kaggle dump is committed**.
Unit tests use `tests/fixtures/mini_twcs.csv` (synthetic schema-compatible fixture).
