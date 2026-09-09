.PHONY: help install test phase0 audit brands docs acquire-instructions data-sample clean build-index eval-phase2 run-agent

PYTHON ?= .venv/bin/python
PIP ?= .venv/bin/pip

help:
	@echo "Targets:"
	@echo "  make install              Create venv and install dependencies"
	@echo "  make acquire-instructions Print dataset acquisition steps"
	@echo "  make data-sample          Download Kaggle sample.csv only (requires credentials)"
	@echo "  make phase0               Run Phase 0 pipeline (requires DATA_PATH or data/raw/*)"
	@echo "  make build-index          Build Phase-2 train-only retrieval index"
	@echo "  make eval-phase2          Run Phase-2 evaluation matrix on golden"
	@echo "  make run-agent            Demo: deterministic agent (set MSG=...)"
	@echo "  make audit                Run dataset audit"
	@echo "  make brands               Rank brand candidates"
	@echo "  make test                 Run unit tests"
	@echo "  make docs                 Regenerate reports/project-engineering-guide.pdf"

install:
	python3 -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

acquire-instructions:
	$(PYTHON) -m src.cli.main acquire --instructions

data-sample:
	$(PYTHON) -m src.cli.main acquire --sample

phase0:
	$(PYTHON) -m src.cli.main phase0

build-index:
	$(PYTHON) -m src.cli.build_index --embedding tfidf_svd

eval-phase2:
	$(PYTHON) -m src.cli.eval_phase2

run-agent:
	$(PYTHON) -m src.cli.run_agent --mode deterministic --message "$(or $(MSG),Hulu keeps buffering when I try to watch live TV)"

audit:
	$(PYTHON) -m src.cli.main audit

brands:
	$(PYTHON) -m src.cli.main brands

test:
	$(PYTHON) -m pytest -q

docs:
	$(PYTHON) -m src.cli.generate_docs

clean:
	rm -rf .pytest_cache htmlcov .coverage
	find . -type d -name '__pycache__' -exec rm -rf {} +

data-sample-mirror:
	$(PYTHON) -m src.cli.main acquire --sample-mirror

data-hf-twcs:
	$(PYTHON) -m src.cli.main acquire --hf-twcs
