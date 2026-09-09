"""Central configuration for the Hiver AI support-agent project.

Precedence (highest last):
1. configs/default.yaml
2. environment variables / .env
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = ROOT / "configs" / "default.yaml"


def _env(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    return value


def _env_int(name: str, default: int | None) -> int | None:
    raw = _env(name)
    if raw is None:
        return default
    return int(raw)


def _env_float(name: str, default: float) -> float:
    raw = _env(name)
    if raw is None:
        return default
    return float(raw)


@dataclass
class Config:
    """Typed project configuration."""

    project_name: str = "hiver-ai-support-agent"
    phase: int = 0
    root: Path = ROOT

    data_path: Path | None = None
    raw_dir: Path = field(default_factory=lambda: ROOT / "data" / "raw")
    interim_dir: Path = field(default_factory=lambda: ROOT / "data" / "interim")
    processed_dir: Path = field(default_factory=lambda: ROOT / "data" / "processed")
    reports_dir: Path = field(default_factory=lambda: ROOT / "reports")
    evaluation_dir: Path = field(default_factory=lambda: ROOT / "evaluation")
    golden_dir: Path = field(default_factory=lambda: ROOT / "evaluation" / "golden")
    results_dir: Path = field(default_factory=lambda: ROOT / "evaluation" / "results")

    selected_brand: str | None = None
    sample_size: int | None = None
    random_seed: int = 42
    train_ratio: float = 0.7
    valid_ratio: float = 0.15
    test_ratio: float = 0.15
    split_strategy: str = "conversation"

    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    llm_model: str | None = None
    retrieval_top_k: int = 5
    min_retrieval_similarity: float = 0.15
    escalation_threshold: float = 0.55

    proposed_min_intents: int = 6
    proposed_max_intents: int = 12
    discovery_n_clusters: int = 10
    discovery_sample_size: int = 2000
    golden_size_min: int = 150
    golden_size_max: int = 250

    def ensure_dirs(self) -> None:
        for path in (
            self.raw_dir,
            self.interim_dir,
            self.processed_dir,
            self.reports_dir,
            self.golden_dir,
            self.results_dir,
            self.reports_dir / "phase0",
        ):
            path.mkdir(parents=True, exist_ok=True)

    def resolve_data_path(self) -> Path:
        """Resolve DATA_PATH or fall back to common raw filenames."""
        if self.data_path is not None and self.data_path.exists():
            return self.data_path

        candidates = [
            self.raw_dir / "twcs.csv",
            self.raw_dir / "sample.csv",
            self.raw_dir / "twcs" / "twcs.csv",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate

        raise FileNotFoundError(
            "No dataset found. Set DATA_PATH or place sample.csv/twcs.csv under "
            f"{self.raw_dir}. See data/raw/README.md for acquisition steps."
        )


def _as_path(value: Any, root: Path) -> Path | None:
    if value is None or value == "":
        return None
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = (root / path).resolve()
    return path


def load_config(config_path: Path | None = None) -> Config:
    """Load YAML defaults, then apply environment overrides."""
    load_dotenv(ROOT / ".env")
    path = config_path or DEFAULT_CONFIG_PATH
    raw: dict[str, Any] = {}
    if path.exists():
        with path.open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}

    paths = raw.get("paths", {})
    data = raw.get("data", {})
    models = raw.get("models", {})
    policy = raw.get("policy", {})
    intent = raw.get("intent", {})
    golden = raw.get("golden", {})
    project = raw.get("project", {})

    cfg = Config(
        project_name=project.get("name", "hiver-ai-support-agent"),
        phase=int(project.get("phase", 0)),
        data_path=_as_path(_env("DATA_PATH", paths.get("data_path")), ROOT),
        raw_dir=_as_path(paths.get("raw_dir", "data/raw"), ROOT) or (ROOT / "data" / "raw"),
        interim_dir=_as_path(paths.get("interim_dir", "data/interim"), ROOT)
        or (ROOT / "data" / "interim"),
        processed_dir=_as_path(paths.get("processed_dir", "data/processed"), ROOT)
        or (ROOT / "data" / "processed"),
        reports_dir=_as_path(paths.get("reports_dir", "reports"), ROOT) or (ROOT / "reports"),
        evaluation_dir=_as_path(paths.get("evaluation_dir", "evaluation"), ROOT)
        or (ROOT / "evaluation"),
        golden_dir=_as_path(paths.get("golden_dir", "evaluation/golden"), ROOT)
        or (ROOT / "evaluation" / "golden"),
        results_dir=_as_path(paths.get("results_dir", "evaluation/results"), ROOT)
        or (ROOT / "evaluation" / "results"),
        selected_brand=_env("SELECTED_BRAND", data.get("selected_brand")),
        sample_size=_env_int("SAMPLE_SIZE", data.get("sample_size")),
        random_seed=_env_int("RANDOM_SEED", data.get("random_seed")) or 42,
        train_ratio=float(data.get("train_ratio", 0.7)),
        valid_ratio=float(data.get("valid_ratio", 0.15)),
        test_ratio=float(data.get("test_ratio", 0.15)),
        split_strategy=str(data.get("split_strategy", "conversation")),
        embedding_model=_env("EMBEDDING_MODEL", models.get("embedding_model"))
        or "sentence-transformers/all-MiniLM-L6-v2",
        llm_model=_env("LLM_MODEL", models.get("llm_model")),
        retrieval_top_k=int(models.get("retrieval_top_k", 5)),
        min_retrieval_similarity=float(models.get("min_retrieval_similarity", 0.15)),
        escalation_threshold=_env_float(
            "ESCALATION_THRESHOLD", float(policy.get("escalation_threshold", 0.55))
        ),
        proposed_min_intents=int(intent.get("proposed_min_intents", 6)),
        proposed_max_intents=int(intent.get("proposed_max_intents", 12)),
        discovery_n_clusters=int(intent.get("discovery_n_clusters", 10)),
        discovery_sample_size=int(intent.get("discovery_sample_size", 2000)),
        golden_size_min=int(golden.get("target_size_min", 150)),
        golden_size_max=int(golden.get("target_size_max", 250)),
    )

    # Validate split ratios approximately sum to 1.
    total = cfg.train_ratio + cfg.valid_ratio + cfg.test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"train/valid/test ratios must sum to 1.0, got {total}")

    return cfg
