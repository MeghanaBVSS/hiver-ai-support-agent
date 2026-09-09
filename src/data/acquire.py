"""Dataset acquisition helpers.

IMPORTANT:
- This module does NOT automatically download the multi-million-row TWCS dump.
- Users must explicitly place Kaggle files under data/raw/ or set DATA_PATH.
- Optional Kaggle CLI helpers are provided for intentional downloads only.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from src.config import Config, load_config

KAGGLE_DATASET = "thoughtvector/customer-support-on-twitter"
HF_TWCS_REPO = "SunidhiSriram/twcs"
ACQUISITION_INSTRUCTIONS = """
Dataset acquisition (manual, intentional)

Source (canonical):
  https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter

Phase 1 reproducible paths used in this repo:
  A) Official small sample (~17KB):
       - Download sample.csv from Kaggle, OR
       - python -m src.cli.main acquire --sample-mirror
  B) Full TWCS (~517MB) via HuggingFace mirror (same schema; explicit):
       - python -m src.cli.main acquire --hf-twcs
       - writes data/raw/twcs.csv
  C) Local path override:
       - export DATA_PATH=/absolute/path/to/twcs.csv

Headline experiment does NOT require keeping the full corpus in memory after
brand extraction: select a brand, then work from data/processed/<brand>_*.

This project never silently downloads the full corpus.
""".strip()


def print_acquisition_instructions() -> str:
    print(ACQUISITION_INSTRUCTIONS)
    return ACQUISITION_INSTRUCTIONS


def kaggle_available() -> bool:
    return shutil.which("kaggle") is not None and (Path.home() / ".kaggle" / "kaggle.json").exists()


def download_sample_csv(dest_dir: Path | None = None, cfg: Config | None = None) -> Path:
    """Download ONLY sample.csv via Kaggle CLI (explicit user action)."""
    cfg = cfg or load_config()
    dest_dir = dest_dir or cfg.raw_dir
    dest_dir.mkdir(parents=True, exist_ok=True)

    if not kaggle_available():
        raise RuntimeError(
            "Kaggle CLI/credentials not available. "
            "Install `kaggle`, place credentials at ~/.kaggle/kaggle.json, "
            "or manually download sample.csv into data/raw/."
        )

    # Download only the small sample file.
    cmd = [
        "kaggle",
        "datasets",
        "download",
        "-d",
        KAGGLE_DATASET,
        "-f",
        "sample.csv",
        "-p",
        str(dest_dir),
        "--force",
    ]
    subprocess.run(cmd, check=True)
    target = dest_dir / "sample.csv"
    if not target.exists():
        # Sometimes Kaggle wraps in a zip even for a single file.
        zips = list(dest_dir.glob("*.zip"))
        if zips:
            import zipfile

            with zipfile.ZipFile(zips[0], "r") as zf:
                zf.extractall(dest_dir)
    if not target.exists():
        raise FileNotFoundError(f"sample.csv not found after download in {dest_dir}")
    return target


def download_full_dataset(dest_dir: Path | None = None, cfg: Config | None = None) -> Path:
    """Download the full dataset via Kaggle CLI.

    Explicit opt-in only. Prefer sample.csv during Phase 0.
    """
    cfg = cfg or load_config()
    dest_dir = dest_dir or cfg.raw_dir
    dest_dir.mkdir(parents=True, exist_ok=True)

    if not kaggle_available():
        raise RuntimeError(
            "Kaggle CLI/credentials not available. See data/raw/README.md."
        )

    cmd = [
        "kaggle",
        "datasets",
        "download",
        "-d",
        KAGGLE_DATASET,
        "-p",
        str(dest_dir),
        "--unzip",
        "--force",
    ]
    print(
        "WARNING: This downloads the full multi-hundred-MB dataset. "
        "Prefer sample.csv for Phase 0 development."
    )
    subprocess.run(cmd, check=True)

    for candidate in (
        dest_dir / "twcs.csv",
        dest_dir / "twcs" / "twcs.csv",
    ):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"twcs.csv not found under {dest_dir} after download")


SAMPLE_MIRROR_URL = (
    "https://raw.githubusercontent.com/Vishesh062/customer-support-tweet-classifier/"
    "main/data/sample.csv"
)


def download_sample_mirror(dest_dir: Path | None = None, cfg: Config | None = None) -> Path:
    """Download the public Kaggle-compatible sample.csv mirror (explicit)."""
    import urllib.request

    cfg = cfg or load_config()
    dest_dir = dest_dir or cfg.raw_dir
    dest_dir.mkdir(parents=True, exist_ok=True)
    target = dest_dir / "sample.csv"
    urllib.request.urlretrieve(SAMPLE_MIRROR_URL, target)
    return target


def download_twcs_from_hf(dest_dir: Path | None = None, cfg: Config | None = None) -> Path:
    """Explicitly download full twcs.csv from the HuggingFace mirror.

    This is intentional and large (~517MB). Prefer brand extraction afterward.
    """
    from huggingface_hub import hf_hub_download

    cfg = cfg or load_config()
    dest_dir = dest_dir or cfg.raw_dir
    dest_dir.mkdir(parents=True, exist_ok=True)
    print(
        "WARNING: Explicit full TWCS download from HuggingFace "
        f"({HF_TWCS_REPO}). This is multi-hundred MB."
    )
    cached = hf_hub_download(
        repo_id=HF_TWCS_REPO, repo_type="dataset", filename="twcs.csv"
    )
    target = dest_dir / "twcs.csv"
    if Path(cached).resolve() != target.resolve():
        shutil.copy2(cached, target)
    return target
