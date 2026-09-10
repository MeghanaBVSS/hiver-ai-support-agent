"""CLI: golden leakage verification with PASS/FAIL printout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.config import ROOT
from src.evaluation.validation_enhancement import format_leakage_text, run_leakage_check


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify golden set is not in train/retrieval/prompts")
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "reports" / "validation" / "leakage_check.json",
    )
    args = parser.parse_args()
    result = run_leakage_check()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    text = format_leakage_text(result)
    (args.out.parent / "leakage_check.txt").write_text(text, encoding="utf-8")
    print(text)
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
