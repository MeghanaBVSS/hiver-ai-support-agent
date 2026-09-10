#!/usr/bin/env python3
"""Convenience entrypoint: python run.py [--message ...] [--mode deterministic|llm]

The repo historically only exposed ``python -m src.cli.run_agent``. This wrapper
lets ``python run.py`` work from the project root.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    from src.cli.run_agent import main as cli_main

    argv = list(sys.argv[1:])
    if "--message" not in argv and "-h" not in argv and "--help" not in argv:
        argv.extend(
            [
                "--message",
                "Hulu keeps buffering when I try to watch live TV",
            ]
        )

    try:
        return int(cli_main(argv) or 0)
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        print("Fix: python -m src.cli.build_index --embedding tfidf_svd", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
