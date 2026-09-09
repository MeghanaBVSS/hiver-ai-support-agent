"""Shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.data.loader import load_tweets

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def mini_csv_path() -> Path:
    return FIXTURES / "mini_twcs.csv"


@pytest.fixture
def mini_df(mini_csv_path: Path) -> pd.DataFrame:
    return load_tweets(mini_csv_path)
