"""Tests for dataset loading and schema validation."""

from pathlib import Path

import pandas as pd
import pytest

from src.data.loader import load_tweets
from src.data.schema import filter_inbound, validate_schema


def test_schema_validation_ok(mini_df):
    result = validate_schema(mini_df)
    assert result.ok
    assert result.missing_columns == ()


def test_schema_validation_missing_column(mini_df):
    bad = mini_df.drop(columns=["inbound"])
    result = validate_schema(bad)
    assert not result.ok
    assert "inbound" in result.missing_columns


def test_load_tweets_sample(mini_csv_path):
    df = load_tweets(mini_csv_path, sample_size=5, random_seed=42)
    assert len(df) == 5


def test_inbound_filtering(mini_df):
    inbound = filter_inbound(mini_df, inbound=True)
    outbound = filter_inbound(mini_df, inbound=False)
    assert len(inbound) + len(outbound) == len(mini_df)
    assert inbound["inbound"].all()
    assert (~outbound["inbound"]).all()


def test_tweet_id_uniqueness_handling(mini_df):
    # Fixture IDs should be unique; duplicates would be dropped in conversation index.
    assert mini_df["tweet_id"].is_unique


def test_load_missing_file():
    with pytest.raises(FileNotFoundError):
        load_tweets(Path("/tmp/does-not-exist-twcs.csv"))
