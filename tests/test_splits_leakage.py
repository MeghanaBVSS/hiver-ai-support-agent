"""Split and leakage tests."""

from src.evaluation.golden_design import contamination_check_utility
from src.evaluation.leakage import (
    check_conversation_split_leakage,
    check_retrieval_index_contamination,
    find_duplicate_texts,
)
from src.evaluation.splits import conversation_level_split, temporal_split
from src.preprocessing.conversations import reconstruct_conversations


def test_conversation_split_no_overlap(mini_df):
    convs = reconstruct_conversations(mini_df)
    ids = [c.conversation_id for c in convs]
    split = conversation_level_split(ids, random_seed=42)
    report = check_conversation_split_leakage(
        split.train_ids, split.valid_ids, split.test_ids
    )
    assert report.ok
    assert set(split.train_ids) | set(split.valid_ids) | set(split.test_ids) == set(ids)


def test_conversation_split_deterministic():
    ids = [f"c{i}" for i in range(20)]
    a = conversation_level_split(ids, random_seed=7)
    b = conversation_level_split(ids, random_seed=7)
    assert a.train_ids == b.train_ids
    assert a.valid_ids == b.valid_ids
    assert a.test_ids == b.test_ids


def test_temporal_split_order():
    mapping = {
        "c1": "2017-01-01",
        "c2": "2017-06-01",
        "c3": "2018-01-01",
        "c4": "2018-06-01",
    }
    split = temporal_split(mapping, train_ratio=0.5, valid_ratio=0.25, test_ratio=0.25)
    assert split.strategy == "temporal"
    assert "c1" in split.train_ids
    assert "c4" in split.test_ids


def test_golden_contamination_detected():
    report = contamination_check_utility(
        golden_ids=["g1", "g2"],
        train_ids=["t1", "g1"],
        retrieval_ids=["r1"],
    )
    assert report["ok"] is False
    assert any("train" in i for i in report["issues"])


def test_golden_contamination_clean():
    report = contamination_check_utility(
        golden_ids=["g1", "g2"],
        train_ids=["t1"],
        retrieval_ids=["r1"],
        valid_ids=["v1"],
        prompt_example_ids=["p1"],
    )
    assert report["ok"] is True


def test_retrieval_index_excludes_test():
    report = check_retrieval_index_contamination(
        retrieval_evidence_ids=["a", "b"],
        test_ids=["b", "c"],
        golden_ids=["d"],
    )
    assert report.ok is False


def test_duplicate_texts():
    dups = find_duplicate_texts(
        ["1", "2", "3"],
        ["hello", "HELLO", "other"],
    )
    # exact lowercase match after strip
    assert "hello" in dups
    assert set(dups["hello"]) == {"1", "2"}
