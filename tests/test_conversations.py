"""Conversation reconstruction tests."""

from src.preprocessing.conversations import (
    build_customer_support_pairs,
    reconstruct_conversations,
)


def test_reconstruct_conversations_basic(mini_df):
    convs = reconstruct_conversations(mini_df)
    assert len(convs) >= 1
    assert all(c.conversation_id for c in convs)
    assert all(c.tweet_ids for c in convs)


def test_missing_parent_flag(mini_df):
    convs = reconstruct_conversations(mini_df)
    orphan_related = [c for c in convs if "30" in c.tweet_ids]
    assert orphan_related
    assert orphan_related[0].has_missing_parent is True


def test_multi_support_responses(mini_df):
    pairs_all = build_customer_support_pairs(mini_df, brand="BetaCare", first_support_only=False)
    pairs_first = build_customer_support_pairs(mini_df, brand="BetaCare", first_support_only=True)
    # Tweet 11 has two support replies (12 and 13).
    multi = [p for p in pairs_all if p.customer_tweet_id == "11"]
    first = [p for p in pairs_first if p.customer_tweet_id == "11"]
    assert len(multi) == 2
    assert len(first) == 1


def test_response_linking_acme(mini_df):
    pairs = build_customer_support_pairs(mini_df, brand="AcmeHelp", first_support_only=True)
    assert len(pairs) >= 3
    assert all(p.brand == "AcmeHelp" for p in pairs)
    assert all(p.customer_message for p in pairs)
    assert all(p.support_response for p in pairs)


def test_conversation_participants(mini_df):
    convs = reconstruct_conversations(mini_df)
    acme = [c for c in convs if c.brand == "AcmeHelp"]
    assert acme
    assert any("AcmeHelp" in c.participant_authors for c in acme)
