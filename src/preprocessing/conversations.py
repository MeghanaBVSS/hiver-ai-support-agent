"""Conversation reconstruction from tweet reply relationships.

Assumptions (documented for interview/review):
1. A conversation is a weakly connected component in the directed reply graph
   formed by edges: child --in_response_to--> parent, and optionally
   parent --response_tweet_id--> child when that ID exists in the frame.
2. Missing parents produce partial trees (orphans kept as roots).
3. Multi-response and multi-turn threads are preserved; pair extraction can
   still flatten to customer→first-support-reply pairs for retrieval baselines.
4. Brand for a conversation is the most frequent non-numeric outbound author,
   if any.
5. Duplicate tweet_ids: first occurrence wins when building the index.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

import pandas as pd

from src.data.schema import assert_columns, parse_response_ids


@dataclass
class TweetNode:
    tweet_id: str
    author_id: str
    inbound: bool | None
    created_at: str | None
    text: str | None
    response_tweet_ids: list[str] = field(default_factory=list)
    in_response_to_tweet_id: str | None = None


@dataclass
class Conversation:
    conversation_id: str
    brand: str | None
    tweet_ids: list[str]
    tweets: list[TweetNode]
    root_tweet_ids: list[str]
    participant_authors: list[str]
    n_turns: int
    has_missing_parent: bool
    customer_messages: list[str]
    support_responses: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "conversation_id": self.conversation_id,
            "brand": self.brand,
            "tweet_ids": self.tweet_ids,
            "root_tweet_ids": self.root_tweet_ids,
            "participant_authors": self.participant_authors,
            "n_turns": self.n_turns,
            "has_missing_parent": self.has_missing_parent,
            "customer_messages": self.customer_messages,
            "support_responses": self.support_responses,
            "tweets": [asdict(t) for t in self.tweets],
        }


@dataclass
class CustomerSupportPair:
    conversation_id: str
    brand: str | None
    customer_tweet_id: str
    support_tweet_id: str
    customer_message: str
    support_response: str
    customer_created_at: str | None
    support_created_at: str | None
    context_customer_messages: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _build_index(df: pd.DataFrame) -> dict[str, TweetNode]:
    assert_columns(
        df,
        [
            "tweet_id",
            "author_id",
            "inbound",
            "created_at",
            "text",
            "response_tweet_id",
            "in_response_to_tweet_id",
        ],
    )
    index: dict[str, TweetNode] = {}
    for row in df.itertuples(index=False):
        tid = str(row.tweet_id) if row.tweet_id is not None else None
        if tid is None or tid in index:
            continue
        parent = row.in_response_to_tweet_id
        parent_id = None if parent is None or (isinstance(parent, float) and pd.isna(parent)) else str(parent)
        if parent_id in {"None", "nan", ""}:
            parent_id = None
        index[tid] = TweetNode(
            tweet_id=tid,
            author_id=str(row.author_id),
            inbound=None if pd.isna(row.inbound) else bool(row.inbound),
            created_at=None if pd.isna(row.created_at) else str(row.created_at),
            text=None if pd.isna(row.text) else str(row.text),
            response_tweet_ids=parse_response_ids(row.response_tweet_id),
            in_response_to_tweet_id=parent_id,
        )
    return index


def _undirected_components(index: dict[str, TweetNode]) -> list[set[str]]:
    """Connected components using parent links and in-file response links."""
    neighbors: dict[str, set[str]] = defaultdict(set)
    for tid, node in index.items():
        neighbors[tid]  # ensure node present
        if node.in_response_to_tweet_id and node.in_response_to_tweet_id in index:
            parent = node.in_response_to_tweet_id
            neighbors[tid].add(parent)
            neighbors[parent].add(tid)
        for rid in node.response_tweet_ids:
            if rid in index:
                neighbors[tid].add(rid)
                neighbors[rid].add(tid)

    seen: set[str] = set()
    components: list[set[str]] = []
    for start in index:
        if start in seen:
            continue
        comp: set[str] = set()
        q: deque[str] = deque([start])
        while q:
            cur = q.popleft()
            if cur in seen:
                continue
            seen.add(cur)
            comp.add(cur)
            for nxt in neighbors[cur]:
                if nxt not in seen:
                    q.append(nxt)
        components.append(comp)
    return components


def _infer_brand(nodes: Iterable[TweetNode]) -> str | None:
    counts: dict[str, int] = defaultdict(int)
    for node in nodes:
        if node.inbound is False and not str(node.author_id).isdigit():
            counts[str(node.author_id)] += 1
    if not counts:
        return None
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]


def reconstruct_conversations(df: pd.DataFrame) -> list[Conversation]:
    """Reconstruct conversations from a TWCS dataframe."""
    index = _build_index(df)
    components = _undirected_components(index)
    conversations: list[Conversation] = []

    for i, comp in enumerate(sorted(components, key=lambda s: min(s))):
        nodes = [index[tid] for tid in sorted(comp)]
        root_ids = [
            n.tweet_id
            for n in nodes
            if n.in_response_to_tweet_id is None or n.in_response_to_tweet_id not in index
        ]
        has_missing_parent = any(
            n.in_response_to_tweet_id is not None and n.in_response_to_tweet_id not in index
            for n in nodes
        )
        customers = [n.text or "" for n in nodes if n.inbound is True]
        support = [n.text or "" for n in nodes if n.inbound is False]
        authors = sorted({n.author_id for n in nodes})
        conversations.append(
            Conversation(
                conversation_id=f"conv_{i:06d}_{min(comp)}",
                brand=_infer_brand(nodes),
                tweet_ids=[n.tweet_id for n in nodes],
                tweets=nodes,
                root_tweet_ids=root_ids,
                participant_authors=authors,
                n_turns=len(nodes),
                has_missing_parent=has_missing_parent,
                customer_messages=customers,
                support_responses=support,
            )
        )
    return conversations


def build_customer_support_pairs(
    df: pd.DataFrame,
    brand: str | None = None,
    first_support_only: bool = True,
) -> list[CustomerSupportPair]:
    """Build customer→support reply pairs.

    A pair exists when an outbound support tweet has in_response_to_tweet_id
    pointing at an inbound customer tweet present in the frame.

    If brand is set, only support tweets authored by that brand are kept.
    If first_support_only, only the earliest support reply per customer tweet
    (by created_at string / tweet_id tie-break) is kept.
    """
    conversations = reconstruct_conversations(df)
    conv_by_tweet = {
        tid: conv.conversation_id for conv in conversations for tid in conv.tweet_ids
    }
    # Precompute inbound customer texts per conversation for O(1) context lookup.
    inbound_by_conv: dict[str, list[tuple[str, str]]] = defaultdict(list)
    index = _build_index(df)
    for tid, node in index.items():
        if node.inbound is True:
            cid = conv_by_tweet.get(tid)
            if cid is not None:
                inbound_by_conv[cid].append((tid, node.text or ""))

    pairs: list[CustomerSupportPair] = []
    for tid, node in index.items():
        if node.inbound is not False:
            continue
        if brand is not None and node.author_id != brand:
            continue
        parent_id = node.in_response_to_tweet_id
        if parent_id is None or parent_id not in index:
            continue
        parent = index[parent_id]
        if parent.inbound is not True:
            continue
        conv_id = conv_by_tweet.get(tid, f"orphan_{tid}")
        context = [
            text
            for other_id, text in inbound_by_conv.get(conv_id, [])
            if other_id != parent_id
        ]
        pairs.append(
            CustomerSupportPair(
                conversation_id=conv_id,
                brand=brand or _infer_brand([node, parent]),
                customer_tweet_id=parent_id,
                support_tweet_id=tid,
                customer_message=parent.text or "",
                support_response=node.text or "",
                customer_created_at=parent.created_at,
                support_created_at=node.created_at,
                context_customer_messages=context,
            )
        )

    if first_support_only:
        best: dict[str, CustomerSupportPair] = {}
        for pair in pairs:
            key = pair.customer_tweet_id
            prev = best.get(key)
            if prev is None:
                best[key] = pair
                continue
            prev_key = (prev.support_created_at or "", prev.support_tweet_id)
            cur_key = (pair.support_created_at or "", pair.support_tweet_id)
            if cur_key < prev_key:
                best[key] = pair
        pairs = list(best.values())

    pairs.sort(key=lambda p: (p.conversation_id, p.customer_tweet_id, p.support_tweet_id))
    return pairs


def conversations_to_frame(conversations: list[Conversation]) -> pd.DataFrame:
    rows = []
    for conv in conversations:
        rows.append(
            {
                "conversation_id": conv.conversation_id,
                "brand": conv.brand,
                "n_turns": conv.n_turns,
                "n_customer_messages": len(conv.customer_messages),
                "n_support_responses": len(conv.support_responses),
                "has_missing_parent": conv.has_missing_parent,
                "root_tweet_ids": ",".join(conv.root_tweet_ids),
                "tweet_ids": ",".join(conv.tweet_ids),
            }
        )
    return pd.DataFrame(rows)


def pairs_to_frame(pairs: list[CustomerSupportPair]) -> pd.DataFrame:
    return pd.DataFrame([p.to_dict() for p in pairs])
