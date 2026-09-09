"""Directed / thread-based conversation reconstruction (conservative).

Compared to undirected connected components, this walks only
``in_response_to_tweet_id`` parent links upward to a root, then collects the
subtree of descendants. Sibling branches that share a root stay together, but
unrelated trees are not merged via undirected response edges alone.
"""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Any

import pandas as pd

from src.preprocessing.conversations import (
    Conversation,
    TweetNode,
    _build_index,
    _infer_brand,
)


def reconstruct_conversations_directed(df: pd.DataFrame) -> list[Conversation]:
    index = _build_index(df)
    children: dict[str | None, list[str]] = defaultdict(list)
    for tid, node in index.items():
        parent = node.in_response_to_tweet_id
        if parent is not None and parent in index:
            children[parent].append(tid)
        elif parent is None or parent not in index:
            children[None].append(tid)  # treat as root candidate

    # Roots: tweets whose parent is missing/absent.
    roots = [
        tid
        for tid, node in index.items()
        if node.in_response_to_tweet_id is None or node.in_response_to_tweet_id not in index
    ]

    seen: set[str] = set()
    conversations: list[Conversation] = []
    for i, root in enumerate(sorted(roots)):
        if root in seen:
            continue
        # BFS down the directed child edges only.
        comp: list[str] = []
        q: deque[str] = deque([root])
        has_missing = False
        while q:
            cur = q.popleft()
            if cur in seen:
                continue
            seen.add(cur)
            comp.append(cur)
            node = index[cur]
            if node.in_response_to_tweet_id is not None and node.in_response_to_tweet_id not in index:
                has_missing = True
            for ch in children.get(cur, []):
                if ch not in seen:
                    q.append(ch)
        nodes = [index[tid] for tid in sorted(comp)]
        customers = [n.text or "" for n in nodes if n.inbound is True]
        support = [n.text or "" for n in nodes if n.inbound is False]
        conversations.append(
            Conversation(
                conversation_id=f"dconv_{i:06d}_{root}",
                brand=_infer_brand(nodes),
                tweet_ids=[n.tweet_id for n in nodes],
                tweets=nodes,
                root_tweet_ids=[root],
                participant_authors=sorted({n.author_id for n in nodes}),
                n_turns=len(nodes),
                has_missing_parent=has_missing
                or any(
                    n.in_response_to_tweet_id is not None and n.in_response_to_tweet_id not in index
                    for n in nodes
                ),
                customer_messages=customers,
                support_responses=support,
            )
        )

    # Orphans already attached as roots; any unseen nodes become singleton convs.
    leftover = [tid for tid in index if tid not in seen]
    base = len(conversations)
    for j, tid in enumerate(sorted(leftover)):
        node = index[tid]
        conversations.append(
            Conversation(
                conversation_id=f"dconv_{base+j:06d}_{tid}",
                brand=_infer_brand([node]),
                tweet_ids=[tid],
                tweets=[node],
                root_tweet_ids=[tid],
                participant_authors=[node.author_id],
                n_turns=1,
                has_missing_parent=node.in_response_to_tweet_id is not None,
                customer_messages=[node.text or ""] if node.inbound else [],
                support_responses=[node.text or ""] if node.inbound is False else [],
            )
        )
    return conversations


def compare_reconstruction_methods(df: pd.DataFrame) -> dict[str, Any]:
    from src.preprocessing.conversations import reconstruct_conversations

    undirected = reconstruct_conversations(df)
    directed = reconstruct_conversations_directed(df)
    u_sizes = [c.n_turns for c in undirected]
    d_sizes = [c.n_turns for c in directed]
    return {
        "undirected": {
            "n_conversations": len(undirected),
            "max_turns": max(u_sizes) if u_sizes else 0,
            "mean_turns": sum(u_sizes) / len(u_sizes) if u_sizes else 0,
            "n_turns_ge_20": sum(1 for x in u_sizes if x >= 20),
            "n_missing_parent": sum(1 for c in undirected if c.has_missing_parent),
        },
        "directed": {
            "n_conversations": len(directed),
            "max_turns": max(d_sizes) if d_sizes else 0,
            "mean_turns": sum(d_sizes) / len(d_sizes) if d_sizes else 0,
            "n_turns_ge_20": sum(1 for x in d_sizes if x >= 20),
            "n_missing_parent": sum(1 for c in directed if c.has_missing_parent),
        },
        "recommendation": (
            "Use directed/thread reconstruction for modeling units when undirected "
            "components produce extreme mergers (shared viral/broadcast reply graphs). "
            "Keep undirected analysis for exploratory connectivity audits."
        ),
    }
