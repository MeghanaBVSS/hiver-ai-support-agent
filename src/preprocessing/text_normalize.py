"""Light text normalization for exploration and baselines."""

from __future__ import annotations

import re

_URL = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_WHITESPACE = re.compile(r"\s+")
_MASK_TOKEN = re.compile(r"__\w+__")


def normalize_tweet_text(text: str, keep_masks: bool = True) -> str:
    """Normalize obvious social-media artifacts without destroying meaning.

    Steps:
    - lowercase
    - strip URLs
    - collapse whitespace
    - optionally preserve privacy masks like __email__
    """
    if text is None:
        return ""
    out = str(text).lower().strip()
    out = _URL.sub(" ", out)
    if not keep_masks:
        out = _MASK_TOKEN.sub(" ", out)
    out = out.replace("\n", " ").replace("\r", " ")
    out = _WHITESPACE.sub(" ", out).strip()
    return out
