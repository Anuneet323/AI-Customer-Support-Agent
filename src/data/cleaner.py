"""
src/data/cleaner.py
───────────────────
Text normalisation and deduplication utilities.

Keeps cleaning minimal and auditable — we do NOT aggressively normalise
customer tweets because their raw style is part of the intent signal.
"""

from __future__ import annotations

import html
import logging
import re
from typing import Sequence

import pandas as pd

logger = logging.getLogger(__name__)

# Regex patterns
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_MENTION_RE = re.compile(r"@\w+")
_HASHTAG_RE = re.compile(r"#(\w+)")
_MULTI_SPACE_RE = re.compile(r"\s+")
_REPEATED_CHAR_RE = re.compile(r"(.)\1{3,}")  # aaaa → a (>3 repeats)


def clean_tweet_text(
    text: str,
    *,
    remove_urls: bool = True,
    remove_mentions: bool = True,
    remove_hashtags: bool = False,
    max_length: int = 1000,
) -> str:
    """Clean a single tweet string.

    Args:
        text: Raw tweet text.
        remove_urls: Strip http/www links.
        remove_mentions: Strip @handle mentions.
        remove_hashtags: Replace #tag with the tag word (default) or strip.
        max_length: Truncate at this many characters.

    Returns:
        Cleaned string; empty string if input is effectively empty.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    # Decode HTML entities (e.g., &amp; → &)
    text = html.unescape(text)

    if remove_urls:
        text = _URL_RE.sub(" ", text)

    if remove_mentions:
        text = _MENTION_RE.sub(" ", text)

    if remove_hashtags:
        text = _HASHTAG_RE.sub(" ", text)
    else:
        # Keep the word, drop the #
        text = _HASHTAG_RE.sub(r"\1", text)

    # Collapse repeated characters: "sooooo bad" → "soo bad"
    text = _REPEATED_CHAR_RE.sub(r"\1\1", text)

    # Collapse whitespace
    text = _MULTI_SPACE_RE.sub(" ", text).strip()

    return text[:max_length]


def clean_dataframe(df: pd.DataFrame, *, min_length: int = 10) -> pd.DataFrame:
    """Apply clean_tweet_text to every row and drop too-short results.

    Args:
        df: DataFrame with a 'text' column.
        min_length: Drop rows where cleaned text is shorter than this.

    Returns:
        DataFrame with added 'clean_text' column and short/null rows dropped.
    """
    logger.info("Cleaning %d rows (min_length=%d)", len(df), min_length)
    df = df.copy()
    df["clean_text"] = df["text"].apply(
        lambda t: clean_tweet_text(t) if isinstance(t, str) else ""
    )
    before = len(df)
    df = df[df["clean_text"].str.len() >= min_length].reset_index(drop=True)
    logger.info(
        "Dropped %d rows with short/empty clean text (%d remain)",
        before - len(df),
        len(df),
    )
    return df


def detect_exact_duplicates(texts: Sequence[str]) -> set[int]:
    """Return indices of duplicate texts (keep first occurrence).

    Args:
        texts: Sequence of strings.

    Returns:
        Set of integer indices that are duplicates (to be removed).
    """
    seen: set[str] = set()
    dupes: set[int] = set()
    for i, t in enumerate(texts):
        normalised = t.lower().strip()
        if normalised in seen:
            dupes.add(i)
        else:
            seen.add(normalised)
    return dupes


def drop_exact_duplicates(df: pd.DataFrame, text_col: str = "clean_text") -> pd.DataFrame:
    """Remove exact duplicate texts from a DataFrame.

    Args:
        df: Input DataFrame.
        text_col: Column to check for duplicates.

    Returns:
        Deduplicated DataFrame.
    """
    before = len(df)
    df = df.drop_duplicates(subset=[text_col], keep="first").reset_index(drop=True)
    logger.info(
        "Exact dedup: removed %d rows (%d remain)",
        before - len(df),
        len(df),
    )
    return df
