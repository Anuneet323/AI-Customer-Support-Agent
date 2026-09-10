"""
src/data/loader.py
──────────────────
Loads and lightly validates the raw Kaggle Twitter Customer Support CSV.

Dataset: thoughtvector/customer-support-on-twitter
Expected columns: tweet_id, author_id, inbound, created_at, text,
                  response_tweet_id, in_response_to_tweet_id
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Columns we actually use
REQUIRED_COLUMNS = {
    "tweet_id",
    "author_id",
    "inbound",
    "created_at",
    "text",
    "response_tweet_id",
    "in_response_to_tweet_id",
}


def load_raw(path: str | Path, nrows: Optional[int] = None) -> pd.DataFrame:
    """Load the raw twcs.csv and do minimal validation.

    Args:
        path: Path to twcs.csv
        nrows: If set, only read this many rows (for fast iteration).

    Returns:
        DataFrame with string tweet_id columns and cleaned dtypes.

    Raises:
        FileNotFoundError: If the CSV does not exist.
        ValueError: If required columns are missing.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}.\n"
            "Download twcs.csv from:\n"
            "  https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter\n"
            "and place it at data/raw/twcs.csv"
        )

    logger.info("Loading raw dataset from %s (nrows=%s)", path, nrows)
    df = pd.read_csv(
        path,
        nrows=nrows,
        dtype=str,          # Everything as string to avoid ID truncation
        na_values=["", "NA", "null", "NULL", "None"],
        keep_default_na=True,
    )

    # Validate columns
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"CSV is missing expected columns: {missing}\n"
            f"Found columns: {list(df.columns)}"
        )

    # Normalise column names to lowercase
    df.columns = [c.lower().strip() for c in df.columns]

    # Parse inbound flag
    df["inbound"] = df["inbound"].map(
        {"True": True, "False": False, "true": True, "false": False}
    )

    # Parse timestamp (UTC)
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True, errors="coerce")

    # Strip whitespace from text
    df["text"] = df["text"].str.strip()

    n_total = len(df)
    n_null_text = df["text"].isna().sum()
    logger.info(
        "Loaded %d rows | %d null text | %d inbound | %d outbound",
        n_total,
        n_null_text,
        df["inbound"].sum(),
        (~df["inbound"].fillna(False)).sum(),
    )
    return df


def get_brand_tweets(df: pd.DataFrame, brand: str) -> pd.DataFrame:
    """Return all tweets where author_id matches the brand handle."""
    mask = df["author_id"].str.lower() == brand.lower()
    result = df[mask].copy()
    logger.info("Brand '%s': %d tweets", brand, len(result))
    return result


def get_all_brands(df: pd.DataFrame) -> pd.Series:
    """Return brand tweet counts (outbound author_ids sorted by count)."""
    outbound = df[df["inbound"] == False]  # noqa: E712
    return outbound["author_id"].value_counts()


def sample_rows(df: pd.DataFrame, n: int, seed: int = 42) -> pd.DataFrame:
    """Random sample of rows, deterministic."""
    if len(df) <= n:
        return df
    return df.sample(n=n, random_state=seed).reset_index(drop=True)
