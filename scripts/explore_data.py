"""
scripts/explore_data.py
────────────────────────
Phase 1 & 2: Dataset exploration and brand selection.

Run this FIRST after placing twcs.csv in data/raw/.

Outputs:
  - Console statistics (copy-pasteable into report)
  - data/brand_selection.json  (scored brand candidates)
  - data/sample/brand_stats.csv

Usage:
    python scripts/explore_data.py
    python scripts/explore_data.py --nrows 500000  # faster for initial look
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from tabulate import tabulate

from src.data.loader import load_raw, get_all_brands
from src.data.cleaner import clean_dataframe

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger(__name__)


def score_brand(brand: str, df: pd.DataFrame, brand_tweet_counts: pd.Series) -> dict:
    """Score a brand on multiple criteria for selection.

    Scoring factors:
      1. Volume: number of brand tweets (more data → better eval)
      2. Resolution rate: fraction of threads where brand replied last
      3. Intent diversity: estimate via keyword variety (proxy)
      4. Noise level: fraction of tweets that are extremely short

    Returns dict with scores and composite.
    """
    brand_tweets = df[df["author_id"].str.lower() == brand.lower()]
    # Customer tweets that mention this brand (inbound to threads brand replied to)
    customer_tweets = df[
        (df["inbound"] == True)  # noqa: E712
        & df["in_response_to_tweet_id"].isin(brand_tweets["tweet_id"])
    ]

    n_brand = len(brand_tweets)
    n_customer = len(customer_tweets)

    # Resolution rate: brand tweets with clean text ≥ 20 chars (actual response)
    resolved = brand_tweets[
        brand_tweets["clean_text"].fillna("").str.len() >= 20
    ]
    resolution_rate = len(resolved) / max(n_brand, 1)

    # Average response length (higher → more informative)
    avg_len = brand_tweets["clean_text"].fillna("").str.len().mean()

    # Noise rate: extremely short brand replies (≤ 10 chars after cleaning)
    noise_rate = (
        brand_tweets["clean_text"].fillna("").str.len() <= 10
    ).mean()

    # Intent diversity proxy: unique first-word distribution in customer tweets
    first_words = (
        customer_tweets["clean_text"]
        .fillna("")
        .str.split()
        .str[0]
        .str.lower()
        .dropna()
    )
    diversity = first_words.nunique() / max(len(first_words), 1)

    # Composite score (higher is better)
    # Weights chosen to favour: volume, resolution quality, diversity, low noise
    score = (
        0.30 * min(n_brand / 5000, 1.0)         # volume (cap at 5k)
        + 0.25 * resolution_rate                  # quality of replies
        + 0.25 * min(avg_len / 100, 1.0)         # reply informativeness
        + 0.10 * min(diversity * 10, 1.0)         # intent diversity
        + 0.10 * (1 - noise_rate)                 # low noise
    )

    return {
        "brand": brand,
        "brand_tweets": n_brand,
        "customer_tweets": n_customer,
        "resolution_rate": round(resolution_rate, 3),
        "avg_reply_length": round(avg_len, 1),
        "noise_rate": round(noise_rate, 3),
        "intent_diversity_proxy": round(diversity, 4),
        "composite_score": round(score, 4),
    }


def main(args: argparse.Namespace) -> None:
    raw_path = Path("data/raw/twcs.csv")
    out_dir = Path("data")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load ──────────────────────────────────────────────────
    df = load_raw(raw_path, nrows=args.nrows)
    df = clean_dataframe(df, min_length=5)  # Light clean for exploration

    total_records = len(df)
    inbound = df[df["inbound"] == True]  # noqa: E712
    outbound = df[df["inbound"] == False]  # noqa: E712

    # ── Brand list ────────────────────────────────────────────
    brand_counts = get_all_brands(df)
    top_brands = brand_counts.head(args.top_n)

    print("\n" + "=" * 60)
    print("DATASET EXPLORATION REPORT")
    print("=" * 60)
    print(f"Total records inspected:        {total_records:>10,}")
    print(f"Inbound (customer) tweets:      {len(inbound):>10,}")
    print(f"Outbound (brand) tweets:        {len(outbound):>10,}")
    print(f"Unique brands (outbound):       {brand_counts.nunique():>10,}")
    print(f"Null text rows (after clean):   {df['clean_text'].isna().sum():>10,}")

    # Timestamp range
    valid_ts = df["created_at"].dropna()
    if len(valid_ts):
        print(f"Date range:  {valid_ts.min().date()} to {valid_ts.max().date()}")

    # Conversation structure
    has_parent = df["in_response_to_tweet_id"].notna().sum()
    print(f"Tweets with parent (replies):   {has_parent:>10,}")
    print(f"Root tweets (no parent):        {(df['in_response_to_tweet_id'].isna()).sum():>10,}")

    print(f"\nTop {args.top_n} brands by tweet count:")
    print(tabulate(
        [[b, c] for b, c in top_brands.items()],
        headers=["Brand", "Tweet Count"],
        tablefmt="simple",
    ))

    # ── Score top brands ──────────────────────────────────────
    print(f"\nScoring top {args.score_n} brands for suitability...")
    candidates = list(brand_counts.head(args.score_n).index)
    scores = []
    for brand in candidates:
        s = score_brand(brand, df, brand_counts)
        scores.append(s)
        print(f"  Scored: {brand} -> {s['composite_score']:.4f}")

    scores.sort(key=lambda x: x["composite_score"], reverse=True)

    print("\nBrand selection scores (sorted):")
    headers = [
        "Brand", "Brand Tweets", "Customer Tweets", "Resolution Rate",
        "Avg Reply Len", "Noise Rate", "Diversity", "Score"
    ]
    rows = [
        [
            s["brand"], s["brand_tweets"], s["customer_tweets"],
            s["resolution_rate"], s["avg_reply_length"],
            s["noise_rate"], s["intent_diversity_proxy"], s["composite_score"]
        ]
        for s in scores
    ]
    print(tabulate(rows, headers=headers, tablefmt="simple", floatfmt=".4f"))

    recommended = scores[0]["brand"]
    print(f"\n[OK] Recommended brand: {recommended}")
    print("  (Highest composite score -- confirm this matches domain intuition)")

    # Save selection data
    selection_output = {
        "methodology": (
            "Brands scored on: volume (30%), resolution rate (25%), "
            "avg reply length/informativeness (25%), intent diversity proxy (10%), "
            "low noise rate (10%). Higher composite score = better evaluation problem."
        ),
        "scores": scores,
        "recommended": recommended,
        "nrows_inspected": args.nrows or "all",
    }
    out_path = out_dir / "brand_selection.json"
    out_path.write_text(json.dumps(selection_output, indent=2))
    print(f"\nSaved brand selection data -> {out_path}")

    # ── Message length distribution ───────────────────────────
    cust_lengths = inbound["clean_text"].fillna("").str.len()
    brand_lengths = outbound["clean_text"].fillna("").str.len()
    print("\nMessage length stats:")
    print(f"  Customer  — mean: {cust_lengths.mean():.0f}, median: {cust_lengths.median():.0f}, "
          f"p95: {cust_lengths.quantile(0.95):.0f}")
    print(f"  Brand     — mean: {brand_lengths.mean():.0f}, median: {brand_lengths.median():.0f}, "
          f"p95: {brand_lengths.quantile(0.95):.0f}")

    # ── Duplicate rate ────────────────────────────────────────
    dup_rate = 1 - df["clean_text"].nunique() / len(df)
    print(f"\nOverall text duplicate rate:    {dup_rate:.2%}")

    print("\n" + "=" * 60)
    print("NEXT STEP: Run  python scripts/prepare_data.py --brand <BRAND>")
    print("           where <BRAND> is the recommended brand above.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Explore the Twitter support dataset.")
    parser.add_argument("--nrows", type=int, default=None,
                        help="Limit rows loaded (default: all)")
    parser.add_argument("--top-n", type=int, default=20,
                        help="Show top N brands by tweet count (default: 20)")
    parser.add_argument("--score-n", type=int, default=30,
                        help="Score top N brands (default: 30)")
    main(parser.parse_args())
