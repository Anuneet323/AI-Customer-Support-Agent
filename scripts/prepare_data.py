"""
scripts/prepare_data.py
────────────────────────
Phase 1–3: Load, clean, reconstruct conversations, and build splits.

Run after explore_data.py has determined the brand.

Outputs:
  data/processed/conversations.jsonl      (all conversations for chosen brand)
  data/processed/train.jsonl              (70% split for retrieval index)
  data/processed/val.jsonl               (15% split for threshold tuning)
  data/processed/held_out.jsonl          (15% split — golden set drawn from this)
  data/sample/sample_conversations.jsonl (small deterministic sample for demos)

Usage:
    python scripts/prepare_data.py --brand AmazonHelp
    python scripts/prepare_data.py --brand AmazonHelp --nrows 300000
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
from tqdm import tqdm

from src.data.loader import load_raw
from src.data.cleaner import clean_dataframe, drop_exact_duplicates
from src.data.conversations import ConversationBuilder, conversations_to_df

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger(__name__)


def load_config() -> dict:
    config_path = Path("configs/config.yaml")
    with open(config_path) as f:
        return yaml.safe_load(f)


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, default=str) + "\n")
    logger.info("Wrote %d records → %s", len(records), path)


def split_conversations(
    conversations: list,
    train_frac: float,
    val_frac: float,
    seed: int,
) -> tuple[list, list, list]:
    """Deterministic 3-way split by conversation_id hash for stability."""
    rng = random.Random(seed)
    shuffled = conversations.copy()
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    train = shuffled[:n_train]
    val = shuffled[n_train: n_train + n_val]
    held_out = shuffled[n_train + n_val:]
    return train, val, held_out


def main(args: argparse.Namespace) -> None:
    cfg = load_config()
    seed = cfg["project"]["random_seed"]
    brand = args.brand or cfg["data"]["brand"]
    nrows = args.nrows or cfg["data"].get("max_conversations_raw", None)

    logger.info("Preparing data for brand: %s", brand)

    # ── Load and clean ────────────────────────────────────────
    df = load_raw("data/raw/twcs.csv", nrows=nrows)
    df = clean_dataframe(df, min_length=cfg["data"]["min_text_length"])
    df = drop_exact_duplicates(df)

    logger.info("After dedup: %d rows", len(df))

    # ── Reconstruct conversations ─────────────────────────────
    logger.info("Reconstructing conversations for brand: %s", brand)
    builder = ConversationBuilder(df, brand=brand)
    conversations = builder.build(
        min_length=cfg["data"]["min_conversation_length"],
        require_brand_reply=True,
        max_conversations=cfg["data"]["max_conversations"],
    )

    if not conversations:
        logger.error(
            "No conversations found for brand '%s'. "
            "Check the brand name matches author_id in the dataset. "
            "Try running: python scripts/explore_data.py to see brand list.",
            brand,
        )
        sys.exit(1)

    # ── Stats ─────────────────────────────────────────────────
    conv_dicts = [c.to_dict() for c in conversations]
    conv_df = conversations_to_df(conversations)

    resolved = conv_df["appears_resolved"].sum()
    total = len(conv_df)

    print("\n" + "=" * 60)
    print("CONVERSATION RECONSTRUCTION STATS")
    print("=" * 60)
    print(f"Brand:                          {brand}")
    print(f"Total conversations:            {total:>10,}")
    print(f"Resolved (heuristic):           {resolved:>10,}  ({resolved/total:.1%})")
    print(f"Average conversation length:    {conv_df['length'].mean():>10.2f} turns")
    print(f"Median conversation length:     {conv_df['length'].median():>10.1f} turns")
    print(f"Max conversation length:        {conv_df['length'].max():>10}")
    print(f"High-confidence resolved:       "
          f"{(conv_df['resolution_confidence'] == 'high').sum():>10,}")
    print("=" * 60 + "\n")

    # ── Save all conversations ────────────────────────────────
    proc_dir = Path("data/processed")
    write_jsonl(proc_dir / "conversations.jsonl", conv_dicts)

    # ── Split ─────────────────────────────────────────────────
    train_frac = cfg["data"]["train_fraction"]
    val_frac = cfg["data"]["val_fraction"]

    train_convs, val_convs, held_out_convs = split_conversations(
        conv_dicts, train_frac, val_frac, seed=seed
    )

    logger.info(
        "Split: train=%d, val=%d, held_out=%d",
        len(train_convs),
        len(val_convs),
        len(held_out_convs),
    )

    write_jsonl(proc_dir / "train.jsonl", train_convs)
    write_jsonl(proc_dir / "val.jsonl", val_convs)
    write_jsonl(proc_dir / "held_out.jsonl", held_out_convs)

    # ── Small deterministic sample for demos ──────────────────
    sample_dir = Path("data/sample")
    sample_size = min(200, len(conv_dicts))
    rng = random.Random(seed)
    sample = rng.sample(conv_dicts, sample_size)
    write_jsonl(sample_dir / "sample_conversations.jsonl", sample)

    # ── Update config with selected brand ─────────────────────
    config_path = Path("configs/config.yaml")
    config_text = config_path.read_text()
    # Replace brand line only if it's a placeholder
    if 'brand: "AmazonHelp"' in config_text or "brand: null" in config_text:
        config_text = config_text.replace(
            'brand: "AmazonHelp"', f'brand: "{brand}"'
        ).replace("brand: null", f'brand: "{brand}"')
        config_path.write_text(config_text)
        logger.info("Updated configs/config.yaml with brand: %s", brand)

    print(f"[OK] Data preparation complete for brand: {brand}")
    print(f"  conversations.jsonl : {len(conv_dicts)}")
    print(f"  train.jsonl         : {len(train_convs)}")
    print(f"  val.jsonl           : {len(val_convs)}")
    print(f"  held_out.jsonl      : {len(held_out_convs)}")
    print(f"  sample (demo)       : {len(sample)}")
    print("\nNEXT: python scripts/discover_intents.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare conversation data.")
    parser.add_argument("--brand", type=str, default=None,
                        help="Brand author_id (overrides config)")
    parser.add_argument("--nrows", type=int, default=None,
                        help="Max raw rows to load")
    main(parser.parse_args())
