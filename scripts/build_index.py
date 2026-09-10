"""
scripts/build_index.py
───────────────────────
Phase 8: Build the FAISS retrieval index from the training split.

CRITICAL LEAKAGE PREVENTION:
  The golden set conversation IDs are loaded and EXCLUDED from the index.
  This ensures evaluation examples cannot retrieve their own answer.

  Documentation:
  - Golden set IDs loaded from evaluation/golden_set.jsonl
  - Excluded by conversation_id match
  - Verified: index size should be < train set size by N(golden set)

Usage:
    python scripts/build_index.py
    python scripts/build_index.py --split train        # default
    python scripts/build_index.py --split train,val    # include val too
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tqdm import tqdm

from src.utils import load_env, setup_logging, load_config, read_jsonl
from src.retrieval.index import FAISSIndex, EmbeddingModel, CaseRecord

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger(__name__)


def load_excluded_ids(golden_set_path: str) -> set[str]:
    """Load conversation IDs from the golden set to exclude from index."""
    golden_path = Path(golden_set_path)
    if not golden_path.exists():
        logger.warning("Golden set not found at %s — no IDs excluded", golden_path)
        return set()

    excluded = set()
    for record in read_jsonl(golden_path):
        cid = record.get("conversation_id") or record.get("id", "").replace("gold_", "")
        if cid:
            excluded.add(cid)

    logger.info(
        "Loaded %d conversation IDs to exclude from index (leakage prevention)",
        len(excluded),
    )
    return excluded


def build_case_record(conv: dict) -> CaseRecord | None:
    """Convert a conversation dict to a CaseRecord for indexing."""
    messages = conv.get("messages", [])
    if not messages:
        return None

    # Customer problem = first customer message
    customer_msgs = [m for m in messages if m.get("speaker") == "customer"]
    brand_msgs = [m for m in messages if m.get("speaker") == "brand"]

    if not customer_msgs or not brand_msgs:
        return None

    # Build customer problem (first customer message + any follow-up context)
    customer_text = customer_msgs[0].get("clean_text", "").strip()
    if not customer_text or len(customer_text) < 10:
        return None

    # Resolution = last brand reply
    last_brand = brand_msgs[-1]
    resolution_text = last_brand.get("clean_text", "").strip()
    if not resolution_text or len(resolution_text) < 10:
        return None

    return CaseRecord(
        case_id=conv["conversation_id"],
        customer_problem=customer_text,
        resolution=resolution_text,
        brand_reply=last_brand.get("text", resolution_text),
        intent=conv.get("intent"),  # May be None if not labelled
        appears_resolved=conv.get("appears_resolved", False),
        resolution_confidence=conv.get("resolution_confidence", "low"),
        conversation_length=conv.get("length", len(messages)),
        metadata={
            "brand": conv.get("brand", ""),
            "n_customer_turns": len(customer_msgs),
            "n_brand_turns": len(brand_msgs),
        },
    )


def main(args: argparse.Namespace) -> None:
    load_env()
    cfg = load_config()

    index_dir = Path(cfg["retrieval"]["index_path"])
    golden_set_path = cfg["evaluation"]["golden_set_path"]

    # ── Load excluded IDs ──────────────────────────────────────────────────────
    excluded_ids = load_excluded_ids(golden_set_path)

    # ── Load conversations ─────────────────────────────────────────────────────
    splits = [s.strip() for s in args.split.split(",")]
    conversations = []
    for split in splits:
        split_path = Path(f"data/processed/{split}.jsonl")
        if not split_path.exists():
            logger.warning("Split file not found: %s", split_path)
            continue
        convs = read_jsonl(split_path)
        logger.info("Loaded %d conversations from %s", len(convs), split_path)
        conversations.extend(convs)

    if not conversations:
        logger.error("No conversations loaded. Run prepare_data.py first.")
        sys.exit(1)

    # ── Filter conversations ───────────────────────────────────────────────────
    before = len(conversations)
    conversations = [c for c in conversations if c["conversation_id"] not in excluded_ids]
    logger.info(
        "After excluding golden set IDs: %d → %d (excluded %d)",
        before, len(conversations), before - len(conversations),
    )

    # Filter to only resolved conversations
    if args.resolved_only:
        conversations = [c for c in conversations if c.get("appears_resolved", False)]
        logger.info("After resolved-only filter: %d conversations", len(conversations))

    # ── Build case records ─────────────────────────────────────────────────────
    records = []
    skipped = 0
    for conv in tqdm(conversations, desc="Building case records"):
        rec = build_case_record(conv)
        if rec:
            records.append(rec)
        else:
            skipped += 1

    logger.info("Built %d case records (%d skipped)", len(records), skipped)

    if not records:
        logger.error("No valid records to index.")
        sys.exit(1)

    # ── Embed ──────────────────────────────────────────────────────────────────
    emb_model = EmbeddingModel(model=cfg["retrieval"]["embedding_model"])
    texts = [r.customer_problem for r in records]

    logger.info("Embedding %d customer problems (batched)...", len(texts))
    embeddings = emb_model.embed_texts(texts, task_type="RETRIEVAL_DOCUMENT")

    # ── Build and save FAISS index ─────────────────────────────────────────────
    index = FAISSIndex(dim=embeddings.shape[1])
    index.add(records, embeddings)
    index.save(index_dir)

    # ── Verify leakage ─────────────────────────────────────────────────────────
    indexed_ids = {r.case_id for r in records}
    overlap = indexed_ids & excluded_ids
    if overlap:
        logger.error(
            "LEAKAGE DETECTED: %d golden set IDs found in index! IDs: %s",
            len(overlap), list(overlap)[:5],
        )
        sys.exit(1)
    else:
        logger.info(
            "[OK] Leakage check passed: 0 golden set IDs in index "
            "(verified %d indexed IDs vs %d excluded IDs)",
            len(indexed_ids), len(excluded_ids),
        )

    print(f"\n[OK] Index built successfully:")
    print(f"  Records indexed:    {len(records)}")
    print(f"  Excluded (golden):  {len(excluded_ids)}")
    print(f"  Embedding dim:      {embeddings.shape[1]}")
    print(f"  Index saved to:     {index_dir}")
    print(f"\nNEXT: python evaluation/evaluate.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build FAISS retrieval index.")
    parser.add_argument("--split", default="train",
                        help="Comma-separated splits to index (default: train)")
    parser.add_argument("--resolved-only", action="store_true",
                        help="Only index conversations that appear resolved")
    main(parser.parse_args())
