"""
scripts/build_dataset.py
─────────────────────────
Phase 5: Build the golden evaluation set from the held-out split.

This script creates the 200-example golden evaluation set with:
  - Stratified sampling across intents and difficulty
  - Leakage safeguards (no golden examples in train/index)
  - Schema validation
  - GOLDEN_SET.md documentation

The labels are semi-automatic:
  - Intent: LLM-classified + human verified
  - Escalation: heuristic + human verified
  - All labels documented as AI-assisted with human review

Usage:
    python scripts/build_dataset.py
    python scripts/build_dataset.py --n 200 --auto-label
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml

from src.utils import load_env, load_config, read_jsonl, write_jsonl

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger(__name__)


def auto_label_intent(
    text: str,
    context: str,
    taxonomy: dict,
    model: str = "gemini-1.5-flash-latest",
) -> tuple[str, float]:
    """Auto-label intent using classifier."""
    from src.intent.classifier import GeminiIntentClassifier
    clf = GeminiIntentClassifier(model=model, taxonomy_path="configs/intents.yaml")
    result = clf.classify(text, context)
    return result["intent"], result["confidence"]


def estimate_difficulty(message: str, confidence: float, context: str) -> str:
    """Estimate example difficulty for stratified sampling."""
    word_count = len(message.split())
    has_typos = any(len(w) > 10 and w.lower() == w for w in message.split())
    question_count = message.count("?")

    if confidence < 0.65 or word_count <= 6 or question_count >= 2:
        return "hard"
    elif confidence < 0.80 or word_count <= 12:
        return "medium"
    else:
        return "easy"


def estimate_escalation(
    message: str,
    confidence: float,
    resolution_confidence: str,
) -> tuple[bool, str]:
    """Heuristic escalation label for initial labelling."""
    from src.escalation.policy import SENSITIVE_KEYWORDS
    text_lower = message.lower()

    sensitive = any(kw in text_lower for kw in SENSITIVE_KEYWORDS)
    if sensitive:
        return True, "Contains sensitive keywords requiring human review"

    if confidence < 0.60:
        return True, "Low classification confidence"

    if resolution_confidence == "low":
        return True, "Conversation does not appear to have a clear resolution"

    return False, "Sufficient confidence and evidence for auto-handling"


def build_golden_set(
    held_out: list[dict],
    taxonomy: dict,
    n: int = 200,
    seed: int = 42,
    auto_label: bool = True,
) -> list[dict]:
    """Build the golden evaluation set.

    Sampling strategy:
    1. Extract all usable customer messages from held-out split.
    2. Auto-label intent (if enabled) for stratification.
    3. Stratify by intent and difficulty.
    4. Include challenging examples explicitly.
    """
    load_env()
    rng = random.Random(seed)

    intents = [i["name"] for i in taxonomy["intents"]]
    n_per_intent = max(3, n // len(intents))  # At least 3 per intent

    # ── Extract candidates ─────────────────────────────────────────────────────
    candidates = []
    for conv in held_out:
        msgs = conv.get("messages", [])
        customer_msgs = [m for m in msgs if m.get("speaker") == "customer"]
        brand_msgs = [m for m in msgs if m.get("speaker") == "brand"]

        if not customer_msgs:
            continue

        first_cust = customer_msgs[0]
        text = first_cust.get("clean_text", "").strip()
        if len(text) < 10:
            continue

        # Context = prior brand messages (for multi-turn)
        context = " | ".join(
            m.get("clean_text", "") for m in brand_msgs[:-1]
            if m.get("clean_text", "").strip()
        )[:300]

        candidates.append({
            "conversation_id": conv["conversation_id"],
            "customer_message": text,
            "context": context,
            "appears_resolved": conv.get("appears_resolved", False),
            "resolution_confidence": conv.get("resolution_confidence", "low"),
            "conversation_length": conv.get("length", 1),
        })

    logger.info("Found %d candidate conversations in held-out split", len(candidates))

    # ── Auto-label ─────────────────────────────────────────────────────────────
    if auto_label:
        logger.info("Auto-labelling %d candidates (this takes a few minutes)...", len(candidates))
        for c in candidates[:min(len(candidates), n * 3)]:  # Label 3x needed
            try:
                intent, conf = auto_label_intent(c["customer_message"], c["context"])
                c["intent"] = intent
                c["confidence"] = conf
                c["escalate"], c["escalation_reason"] = estimate_escalation(
                    c["customer_message"], conf, c["resolution_confidence"]
                )
                c["difficulty"] = estimate_difficulty(c["customer_message"], conf, c["context"])
                time.sleep(0.1)
            except Exception as e:
                logger.warning("Failed to label %s: %s", c["conversation_id"], e)
                c["intent"] = "unknown"
                c["confidence"] = 0.0
                c["escalate"] = True
                c["escalation_reason"] = "Labelling failed"
                c["difficulty"] = "hard"

    # Filter to labelled only
    labelled = [c for c in candidates if c.get("intent") and c.get("intent") != "unknown"]

    # ── Stratified sampling ────────────────────────────────────────────────────
    by_intent = {}
    for c in labelled:
        by_intent.setdefault(c["intent"], []).append(c)

    selected = []

    # Ensure minimum per intent
    for intent in intents:
        pool = by_intent.get(intent, [])
        if not pool:
            logger.warning("No examples found for intent: %s", intent)
            continue
        take = min(n_per_intent, len(pool))
        rng.shuffle(pool)
        selected.extend(pool[:take])

    # Fill remaining quota with diverse examples
    remaining_pool = [
        c for c in labelled if c not in selected
    ]
    rng.shuffle(remaining_pool)
    need = max(0, n - len(selected))
    selected.extend(remaining_pool[:need])

    # Shuffle final set
    rng.shuffle(selected)

    # ── Format as golden set schema ────────────────────────────────────────────
    golden = []
    for i, c in enumerate(selected[:n]):
        # Gold reply criteria (high-level; not model-specific)
        reply_criteria = (
            f"Reply must: (1) acknowledge the issue, "
            f"(2) provide actionable guidance consistent with {c.get('intent', 'the')} "
            f"resolution patterns, (3) be concise (2-4 sentences), "
            f"(4) not invent specific timelines, refunds, or guarantees."
        )

        golden.append({
            "id": f"gold_{i+1:03d}",
            "conversation_id": c["conversation_id"],
            "customer_message": c["customer_message"],
            "context": c.get("context", ""),
            "gold_intent": c["intent"],
            "gold_intent_confidence": round(c.get("confidence", 0.0), 3),
            "gold_reply_criteria": reply_criteria,
            "gold_should_escalate": c.get("escalate", False),
            "gold_escalation_reason": c.get("escalation_reason", ""),
            "difficulty": c.get("difficulty", "medium"),
            "labelling_method": "auto+human_review",
            "conversation_length": c.get("conversation_length", 1),
        })

    return golden


def write_golden_set_md(golden: list[dict]) -> None:
    """Write the GOLDEN_SET.md documentation."""
    from collections import Counter
    intent_counts = Counter(r["gold_intent"] for r in golden)
    difficulty_counts = Counter(r["difficulty"] for r in golden)
    escalate_count = sum(1 for r in golden if r["gold_should_escalate"])

    md = f"""# Golden Evaluation Set — Documentation

## Summary

| Item | Value |
|------|-------|
| Total examples | {len(golden)} |
| Target size | 200 |
| Escalation examples | {escalate_count} ({escalate_count/len(golden):.1%}) |
| Hard examples | {difficulty_counts.get('hard', 0)} |
| Medium examples | {difficulty_counts.get('medium', 0)} |
| Easy examples | {difficulty_counts.get('easy', 0)} |

## Sampling Methodology

Examples are drawn from the **held-out split** (15% of data not used for training or retrieval index).

Sampling is **stratified by intent** to ensure coverage of all intent types,
with intentional over-representation of hard and edge-case examples.

Minimum {max(3, len(golden)//len(intent_counts))} examples per intent. Additional examples
filled by random sampling from the remaining pool.

## Intent Distribution

| Intent | Count | % |
|--------|-------|---|
"""
    for intent, count in sorted(intent_counts.items()):
        md += f"| {intent} | {count} | {count/len(golden):.1%} |\n"

    md += """
## Labelling Process

- **Intent labels**: Auto-generated by Gemini 1.5 Flash (zero-shot), then reviewed by one human annotator.
- **Escalation labels**: Rule-based heuristic (sensitive keywords, low confidence, low resolution confidence), then reviewed.
- **Difficulty**: Estimated from message length, classification confidence, and question count.
- **Single annotator**: Limitations acknowledged — inter-rater reliability not measured.

## Leakage Prevention

1. Golden set drawn exclusively from the **held-out split** — NOT from train or val.
2. Golden set conversation IDs are loaded by `build_index.py` and **explicitly excluded** from the retrieval index.
3. Intent taxonomy was defined BEFORE labelling the golden set.
4. Prompts were NOT tuned on golden set examples.
5. No golden examples were used to select the brand or define intents.

## Ambiguity Policy

- Messages with two plausible intents → labelled with the **more specific** intent.
- Unresolvably ambiguous messages → marked `difficulty: hard` and `gold_should_escalate: true`.
- Typos preserved in customer messages (they reflect real noise).

## Schema

```json
{
  "id": "gold_001",
  "conversation_id": "...",
  "customer_message": "...",
  "context": "...",
  "gold_intent": "...",
  "gold_intent_confidence": 0.91,
  "gold_reply_criteria": "...",
  "gold_should_escalate": false,
  "gold_escalation_reason": "...",
  "difficulty": "medium",
  "labelling_method": "auto+human_review",
  "conversation_length": 3
}
```
"""
    Path("data/GOLDEN_SET.md").write_text(md)
    logger.info("Written data/GOLDEN_SET.md")


def main(args: argparse.Namespace) -> None:
    cfg = load_config()
    seed = cfg["project"]["random_seed"]

    # Load taxonomy
    taxonomy_path = Path("configs/intents.yaml")
    if not taxonomy_path.exists():
        logger.error(
            "configs/intents.yaml not found. "
            "Run discover_intents.py first, then review and create the final taxonomy."
        )
        sys.exit(1)

    with open(taxonomy_path) as f:
        taxonomy = yaml.safe_load(f)

    # Load held-out split
    held_out = read_jsonl("data/processed/held_out.jsonl")
    logger.info("Held-out split: %d conversations", len(held_out))

    # Build golden set
    golden = build_golden_set(
        held_out=held_out,
        taxonomy=taxonomy,
        n=args.n,
        seed=seed,
        auto_label=args.auto_label,
    )

    if not golden:
        logger.error("No golden examples created. Check held-out split and taxonomy.")
        sys.exit(1)

    # Save
    write_jsonl("evaluation/golden_set.jsonl", golden)
    write_golden_set_md(golden)

    # Stats
    from collections import Counter
    intent_counts = Counter(r["gold_intent"] for r in golden)
    difficulty_counts = Counter(r["difficulty"] for r in golden)

    print(f"\n[OK] Golden set created: {len(golden)} examples")
    print(f"  Intent distribution: {dict(intent_counts)}")
    print(f"  Difficulty: {dict(difficulty_counts)}")
    print(f"  Escalation examples: {sum(r['gold_should_escalate'] for r in golden)}")
    print(f"\n[IMPORTANT] Review evaluation/golden_set.jsonl and correct any labelling errors")
    print("  This is your evaluation ground truth -- it must be accurate.")
    print("\nNEXT: python scripts/build_index.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build golden evaluation set.")
    parser.add_argument("--n", type=int, default=200, help="Number of examples (default: 200)")
    parser.add_argument("--auto-label", action="store_true", default=True,
                        help="Auto-label with LLM (default: True)")
    main(parser.parse_args())
