"""
scripts/demo.py
────────────────
Interactive demo of the full pipeline.

Shows a single example with all intermediate outputs.

Usage:
    python scripts/demo.py
    python scripts/demo.py --message "I was charged twice for my order"
    python scripts/demo.py --interactive
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import load_env, setup_logging

DEMO_MESSAGES = [
    "I was charged twice and still haven't received my money back. It's been 3 days!",
    "My package says delivered but I never received it",
    "How do I cancel my Prime membership?",
    "The item I received is broken, I want a refund",
    "I can't log into my account, keeps saying wrong password",
]


def run_demo(message: str) -> None:
    from src.pipeline import SupportAgentPipeline

    pipeline = SupportAgentPipeline()
    output = pipeline.run(message)

    print("\n" + "=" * 70)
    print("SUPPORT AGENT DEMO")
    print("=" * 70)
    print(f"\nCustomer:\n  {message}")
    print(f"\nIntent:              {output.intent}")
    print(f"Confidence:          {output.intent_confidence:.2f}")
    print(f"Reasoning:           {output.intent_reasoning}")
    print(f"\nEvidence quality:    {output.evidence_quality}")
    print(f"Best similarity:     {output.best_similarity:.3f}")
    print(f"Retrieved cases:     {len(output.evidence)}")

    if output.evidence:
        print("\nTop retrieved case:")
        e = output.evidence[0]
        print(f"  Case ID:    {e['case_id']}")
        print(f"  Similarity: {e['similarity']:.3f}")
        print(f"  Problem:    {e['customer_problem'][:100]}...")
        print(f"  Resolution: {e['resolution'][:150]}...")

    print(f"\nDraft reply:\n  {output.reply}")
    print(f"\nEscalate:            {output.escalate}")
    if output.escalate:
        print(f"Escalation reason:   {output.escalation_reason}")
    print(f"Hallucination risk:  {output.hallucination_risk}")
    print(f"Latency:             {output.latency_ms:.0f}ms")

    if output.errors:
        print(f"\nErrors:              {output.errors}")

    print("=" * 70 + "\n")

    # Save as JSON too
    out_path = Path("experiments/results/demo_output.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output.to_dict(), indent=2))
    print(f"Full output saved -> {out_path}")


def main(args: argparse.Namespace) -> None:
    load_env()
    setup_logging()

    if args.interactive:
        print("Interactive mode. Type 'quit' to exit.")
        while True:
            msg = input("\nCustomer message: ").strip()
            if msg.lower() in ("quit", "exit", "q"):
                break
            if msg:
                run_demo(msg)
    elif args.message:
        run_demo(args.message)
    else:
        # Default: run all demo messages
        for msg in DEMO_MESSAGES:
            run_demo(msg)
            print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Demo the support agent pipeline.")
    parser.add_argument("--message", type=str, default=None,
                        help="Single message to process")
    parser.add_argument("--interactive", action="store_true",
                        help="Interactive mode")
    main(parser.parse_args())
