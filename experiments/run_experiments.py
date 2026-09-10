"""
experiments/run_experiments.py
───────────────────────────────
Ablation studies comparing system configurations.

Ablations:
  A. No retrieval (LLM-only, no evidence)
  B. Retrieval + LLM (full, no escalation gate)
  C. Retrieval + escalation (full system) — default
  D. k=1 vs k=3 vs k=5 retrieval
  E. No evidence quality gate

Results saved to experiments/results/ablations_*.json.
Summary table written to experiments/results/ablation_summary.json.

Usage:
    python experiments/run_experiments.py
    python experiments/run_experiments.py --ablation no_retrieval
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import load_env, setup_logging, load_config, read_jsonl, write_jsonl
from evaluation.metrics import intent_metrics, escalation_metrics
from evaluation.bootstrap import bootstrap_ci, macro_f1_fn, escalation_f1_fn

logger = logging.getLogger(__name__)


def run_no_retrieval_ablation(golden: list[dict], cfg: dict) -> list[dict]:
    """Ablation A: LLM classifies and generates without any retrieval."""
    from src.intent.classifier import GeminiIntentClassifier
    from src.generation.reply_generator import ReplyGenerator
    from src.escalation.policy import EscalationPolicy

    clf = GeminiIntentClassifier()
    gen = ReplyGenerator()
    esc = EscalationPolicy(
        confidence_threshold=cfg["intent"]["confidence_threshold"]
    )

    results = []
    for ex in golden:
        cl = clf.classify(ex["customer_message"], ex.get("context", ""))
        gr = gen.generate(ex["customer_message"], cl["intent"], evidence=[], context=ex.get("context", ""))
        ed = esc.decide(
            customer_message=ex["customer_message"],
            intent=cl["intent"],
            intent_confidence=cl["confidence"],
            evidence_passes=False,  # No retrieval
            evidence_reason_code="NO_RETRIEVAL",
            best_evidence_similarity=0.0,
            hallucination_risk=gr["hallucination_risk"],
            generation_error=gr.get("error"),
            classification_error=cl.get("error"),
        )
        results.append({
            "id": ex["id"],
            "customer_message": ex["customer_message"],
            "gold_intent": ex["gold_intent"],
            "gold_escalate": ex["gold_should_escalate"],
            "difficulty": ex.get("difficulty", "medium"),
            "predicted_intent": cl["intent"],
            "intent_confidence": cl["confidence"],
            "predicted_escalate": ed.escalate,
            "reply": gr["reply"],
            "evidence": [],
            "evidence_quality": "NO_RETRIEVAL",
            "best_similarity": 0.0,
            "grounding_used": False,
            "hallucination_risk": gr["hallucination_risk"],
            "latency_ms": 0.0,
            "errors": [],
            "ablation": "no_retrieval",
        })
        time.sleep(0.01)
    return results


def run_topk_ablation(golden: list[dict], cfg: dict, k: int) -> list[dict]:
    """Ablation D: vary retrieval k."""
    from src.pipeline import SupportAgentPipeline
    pipeline = SupportAgentPipeline(
        index_dir=cfg["retrieval"]["index_path"],
        taxonomy_path=cfg["intent"]["taxonomy_path"],
        config_path="configs/config.yaml",
    )
    results = []
    for ex in golden:
        output = pipeline.run(
            customer_message=ex["customer_message"],
            context=ex.get("context", ""),
            top_k=k,
        )
        results.append({
            "id": ex["id"],
            "customer_message": ex["customer_message"],
            "gold_intent": ex["gold_intent"],
            "gold_escalate": ex["gold_should_escalate"],
            "difficulty": ex.get("difficulty", "medium"),
            "predicted_intent": output.intent,
            "intent_confidence": output.intent_confidence,
            "predicted_escalate": output.escalate,
            "reply": output.reply,
            "evidence": output.evidence,
            "evidence_quality": output.evidence_quality,
            "best_similarity": output.best_similarity,
            "grounding_used": output.grounding_used,
            "hallucination_risk": output.hallucination_risk,
            "latency_ms": output.latency_ms,
            "errors": output.errors,
            "ablation": f"top_k_{k}",
        })
        time.sleep(0.01)
    return results


def compute_summary_metrics(results: list[dict]) -> dict:
    gold_intents = [r["gold_intent"] for r in results]
    pred_intents = [r["predicted_intent"] for r in results]
    gold_esc = [bool(r["gold_escalate"]) for r in results]
    pred_esc = [bool(r["predicted_escalate"]) for r in results]

    im = intent_metrics(gold_intents, pred_intents)
    em = escalation_metrics(gold_esc, pred_esc)

    return {
        "intent_accuracy": im["accuracy"],
        "intent_macro_f1": im["macro_f1"],
        "escalation_f1": em["f1"],
        "false_auto_handle_rate": em["false_auto_handle_rate"],
        "n_samples": len(results),
    }


def main(args: argparse.Namespace) -> None:
    load_env()
    setup_logging()
    cfg = load_config()

    results_dir = Path(cfg["evaluation"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    golden = read_jsonl(cfg["evaluation"]["golden_set_path"])
    logger.info("Golden set: %d examples", len(golden))

    ablation = args.ablation
    summary = {}

    if ablation in ("no_retrieval", "all"):
        logger.info("Running ablation A: No retrieval...")
        results = run_no_retrieval_ablation(golden, cfg)
        write_jsonl(results_dir / "ablation_no_retrieval.jsonl", results)
        summary["no_retrieval"] = compute_summary_metrics(results)
        logger.info("Ablation A complete: %s", summary["no_retrieval"])

    if ablation in ("topk", "all"):
        for k in [1, 3, 5]:
            logger.info("Running ablation D: top_k=%d...", k)
            results = run_topk_ablation(golden, cfg, k)
            write_jsonl(results_dir / f"ablation_top_k_{k}.jsonl", results)
            summary[f"top_k_{k}"] = compute_summary_metrics(results)
            logger.info("k=%d: %s", k, summary[f"top_k_{k}"])

    # Save summary
    from tabulate import tabulate
    summary_path = results_dir / "ablation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    if summary:
        print("\n" + "=" * 70)
        print("ABLATION STUDY RESULTS")
        print("=" * 70)
        rows = [
            [name, m["intent_accuracy"], m["intent_macro_f1"],
             m["escalation_f1"], m["false_auto_handle_rate"]]
            for name, m in summary.items()
        ]
        print(tabulate(
            rows,
            headers=["Ablation", "Intent Acc", "Macro F1", "Escalation F1", "False Auto-Handle"],
            tablefmt="simple", floatfmt=".4f",
        ))
        print("=" * 70 + "\n")

    logger.info("Ablation results saved -> %s", summary_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ablation experiments.")
    parser.add_argument("--ablation", default="all",
                        choices=["no_retrieval", "topk", "all"],
                        help="Which ablation to run (default: all)")
    main(parser.parse_args())
