"""
evaluation/evaluate.py
───────────────────────
Orchestrates the complete evaluation pipeline.

Runs:
  1. Intent metrics (accuracy, macro F1, per-class)
  2. Escalation metrics
  3. Retrieval metrics
  4. LLM judge scores on all examples
  5. Bootstrap confidence intervals
  6. Failure analysis
  7. Human-LLM agreement (if human_judgments.json exists)

Usage:
    python evaluation/evaluate.py
    python evaluation/evaluate.py --system proposed   # default
    python evaluation/evaluate.py --system majority   # trivial baseline
    python evaluation/evaluate.py --system tfidf      # simple baseline
    python evaluation/evaluate.py --no-judge          # skip LLM judge (faster)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from tabulate import tabulate

from src.utils import load_env, setup_logging, load_config, read_jsonl, write_jsonl
from evaluation.metrics import intent_metrics, escalation_metrics, retrieval_metrics, reply_length_stats
from evaluation.bootstrap import bootstrap_ci, macro_f1_fn, accuracy_fn, escalation_f1_fn
from evaluation.failure_analysis import find_failure_modes, print_failure_report

logger = logging.getLogger(__name__)


def run_proposed_system(
    golden_set: list[dict],
    config: dict,
) -> list[dict]:
    """Run the full proposed system on all golden examples.

    Args:
        golden_set: List of golden set records.
        config: Config dict.

    Returns:
        List of result dicts with predictions and metadata.
    """
    from src.pipeline import SupportAgentPipeline

    pipeline = SupportAgentPipeline(
        index_dir=config["retrieval"]["index_path"],
        taxonomy_path=config["intent"]["taxonomy_path"],
        config_path="configs/config.yaml",
        brand_name=config["data"].get("brand", "our support team"),
    )

    results = []
    for example in golden_set:
        output = pipeline.run(
            customer_message=example["customer_message"],
            context=example.get("context", ""),
        )
        results.append({
            "id": example["id"],
            "customer_message": example["customer_message"],
            "context": example.get("context", ""),
            "gold_intent": example["gold_intent"],
            "gold_escalate": example["gold_should_escalate"],
            "gold_escalation_reason": example.get("gold_escalation_reason", ""),
            "difficulty": example.get("difficulty", "medium"),
            # Predictions
            "predicted_intent": output.intent,
            "intent_confidence": output.intent_confidence,
            "predicted_escalate": output.escalate,
            "escalation_reason_code": output.escalation_reason_code,
            "escalation_reason": output.escalation_reason,
            "reply": output.reply,
            "evidence": output.evidence,
            "evidence_quality": output.evidence_quality,
            "best_similarity": output.best_similarity,
            "grounding_used": output.grounding_used,
            "hallucination_risk": output.hallucination_risk,
            "latency_ms": output.latency_ms,
            "errors": output.errors,
        })
        # Rate limit protection
        time.sleep(0.2)

    return results


def run_majority_baseline(golden_set: list[dict], train_set: list[dict]) -> list[dict]:
    """Run majority class baseline."""
    from src.intent.baseline import MajorityClassifier

    # Fit on training intent labels (need labelled training data)
    train_labels = [r.get("intent") for r in train_set if r.get("intent")]
    if not train_labels:
        # Fall back: use golden set's own distribution (biased but explicit)
        logger.warning("No labelled training data for majority baseline — using golden distribution")
        train_labels = [r["gold_intent"] for r in golden_set]

    clf = MajorityClassifier()
    clf.fit([""] * len(train_labels), train_labels)
    majority = clf.majority_class

    results = []
    for example in golden_set:
        results.append({
            "id": example["id"],
            "customer_message": example["customer_message"],
            "gold_intent": example["gold_intent"],
            "gold_escalate": example["gold_should_escalate"],
            "difficulty": example.get("difficulty", "medium"),
            "predicted_intent": majority,
            "intent_confidence": 1.0,
            "predicted_escalate": False,  # Majority baseline never escalates
            "reply": "Thank you for reaching out. Our team will look into this.",
            "evidence": [],
            "evidence_quality": "N/A",
            "best_similarity": 0.0,
            "grounding_used": False,
            "hallucination_risk": "N/A",
            "latency_ms": 0.0,
            "errors": [],
        })
    return results


def run_tfidf_baseline(
    golden_set: list[dict],
    train_set: list[dict],
) -> list[dict]:
    """Run TF-IDF + LR baseline."""
    from src.intent.baseline import TFIDFLogisticRegression

    train_pairs = []
    for r in train_set:
        intent = r.get("intent")
        if not intent:
            continue
        text = r.get("first_customer_text", "")
        if not text:
            for msg in r.get("messages", []):
                if msg.get("speaker") == "customer":
                    text = msg.get("clean_text") or msg.get("text", "")
                    break
        if text and text.strip():
            train_pairs.append((text, intent))

    if not train_pairs:
        logger.error("No training data with intent labels for TF-IDF baseline")
        return []

    train_texts = [p[0] for p in train_pairs]
    train_labels = [p[1] for p in train_pairs]

    clf = TFIDFLogisticRegression()
    clf.fit(train_texts, train_labels)

    messages = [r["customer_message"] for r in golden_set]
    predictions = clf.predict(messages)
    probas = clf.predict_proba(messages)

    results = []
    for example, pred, prob in zip(golden_set, predictions, probas):
        confidence = prob.get(pred, 0.5)
        results.append({
            "id": example["id"],
            "customer_message": example["customer_message"],
            "gold_intent": example["gold_intent"],
            "gold_escalate": example["gold_should_escalate"],
            "difficulty": example.get("difficulty", "medium"),
            "predicted_intent": pred,
            "intent_confidence": confidence,
            "predicted_escalate": confidence < 0.65,  # Use same threshold as proposed
            "reply": "Thank you for reaching out. Our team will look into this.",
            "evidence": [],
            "evidence_quality": "N/A",
            "best_similarity": 0.0,
            "grounding_used": False,
            "hallucination_risk": "N/A",
            "latency_ms": 0.0,
            "errors": [],
        })
    return results


def score_with_judge(results: list[dict]) -> list[dict]:
    """Add LLM judge scores to results."""
    from evaluation.judge import LLMJudge

    judge = LLMJudge()
    enriched = []
    for r in results:
        score = judge.score(
            customer_message=r["customer_message"],
            intent=r["gold_intent"],
            reply=r["reply"],
            evidence=r.get("evidence", []),
            context=r.get("context", ""),
            grounding_used=r.get("grounding_used", False),
        )
        enriched.append({
            **r,
            "judge_correctness": score.correctness,
            "judge_groundedness": score.groundedness,
            "judge_resolution": score.resolution,
            "judge_tone": score.tone,
            "judge_non_hallucination": score.non_hallucination,
            "judge_overall": score.overall,
            "judge_mean_dimension": score.mean_dimension_score,
            "judge_critique": score.critique,
            "judge_worst_issue": score.worst_issue,
        })
        time.sleep(0.3)
    return enriched


def compute_all_metrics(results: list[dict], ci: bool = True) -> dict:
    """Compute all metrics from results."""
    gold_intents = [r["gold_intent"] for r in results]
    pred_intents = [r["predicted_intent"] for r in results]
    gold_escalate = [bool(r["gold_escalate"]) for r in results]
    pred_escalate = [bool(r["predicted_escalate"]) for r in results]

    all_metrics = {}

    # Intent metrics
    labels = sorted(set(gold_intents))
    all_metrics["intent"] = intent_metrics(gold_intents, pred_intents, labels=labels)

    # Escalation metrics
    all_metrics["escalation"] = escalation_metrics(gold_escalate, pred_escalate)

    # Reply quality (judge)
    judge_scores = [r.get("judge_overall") for r in results if r.get("judge_overall") is not None]
    if judge_scores:
        all_metrics["reply_quality"] = {
            "mean_overall": round(float(np.mean(judge_scores)), 3),
            "median_overall": round(float(np.median(judge_scores)), 3),
            "mean_dimensions": {
                dim: round(float(np.mean([r.get(f"judge_{dim}", 3) for r in results])), 3)
                for dim in ["correctness", "groundedness", "resolution", "tone", "non_hallucination"]
            },
            "n_scored": len(judge_scores),
        }

    # Bootstrap CIs
    if ci:
        _, ci_lo, ci_hi = bootstrap_ci(macro_f1_fn, gold_intents, pred_intents)
        all_metrics["intent"]["macro_f1_ci"] = [round(ci_lo, 4), round(ci_hi, 4)]

        _, esc_lo, esc_hi = bootstrap_ci(escalation_f1_fn, gold_escalate, pred_escalate)
        all_metrics["escalation"]["f1_ci"] = [round(esc_lo, 4), round(esc_hi, 4)]

    return all_metrics


def print_results_table(system_metrics: dict[str, dict]) -> None:
    """Print comparison table across systems."""
    from tabulate import tabulate
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS SUMMARY")
    print("=" * 80)

    rows = []
    for system, m in system_metrics.items():
        intent = m.get("intent", {})
        esc = m.get("escalation", {})
        rq = m.get("reply_quality", {})
        rows.append([
            system,
            f"{intent.get('accuracy', 0):.3f}",
            f"{intent.get('macro_f1', 0):.3f}",
            f"{esc.get('f1', 0):.3f}",
            f"{esc.get('false_auto_handle_rate', 0):.3f}",
            f"{rq.get('mean_overall', 'N/A')}",
        ])

    print(tabulate(
        rows,
        headers=["System", "Intent Acc", "Intent Macro-F1", "Escalation F1",
                 "False Auto-Handle", "Reply Quality (1-5)"],
        tablefmt="simple",
    ))
    print("=" * 80 + "\n")


def main(args: argparse.Namespace) -> None:
    load_env()
    setup_logging()
    cfg = load_config()

    results_dir = Path(cfg["evaluation"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    golden_set = read_jsonl(cfg["evaluation"]["golden_set_path"])
    logger.info("Golden set: %d examples", len(golden_set))

    system = args.system
    use_judge = not args.no_judge
    all_system_metrics = {}

    # ── Run specified system ───────────────────────────────────────────────────
    if system in ("proposed", "all"):
        logger.info("Running proposed system...")
        results = run_proposed_system(golden_set, cfg)
        if use_judge:
            results = score_with_judge(results)
        write_jsonl(results_dir / "results_proposed.jsonl", results)
        all_system_metrics["proposed"] = compute_all_metrics(results)

    if system in ("majority", "all"):
        logger.info("Running majority baseline...")
        train_set = read_jsonl("data/processed/train_labelled.jsonl") if \
            Path("data/processed/train_labelled.jsonl").exists() else []
        results = run_majority_baseline(golden_set, train_set)
        if use_judge:
            results = score_with_judge(results)
        write_jsonl(results_dir / "results_majority.jsonl", results)
        all_system_metrics["majority"] = compute_all_metrics(results, ci=False)

    if system in ("tfidf", "all"):
        logger.info("Running TF-IDF baseline...")
        train_set = read_jsonl("data/processed/train_labelled.jsonl") if \
            Path("data/processed/train_labelled.jsonl").exists() else []
        results = run_tfidf_baseline(golden_set, train_set)
        if use_judge:
            results = score_with_judge(results)
        write_jsonl(results_dir / "results_tfidf.jsonl", results)
        all_system_metrics["tfidf"] = compute_all_metrics(results, ci=False)

    # ── Print results ──────────────────────────────────────────────────────────
    if all_system_metrics:
        print_results_table(all_system_metrics)

    # ── Failure analysis ───────────────────────────────────────────────────────
    proposed_results_path = results_dir / "results_proposed.jsonl"
    if proposed_results_path.exists():
        proposed_results = read_jsonl(proposed_results_path)
        failure_modes = find_failure_modes(proposed_results, top_n=5)
        print_failure_report(failure_modes)
        (results_dir / "failure_modes.json").write_text(
            json.dumps(failure_modes, indent=2, default=str)
        )

    # ── Human-LLM agreement ────────────────────────────────────────────────────
    human_judgments_path = Path("evaluation/human_judgments.json")
    if human_judgments_path.exists():
        from evaluation.judge_agreement import (
            load_human_judgments, compute_agreement, print_agreement_report
        )
        human = load_human_judgments()
        # Load corresponding LLM judge scores
        llm_results_path = results_dir / "llm_judge_scores_agreement_subset.json"
        if llm_results_path.exists():
            llm_scores = json.loads(llm_results_path.read_text())
            agreement = compute_agreement(human, llm_scores)
            print_agreement_report(agreement)
            (results_dir / "agreement_stats.json").write_text(
                json.dumps(agreement, indent=2)
            )

    # ── Save all metrics ───────────────────────────────────────────────────────
    summary_path = results_dir / "summary_metrics.json"
    summary_path.write_text(json.dumps(all_system_metrics, indent=2))
    logger.info("All metrics saved -> %s", summary_path)

    print(f"\n[OK] Evaluation complete. Results in {results_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run complete evaluation.")
    parser.add_argument(
        "--system", default="proposed",
        choices=["proposed", "majority", "tfidf", "all"],
        help="Which system to evaluate",
    )
    parser.add_argument("--no-judge", action="store_true",
                        help="Skip LLM judge (faster, no API cost)")
    main(parser.parse_args())
