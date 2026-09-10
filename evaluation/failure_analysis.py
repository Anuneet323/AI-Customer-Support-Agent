"""
evaluation/failure_analysis.py
────────────────────────────────
Systematically finds and categorises system failures.

Failure categories are discovered FROM the evaluation outputs,
NOT pre-defined and then force-fit. The categories listed are
candidates — the actual categories printed are driven by data.

Top failure modes are identified by:
1. Severity of failure (intent wrong + escalation wrong + poor reply)
2. Frequency (how many examples share the same failure pattern)
3. Instructiveness (what they tell us about system weaknesses)
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def load_results(path: str | Path) -> list[dict]:
    """Load evaluation results JSONL file."""
    path = Path(path)
    results = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results


def classify_failure_type(result: dict) -> Optional[str]:
    """Classify a single result's failure type.

    Returns None if no failure, else a category string.
    """
    gold_intent = result.get("gold_intent", "")
    pred_intent = result.get("predicted_intent", "")
    gold_escalate = result.get("gold_escalate", False)
    pred_escalate = result.get("predicted_escalate", False)
    judge_overall = result.get("judge_overall", 5)
    confidence = result.get("intent_confidence", 1.0)
    similarity = result.get("best_similarity", 1.0)
    message = result.get("customer_message", "")

    # Determine failure presence
    intent_wrong = gold_intent != pred_intent
    escalation_wrong = bool(gold_escalate) != bool(pred_escalate)
    reply_poor = judge_overall is not None and judge_overall <= 2

    if not intent_wrong and not escalation_wrong and not reply_poor:
        return None  # No failure

    # Classify failure type
    # Order matters — most specific first

    # 1. False auto-handle: should escalate but didn't
    if gold_escalate and not pred_escalate:
        return "false_auto_handle"

    # 2. Context-dependent misclassification: short messages
    if intent_wrong and len(message.split()) <= 8:
        return "short_ambiguous_message"

    # 3. Multi-intent confusion: multiple question marks or long messages
    if intent_wrong and (message.count("?") >= 2 or len(message.split()) > 50):
        return "multi_intent_complexity"

    # 4. Retrieval mismatch: intent correct but reply poor despite passing evidence
    if not intent_wrong and reply_poor and similarity > 0.60:
        return "retrieval_mismatch"

    # 5. Low confidence misclassification
    if intent_wrong and confidence < 0.65:
        return "low_confidence_misclassification"

    # 6. Reply hallucination: reply poor despite correct intent and good evidence
    if not intent_wrong and reply_poor and result.get("grounding_used", True):
        return "reply_hallucination"

    # Default
    if intent_wrong:
        return "intent_misclassification"
    if escalation_wrong:
        return "escalation_error"
    if reply_poor:
        return "poor_reply_quality"

    return "other"


def find_failure_modes(
    results: list[dict],
    top_n: int = 5,
) -> list[dict]:
    """Find and describe top N failure modes.

    Args:
        results: List of evaluation results with gold labels and predictions.
        top_n: Number of top failure modes to return.

    Returns:
        List of failure mode dicts with name, examples, frequency.
    """
    # Classify each result
    failure_map: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        ftype = classify_failure_type(r)
        if ftype:
            failure_map[ftype].append(r)

    # Sort by frequency
    sorted_failures = sorted(failure_map.items(), key=lambda x: len(x[1]), reverse=True)

    modes = []
    for name, examples in sorted_failures[:top_n]:
        worst_example = min(
            examples,
            key=lambda r: (r.get("judge_overall", 5), -r.get("intent_confidence", 1)),
        )

        modes.append({
            "rank": len(modes) + 1,
            "name": name,
            "frequency": len(examples),
            "frequency_pct": round(len(examples) / len(results) * 100, 1),
            "worst_example": {
                "id": worst_example.get("id", "?"),
                "customer_message": worst_example.get("customer_message", ""),
                "gold_intent": worst_example.get("gold_intent", ""),
                "predicted_intent": worst_example.get("predicted_intent", ""),
                "gold_escalate": worst_example.get("gold_escalate", False),
                "predicted_escalate": worst_example.get("predicted_escalate", False),
                "reply": worst_example.get("reply", ""),
                "judge_overall": worst_example.get("judge_overall", "N/A"),
                "intent_confidence": worst_example.get("intent_confidence", "N/A"),
                "evidence_quality": worst_example.get("evidence_quality", "N/A"),
            },
            "pattern": _describe_pattern(name),
            "proposed_fix": _propose_fix(name),
        })

    return modes


def _describe_pattern(name: str) -> str:
    patterns = {
        "false_auto_handle": (
            "System auto-handled a message that required human escalation. "
            "Most dangerous failure — customer gets wrong/insufficient help."
        ),
        "short_ambiguous_message": (
            "Very short messages lack context for reliable classification. "
            "Single-word or 2–3 word messages are inherently ambiguous."
        ),
        "multi_intent_complexity": (
            "Customer raised multiple issues in one message. "
            "System classifies the dominant intent but misses secondary issues."
        ),
        "retrieval_mismatch": (
            "Retrieved cases match intent label but not the specific problem variant. "
            "Embedding similarity captures topic but not sub-problem nuance."
        ),
        "low_confidence_misclassification": (
            "Classifier was uncertain (low confidence) AND wrong. "
            "These should ideally be caught by the escalation policy."
        ),
        "reply_hallucination": (
            "Reply generated specific claims not grounded in the evidence. "
            "Particularly dangerous for financial or account-related intents."
        ),
        "intent_misclassification": (
            "System classified intent incorrectly with moderate confidence. "
            "Often caused by ambiguous phrasing or overlapping intent boundaries."
        ),
        "escalation_error": (
            "Escalation decision was wrong (unnecessary escalation or missed escalation). "
            "May indicate threshold miscalibration."
        ),
        "poor_reply_quality": (
            "Reply is technically correct but unhelpful, generic, or poorly phrased. "
            "Often when evidence is weak but above minimum threshold."
        ),
    }
    return patterns.get(name, "Unclassified failure pattern.")


def _propose_fix(name: str) -> str:
    fixes = {
        "false_auto_handle": (
            "Lower escalation threshold for low-evidence cases. "
            "Add explicit check: if resolution_confidence='low', always escalate."
        ),
        "short_ambiguous_message": (
            "Request more context from customer before classifying. "
            "OR: default to escalation when message length < 5 tokens."
        ),
        "multi_intent_complexity": (
            "Improve multi-intent detector in escalation policy. "
            "Consider splitting messages at sentence boundaries and classifying each."
        ),
        "retrieval_mismatch": (
            "Add sub-intent or issue-type metadata to retrieval records. "
            "Improve evidence gate to check problem-specific match, not just intent."
        ),
        "low_confidence_misclassification": (
            "Ensure confidence_threshold is high enough to catch these. "
            "If threshold is already 0.65+, investigate why classification was uncertain."
        ),
        "reply_hallucination": (
            "Strengthen grounding prompt: explicitly list forbidden claim types. "
            "Add post-generation hallucination check against evidence."
        ),
        "intent_misclassification": (
            "Add more few-shot examples for confused intent pairs. "
            "Consider fine-tuning or explicit disambiguation rules."
        ),
        "escalation_error": (
            "Re-examine threshold calibration on validation set. "
            "Check if specific intents are systematically mis-escalated."
        ),
        "poor_reply_quality": (
            "Raise minimum evidence similarity threshold. "
            "If evidence is marginal, escalate instead of generating weak reply."
        ),
    }
    return fixes.get(name, "Investigate root cause from examples before proposing fix.")


def print_failure_report(modes: list[dict]) -> None:
    """Print a formatted failure analysis report."""
    print("\n" + "=" * 70)
    print("TOP FAILURE MODES")
    print("=" * 70)

    for mode in modes:
        ex = mode["worst_example"]
        print(f"\n{'-' * 60}")
        print(f"#{mode['rank']} {mode['name'].upper()}")
        print(f"   Frequency: {mode['frequency']} examples ({mode['frequency_pct']}%)")
        print(f"   Pattern: {mode['pattern']}")
        print(f"\n   WORST EXAMPLE (id={ex['id']}):")
        print(f"   Message: {ex['customer_message'][:150]}")
        print(f"   Gold intent: {ex['gold_intent']} | Predicted: {ex['predicted_intent']}")
        print(f"   Gold escalate: {ex['gold_escalate']} | Predicted: {ex['predicted_escalate']}")
        print(f"   Reply: {str(ex['reply'])[:150]}...")
        print(f"   Judge overall: {ex['judge_overall']}")
        print(f"\n   Proposed fix: {mode['proposed_fix']}")

    print("\n" + "=" * 70 + "\n")
