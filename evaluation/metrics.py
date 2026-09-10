"""
evaluation/metrics.py
──────────────────────
All evaluation metrics: intent, escalation, retrieval.
No LLM calls here — pure computation from predictions and gold labels.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
)


# ── Intent metrics ─────────────────────────────────────────────────────────────

def intent_metrics(
    gold: list[str],
    predicted: list[str],
    labels: Optional[list[str]] = None,
) -> dict:
    """Compute intent classification metrics.

    Args:
        gold: Ground-truth intent labels.
        predicted: Predicted intent labels.
        labels: Ordered list of class names (for consistent confusion matrix).

    Returns:
        Dict with accuracy, macro_f1, weighted_f1, per_intent, confusion_matrix.
    """
    unique = sorted(set(gold) | set(predicted))
    labels = labels or unique

    acc = accuracy_score(gold, predicted)
    macro_f1 = f1_score(gold, predicted, average="macro", labels=labels, zero_division=0)
    weighted_f1 = f1_score(gold, predicted, average="weighted", labels=labels, zero_division=0)

    # Per-intent metrics
    prec = precision_score(gold, predicted, average=None, labels=labels, zero_division=0)
    rec = recall_score(gold, predicted, average=None, labels=labels, zero_division=0)
    f1 = f1_score(gold, predicted, average=None, labels=labels, zero_division=0)

    per_intent = {
        label: {
            "precision": round(float(p), 4),
            "recall": round(float(r), 4),
            "f1": round(float(f), 4),
            "support": int(gold.count(label)),
        }
        for label, p, r, f in zip(labels, prec, rec, f1)
    }

    cm = confusion_matrix(gold, predicted, labels=labels).tolist()

    return {
        "accuracy": round(float(acc), 4),
        "macro_f1": round(float(macro_f1), 4),
        "weighted_f1": round(float(weighted_f1), 4),
        "per_intent": per_intent,
        "confusion_matrix": cm,
        "labels": labels,
        "n_samples": len(gold),
    }


# ── Escalation metrics ─────────────────────────────────────────────────────────

def escalation_metrics(
    gold_escalate: list[bool],
    pred_escalate: list[bool],
) -> dict:
    """Compute escalation decision metrics.

    Args:
        gold_escalate: Ground-truth escalation labels (True = should escalate).
        pred_escalate: Predicted escalation decisions.

    Returns:
        Dict with precision, recall, f1, false_auto_handle_rate,
              unnecessary_escalation_rate, confusion_matrix.
    """
    gold_int = [int(b) for b in gold_escalate]
    pred_int = [int(b) for b in pred_escalate]

    prec = precision_score(gold_int, pred_int, zero_division=0)
    rec = recall_score(gold_int, pred_int, zero_division=0)
    f1 = f1_score(gold_int, pred_int, zero_division=0)

    cm = confusion_matrix(gold_int, pred_int, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)

    # False auto-handle: should have escalated but didn't (fn)
    false_auto_handle_rate = fn / max(fn + tp, 1)
    # Unnecessary escalation: didn't need to but did (fp)
    unnecessary_escalation_rate = fp / max(fp + tn, 1)

    return {
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1": round(float(f1), 4),
        "true_positives": int(tp),
        "true_negatives": int(tn),
        "false_positives": int(fp),   # Unnecessary escalations
        "false_negatives": int(fn),   # False auto-handles (most dangerous)
        "false_auto_handle_rate": round(float(false_auto_handle_rate), 4),
        "unnecessary_escalation_rate": round(float(unnecessary_escalation_rate), 4),
        "n_samples": len(gold_escalate),
        "confusion_matrix": cm.tolist(),
    }


# ── Retrieval metrics ──────────────────────────────────────────────────────────

def retrieval_metrics(
    query_intents: list[str],
    retrieved_intents_per_query: list[list[str]],
) -> dict:
    """Measure retrieval quality by intent match.

    Args:
        query_intents: True intent for each query.
        retrieved_intents_per_query: List of retrieved case intents per query.

    Returns:
        Dict with recall@1, recall@3, mrr, intent_match_rate.
    """
    n = len(query_intents)
    r_at_1 = []
    r_at_3 = []
    rr = []

    for gold, retrieved in zip(query_intents, retrieved_intents_per_query):
        if not retrieved:
            r_at_1.append(0)
            r_at_3.append(0)
            rr.append(0.0)
            continue

        r1 = int(retrieved[0] == gold) if len(retrieved) >= 1 else 0
        r3 = int(any(r == gold for r in retrieved[:3]))
        r_at_1.append(r1)
        r_at_3.append(r3)

        # MRR
        rank = next((i + 1 for i, r in enumerate(retrieved) if r == gold), None)
        rr.append(1.0 / rank if rank else 0.0)

    return {
        "recall_at_1": round(float(np.mean(r_at_1)), 4),
        "recall_at_3": round(float(np.mean(r_at_3)), 4),
        "mrr": round(float(np.mean(rr)), 4),
        "n_queries": n,
    }


# ── Reply quality (non-LLM) ────────────────────────────────────────────────────

def reply_length_stats(replies: list[str]) -> dict:
    """Basic length statistics for replies."""
    lengths = [len(r.split()) for r in replies]
    return {
        "mean_words": round(float(np.mean(lengths)), 1),
        "median_words": round(float(np.median(lengths)), 1),
        "min_words": int(np.min(lengths)),
        "max_words": int(np.max(lengths)),
        "p95_words": round(float(np.percentile(lengths, 95)), 1),
    }
