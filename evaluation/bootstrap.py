"""
evaluation/bootstrap.py
────────────────────────
Bootstrap confidence intervals for evaluation metrics.

Why bootstrapping:
- Our golden set has 200 examples — small enough that point estimates are uncertain.
- Bootstrap is non-parametric — no distribution assumption.
- 95% CI width shows how much to trust the headline number.

For the report: if CI is wide (e.g., F1 = 0.74 ± 0.08), we say the metric
is directionally meaningful but not precise. This is honest.
"""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np

logger = logging.getLogger(__name__)


def bootstrap_ci(
    metric_fn: Callable[[list, list], float],
    gold: list,
    predicted: list,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Compute bootstrap confidence interval for a metric.

    Args:
        metric_fn: Function taking (gold, predicted) → float.
        gold: Ground truth labels.
        predicted: Model predictions.
        n_bootstrap: Number of bootstrap samples.
        ci: Confidence interval level (e.g., 0.95).
        seed: Random seed.

    Returns:
        Tuple of (point_estimate, ci_low, ci_high).
    """
    rng = np.random.default_rng(seed)
    n = len(gold)
    gold_arr = np.array(gold)
    pred_arr = np.array(predicted)

    point = metric_fn(list(gold_arr), list(pred_arr))

    bootstrap_scores = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        sample_gold = list(gold_arr[idx])
        sample_pred = list(pred_arr[idx])
        try:
            score = metric_fn(sample_gold, sample_pred)
            bootstrap_scores.append(score)
        except Exception:
            pass  # Skip degenerate samples

    alpha = 1.0 - ci
    ci_low = float(np.percentile(bootstrap_scores, 100 * alpha / 2))
    ci_high = float(np.percentile(bootstrap_scores, 100 * (1 - alpha / 2)))

    logger.debug(
        "Bootstrap CI (n=%d, ci=%.0f%%): %.4f [%.4f, %.4f]",
        n_bootstrap, ci * 100, point, ci_low, ci_high,
    )
    return point, ci_low, ci_high


def macro_f1_fn(gold: list[str], pred: list[str]) -> float:
    from sklearn.metrics import f1_score
    return f1_score(gold, pred, average="macro", zero_division=0)


def accuracy_fn(gold: list, pred: list) -> float:
    return sum(g == p for g, p in zip(gold, pred)) / len(gold)


def escalation_f1_fn(gold: list[bool], pred: list[bool]) -> float:
    from sklearn.metrics import f1_score
    return f1_score(
        [int(g) for g in gold],
        [int(p) for p in pred],
        zero_division=0,
    )


def mean_score_fn(scores: list[float], _: list) -> float:
    return float(np.mean(scores))
