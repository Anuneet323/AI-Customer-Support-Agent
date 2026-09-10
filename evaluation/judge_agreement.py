"""
evaluation/judge_agreement.py
──────────────────────────────
Measures agreement between the LLM judge and human ratings.

This is MANDATORY per the assignment.

Methodology:
1. Select 50 evaluation examples with system outputs.
2. Human rates each on the same 5-dimension rubric (1–5 scale).
3. LLM judge rates the same 50 examples.
4. Compute agreement statistics.

Agreement statistics:
  - Mean absolute difference (MAD) per dimension and overall
  - Exact agreement rate (same score)
  - Within-1-point agreement rate
  - Spearman correlation (overall score)
  - Weighted Cohen's kappa (overall, treating as ordinal)

Limitations documented:
  - Single human rater (one annotator)
  - 50 examples may not capture full distribution
  - LLM judge may share biases with the generator (same model family)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)

DIMENSIONS = ["correctness", "groundedness", "resolution", "tone", "non_hallucination", "overall"]


def load_human_judgments(path: str | Path = "evaluation/human_judgments.json") -> list[dict]:
    """Load human judgment file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Human judgments not found at {path}.\n"
            "Run: python evaluation/label_human_judgments.py to create them."
        )
    return json.loads(path.read_text())


def compute_agreement(
    human_scores: list[dict],
    llm_scores: list[dict],
) -> dict:
    """Compute human-LLM agreement statistics.

    Args:
        human_scores: List of dicts with dimension scores (human).
        llm_scores: List of dicts with dimension scores (LLM).
            Both must be same length and same order.

    Returns:
        Dict with agreement statistics per dimension and overall.
    """
    assert len(human_scores) == len(llm_scores), "Must have same number of scores"
    n = len(human_scores)

    results = {"n": n, "dimensions": {}, "overall": {}}

    for dim in DIMENSIONS:
        h = np.array([s.get(dim, 3) for s in human_scores], dtype=float)
        l = np.array([s.get(dim, 3) for s in llm_scores], dtype=float)

        mad = float(np.mean(np.abs(h - l)))
        exact = float(np.mean(h == l))
        within_one = float(np.mean(np.abs(h - l) <= 1))

        # Spearman
        try:
            spearman_r, spearman_p = stats.spearmanr(h, l)
        except Exception:
            spearman_r, spearman_p = 0.0, 1.0

        results["dimensions"][dim] = {
            "mean_absolute_difference": round(mad, 3),
            "exact_agreement": round(exact, 3),
            "within_one_agreement": round(within_one, 3),
            "spearman_r": round(float(spearman_r), 3),
            "spearman_p": round(float(spearman_p), 4),
            "human_mean": round(float(h.mean()), 2),
            "llm_mean": round(float(l.mean()), 2),
        }

    # Overall summary
    h_overall = np.array([s.get("overall", 3) for s in human_scores], dtype=float)
    l_overall = np.array([s.get("overall", 3) for s in llm_scores], dtype=float)

    # Weighted Cohen's kappa for the ordinal overall score
    try:
        kappa = _weighted_kappa(
            [int(v) for v in h_overall],
            [int(v) for v in l_overall],
            weights="linear",
        )
    except Exception as e:
        logger.warning("Kappa computation failed: %s", e)
        kappa = None

    overall_mad = float(np.mean(np.abs(h_overall - l_overall)))
    overall_exact = float(np.mean(h_overall == l_overall))
    overall_w1 = float(np.mean(np.abs(h_overall - l_overall) <= 1))
    spearman_r, spearman_p = stats.spearmanr(h_overall, l_overall)

    results["overall"] = {
        "mean_absolute_difference": round(overall_mad, 3),
        "exact_agreement": round(overall_exact, 3),
        "within_one_agreement": round(overall_w1, 3),
        "spearman_r": round(float(spearman_r), 3),
        "spearman_p": round(float(spearman_p), 4),
        "weighted_kappa": round(float(kappa), 3) if kappa is not None else None,
        "human_mean": round(float(h_overall.mean()), 2),
        "llm_mean": round(float(l_overall.mean()), 2),
        "interpretation": _interpret_agreement(
            overall_w1, float(spearman_r), kappa
        ),
    }

    return results


def _weighted_kappa(y1: list[int], y2: list[int], weights: str = "linear") -> float:
    """Compute weighted Cohen's kappa."""
    from sklearn.metrics import cohen_kappa_score
    return float(cohen_kappa_score(y1, y2, weights=weights))


def _interpret_agreement(within_one: float, spearman: float, kappa: Optional[float]) -> str:
    """Provide a human-readable interpretation."""
    if within_one >= 0.85 and spearman >= 0.7:
        return "Strong agreement -- LLM judge is a reliable proxy for human judgment"
    elif within_one >= 0.70 and spearman >= 0.5:
        return "Moderate agreement -- LLM judge is directionally useful with some noise"
    elif within_one >= 0.55:
        return "Weak agreement -- LLM judge should be used with caution; supplement with human review"
    else:
        return "Poor agreement -- LLM judge is not a reliable substitute for human evaluation"


def print_agreement_report(agreement: dict) -> None:
    """Print a formatted agreement report."""
    from tabulate import tabulate
    print("\n" + "=" * 70)
    print("HUMAN vs LLM JUDGE AGREEMENT REPORT")
    print(f"N = {agreement['n']} examples")
    print("=" * 70)

    rows = []
    for dim, stats_dict in agreement["dimensions"].items():
        rows.append([
            dim,
            f"{stats_dict['human_mean']:.2f}",
            f"{stats_dict['llm_mean']:.2f}",
            f"{stats_dict['mean_absolute_difference']:.3f}",
            f"{stats_dict['exact_agreement']:.1%}",
            f"{stats_dict['within_one_agreement']:.1%}",
            f"{stats_dict['spearman_r']:.3f}",
        ])

    print(tabulate(
        rows,
        headers=["Dimension", "Human Mean", "LLM Mean", "MAD", "Exact", "Within-1", "Spearman r"],
        tablefmt="simple",
    ))

    ov = agreement["overall"]
    print(f"\nOverall: MAD={ov['mean_absolute_difference']:.3f}, "
          f"Within-1={ov['within_one_agreement']:.1%}, "
          f"Spearman-r={ov['spearman_r']:.3f}, "
          f"Kappa={ov['weighted_kappa']}")
    print(f"\nInterpretation: {ov['interpretation']}")
    print("=" * 70 + "\n")
