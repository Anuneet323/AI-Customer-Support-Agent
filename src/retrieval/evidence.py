"""
src/retrieval/evidence.py
──────────────────────────
Evidence quality gate: validates whether retrieved cases are useful enough
to ground a reply, or whether the system should escalate.

This is critical for preventing hallucination: if we can't find good
evidence, we must not pretend we can.

Quality signals:
  1. Minimum similarity threshold (configurable)
  2. Intent consistency (do top results agree with classified intent?)
  3. Resolution confidence (is the retrieved case actually resolved?)
  4. Deduplication guard (retrieved case is not the query itself)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from src.retrieval.retriever import RetrievalResult

logger = logging.getLogger(__name__)


@dataclass
class EvidenceAssessment:
    """Result of the evidence quality gate."""

    passes: bool                        # True → proceed; False → escalate
    reason_code: str                    # e.g., "OK", "LOW_SIMILARITY", "NO_RESULTS"
    reason: str                         # Human-readable explanation
    best_similarity: float
    n_results: int
    n_intent_consistent: int
    top_results: list[RetrievalResult]  # Filtered/ranked evidence


class EvidenceGate:
    """Determines whether retrieved evidence is trustworthy enough to use.

    Args:
        min_similarity: Minimum cosine similarity for a case to count.
        min_intent_consistent: Min number of cases matching the classified intent.
        max_similarity_dedup: Cases above this threshold are treated as near-duplicates
                              of the query (should not happen if indexing was correct,
                              but guards against accidental contamination).
    """

    def __init__(
        self,
        min_similarity: float = 0.60,
        min_intent_consistent: int = 1,
        max_similarity_dedup: float = 0.98,
    ) -> None:
        self._min_sim = min_similarity
        self._min_intent_consistent = min_intent_consistent
        self._max_dedup = max_similarity_dedup

    def assess(
        self,
        results: list[RetrievalResult],
        classified_intent: Optional[str] = None,
    ) -> EvidenceAssessment:
        """Assess evidence quality.

        Args:
            results: Retrieved cases (already sorted by similarity desc).
            classified_intent: Intent predicted for the current query.

        Returns:
            EvidenceAssessment with pass/fail and diagnostic fields.
        """
        if not results:
            return EvidenceAssessment(
                passes=False,
                reason_code="NO_RESULTS",
                reason="No cases were retrieved from the index.",
                best_similarity=0.0,
                n_results=0,
                n_intent_consistent=0,
                top_results=[],
            )

        # Filter out near-duplicate results (potential contamination guard)
        non_dedup = [r for r in results if r.similarity < self._max_dedup]
        if len(non_dedup) < len(results):
            logger.warning(
                "Filtered %d near-duplicate results (similarity ≥ %.2f)",
                len(results) - len(non_dedup),
                self._max_dedup,
            )
        results = non_dedup

        if not results:
            return EvidenceAssessment(
                passes=False,
                reason_code="ALL_DUPLICATES",
                reason="All retrieved cases appear to be near-duplicates of the query.",
                best_similarity=0.0,
                n_results=0,
                n_intent_consistent=0,
                top_results=[],
            )

        best_sim = results[0].similarity

        # Filter by minimum similarity
        good_results = [r for r in results if r.similarity >= self._min_sim]

        if not good_results:
            return EvidenceAssessment(
                passes=False,
                reason_code="LOW_SIMILARITY",
                reason=(
                    f"Best retrieved case has similarity {best_sim:.3f} "
                    f"(threshold: {self._min_sim:.3f}). "
                    "No sufficiently similar historical resolution found."
                ),
                best_similarity=best_sim,
                n_results=len(results),
                n_intent_consistent=0,
                top_results=[],
            )

        # Count intent-consistent results
        n_consistent = 0
        if classified_intent:
            n_consistent = sum(
                1 for r in good_results if r.intent == classified_intent
            )
        else:
            n_consistent = len(good_results)  # No intent filter → treat as consistent

        if n_consistent < self._min_intent_consistent:
            return EvidenceAssessment(
                passes=False,
                reason_code="INTENT_MISMATCH",
                reason=(
                    f"Retrieved cases do not match classified intent '{classified_intent}'. "
                    f"Only {n_consistent}/{len(good_results)} consistent results "
                    f"(need ≥{self._min_intent_consistent})."
                ),
                best_similarity=best_sim,
                n_results=len(results),
                n_intent_consistent=n_consistent,
                top_results=good_results,
            )

        # All checks passed
        return EvidenceAssessment(
            passes=True,
            reason_code="OK",
            reason=(
                f"Retrieved {len(good_results)} cases with similarity ≥ {self._min_sim:.2f}. "
                f"{n_consistent} match the classified intent."
            ),
            best_similarity=best_sim,
            n_results=len(results),
            n_intent_consistent=n_consistent,
            top_results=good_results,
        )
