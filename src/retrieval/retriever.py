"""
src/retrieval/retriever.py
───────────────────────────
Query interface over the FAISS index.

At inference time:
1. Embed the customer message (RETRIEVAL_QUERY task type).
2. Search index for top-k nearest cases.
3. Optionally filter by intent for intent-conditioned retrieval.
4. Return results for evidence validation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from src.retrieval.index import FAISSIndex, EmbeddingModel, CaseRecord

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """A single retrieved case with its similarity score."""

    case_id: str
    similarity: float           # cosine similarity 0..1
    customer_problem: str
    resolution: str
    brand_reply: str
    intent: Optional[str]
    resolution_confidence: str
    conversation_length: int

    def to_dict(self) -> dict:
        return {
            "case_id": self.case_id,
            "similarity": round(self.similarity, 4),
            "customer_problem": self.customer_problem,
            "resolution": self.resolution,
            "brand_reply": self.brand_reply,
            "intent": self.intent,
            "resolution_confidence": self.resolution_confidence,
        }


class Retriever:
    """Retrieves historically similar resolved cases for a new customer message.

    Args:
        index: Loaded FAISSIndex.
        embedding_model: EmbeddingModel for query embedding.
        top_k: Default number of results to return.
    """

    def __init__(
        self,
        index: FAISSIndex,
        embedding_model: EmbeddingModel,
        top_k: int = 3,
    ) -> None:
        self._index = index
        self._emb = embedding_model
        self._top_k = top_k
        logger.info("Retriever ready. Index size: %d, top_k: %d", index.size, top_k)

    def retrieve(
        self,
        query: str,
        k: Optional[int] = None,
        filter_intent: Optional[str] = None,
    ) -> list[RetrievalResult]:
        """Retrieve top-k cases similar to query.

        Args:
            query: Customer message text to search for.
            k: Override default top_k.
            filter_intent: If set, also retrieves extra candidates and
                           re-ranks to prefer matching intent.

        Returns:
            List of RetrievalResult sorted by similarity descending.
        """
        k_actual = k or self._top_k

        # Embed query
        q_emb = self._emb.embed_query(query)

        # Search (get extra candidates if filtering)
        search_k = k_actual * 3 if filter_intent else k_actual
        raw_results = self._index.search(q_emb, k=search_k)

        # Convert to RetrievalResult
        results = [
            RetrievalResult(
                case_id=case.case_id,
                similarity=sim,
                customer_problem=case.customer_problem,
                resolution=case.resolution,
                brand_reply=case.brand_reply,
                intent=case.intent,
                resolution_confidence=case.resolution_confidence,
                conversation_length=case.conversation_length,
            )
            for case, sim in raw_results
        ]

        # Intent-aware re-ranking: put matching-intent results first
        if filter_intent and results:
            matching = [r for r in results if r.intent == filter_intent]
            non_matching = [r for r in results if r.intent != filter_intent]
            results = (matching + non_matching)[:k_actual]
        else:
            results = results[:k_actual]

        logger.debug(
            "Retrieved %d cases for query '%.50s...' (top sim=%.3f)",
            len(results),
            query,
            results[0].similarity if results else 0.0,
        )
        return results
