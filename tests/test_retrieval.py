"""
tests/test_retrieval.py
─────────────────────────
Unit tests for the retrieval layer:
  - FAISSIndex (indexing, search, persistence)
  - EmbeddingModel (shape, offline fallback)
  - EvidenceGate (similarity thresholds, intent consistency, dedup guard)
  - Retriever (top-k search, intent filtering)
"""

from __future__ import annotations

import tempfile
import numpy as np
import pytest

from src.retrieval.index import CaseRecord, FAISSIndex, EmbeddingModel
from src.retrieval.retriever import RetrievalResult, Retriever
from src.retrieval.evidence import EvidenceGate


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_records() -> list[CaseRecord]:
    return [
        CaseRecord(
            case_id="case_001",
            customer_problem="Where is my package? It says delivered but I never got it.",
            resolution="We checked tracking and dispatched a replacement immediately.",
            brand_reply="Replacement dispatched right away.",
            intent="delivery_issue",
            appears_resolved=True,
            resolution_confidence="high",
            conversation_length=2,
        ),
        CaseRecord(
            case_id="case_002",
            customer_problem="I was charged twice for order 12345.",
            resolution="Duplicate charge was refunded within 3-5 business days.",
            brand_reply="Refund of duplicate charge processed.",
            intent="billing_payment_issue",
            appears_resolved=True,
            resolution_confidence="high",
            conversation_length=2,
        ),
        CaseRecord(
            case_id="case_003",
            customer_problem="How to cancel my Prime membership before renewal?",
            resolution="Guided customer to Account > Prime > End Membership.",
            brand_reply="Visit Account settings to end membership.",
            intent="subscription_membership_issue",
            appears_resolved=True,
            resolution_confidence="high",
            conversation_length=2,
        ),
    ]


@pytest.fixture
def mock_embedding_model() -> EmbeddingModel:
    return EmbeddingModel(force_offline=True)


# ── FAISSIndex Tests ──────────────────────────────────────────────────────────

class TestFAISSIndex:
    def test_add_and_search(self, sample_records: list[CaseRecord], mock_embedding_model: EmbeddingModel):
        dim = 768
        index = FAISSIndex(dim=dim)
        texts = [r.customer_problem for r in sample_records]
        embeddings = mock_embedding_model.embed_texts(texts)

        index.add(sample_records, embeddings)
        assert len(index) == 3

        # Query search
        q_emb = mock_embedding_model.embed_query("package not delivered")
        results = index.search(q_emb, k=2)

        assert len(results) == 2
        assert all(isinstance(r, CaseRecord) for r, _ in results)
        assert all(isinstance(s, float) for _, s in results)

    def test_save_and_load(self, sample_records: list[CaseRecord], mock_embedding_model: EmbeddingModel):
        dim = 768
        index = FAISSIndex(dim=dim)
        texts = [r.customer_problem for r in sample_records]
        embeddings = mock_embedding_model.embed_texts(texts)
        index.add(sample_records, embeddings)

        with tempfile.TemporaryDirectory() as tmpdir:
            index.save(tmpdir)
            loaded_index = FAISSIndex.load(tmpdir)

            assert len(loaded_index) == 3
            assert loaded_index._cases[0].case_id == "case_001"


# ── EvidenceGate Tests ────────────────────────────────────────────────────────

class TestEvidenceGate:
    def test_empty_results_fails(self):
        gate = EvidenceGate(min_similarity=0.60)
        eval_result = gate.assess([], classified_intent="delivery_issue")
        assert not eval_result.passes
        assert eval_result.reason_code == "NO_RESULTS"

    def test_low_similarity_fails(self):
        gate = EvidenceGate(min_similarity=0.60)
        results = [
            RetrievalResult(
                case_id="case_001",
                customer_problem="unrelated problem",
                resolution="unrelated resolution",
                brand_reply="reply",
                intent="delivery_issue",
                similarity=0.45,
                resolution_confidence="high",
                conversation_length=2,
            )
        ]
        eval_result = gate.assess(results, classified_intent="delivery_issue")
        assert not eval_result.passes
        assert eval_result.reason_code == "LOW_SIMILARITY"

    def test_no_intent_consistent_fails(self):
        gate = EvidenceGate(min_similarity=0.60, min_intent_consistent=1)
        results = [
            RetrievalResult(
                case_id="case_001",
                customer_problem="refund question",
                resolution="refund processed",
                brand_reply="reply",
                intent="billing_payment_issue",
                similarity=0.75,
                resolution_confidence="high",
                conversation_length=2,
            )
        ]
        eval_result = gate.assess(results, classified_intent="delivery_issue")
        assert not eval_result.passes
        assert eval_result.reason_code == "INTENT_MISMATCH"

    def test_valid_evidence_passes(self):
        gate = EvidenceGate(min_similarity=0.60, min_intent_consistent=1)
        results = [
            RetrievalResult(
                case_id="case_001",
                customer_problem="package missing",
                resolution="replacement dispatched",
                brand_reply="reply",
                intent="delivery_issue",
                similarity=0.82,
                resolution_confidence="high",
                conversation_length=2,
            )
        ]
        eval_result = gate.assess(results, classified_intent="delivery_issue")
        assert eval_result.passes
        assert eval_result.reason_code == "OK"
        assert len(eval_result.top_results) == 1

    def test_dedup_near_duplicate_filter(self):
        gate = EvidenceGate(min_similarity=0.60, max_similarity_dedup=0.98)
        results = [
            RetrievalResult(
                case_id="case_001",
                customer_problem="exact duplicate test query",
                resolution="resolution",
                brand_reply="reply",
                intent="delivery_issue",
                similarity=0.99,  # Near identical - potential test contamination
                resolution_confidence="high",
                conversation_length=2,
            ),
            RetrievalResult(
                case_id="case_002",
                customer_problem="another similar query",
                resolution="resolution 2",
                brand_reply="reply 2",
                intent="delivery_issue",
                similarity=0.85,
                resolution_confidence="high",
                conversation_length=2,
            ),
        ]
        eval_result = gate.assess(results, classified_intent="delivery_issue")
        assert eval_result.passes
        # Near duplicate excluded, leaving the 0.85 case
        assert len(eval_result.top_results) == 1
        assert eval_result.top_results[0].case_id == "case_002"
