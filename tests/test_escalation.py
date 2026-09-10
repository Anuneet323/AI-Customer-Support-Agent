"""
tests/test_escalation.py
─────────────────────────
Unit tests for the rule-based EscalationPolicy.
Tests all 7 prioritized signals:
  1. API/generation error -> always escalate
  2. Unknown intent -> always escalate
  3. Sensitive keywords -> always escalate (fraud, legal, hacked, chargeback)
  4. Low intent confidence -> escalate (< 0.65 threshold)
  5. Evidence gate failed -> escalate (low similarity, no intent match)
  6. High hallucination risk -> escalate
  7. Multi-intent heuristic -> escalate
  8. Clean case -> auto-handle (escalate = False)
"""

from __future__ import annotations

import pytest
from src.escalation.policy import EscalationPolicy, ReasonCode


@pytest.fixture
def policy() -> EscalationPolicy:
    return EscalationPolicy(
        confidence_threshold=0.65,
        min_similarity=0.60,
    )


class TestEscalationPolicy:
    def test_signal_1_api_error(self, policy: EscalationPolicy):
        dec = policy.decide(
            customer_message="Where is my order?",
            intent="delivery_issue",
            intent_confidence=0.95,
            evidence_passes=True,
            best_evidence_similarity=0.85,
            classification_error="ConnectionTimeout",
        )
        assert dec.escalate
        assert dec.reason_code == ReasonCode.API_ERROR

    def test_signal_2_unknown_intent(self, policy: EscalationPolicy):
        dec = policy.decide(
            customer_message="Blah blah gibberish xyz",
            intent="unknown",
            intent_confidence=0.20,
            evidence_passes=True,
            best_evidence_similarity=0.85,
        )
        assert dec.escalate
        assert dec.reason_code == ReasonCode.UNKNOWN_INTENT

    @pytest.mark.parametrize("keyword", ["fraud", "lawsuit", "police", "hacked", "stolen", "chargeback"])
    def test_signal_3_sensitive_keywords(self, policy: EscalationPolicy, keyword: str):
        msg = f"Someone is committing {keyword} on my account, help immediately!"
        dec = policy.decide(
            customer_message=msg,
            intent="billing_payment_issue",
            intent_confidence=0.95,
            evidence_passes=True,
            best_evidence_similarity=0.88,
        )
        assert dec.escalate
        assert dec.reason_code == ReasonCode.SENSITIVE_CONTENT

    def test_signal_4_low_intent_confidence(self, policy: EscalationPolicy):
        dec = policy.decide(
            customer_message="I might need something else",
            intent="general_inquiry",
            intent_confidence=0.50,  # Below 0.65 threshold
            evidence_passes=True,
            best_evidence_similarity=0.80,
        )
        assert dec.escalate
        assert dec.reason_code == ReasonCode.LOW_CONFIDENCE

    def test_signal_5_evidence_gate_failed(self, policy: EscalationPolicy):
        dec = policy.decide(
            customer_message="Can you explain this unusual charge?",
            intent="billing_payment_issue",
            intent_confidence=0.90,
            evidence_passes=False,
            evidence_reason_code="LOW_SIMILARITY",
            best_evidence_similarity=0.45,
        )
        assert dec.escalate
        assert dec.reason_code == ReasonCode.LOW_EVIDENCE

    def test_signal_6_high_hallucination_risk(self, policy: EscalationPolicy):
        dec = policy.decide(
            customer_message="How do I return this?",
            intent="order_cancellation_return",
            intent_confidence=0.92,
            evidence_passes=True,
            best_evidence_similarity=0.85,
            hallucination_risk="high",
        )
        assert dec.escalate
        assert dec.reason_code == ReasonCode.HIGH_HALLUCINATION

    def test_signal_7_multi_intent_detected(self, policy: EscalationPolicy):
        dec = policy.decide(
            customer_message="Where is my package? Also how do I cancel my Prime membership?",
            intent="delivery_issue",
            intent_confidence=0.90,
            evidence_passes=True,
            best_evidence_similarity=0.85,
        )
        assert dec.escalate
        assert dec.reason_code == ReasonCode.MULTI_INTENT

    def test_clean_case_auto_handled(self, policy: EscalationPolicy):
        dec = policy.decide(
            customer_message="My package says delivered but I never received it. Order #12345.",
            intent="delivery_issue",
            intent_confidence=0.92,
            evidence_passes=True,
            evidence_reason_code="OK",
            best_evidence_similarity=0.85,
            hallucination_risk="low",
        )
        assert not dec.escalate
        assert dec.reason_code == ReasonCode.OK
        assert dec.signals["evidence_passes"] is True
        assert dec.signals["hallucination_risk"] == "low"
