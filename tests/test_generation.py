"""
tests/test_generation.py
─────────────────────────
Unit tests for the generation layer:
  - build_generation_prompt (formatting, constraints, evidence insertion)
  - ReplyGenerator parsing (JSON parsing, markdown code block stripping, error fallback)
  - Grounding and hallucination risk reporting
"""

from __future__ import annotations

import json
import pytest

from src.generation.reply_generator import ReplyGenerator, build_generation_prompt
from src.retrieval.evidence import RetrievalResult


@pytest.fixture
def sample_evidence() -> list[RetrievalResult]:
    return [
        RetrievalResult(
            case_id="case_101",
            similarity=0.89,
            customer_problem="Package missing after being marked delivered.",
            resolution="Carrier investigated and found parcel in secure parcel locker.",
            brand_reply="Please check your parcel locker.",
            intent="delivery_issue",
            resolution_confidence="high",
            conversation_length=4,
        )
    ]


class TestPromptBuilder:
    def test_prompt_includes_evidence(self, sample_evidence: list[RetrievalResult]):
        prompt = build_generation_prompt(
            customer_message="Where is my parcel?",
            context="",
            intent="delivery_issue",
            evidence=sample_evidence,
            brand_name="AmazonHelp",
        )
        assert "HISTORICAL EVIDENCE" in prompt
        assert "parcel locker" in prompt
        assert "delivery_issue" in prompt
        assert "Where is my parcel?" in prompt

    def test_prompt_handles_empty_evidence(self):
        prompt = build_generation_prompt(
            customer_message="How to return?",
            context="",
            intent="order_cancellation_return",
            evidence=[],
            brand_name="AmazonHelp",
        )
        assert "None available" in prompt


class TestReplyGenerator:
    def test_parse_clean_json(self):
        gen = ReplyGenerator()
        raw = '{"reply": "We are investigating your package.", "grounding_used": true, "confidence": 0.95, "hallucination_risk": "low"}'
        parsed = gen._parse_response(raw)
        assert parsed["reply"] == "We are investigating your package."
        assert parsed["grounding_used"] is True
        assert parsed["confidence"] == 0.95
        assert parsed["hallucination_risk"] == "low"

    def test_parse_markdown_wrapped_json(self):
        gen = ReplyGenerator()
        raw = '```json\n{"reply": "Please check your parcel locker.", "grounding_used": true, "confidence": 0.88, "hallucination_risk": "low"}\n```'
        parsed = gen._parse_response(raw)
        assert parsed["reply"] == "Please check your parcel locker."
        assert parsed["grounding_used"] is True

    def test_offline_fallback_generation(self, sample_evidence: list[RetrievalResult]):
        gen = ReplyGenerator()
        gen._offline = True
        res = gen.generate(
            customer_message="Where is my order?",
            intent="delivery_issue",
            evidence=sample_evidence,
        )
        assert "parcel locker" in res["reply"]
        assert res["grounding_used"] is True
        assert res["hallucination_risk"] == "low"
        assert res["error"] is None

    def test_offline_fallback_without_evidence(self):
        gen = ReplyGenerator()
        gen._offline = True
        res = gen.generate(
            customer_message="How do I cancel Prime?",
            intent="subscription_membership_issue",
            evidence=[],
        )
        assert "subscription membership issue" in res["reply"]
        assert res["grounding_used"] is False
        assert res["hallucination_risk"] == "medium"
