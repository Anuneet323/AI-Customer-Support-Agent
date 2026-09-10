"""
src/escalation/policy.py
─────────────────────────
Escalation decision engine.

Design philosophy:
  Escalation is NOT a prompt asking "should I escalate?"
  It is an explicit, auditable rule-based system over measurable signals,
  with a final LLM check for nuanced cases.

  This matters because:
  - Rule-based signals are explainable and debuggable.
  - We can tune thresholds on the validation split (not the golden set).
  - A reviewer can understand exactly WHY a message was escalated.

Escalation signals (from strongest to weakest):
  1. Sensitive keyword detected (always escalates)
  2. API/generation error (always escalates — safe failure)
  3. Low intent confidence
  4. Evidence gate failed (no good historical cases)
  5. High hallucination risk flagged by generator
  6. Multi-intent complexity (heuristic)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

# ── Sensitive keywords — always trigger escalation ─────────────────────────────
SENSITIVE_KEYWORDS = {
    "fraud", "unauthorized", "lawsuit", "legal", "police",
    "hack", "hacked", "stolen", "chargeback", "death", "disability",
    "emergency", "attorney", "lawyer", "sue", "court",
    "discrimination", "harassment", "abuse",
}


@dataclass
class EscalationDecision:
    """The output of the escalation policy."""

    escalate: bool
    reason_code: str        # Machine-readable code (see REASON_CODES below)
    reason: str             # Human-readable explanation
    signals: dict           # All evaluated signals and their values

    def to_dict(self) -> dict:
        return {
            "escalate": self.escalate,
            "reason_code": self.reason_code,
            "reason": self.reason,
            "signals": self.signals,
        }


# ── Reason codes ───────────────────────────────────────────────────────────────
class ReasonCode:
    OK = "OK"                               # No escalation needed
    SENSITIVE_CONTENT = "SENSITIVE_CONTENT" # Keyword detected
    LOW_CONFIDENCE = "LOW_CONFIDENCE"       # Intent confidence too low
    LOW_EVIDENCE = "LOW_EVIDENCE"           # Retrieval quality gate failed
    HIGH_HALLUCINATION = "HIGH_HALLUCINATION"  # Generator flagged risk
    MULTI_INTENT = "MULTI_INTENT"           # Message spans multiple intents
    API_ERROR = "API_ERROR"                 # System error → safe failure
    UNKNOWN_INTENT = "UNKNOWN_INTENT"       # Intent classifier returned 'unknown'


class EscalationPolicy:
    """Evaluates whether a request should be auto-handled or escalated.

    Args:
        confidence_threshold: Intent confidence below this → escalate.
        min_similarity: Evidence similarity below this → escalate
            (this should match EvidenceGate.min_similarity).
    """

    def __init__(
        self,
        confidence_threshold: float = 0.65,
        min_similarity: float = 0.60,
        check_sensitive: bool = True,
        check_multi_intent: bool = True,
    ) -> None:
        self._conf_threshold = confidence_threshold
        self._min_sim = min_similarity
        self._check_sensitive = check_sensitive
        self._check_multi_intent = check_multi_intent
        logger.info(
            "EscalationPolicy: conf_threshold=%.2f, min_similarity=%.2f",
            confidence_threshold,
            min_similarity,
        )

    def decide(
        self,
        *,
        customer_message: str,
        intent: str,
        intent_confidence: float,
        evidence_passes: bool,
        evidence_reason_code: str = "OK",
        best_evidence_similarity: float = 0.0,
        hallucination_risk: str = "low",
        generation_error: Optional[str] = None,
        classification_error: Optional[str] = None,
    ) -> EscalationDecision:
        """Make escalation decision from all available signals.

        Args:
            customer_message: Raw customer message text.
            intent: Classified intent label.
            intent_confidence: Classifier confidence [0, 1].
            evidence_passes: Whether the evidence gate passed.
            evidence_reason_code: Code from EvidenceAssessment.
            best_evidence_similarity: Best retrieved case similarity.
            hallucination_risk: From generator: "low" | "medium" | "high".
            generation_error: None if generation succeeded, else error string.
            classification_error: None if classification succeeded.

        Returns:
            EscalationDecision.
        """
        signals = {
            "intent": intent,
            "intent_confidence": intent_confidence,
            "evidence_passes": evidence_passes,
            "evidence_reason_code": evidence_reason_code,
            "best_evidence_similarity": best_evidence_similarity,
            "hallucination_risk": hallucination_risk,
            "generation_error": generation_error,
            "classification_error": classification_error,
        }

        # ── Signal 1: API errors — always escalate ─────────────────────────
        if generation_error or classification_error:
            return EscalationDecision(
                escalate=True,
                reason_code=ReasonCode.API_ERROR,
                reason=(
                    "System error prevented reliable processing. "
                    f"Error: {generation_error or classification_error}"
                ),
                signals=signals,
            )

        # ── Signal 2: Unknown intent ───────────────────────────────────────
        if intent == "unknown":
            return EscalationDecision(
                escalate=True,
                reason_code=ReasonCode.UNKNOWN_INTENT,
                reason="Intent classifier could not determine the request type.",
                signals=signals,
            )

        # ── Signal 3: Sensitive content ────────────────────────────────────
        if self._check_sensitive and self._has_sensitive_content(customer_message):
            detected = self._get_sensitive_keywords(customer_message)
            return EscalationDecision(
                escalate=True,
                reason_code=ReasonCode.SENSITIVE_CONTENT,
                reason=(
                    f"Message contains sensitive keywords ({', '.join(detected)}). "
                    "Requires human review per policy."
                ),
                signals={**signals, "sensitive_keywords": list(detected)},
            )

        # ── Signal 4: Low intent confidence ───────────────────────────────
        if intent_confidence < self._conf_threshold:
            return EscalationDecision(
                escalate=True,
                reason_code=ReasonCode.LOW_CONFIDENCE,
                reason=(
                    f"Intent confidence {intent_confidence:.2f} is below threshold "
                    f"{self._conf_threshold:.2f}. Cannot reliably classify the request."
                ),
                signals=signals,
            )

        # ── Signal 5: Evidence gate failed ────────────────────────────────
        if not evidence_passes:
            return EscalationDecision(
                escalate=True,
                reason_code=ReasonCode.LOW_EVIDENCE,
                reason=(
                    f"Evidence quality gate failed ({evidence_reason_code}). "
                    f"Best similarity: {best_evidence_similarity:.3f} "
                    f"(threshold: {self._min_sim:.3f}). "
                    "No reliable historical resolution found."
                ),
                signals=signals,
            )

        # ── Signal 6: High hallucination risk ─────────────────────────────
        if hallucination_risk == "high":
            return EscalationDecision(
                escalate=True,
                reason_code=ReasonCode.HIGH_HALLUCINATION,
                reason=(
                    "Reply generator flagged high hallucination risk. "
                    "The generated response may contain unverified claims."
                ),
                signals=signals,
            )

        # ── Signal 7: Multi-intent heuristic ──────────────────────────────
        if self._check_multi_intent and self._looks_multi_intent(customer_message):
            return EscalationDecision(
                escalate=True,
                reason_code=ReasonCode.MULTI_INTENT,
                reason=(
                    "Message appears to contain multiple distinct issues. "
                    "Auto-handling is unreliable for complex multi-issue requests."
                ),
                signals=signals,
            )

        # ── All signals passed → auto-handle ──────────────────────────────
        return EscalationDecision(
            escalate=False,
            reason_code=ReasonCode.OK,
            reason=(
                f"All signals within thresholds: "
                f"confidence={intent_confidence:.2f}, "
                f"similarity={best_evidence_similarity:.2f}, "
                f"hallucination_risk={hallucination_risk}."
            ),
            signals=signals,
        )

    def _has_sensitive_content(self, text: str) -> bool:
        text_lower = text.lower()
        return any(kw in text_lower for kw in SENSITIVE_KEYWORDS)

    def _get_sensitive_keywords(self, text: str) -> set[str]:
        text_lower = text.lower()
        return {kw for kw in SENSITIVE_KEYWORDS if kw in text_lower}

    def _looks_multi_intent(self, text: str) -> bool:
        """Heuristic: multiple questions or issue markers in one message."""
        # Count question marks
        question_marks = text.count("?")
        # Look for "and also", "additionally", "also", "another issue" etc.
        multi_markers = re.findall(
            r"\b(also|additionally|furthermore|another issue|besides|plus)\b",
            text.lower(),
        )
        return question_marks >= 2 or len(multi_markers) >= 2
