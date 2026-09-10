"""
src/pipeline.py
────────────────
End-to-end inference pipeline.

Input:  customer message + optional context
Output: structured JSON with intent, reply, escalation decision, evidence

This is the single entry point for inference — evaluation, demo, and API
all call this. Keeps all components loosely coupled.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

from src.intent.classifier import GeminiIntentClassifier
from src.retrieval.index import FAISSIndex, EmbeddingModel
from src.retrieval.retriever import Retriever
from src.retrieval.evidence import EvidenceGate
from src.generation.reply_generator import ReplyGenerator
from src.escalation.policy import EscalationPolicy

logger = logging.getLogger(__name__)


@dataclass
class PipelineOutput:
    """Structured output from the full pipeline."""

    request_id: str
    intent: str
    intent_confidence: float
    intent_reasoning: str
    reply: str
    escalate: bool
    escalation_reason_code: str
    escalation_reason: str
    evidence: list[dict]          # List of retrieved case dicts
    evidence_quality: str         # "OK" | reason code
    best_similarity: float
    grounding_used: bool
    hallucination_risk: str
    latency_ms: float
    model: str
    errors: list[str]

    def to_dict(self) -> dict:
        return {
            "request_id": self.request_id,
            "intent": self.intent,
            "intent_confidence": round(self.intent_confidence, 4),
            "intent_reasoning": self.intent_reasoning,
            "reply": self.reply,
            "escalate": self.escalate,
            "escalation_reason_code": self.escalation_reason_code,
            "escalation_reason": self.escalation_reason,
            "evidence": self.evidence,
            "evidence_quality": self.evidence_quality,
            "best_similarity": round(self.best_similarity, 4),
            "grounding_used": self.grounding_used,
            "hallucination_risk": self.hallucination_risk,
            "latency_ms": round(self.latency_ms, 1),
            "model": self.model,
            "errors": self.errors,
        }


class SupportAgentPipeline:
    """Full pipeline: classify → retrieve → validate → generate → escalate.

    Designed to be instantiated once and called many times (components
    are stateful after loading the index).
    """

    def __init__(
        self,
        index_dir: str = "data/faiss_index",
        taxonomy_path: str = "configs/intents.yaml",
        config_path: str = "configs/config.yaml",
        llm_model: Optional[str] = None,
        brand_name: str = "our support team",
    ) -> None:
        import yaml
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        intent_cfg = cfg["intent"]
        ret_cfg = cfg["retrieval"]
        gen_cfg = cfg["generation"]
        esc_cfg = cfg["escalation"]

        if llm_model is None:
            llm_model = gen_cfg.get("model", "models/gemini-flash-latest")

        # ── Components ─────────────────────────────────────────────────────
        self._classifier = GeminiIntentClassifier(
            model=llm_model,
            taxonomy_path=taxonomy_path,
            temperature=intent_cfg["llm_temperature"],
            max_retries=intent_cfg["max_retries"],
        )

        self._embedding_model = EmbeddingModel(
            model=ret_cfg["embedding_model"]
        )
        self._index = FAISSIndex.load(index_dir)
        self._retriever = Retriever(
            index=self._index,
            embedding_model=self._embedding_model,
            top_k=ret_cfg["top_k"],
        )
        self._evidence_gate = EvidenceGate(
            min_similarity=ret_cfg["min_similarity"],
            min_intent_consistent=ret_cfg["min_intent_consistent"],
            max_similarity_dedup=ret_cfg["max_similarity_for_dedup"],
        )
        self._generator = ReplyGenerator(
            model=gen_cfg["model"],
            temperature=gen_cfg["temperature"],
            max_output_tokens=gen_cfg["max_output_tokens"],
            max_retries=gen_cfg["max_retries"],
            brand_name=brand_name,
        )
        self._escalation = EscalationPolicy(
            confidence_threshold=intent_cfg["confidence_threshold"],
            min_similarity=ret_cfg["min_similarity"],
        )

        self._model_name = llm_model
        logger.info("SupportAgentPipeline ready.")

    def run(
        self,
        customer_message: str,
        context: str = "",
        top_k: Optional[int] = None,
    ) -> PipelineOutput:
        """Process a customer message end to end.

        Args:
            customer_message: Raw customer text.
            context: Optional prior conversation context.
            top_k: Override retrieval k.

        Returns:
            PipelineOutput with all fields populated.
        """
        start = time.time()
        request_id = str(uuid.uuid4())[:8]
        errors: list[str] = []

        logger.info("[%s] Processing: %.80s...", request_id, customer_message)

        # ── Step 1: Classify intent ────────────────────────────────────────
        classification = self._classifier.classify(customer_message, context)
        intent = classification["intent"]
        intent_confidence = classification["confidence"]
        intent_reasoning = classification.get("reasoning", "")
        classification_error = classification.get("error")
        if classification_error:
            errors.append(f"classification: {classification_error}")

        # ── Step 2: Retrieve similar cases ────────────────────────────────
        retrieval_results = self._retriever.retrieve(
            query=customer_message,
            k=top_k,
            filter_intent=intent,
        )

        # ── Step 3: Evidence quality gate ─────────────────────────────────
        evidence_assessment = self._evidence_gate.assess(
            results=retrieval_results,
            classified_intent=intent,
        )

        # ── Step 4: Generate reply ─────────────────────────────────────────
        # Generate even if evidence failed — the generator will flag it
        generation = self._generator.generate(
            customer_message=customer_message,
            intent=intent,
            evidence=evidence_assessment.top_results,
            context=context,
        )
        generation_error = generation.get("error")
        if generation_error:
            errors.append(f"generation: {generation_error}")

        # ── Step 5: Escalation decision ───────────────────────────────────
        escalation = self._escalation.decide(
            customer_message=customer_message,
            intent=intent,
            intent_confidence=intent_confidence,
            evidence_passes=evidence_assessment.passes,
            evidence_reason_code=evidence_assessment.reason_code,
            best_evidence_similarity=evidence_assessment.best_similarity,
            hallucination_risk=generation["hallucination_risk"],
            generation_error=generation_error,
            classification_error=classification_error,
        )

        # ── Assemble output ────────────────────────────────────────────────
        latency_ms = (time.time() - start) * 1000

        logger.info(
            "[%s] intent=%s conf=%.2f escalate=%s evidence=%s latency=%.0fms",
            request_id,
            intent,
            intent_confidence,
            escalation.escalate,
            evidence_assessment.reason_code,
            latency_ms,
        )

        return PipelineOutput(
            request_id=request_id,
            intent=intent,
            intent_confidence=intent_confidence,
            intent_reasoning=intent_reasoning,
            reply=generation["reply"],
            escalate=escalation.escalate,
            escalation_reason_code=escalation.reason_code,
            escalation_reason=escalation.reason,
            evidence=[r.to_dict() for r in evidence_assessment.top_results],
            evidence_quality=evidence_assessment.reason_code,
            best_similarity=evidence_assessment.best_similarity,
            grounding_used=generation["grounding_used"],
            hallucination_risk=generation["hallucination_risk"],
            latency_ms=latency_ms,
            model=self._model_name,
            errors=errors,
        )
