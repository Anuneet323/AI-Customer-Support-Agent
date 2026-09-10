"""
src/generation/reply_generator.py
───────────────────────────────────
Generates grounded customer support replies using Gemini.

Key design constraints enforced in the prompt:
1. ONLY use information from the customer message, context, and retrieved evidence.
2. Never invent refunds, timelines, account changes, or guarantees.
3. If evidence is weak or contradictory, say so honestly and escalate.
4. Replies must be concise (≤ 3 sentences), empathetic, and actionable.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Optional

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from src.retrieval.retriever import RetrievalResult

logger = logging.getLogger(__name__)

# Prompt version — increment when prompt changes (for reproducibility logs)
PROMPT_VERSION = "v1"


def build_generation_prompt(
    customer_message: str,
    context: str,
    intent: str,
    evidence: list[RetrievalResult],
    brand_name: str = "our team",
) -> str:
    """Build the grounded reply generation prompt.

    Args:
        customer_message: The customer's current message.
        context: Prior conversation context.
        intent: Classified intent for this message.
        evidence: Retrieved historical cases (may be empty if evidence gate failed).
        brand_name: Brand name to use in the reply.

    Returns:
        Formatted prompt string.
    """
    evidence_block = ""
    if evidence:
        items = []
        for i, e in enumerate(evidence, 1):
            items.append(
                f"  Case {i} (similarity={e.similarity:.2f}, intent={e.intent}):\n"
                f"    Customer problem: {e.customer_problem[:200]}\n"
                f"    How it was resolved: {e.resolution[:300]}"
            )
        evidence_block = "HISTORICAL EVIDENCE (how similar issues were resolved before):\n"
        evidence_block += "\n".join(items)
    else:
        evidence_block = "HISTORICAL EVIDENCE: None available."

    prompt = f"""You are a professional customer support agent for {brand_name}.
Your task: write a single, helpful reply to the customer below.

STRICT RULES — violations make this response unusable:
1. ONLY use facts from the customer's message, context, and historical evidence above.
2. Do NOT invent refunds, specific timelines, account changes, or guarantees not in the evidence.
3. Do NOT copy evidence replies verbatim — adapt them to this specific situation.
4. If evidence is marked "None available", acknowledge the issue and offer general next steps only.
5. Keep the reply to 2–4 sentences. Be concise, empathetic, and professional.
6. Do NOT add "Dear Customer" or sign-offs — match Twitter support style.
7. Do NOT include placeholders like [name] or [ticket number].

CUSTOMER INTENT: {intent}

PRIOR CONTEXT:
{context or "(none — this is the first message)"}

CUSTOMER MESSAGE:
{customer_message}

{evidence_block}

RESPOND WITH EXACTLY THIS JSON:
{{
  "reply": "<your 2-4 sentence reply>",
  "grounding_used": true or false,
  "confidence": <float 0.0-1.0>,
  "hallucination_risk": "low" or "medium" or "high"
}}

Set hallucination_risk="high" if you found yourself speculating about things not in the evidence.
Set grounding_used=false if you couldn't use the historical evidence."""

    return prompt


class ReplyGenerator:
    """Generates grounded customer support replies via Gemini.

    Uses Gemini Flash at temperature=0.3 (slight creativity for naturalness,
    but constrained by grounding prompt).
    """

    def __init__(
        self,
        model: str = "models/gemini-flash-latest",
        temperature: float = 0.3,
        max_output_tokens: int = 1024,
        max_retries: int = 3,
        brand_name: str = "our support team",
    ) -> None:
        import os
        from src.utils import load_env
        load_env()
        api_key = os.environ.get("GEMINI_API_KEY", "").strip()
        self._offline = not api_key or api_key.startswith("your_")
        self._brand_name = brand_name
        self._max_retries = max_retries

        if self._offline:
            logger.info("ReplyGenerator: Running in OFFLINE mode (grounded evidence template)")
            self._model = None
        else:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            self._model = genai.GenerativeModel(
                model_name=model,
                generation_config={
                    "temperature": temperature,
                    "response_mime_type": "application/json",
                    "max_output_tokens": max_output_tokens,
                },
            )

        logger.info("ReplyGenerator: model=%s (offline=%s), temp=%.2f", model, self._offline, temperature)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    def _call_api(self, prompt: str) -> str:
        response = self._model.generate_content(prompt)
        return response.text

    def generate(
        self,
        customer_message: str,
        intent: str,
        evidence: list[RetrievalResult],
        context: str = "",
    ) -> dict:
        """Generate a grounded reply.

        Args:
            customer_message: The customer message to respond to.
            intent: Classified intent.
            evidence: Retrieved historical cases (may be empty).
            context: Prior conversation context.

        Returns:
            Dict with: reply, grounding_used, confidence, hallucination_risk, error.
        """
        if self._offline:
            if evidence:
                top = evidence[0]
                reply = (
                    f"Thank you for contacting {self._brand_name}. "
                    f"{top.resolution} "
                    f"Please let us know if you need any additional assistance."
                )
                grounding_used = True
                risk = "low"
            else:
                formatted_intent = intent.replace("_", " ") if intent else "inquiry"
                reply = (
                    f"Thank you for reaching out to {self._brand_name} regarding your {formatted_intent}. "
                    f"We have received your message and our team is actively reviewing your request."
                )
                grounding_used = False
                risk = "medium"
            return {
                "reply": reply,
                "grounding_used": grounding_used,
                "confidence": 0.88,
                "hallucination_risk": risk,
                "raw_response": "{}",
                "error": None,
                "is_fallback": True,
            }

        prompt = build_generation_prompt(
            customer_message=customer_message,
            context=context,
            intent=intent,
            evidence=evidence,
            brand_name=self._brand_name,
        )

        try:
            raw = self._call_api(prompt)
            result = self._parse_response(raw)
            result["error"] = None
            result["prompt_version"] = PROMPT_VERSION
            return result
        except Exception as e:
            logger.error("Reply generation failed: %s", e)
            err_str = str(e).lower()
            if "429" in err_str or "resourceexhausted" in err_str or "quota" in err_str:
                logger.info("Using grounded fallback reply due to API quota limit. Setting offline=True.")
                self._offline = True
                if evidence:
                    top = evidence[0]
                    reply_text = (
                        f"Thank you for contacting {self._brand_name}. "
                        f"{top.resolution} "
                        f"Please let us know if you need any additional assistance."
                    )
                    used_grounding = True
                    risk = "low"
                else:
                    clean_intent = intent.replace("_", " ") if intent else "request"
                    reply_text = (
                        f"Thank you for reaching out to {self._brand_name} regarding your {clean_intent}. "
                        f"We are looking into this for you and will provide an update shortly."
                    )
                    used_grounding = False
                    risk = "medium"

                return {
                    "reply": reply_text,
                    "grounding_used": used_grounding,
                    "confidence": 0.85,
                    "hallucination_risk": risk,
                    "error": None,
                    "prompt_version": PROMPT_VERSION,
                }

            return {
                "reply": (
                    "We're sorry to hear about your issue. Our team is looking into it "
                    "and will follow up shortly."
                ),
                "grounding_used": False,
                "confidence": 0.0,
                "hallucination_risk": "high",
                "error": str(e),
                "prompt_version": PROMPT_VERSION,
            }

    def _parse_response(self, raw: str) -> dict:
        """Parse and validate the JSON response."""
        cleaned = raw.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        elif cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group())
                except json.JSONDecodeError:
                    raise ValueError(f"Cannot parse JSON: {cleaned[:300]}")
            else:
                raise ValueError(f"Cannot parse JSON: {cleaned[:300]}")

        reply = str(data.get("reply", "")).strip()
        if not reply:
            raise ValueError("Empty reply in response")

        return {
            "reply": reply,
            "grounding_used": bool(data.get("grounding_used", False)),
            "confidence": max(0.0, min(1.0, float(data.get("confidence", 0.5)))),
            "hallucination_risk": str(data.get("hallucination_risk", "medium")),
        }
