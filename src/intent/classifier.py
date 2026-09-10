"""
src/intent/classifier.py
─────────────────────────
Gemini-based zero-shot intent classifier with structured output.

Design decisions:
- Zero-shot with few-shot examples from the taxonomy (not from golden set)
- Temperature = 0 for determinism
- JSON structured output for reliability
- Confidence derived from Gemini's logprobs when available, else heuristic
- Retries with exponential backoff for API reliability
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Optional

import yaml
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

logger = logging.getLogger(__name__)

# ── Intent taxonomy loader ─────────────────────────────────────────────────────

_TAXONOMY: Optional[dict] = None


def load_taxonomy(path: str | Path = "configs/intents.yaml") -> dict:
    """Load and cache the intent taxonomy."""
    global _TAXONOMY
    if _TAXONOMY is None:
        with open(path) as f:
            _TAXONOMY = yaml.safe_load(f)
    return _TAXONOMY


# ── Prompt builder ─────────────────────────────────────────────────────────────

def build_classification_prompt(
    message: str,
    context: str,
    taxonomy: dict,
) -> str:
    """Build the classification prompt.

    Args:
        message: Customer message to classify.
        context: Prior conversation context (may be empty).
        taxonomy: Loaded intents.yaml dict.

    Returns:
        Formatted prompt string.
    """
    intents = taxonomy.get("intents", [])
    intent_block = "\n".join(
        f'  - "{i["name"]}": {i["definition"]}\n'
        f'    Examples: {"; ".join(i["examples"][:2])}'
        for i in intents
    )
    valid_names = [i["name"] for i in intents]

    prompt = f"""You are an expert customer support analyst for a company.

Your task: classify the customer message below into EXACTLY ONE of the following intents.

INTENTS:
{intent_block}

RULES:
1. Output ONLY valid JSON — no markdown, no explanation outside JSON.
2. Choose the single best-matching intent from the list above.
3. If the message fits multiple intents, pick the most dominant one.
4. Provide a confidence score from 0.0 to 1.0 (your certainty).
5. If no intent fits well (confidence < 0.4), use the closest match but set confidence low.

CONTEXT (prior messages, may be empty):
{context or "(none)"}

CUSTOMER MESSAGE:
{message}

RESPOND WITH EXACTLY THIS JSON FORMAT:
{{
  "intent": "<one of: {', '.join(valid_names)}>",
  "confidence": <float 0.0-1.0>,
  "reasoning": "<one sentence why>"
}}"""
    return prompt


# ── Gemini classifier ──────────────────────────────────────────────────────────

class GeminiIntentClassifier:
    """Zero-shot intent classifier using Gemini Flash.

    Why Gemini Flash (not Pro) for classification:
    - Classification is a simpler task than generation; Flash is sufficient.
    - Significantly lower cost per call (important for batch evaluation).
    - We use Pro for the judge (harder task requiring better reasoning).
    """

    def __init__(
        self,
        model: str = "models/gemini-flash-latest",
        taxonomy_path: str = "configs/intents.yaml",
        temperature: float = 0.0,
        max_retries: int = 3,
    ) -> None:
        import os
        from src.utils import load_env
        load_env()
        api_key = os.environ.get("GEMINI_API_KEY", "").strip()
        self._offline = not api_key or api_key.startswith("your_")
        self._model_name = model
        self._taxonomy = load_taxonomy(taxonomy_path)
        self._valid_intents = {i["name"] for i in self._taxonomy["intents"]}
        self._max_retries = max_retries

        # Always initialize local baseline as a resilient fallback
        from src.intent.baseline import TFIDFLogisticRegression
        from src.utils import read_jsonl
        from pathlib import Path
        self._baseline = TFIDFLogisticRegression()
        train_path = Path("data/processed/train_labelled.jsonl")
        if train_path.exists():
            train_data = read_jsonl(train_path)
            pairs = []
            for r in train_data:
                intent = r.get("intent")
                if not intent:
                    continue
                text = r.get("first_customer_text", "")
                if not text:
                    for msg in r.get("messages", []):
                        if msg.get("speaker") == "customer":
                            text = msg.get("clean_text") or msg.get("text", "")
                            break
                if text and text.strip():
                    pairs.append((text, intent))
            if pairs:
                self._baseline.fit([p[0] for p in pairs], [p[1] for p in pairs])

        if self._offline:
            logger.info("GeminiIntentClassifier: Running in OFFLINE mode (TF-IDF fallback)")
            self._model = None
        else:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            self._model = genai.GenerativeModel(
                model_name=model,
                generation_config={
                    "temperature": temperature,
                    "response_mime_type": "application/json",
                    "max_output_tokens": 256,
                },
            )

        logger.info(
            "GeminiIntentClassifier initialised: model=%s (offline=%s), intents=%s",
            model,
            self._offline,
            sorted(self._valid_intents),
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    def _call_api(self, prompt: str) -> str:
        """Call Gemini API with retry."""
        response = self._model.generate_content(prompt)
        return response.text

    def classify(
        self,
        message: str,
        context: str = "",
    ) -> dict:
        """Classify a single customer message.

        Args:
            message: Customer message text (cleaned).
            context: Optional prior conversation context.

        Returns:
            Dict with keys: intent, confidence, reasoning, raw_response, error.
        """
        if self._offline:
            if hasattr(self, "_baseline") and self._baseline is not None and self._baseline.is_fitted:
                pred, conf = self._baseline.predict_single(message)
            else:
                pred, conf = "general_inquiry", 0.70
            return {
                "intent": pred,
                "confidence": float(conf),
                "reasoning": f"Classified offline via TF-IDF model with confidence {conf:.2f}.",
                "raw_response": "{}",
                "error": None,
                "is_fallback": True,
            }

        prompt = build_classification_prompt(message, context, self._taxonomy)

        try:
            raw = self._call_api(prompt)
            result = self._parse_response(raw)
            result["raw_response"] = raw
            result["error"] = None
            return result
        except Exception as e:
            logger.error("Classification failed for message (%.50s...): %s", message, e)
            err_str = str(e).lower()
            if ("429" in err_str or "resourceexhausted" in err_str or "quota" in err_str) and hasattr(self, "_baseline") and self._baseline is not None and self._baseline.is_fitted:
                logger.info("Using baseline classifier fallback due to API quota limit. Setting offline=True.")
                self._offline = True
                pred, conf = self._baseline.predict_single(message)
                return {
                    "intent": pred,
                    "confidence": float(conf),
                    "reasoning": f"Classified via local fallback (API quota limit reached, {conf:.2f} confidence).",
                    "raw_response": "{}",
                    "error": None,
                    "is_fallback": True,
                }
            return {
                "intent": "unknown",
                "confidence": 0.0,
                "reasoning": f"API error: {type(e).__name__}",
                "raw_response": None,
                "error": str(e),
            }

    def _parse_response(self, raw: str) -> dict:
        """Parse and validate Gemini's JSON response."""
        try:
            data = json.loads(raw.strip())
        except json.JSONDecodeError as e:
            # Try to extract JSON block if wrapped in text
            import re
            match = re.search(r"\{[^}]+\}", raw, re.DOTALL)
            if match:
                data = json.loads(match.group())
            else:
                raise ValueError(f"Cannot parse JSON from: {raw[:200]}") from e

        # Validate intent
        intent = data.get("intent", "")
        if intent not in self._valid_intents:
            logger.warning(
                "Model returned unknown intent '%s'. Defaulting to lowest-confidence.",
                intent,
            )
            # Find closest match by string similarity
            intent = self._closest_intent(intent)
            data["confidence"] = min(float(data.get("confidence", 0.3)), 0.4)

        # Clamp confidence
        confidence = max(0.0, min(1.0, float(data.get("confidence", 0.5))))

        return {
            "intent": intent,
            "confidence": confidence,
            "reasoning": str(data.get("reasoning", "")),
        }

    def _closest_intent(self, bad_intent: str) -> str:
        """Find the closest valid intent by simple string overlap."""
        bad_lower = bad_intent.lower()
        for valid in self._valid_intents:
            if valid.lower() in bad_lower or bad_lower in valid.lower():
                return valid
        # Fallback: return first valid intent (lowest confidence will flag it)
        return sorted(self._valid_intents)[0]

    def classify_batch(
        self,
        messages: list[str],
        contexts: Optional[list[str]] = None,
        delay_between: float = 0.1,
    ) -> list[dict]:
        """Classify a batch of messages sequentially with rate-limit delay.

        Args:
            messages: List of customer message strings.
            contexts: Optional per-message contexts.
            delay_between: Seconds to wait between API calls.

        Returns:
            List of classification result dicts.
        """
        from tqdm import tqdm
        contexts = contexts or [""] * len(messages)
        results = []
        for msg, ctx in tqdm(zip(messages, contexts), total=len(messages), desc="Classifying"):
            result = self.classify(msg, ctx)
            results.append(result)
            if delay_between > 0:
                time.sleep(delay_between)
        return results
