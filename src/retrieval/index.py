"""
src/retrieval/index.py
───────────────────────
Builds and persists the FAISS embedding index over historical resolved conversations.

Design choices:
- FlatL2 index (exact search) — dataset is small enough, reproducible.
- text-embedding-004 (768 dims) from Google — matches our Gemini stack.
- Index stores metadata separately (JSON sidecar) — FAISS only stores vectors.
- Golden set conversation IDs are excluded to prevent retrieval contamination.

Embedding what:
  "customer_problem" = first customer message + brief context
  This matches what happens at inference time (we embed the new customer message).
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CaseRecord:
    """A single retrievable case in the index."""

    case_id: str                    # conversation_id
    customer_problem: str           # clean first customer message
    resolution: str                 # clean last brand reply
    brand_reply: str                # raw last brand reply
    intent: Optional[str]           # assigned intent (if labelled)
    appears_resolved: bool
    resolution_confidence: str      # low / medium / high
    conversation_length: int
    metadata: dict = field(default_factory=dict)


class EmbeddingModel:
    """Wrapper for Google text-embedding-004 with local offline fallback."""

    BATCH_SIZE = 100       # Google API batch limit
    EMBED_DIM = 768

    def __init__(self, model: str = "models/text-embedding-004", force_offline: bool = False) -> None:
        import os
        from src.utils import load_env
        load_env()
        api_key = os.environ.get("GEMINI_API_KEY", "").strip()
        self._offline = force_offline or not api_key or api_key.startswith("your_")
        self._model = model

        if self._offline:
            from sklearn.feature_extraction.text import HashingVectorizer
            logger.info("EmbeddingModel: Running in OFFLINE mode (HashingVectorizer 768-dim)")
            self._vectorizer = HashingVectorizer(n_features=self.EMBED_DIM, norm="l2", alternate_sign=False)
            self._genai = None
        else:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            self._genai = genai
            self._vectorizer = None
            logger.info("EmbeddingModel: %s (dim=%d, online)", model, self.EMBED_DIM)

    def embed_texts(self, texts: list[str], task_type: str = "RETRIEVAL_DOCUMENT") -> np.ndarray:
        """Embed a list of texts.

        Args:
            texts: List of strings to embed.
            task_type: "RETRIEVAL_DOCUMENT" for index, "RETRIEVAL_QUERY" for queries.

        Returns:
            numpy array of shape (len(texts), EMBED_DIM)
        """
        if not texts:
            return np.zeros((0, self.EMBED_DIM), dtype=np.float32)

        if self._offline:
            arr = self._vectorizer.transform(texts).toarray().astype(np.float32)
            logger.debug("Embedded %d texts offline -> shape %s", len(texts), arr.shape)
            return arr

        all_embeddings = []
        for i in range(0, len(texts), self.BATCH_SIZE):
            batch = texts[i: i + self.BATCH_SIZE]
            result = None
            for attempt in range(4):
                try:
                    result = self._genai.embed_content(
                        model=self._model,
                        content=batch,
                        task_type=task_type,
                        output_dimensionality=self.EMBED_DIM,
                    )
                    break
                except Exception as e:
                    if ("ResourceExhausted" in type(e).__name__ or "429" in str(e)) and attempt < 3:
                        wait_sec = 35 * (attempt + 1)
                        logger.warning("Free tier rate limit reached on batch %d. Waiting %ds...", i, wait_sec)
                        time.sleep(wait_sec)
                    else:
                        logger.warning("Live batch embedding failed (%s), falling back to local hashing: %s", type(e).__name__, e)
                        from sklearn.feature_extraction.text import HashingVectorizer
                        return HashingVectorizer(n_features=self.EMBED_DIM, norm="l2", alternate_sign=False).transform(texts).toarray().astype(np.float32)

            raw_emb = result["embedding"]
            if raw_emb and isinstance(raw_emb[0], dict) and "values" in raw_emb[0]:
                batch_embs = [e["values"] for e in raw_emb]
            elif raw_emb and isinstance(raw_emb[0], list):
                batch_embs = raw_emb
            else:
                batch_embs = [raw_emb]
            all_embeddings.extend(batch_embs)
            if i + self.BATCH_SIZE < len(texts):
                time.sleep(1.0)  # Rate limit courtesy pause

        arr = np.array(all_embeddings, dtype=np.float32)
        logger.debug("Embedded %d texts -> shape %s", len(texts), arr.shape)
        return arr

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a single query string."""
        if self._offline:
            return self._vectorizer.transform([text]).toarray()[0].astype(np.float32)
        try:
            result = self._genai.embed_content(
                model=self._model,
                content=[text],
                task_type="RETRIEVAL_QUERY",
                output_dimensionality=self.EMBED_DIM,
            )
            raw_emb = result["embedding"]
            if raw_emb and isinstance(raw_emb[0], dict) and "values" in raw_emb[0]:
                vec = raw_emb[0]["values"]
            elif raw_emb and isinstance(raw_emb[0], list):
                vec = raw_emb[0]
            else:
                vec = raw_emb
            return np.array(vec, dtype=np.float32)
        except Exception as e:
            logger.warning("Live query embedding failed (%s), falling back to local hashing: %s", type(e).__name__, e)
            from sklearn.feature_extraction.text import HashingVectorizer
            return HashingVectorizer(n_features=self.EMBED_DIM, norm="l2", alternate_sign=False).transform([text]).toarray()[0].astype(np.float32)


class FAISSIndex:
    """FAISS flat L2 index with JSON metadata sidecar.

    Flat (brute-force) search is used because:
    1. Our index size (~3–5k records) is small enough for exact search.
    2. Reproducibility — approximate indexes can differ across builds.
    3. No hyperparameter tuning needed (unlike HNSW ef, M).
    """

    def __init__(self, dim: int = 768) -> None:
        import faiss
        self._dim = dim
        self._index = faiss.IndexFlatIP(dim)  # Inner product (cosine after normalize)
        self._cases: list[CaseRecord] = []
        self._is_trained = False

    def __len__(self) -> int:
        return len(self._cases)

    def add(self, records: list[CaseRecord], embeddings: np.ndarray) -> None:
        """Add records and their embeddings to the index.

        Args:
            records: CaseRecord objects (same order as embeddings).
            embeddings: float32 array of shape (n, dim).
        """
        import faiss
        assert len(records) == len(embeddings), "Records and embeddings must match"
        assert embeddings.shape[1] == self._dim, f"Expected dim {self._dim}"

        # Normalize for cosine similarity via inner product
        faiss.normalize_L2(embeddings)
        self._index.add(embeddings)
        self._cases.extend(records)
        self._is_trained = True
        logger.info("Index now has %d records", len(self._cases))

    def search(self, query_embedding: np.ndarray, k: int = 3) -> list[tuple[CaseRecord, float]]:
        """Search for top-k nearest neighbours.

        Args:
            query_embedding: 1D float32 array of shape (dim,).
            k: Number of results to return.

        Returns:
            List of (CaseRecord, similarity_score) sorted by similarity desc.
        """
        import faiss
        if not self._is_trained or len(self._cases) == 0:
            return []

        q = query_embedding.reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(q)
        k_actual = min(k, len(self._cases))
        scores, indices = self._index.search(q, k_actual)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:
                results.append((self._cases[idx], float(score)))
        return results

    def save(self, dir_path: str | Path) -> None:
        """Persist index and metadata to disk."""
        import faiss
        dir_path = Path(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self._index, str(dir_path / "index.faiss"))
        metadata = [
            {
                "case_id": r.case_id,
                "customer_problem": r.customer_problem,
                "resolution": r.resolution,
                "brand_reply": r.brand_reply,
                "intent": r.intent,
                "appears_resolved": r.appears_resolved,
                "resolution_confidence": r.resolution_confidence,
                "conversation_length": r.conversation_length,
                "metadata": r.metadata,
            }
            for r in self._cases
        ]
        (dir_path / "metadata.json").write_text(
            json.dumps(metadata, indent=2, default=str)
        )
        logger.info("Saved index to %s (%d records)", dir_path, len(self._cases))

    @classmethod
    def load(cls, dir_path: str | Path) -> "FAISSIndex":
        """Load a saved index from disk."""
        import faiss
        dir_path = Path(dir_path)
        obj = cls.__new__(cls)
        obj._dim = 768
        obj._index = faiss.read_index(str(dir_path / "index.faiss"))
        obj._is_trained = True

        raw = json.loads((dir_path / "metadata.json").read_text())
        obj._cases = [
            CaseRecord(
                case_id=r["case_id"],
                customer_problem=r["customer_problem"],
                resolution=r["resolution"],
                brand_reply=r["brand_reply"],
                intent=r.get("intent"),
                appears_resolved=r.get("appears_resolved", False),
                resolution_confidence=r.get("resolution_confidence", "low"),
                conversation_length=r.get("conversation_length", 0),
                metadata=r.get("metadata", {}),
            )
            for r in raw
        ]
        logger.info("Loaded index from %s (%d records)", dir_path, len(obj._cases))
        return obj

    @property
    def size(self) -> int:
        return len(self._cases)
