"""
src/intent/baseline.py
───────────────────────
Baseline intent classifiers evaluated on the golden evaluation set.

Baselines:
  1. MajorityClassifier  — trivial: always predict most frequent intent
  2. TFIDFLogisticRegression — simple ML: TF-IDF bag-of-words + LR

Both share a common interface so the evaluator can call them identically.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)


class BaseClassifier:
    """Common interface for all classifiers."""

    def fit(self, texts: list[str], labels: list[str]) -> "BaseClassifier":
        raise NotImplementedError

    def predict(self, texts: list[str]) -> list[str]:
        raise NotImplementedError

    def predict_proba(self, texts: list[str]) -> list[dict[str, float]]:
        """Return dict of intent → probability for each text."""
        raise NotImplementedError

    def predict_single(self, text: str) -> tuple[str, float]:
        """Return (intent, confidence) for a single text."""
        predictions = self.predict([text])
        probas = self.predict_proba([text])
        intent = predictions[0]
        confidence = probas[0].get(intent, 0.0)
        return intent, confidence


class MajorityClassifier(BaseClassifier):
    """Trivial baseline: always predicts the majority class.

    Rationale for inclusion:
    - Sets a floor — any system below this is strictly useless.
    - Reveals class imbalance: if majority baseline scores "well",
      the dataset is skewed and macro-F1 becomes the honest metric.
    """

    def __init__(self) -> None:
        self._majority_class: Optional[str] = None
        self._class_distribution: dict[str, float] = {}

    def fit(self, texts: list[str], labels: list[str]) -> "MajorityClassifier":
        from collections import Counter
        counts = Counter(labels)
        total = len(labels)
        self._majority_class = counts.most_common(1)[0][0]
        self._class_distribution = {k: v / total for k, v in counts.items()}
        logger.info(
            "MajorityClassifier: majority class = '%s' (%.1f%% of training data)",
            self._majority_class,
            self._class_distribution[self._majority_class] * 100,
        )
        return self

    def predict(self, texts: list[str]) -> list[str]:
        if self._majority_class is None:
            raise RuntimeError("Call fit() before predict()")
        return [self._majority_class] * len(texts)

    def predict_proba(self, texts: list[str]) -> list[dict[str, float]]:
        return [self._class_distribution.copy() for _ in texts]

    @property
    def majority_class(self) -> Optional[str]:
        return self._majority_class


class TFIDFLogisticRegression(BaseClassifier):
    """TF-IDF + Logistic Regression intent classifier.

    Why chosen as Simple Baseline:
    - Interpretable: feature weights directly map to words.
    - No API cost, no network, fully reproducible.
    - Well-understood failure modes (OOV words, synonym blindness).
    - Strong in-domain when training data is sufficient.
    - Establishes whether semantics beyond word frequency matter.

    Hyperparameters: chosen by brief grid search on val split.
    NOT tuned on the golden set.
    """

    def __init__(
        self,
        max_features: int = 20_000,
        ngram_range: tuple[int, int] = (1, 2),
        min_df: int = 2,
        C: float = 1.0,
        max_iter: int = 1000,
        random_state: int = 42,
    ) -> None:
        self._pipeline = Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        max_features=max_features,
                        ngram_range=ngram_range,
                        min_df=min_df,
                        sublinear_tf=True,
                        strip_accents="unicode",
                        analyzer="word",
                        token_pattern=r"\w{2,}",
                    ),
                ),
                (
                    "clf",
                    LogisticRegression(
                        C=C,
                        max_iter=max_iter,
                        random_state=random_state,
                        class_weight="balanced",
                        solver="lbfgs",
                    ),
                ),
            ]
        )
        self._classes: Optional[np.ndarray] = None

    @property
    def is_fitted(self) -> bool:
        return self._classes is not None

    def fit(self, texts: list[str], labels: list[str]) -> "TFIDFLogisticRegression":
        logger.info(
            "TF-IDF + LR: training on %d examples, %d classes",
            len(texts),
            len(set(labels)),
        )
        self._pipeline.fit(texts, labels)
        self._classes = self._pipeline.classes_
        logger.info("Training complete. Classes: %s", list(self._classes))
        return self

    def predict(self, texts: list[str]) -> list[str]:
        return list(self._pipeline.predict(texts))

    def predict_proba(self, texts: list[str]) -> list[dict[str, float]]:
        if self._classes is None:
            raise RuntimeError("Call fit() first")
        proba_matrix = self._pipeline.predict_proba(texts)
        return [
            {cls: float(prob) for cls, prob in zip(self._classes, row)}
            for row in proba_matrix
        ]

    def get_top_features(self, intent: str, n: int = 10) -> list[tuple[str, float]]:
        """Return the top n TF-IDF features for a given intent class.

        Useful for debugging and explaining what the classifier learned.
        """
        clf = self._pipeline.named_steps["clf"]
        tfidf = self._pipeline.named_steps["tfidf"]
        classes = list(clf.classes_)
        if intent not in classes:
            return []
        idx = classes.index(intent)
        coefs = clf.coef_[idx]
        feature_names = tfidf.get_feature_names_out()
        top_indices = np.argsort(coefs)[::-1][:n]
        return [(feature_names[i], float(coefs[i])) for i in top_indices]
