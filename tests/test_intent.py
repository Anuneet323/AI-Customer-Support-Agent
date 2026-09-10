"""
tests/test_intent.py
─────────────────────
Tests for intent classification components.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.intent.baseline import MajorityClassifier, TFIDFLogisticRegression


class TestMajorityClassifier:
    def setup_method(self):
        self.clf = MajorityClassifier()
        self.texts = ["msg1", "msg2", "msg3", "msg4", "msg5"]
        self.labels = ["A", "A", "A", "B", "B"]

    def test_predict_majority(self):
        self.clf.fit(self.texts, self.labels)
        preds = self.clf.predict(["anything"])
        assert preds == ["A"]

    def test_majority_class_is_correct(self):
        self.clf.fit(self.texts, self.labels)
        assert self.clf.majority_class == "A"

    def test_predict_proba_sums_to_one(self):
        self.clf.fit(self.texts, self.labels)
        proba = self.clf.predict_proba(["x"])[0]
        assert abs(sum(proba.values()) - 1.0) < 1e-6

    def test_predict_single(self):
        self.clf.fit(self.texts, self.labels)
        intent, confidence = self.clf.predict_single("test")
        assert intent == "A"
        assert 0 <= confidence <= 1

    def test_not_fitted_raises(self):
        clf = MajorityClassifier()
        with pytest.raises(RuntimeError):
            clf.predict(["test"])

    def test_all_same_label(self):
        self.clf.fit(["a", "b", "c"], ["X", "X", "X"])
        assert self.clf.majority_class == "X"


class TestTFIDFLogisticRegression:
    def setup_method(self):
        self.clf = TFIDFLogisticRegression()
        self.texts = [
            "my order is missing please help",
            "package not delivered to my address",
            "I was charged twice for this order",
            "billing error please refund me",
            "my account is locked cannot login",
            "password reset not working for account",
            "item arrived broken damaged",
            "product quality is terrible broken",
        ]
        self.labels = [
            "delivery_issue", "delivery_issue",
            "billing_issue", "billing_issue",
            "account_issue", "account_issue",
            "product_issue", "product_issue",
        ]

    def test_fits_and_predicts(self):
        self.clf.fit(self.texts, self.labels)
        preds = self.clf.predict(self.texts)
        assert len(preds) == len(self.texts)
        assert all(p in self.labels for p in preds)

    def test_predict_proba_has_all_classes(self):
        self.clf.fit(self.texts, self.labels)
        probas = self.clf.predict_proba(["delivery problem"])
        assert len(probas) == 1
        assert set(probas[0].keys()) == {"delivery_issue", "billing_issue", "account_issue", "product_issue"}

    def test_predict_single_returns_tuple(self):
        self.clf.fit(self.texts, self.labels)
        intent, confidence = self.clf.predict_single("account locked")
        assert isinstance(intent, str)
        assert 0 <= confidence <= 1

    def test_not_fitted_raises(self):
        clf = TFIDFLogisticRegression()
        with pytest.raises(Exception):
            clf.predict_proba(["test"])

    def test_top_features(self):
        self.clf.fit(self.texts, self.labels)
        features = self.clf.get_top_features("delivery_issue", n=5)
        assert len(features) > 0
        assert all(isinstance(f[0], str) and isinstance(f[1], float) for f in features)

    def test_invalid_intent_top_features(self):
        self.clf.fit(self.texts, self.labels)
        result = self.clf.get_top_features("nonexistent_intent")
        assert result == []
