"""
tests/test_evaluation.py
──────────────────────────
Unit tests for the evaluation harness:
  - Intent classification metrics (accuracy, macro-F1, per-class metrics)
  - Escalation metrics (precision, recall, F1, false auto-handle rate)
  - Bootstrap confidence intervals
  - Human-LLM judge agreement statistics
"""

from __future__ import annotations

import numpy as np
import pytest

from evaluation.metrics import intent_metrics, escalation_metrics
from evaluation.bootstrap import bootstrap_ci, macro_f1_fn, escalation_f1_fn
from evaluation.judge_agreement import compute_agreement


class TestIntentMetrics:
    def test_perfect_accuracy(self):
        gold = ["delivery_issue", "billing_payment_issue", "general_inquiry"]
        pred = ["delivery_issue", "billing_payment_issue", "general_inquiry"]
        m = intent_metrics(gold, pred)
        assert m["accuracy"] == 1.0
        assert m["macro_f1"] == 1.0

    def test_partial_accuracy(self):
        gold = ["delivery_issue", "delivery_issue", "billing_payment_issue", "general_inquiry"]
        pred = ["delivery_issue", "billing_payment_issue", "billing_payment_issue", "general_inquiry"]
        m = intent_metrics(gold, pred)
        assert m["accuracy"] == 0.75
        assert 0.0 < m["macro_f1"] < 1.0
        assert "delivery_issue" in m["per_intent"]


class TestEscalationMetrics:
    def test_escalation_precision_recall_f1(self):
        gold = [True, True, False, False]
        pred = [True, False, False, False]
        m = escalation_metrics(gold, pred)
        assert m["precision"] == 1.0
        assert m["recall"] == 0.5
        assert round(m["f1"], 2) == 0.67
        # One gold True predicted False -> False auto-handle
        assert m["false_negatives"] == 1
        assert m["false_auto_handle_rate"] == 0.5

    def test_zero_division_safety(self):
        gold = [False, False]
        pred = [False, False]
        m = escalation_metrics(gold, pred)
        assert m["precision"] == 0.0
        assert m["recall"] == 0.0
        assert m["f1"] == 0.0
        assert m["false_auto_handle_rate"] == 0.0


class TestBootstrapCI:
    def test_bootstrap_ci_returns_valid_bounds(self):
        gold = ["a", "b", "a", "b", "a", "b", "a", "b"]
        pred = ["a", "b", "a", "a", "a", "b", "b", "b"]
        mean_val, lo, hi = bootstrap_ci(macro_f1_fn, gold, pred, n_bootstrap=100, ci=0.95, seed=42)
        assert 0.0 <= lo <= mean_val <= hi <= 1.0


class TestJudgeAgreement:
    def test_compute_agreement_identical(self):
        scores = [
            {"correctness": 5, "groundedness": 5, "resolution": 5, "tone": 5, "non_hallucination": 5, "overall": 5},
            {"correctness": 4, "groundedness": 4, "resolution": 3, "tone": 4, "non_hallucination": 4, "overall": 4},
        ]
        res = compute_agreement(scores, scores)
        assert res["n"] == 2
        assert res["overall"]["mean_absolute_difference"] == 0.0
        assert res["overall"]["exact_agreement"] == 1.0
        assert res["overall"]["within_one_agreement"] == 1.0

    def test_compute_agreement_variation(self):
        human = [
            {"correctness": 5, "groundedness": 5, "resolution": 4, "tone": 5, "non_hallucination": 5, "overall": 5},
            {"correctness": 3, "groundedness": 3, "resolution": 2, "tone": 4, "non_hallucination": 3, "overall": 3},
        ]
        llm = [
            {"correctness": 4, "groundedness": 5, "resolution": 4, "tone": 5, "non_hallucination": 4, "overall": 4},
            {"correctness": 3, "groundedness": 2, "resolution": 3, "tone": 4, "non_hallucination": 3, "overall": 3},
        ]
        res = compute_agreement(human, llm)
        assert res["overall"]["within_one_agreement"] == 1.0
        assert res["overall"]["mean_absolute_difference"] == 0.5
