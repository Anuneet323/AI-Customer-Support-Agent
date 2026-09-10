"""
tests/test_data.py
───────────────────
Unit tests for data layer: loader, cleaner, conversation reconstruction.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.cleaner import clean_tweet_text, detect_exact_duplicates, drop_exact_duplicates
from src.data.conversations import ConversationBuilder, Message, Conversation


# ── Cleaner tests ──────────────────────────────────────────────────────────────

class TestCleanTweetText:
    def test_removes_urls(self):
        result = clean_tweet_text("Check https://example.com for help", remove_urls=True)
        assert "https" not in result
        assert "example" not in result

    def test_removes_mentions(self):
        result = clean_tweet_text("Hi @support please help me", remove_mentions=True)
        assert "@support" not in result
        assert "please help me" in result

    def test_keeps_hashtag_word(self):
        result = clean_tweet_text("Love #amazon", remove_hashtags=False)
        assert "amazon" in result
        assert "#" not in result

    def test_empty_string(self):
        assert clean_tweet_text("") == ""

    def test_none_returns_empty(self):
        assert clean_tweet_text(None) == ""  # type: ignore

    def test_html_entities(self):
        result = clean_tweet_text("AT&amp;T is charging me")
        assert "AT&T" in result

    def test_repeated_chars_collapsed(self):
        result = clean_tweet_text("sooooooo frustrating")
        assert "soo" in result
        # Should not have 6 o's
        assert "oooooo" not in result

    def test_max_length_truncates(self):
        long_text = "x " * 600  # 1200 chars
        result = clean_tweet_text(long_text, max_length=100)
        assert len(result) <= 100


# ── Deduplication tests ────────────────────────────────────────────────────────

class TestDeduplication:
    def test_detect_exact_duplicates(self):
        texts = ["hello world", "foo bar", "hello world", "baz"]
        dupes = detect_exact_duplicates(texts)
        assert 2 in dupes  # Third element is duplicate
        assert 0 not in dupes  # First occurrence kept

    def test_no_duplicates(self):
        texts = ["hello", "world", "foo"]
        dupes = detect_exact_duplicates(texts)
        assert len(dupes) == 0

    def test_drop_exact_duplicates_df(self):
        df = pd.DataFrame({
            "clean_text": ["a b c", "d e f", "a b c", "g h i"]
        })
        result = drop_exact_duplicates(df)
        assert len(result) == 3


# ── Conversation reconstruction tests ─────────────────────────────────────────

def make_test_df() -> pd.DataFrame:
    """Create a minimal test DataFrame with a simple conversation chain."""
    return pd.DataFrame({
        "tweet_id": ["1", "2", "3", "4"],
        "author_id": ["CustomerA", "BrandX", "CustomerA", "BrandX"],
        "inbound": [True, False, True, False],
        "created_at": pd.to_datetime([
            "2021-01-01 10:00", "2021-01-01 10:05",
            "2021-01-01 10:10", "2021-01-01 10:15",
        ], utc=True),
        "text": [
            "My order is missing",
            "We're looking into it",
            "Any update?",
            "It's been shipped now",
        ],
        "clean_text": [
            "My order is missing",
            "looking into it",
            "Any update",
            "been shipped now",
        ],
        "in_response_to_tweet_id": [None, "1", "2", "3"],
        "response_tweet_id": ["2", "3", "4", None],
        "author_id_lower": ["customera", "brandx", "customera", "brandx"],
    })


class TestConversationBuilder:
    def test_builds_conversation(self):
        df = make_test_df()
        builder = ConversationBuilder(df, brand="BrandX")
        convs = builder.build(min_length=2)
        assert len(convs) == 1
        assert convs[0].conversation_id == "1"
        assert len(convs[0].messages) == 4

    def test_customer_brand_speakers(self):
        df = make_test_df()
        builder = ConversationBuilder(df, brand="BrandX")
        convs = builder.build()
        conv = convs[0]
        speakers = [m.speaker for m in conv.messages]
        assert "customer" in speakers
        assert "brand" in speakers

    def test_appears_resolved(self):
        df = make_test_df()
        builder = ConversationBuilder(df, brand="BrandX")
        convs = builder.build()
        assert convs[0].appears_resolved is True

    def test_min_length_filter(self):
        df = make_test_df()
        builder = ConversationBuilder(df, brand="BrandX")
        convs = builder.build(min_length=10)  # Too high
        assert len(convs) == 0

    def test_require_brand_reply(self):
        # Make a customer-only thread
        df = make_test_df().copy()
        df["author_id"] = ["CustomerA", "CustomerA", "CustomerA", "CustomerA"]
        df["inbound"] = [True, True, True, True]
        df["author_id_lower"] = ["customera"] * 4
        builder = ConversationBuilder(df, brand="BrandX")
        convs = builder.build(require_brand_reply=True)
        assert len(convs) == 0

    def test_empty_df(self):
        """Empty dataframe should produce no conversations."""
        df = pd.DataFrame(columns=make_test_df().columns)
        builder = ConversationBuilder(df, brand="BrandX")
        convs = builder.build()
        assert convs == []


class TestConversationProperties:
    def test_first_customer_message(self):
        conv = Conversation(conversation_id="1", brand="brandx")
        conv.messages = [
            Message("1", "customer", "hello", "hello", None, "cust"),
            Message("2", "brand", "hi", "hi", None, "brand"),
        ]
        assert conv.first_customer_message.tweet_id == "1"

    def test_last_brand_message(self):
        conv = Conversation(conversation_id="1", brand="brandx")
        conv.messages = [
            Message("1", "customer", "hello", "hello", None, "cust"),
            Message("2", "brand", "reply1", "reply1", None, "brand"),
            Message("3", "brand", "reply2", "reply2", None, "brand"),
        ]
        assert conv.last_brand_message.tweet_id == "3"

    def test_resolution_confidence_high(self):
        conv = Conversation(conversation_id="1", brand="brandx")
        conv.messages = [
            Message("1", "customer", "q", "q", None, "c"),
            Message("2", "brand", "a1", "a1", None, "b"),
            Message("3", "brand", "a2", "a2", None, "b"),
        ]
        assert conv.appears_resolved
        assert conv.resolution_confidence == "high"
