"""
src/data/conversations.py
──────────────────────────
Reconstructs multi-turn Twitter support conversations from flat tweet rows.

The raw dataset is a flat list of tweets with in_response_to_tweet_id links.
We reconstruct threads by following the reply chain.

Key assumptions (documented):
1. A "conversation" starts with an inbound (customer) tweet.
2. Brand replies are outbound tweets whose in_response_to_tweet_id points
   to a tweet in the same thread.
3. We follow the chain breadth-first, limiting depth to avoid cycles.
4. Tweets with no in_response_to_tweet_id are conversation starters.
5. We cannot reliably infer "resolved" without signal — we use heuristics:
   - Final turn is a brand tweet → "potentially resolved"
   - Thread has ≥ 2 turns → "has resolution"
   - Absence of follow-up customer message → "appears closed"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class Message:
    """A single turn in a conversation."""

    tweet_id: str
    speaker: str          # "customer" | "brand"
    text: str
    clean_text: str
    timestamp: Optional[pd.Timestamp]
    author_id: str


@dataclass
class Conversation:
    """A reconstructed multi-turn support conversation."""

    conversation_id: str          # ID of the root (first) tweet
    brand: str
    messages: list[Message] = field(default_factory=list)

    @property
    def customer_messages(self) -> list[Message]:
        return [m for m in self.messages if m.speaker == "customer"]

    @property
    def brand_messages(self) -> list[Message]:
        return [m for m in self.messages if m.speaker == "brand"]

    @property
    def first_customer_message(self) -> Optional[Message]:
        msgs = self.customer_messages
        return msgs[0] if msgs else None

    @property
    def last_brand_message(self) -> Optional[Message]:
        msgs = self.brand_messages
        return msgs[-1] if msgs else None

    @property
    def length(self) -> int:
        return len(self.messages)

    @property
    def appears_resolved(self) -> bool:
        """Heuristic: last message is from brand and thread has ≥2 turns."""
        return (
            len(self.messages) >= 2
            and self.messages[-1].speaker == "brand"
        )

    @property
    def resolution_confidence(self) -> str:
        """Low / Medium / High — based on available signals."""
        if not self.appears_resolved:
            return "low"
        if len(self.brand_messages) >= 2:
            return "high"
        return "medium"

    def to_dict(self) -> dict:
        return {
            "conversation_id": self.conversation_id,
            "brand": self.brand,
            "length": self.length,
            "appears_resolved": self.appears_resolved,
            "resolution_confidence": self.resolution_confidence,
            "messages": [
                {
                    "tweet_id": m.tweet_id,
                    "speaker": m.speaker,
                    "text": m.text,
                    "clean_text": m.clean_text,
                    "timestamp": m.timestamp.isoformat() if m.timestamp else None,
                    "author_id": m.author_id,
                }
                for m in self.messages
            ],
        }


class ConversationBuilder:
    """Builds Conversation objects from a flat DataFrame of tweets.

    Usage:
        builder = ConversationBuilder(df, brand="AmazonHelp")
        conversations = builder.build()
    """

    MAX_DEPTH = 20  # Guard against infinite reply chains

    def __init__(self, df: pd.DataFrame, brand: str):
        """
        Args:
            df: DataFrame with columns: tweet_id, author_id, inbound,
                created_at, text, clean_text, in_response_to_tweet_id.
            brand: Brand author_id string (case-insensitive).
        """
        self.brand = brand.lower()
        self._df = df.copy()
        self._df["author_id_lower"] = self._df["author_id"].str.lower()

        # Build lookup by tweet_id
        self._by_id: dict[str, pd.Series] = {
            row["tweet_id"]: row
            for _, row in self._df.iterrows()
            if pd.notna(row["tweet_id"])
        }

        # Build parent → children mapping
        self._children: dict[str, list[str]] = {}
        for _, row in self._df.iterrows():
            parent = row.get("in_response_to_tweet_id")
            if pd.notna(parent) and pd.notna(row["tweet_id"]):
                self._children.setdefault(str(parent), []).append(str(row["tweet_id"]))

        logger.info(
            "ConversationBuilder: %d tweets, %d unique parents",
            len(self._df),
            len(self._children),
        )

    def _make_message(self, row: pd.Series) -> Message:
        is_brand = row["author_id_lower"] == self.brand
        return Message(
            tweet_id=str(row["tweet_id"]),
            speaker="brand" if is_brand else "customer",
            text=str(row.get("text", "")),
            clean_text=str(row.get("clean_text", "")),
            timestamp=row.get("created_at"),
            author_id=str(row["author_id"]),
        )

    def _get_thread(self, root_id: str, depth: int = 0) -> list[str]:
        """DFS to collect tweet IDs in a thread, ordered by reply chain."""
        if depth > self.MAX_DEPTH:
            return []
        result = [root_id]
        for child_id in self._children.get(root_id, []):
            result.extend(self._get_thread(child_id, depth + 1))
        return result

    def build(
        self,
        min_length: int = 2,
        require_brand_reply: bool = True,
        max_conversations: Optional[int] = None,
    ) -> list[Conversation]:
        """Reconstruct all conversations.

        Args:
            min_length: Minimum number of turns to include.
            require_brand_reply: Only include conversations where the brand
                replied at least once.
            max_conversations: Cap total conversations (deterministic subset).

        Returns:
            List of Conversation objects sorted by conversation_id.
        """
        # Find root tweets: inbound tweets with no parent, or parent not in dataset
        inbound_mask = self._df["inbound"] == True  # noqa: E712
        inbound_df = self._df[inbound_mask]

        root_ids: list[str] = []
        for _, row in inbound_df.iterrows():
            tid = str(row["tweet_id"])
            parent = row.get("in_response_to_tweet_id")
            if pd.isna(parent) or str(parent) not in self._by_id:
                root_ids.append(tid)

        logger.info("Found %d root (conversation-starting) tweets", len(root_ids))

        conversations: list[Conversation] = []
        skipped = 0

        for root_id in sorted(root_ids):  # Sorted for determinism
            if max_conversations and len(conversations) >= max_conversations:
                break

            thread_ids = self._get_thread(root_id)
            # Sort by timestamp where possible
            thread_rows = [
                self._by_id[tid]
                for tid in thread_ids
                if tid in self._by_id
            ]
            thread_rows.sort(
                key=lambda r: r["created_at"]
                if pd.notna(r.get("created_at"))
                else pd.Timestamp.min
            )

            messages = [self._make_message(r) for r in thread_rows]

            if len(messages) < min_length:
                skipped += 1
                continue

            has_brand_reply = any(m.speaker == "brand" for m in messages)
            if require_brand_reply and not has_brand_reply:
                skipped += 1
                continue

            conv = Conversation(
                conversation_id=root_id,
                brand=self.brand,
                messages=messages,
            )
            conversations.append(conv)

        logger.info(
            "Built %d conversations (%d skipped)",
            len(conversations),
            skipped,
        )
        return conversations


def conversations_to_df(conversations: list[Conversation]) -> pd.DataFrame:
    """Flatten conversations to a DataFrame with one row per conversation.

    Useful for analysis and sampling.
    """
    rows = []
    for conv in conversations:
        fcm = conv.first_customer_message
        lbm = conv.last_brand_message
        rows.append(
            {
                "conversation_id": conv.conversation_id,
                "brand": conv.brand,
                "length": conv.length,
                "appears_resolved": conv.appears_resolved,
                "resolution_confidence": conv.resolution_confidence,
                "customer_count": len(conv.customer_messages),
                "brand_count": len(conv.brand_messages),
                "first_customer_text": fcm.clean_text if fcm else "",
                "last_brand_text": lbm.clean_text if lbm else "",
                "first_timestamp": (
                    conv.messages[0].timestamp if conv.messages else None
                ),
            }
        )
    return pd.DataFrame(rows)
