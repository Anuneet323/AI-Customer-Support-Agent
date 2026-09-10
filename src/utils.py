"""
src/utils.py
─────────────
Shared utilities: config loading, logging setup, Gemini client init.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

_CONFIG_CACHE: dict | None = None


def load_config(path: str | Path = "configs/config.yaml") -> dict:
    """Load and cache the master YAML config."""
    global _CONFIG_CACHE
    if _CONFIG_CACHE is None:
        with open(path) as f:
            _CONFIG_CACHE = yaml.safe_load(f)
    return _CONFIG_CACHE


def setup_logging(level: str = "INFO") -> None:
    """Configure root logger with coloured output if colorlog is available."""
    try:
        import colorlog
        handler = colorlog.StreamHandler()
        handler.setFormatter(
            colorlog.ColoredFormatter(
                "%(log_color)s%(asctime)s | %(levelname)-8s%(reset)s | %(name)s | %(message)s"
            )
        )
        logging.root.handlers = [handler]
    except ImportError:
        logging.basicConfig(
            format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
        )
    logging.root.setLevel(getattr(logging, level.upper(), logging.INFO))


def load_env() -> None:
    """Load .env file if present (python-dotenv)."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass


def get_gemini_api_key() -> str:
    """Return Gemini API key from environment. Raises if missing."""
    load_env()
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key:
        raise EnvironmentError(
            "GEMINI_API_KEY is not set.\n"
            "Copy .env.example -> .env and fill in your key."
        )
    return key


def read_jsonl(path: str | Path) -> list[dict]:
    """Read a JSONL file into a list of dicts."""
    import json
    path = Path(path)
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(path: str | Path, records: list[dict]) -> None:
    """Write a list of dicts to a JSONL file."""
    import json
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, default=str) + "\n")
