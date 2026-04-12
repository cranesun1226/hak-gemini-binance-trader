"""Helpers for reading runtime credentials from the environment or `.env`."""

from __future__ import annotations

import os
from typing import Iterator, Optional

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
ENV_PATH = os.path.join(ROOT_DIR, ".env")


def _strip_env_value(value: str) -> str:
    return value.strip().strip('"').strip("'")


def _iter_env_file_entries(env_path: str) -> Iterator[tuple[str, str]]:
    if not os.path.exists(env_path):
        return

    try:
        with open(env_path, "r", encoding="utf-8") as file_obj:
            for raw_line in file_obj:
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue

                env_key, env_value = line.split("=", 1)
                yield env_key.strip(), _strip_env_value(env_value)
    except OSError:
        return


def load_env_var(key: str, default: Optional[str] = None) -> Optional[str]:
    """Load an environment variable, falling back to the repository `.env` file."""
    value = os.getenv(key)
    if value:
        return _strip_env_value(value)

    for env_key, env_value in _iter_env_file_entries(ENV_PATH):
        if env_key == key:
            return env_value

    return default


def get_gemini_api_key() -> str:
    """Return the configured Gemini API key."""
    api_key = load_env_var("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in .env file or environment")
    return api_key


def get_binance_credentials() -> tuple[str, str]:
    """Return the configured Binance Futures API credentials."""
    api_key = load_env_var("BINANCE_API_KEY")
    api_secret = load_env_var("BINANCE_API_SECRET")

    if not api_key:
        raise ValueError("BINANCE_API_KEY not found in .env file or environment")
    if not api_secret:
        raise ValueError("BINANCE_API_SECRET not found in .env file or environment")

    return api_key, api_secret
