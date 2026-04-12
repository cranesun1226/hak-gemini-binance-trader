"""Shared Binance Futures utility helpers."""

from __future__ import annotations

from typing import Any, Optional

from src.infra.env_loader import load_env_var

DEFAULT_BINANCE_TESTNET = False
DEFAULT_BINANCE_RECV_WINDOW_MS = 5000


def safe_float(value: Any, default: float = 0.0) -> float:
    """Convert a value to float while preserving a caller-supplied default."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value: Any, default: int = 0) -> int:
    """Convert a value to int while preserving a caller-supplied default."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def get_binance_futures_base_url(default_testnet: bool = DEFAULT_BINANCE_TESTNET) -> str:
    """
    USDT-M Futures base URL.

    - Production: https://fapi.binance.com
    - Testnet:    https://demo-fapi.binance.com
    """
    default_raw = "true" if default_testnet else "false"
    testnet = (load_env_var("BINANCE_TESTNET", default_raw) or default_raw).strip().lower() == "true"
    return "https://demo-fapi.binance.com" if testnet else "https://fapi.binance.com"


def get_recv_window_ms(default: int = DEFAULT_BINANCE_RECV_WINDOW_MS) -> int:
    """Read the Binance receive window from the environment with validation."""
    raw = load_env_var("BINANCE_RECV_WINDOW", str(default))
    try:
        value = int(str(raw).strip())
    except (TypeError, ValueError):
        return int(default)
    if value <= 0:
        return int(default)
    return value


def to_binance_kline_interval(interval: Any) -> str:
    """
    Convert legacy intervals used by the bot (e.g. "60", "240") into Binance kline intervals.
    """
    if interval is None:
        return "1h"

    value = str(interval).strip()
    if not value:
        return "1h"

    upper = value.upper()
    if upper == "W":
        return "1w"
    if upper == "M":
        return "1M"

    # Already Binance style.
    if value.endswith(("m", "h", "d", "w", "M")) and value[:-1].isdigit():
        return value

    if value.isdigit():
        minutes = int(value)
        minute_map = {
            1: "1m",
            3: "3m",
            5: "5m",
            15: "15m",
            30: "30m",
            60: "1h",
            120: "2h",
            240: "4h",
            360: "6h",
            480: "8h",
            720: "12h",
        }
        mapped = minute_map.get(minutes)
        if mapped:
            return mapped

    # Best-effort: return as-is (Binance will reject invalid values).
    return value


def interval_to_minutes(interval: Any) -> Optional[int]:
    """
    Convert supported interval representations to minutes for staleness checks.
    Returns None for unsupported values.
    """
    if interval is None:
        return None

    value = str(interval).strip()
    if not value:
        return None

    upper = value.upper()
    if upper == "W" or value.endswith("w"):
        return 60 * 24 * 7 * safe_int(value[:-1] or 1, 1) if value.endswith("w") else 60 * 24 * 7
    if upper == "M" or value.endswith("M"):
        return 60 * 24 * 30 * safe_int(value[:-1] or 1, 1) if value.endswith("M") else 60 * 24 * 30

    if value.endswith("m") and value[:-1].isdigit():
        return int(value[:-1])
    if value.endswith("h") and value[:-1].isdigit():
        return int(value[:-1]) * 60
    if value.endswith("d") and value[:-1].isdigit():
        return int(value[:-1]) * 60 * 24

    if value.isdigit():
        return int(value)

    return None
