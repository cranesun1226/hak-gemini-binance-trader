"""Binance Futures market-data fetch and parsing helpers."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional, Sequence

try:
    import requests
except ModuleNotFoundError:  # pragma: no cover - exercised in lightweight test environments
    class _RequestsFallback:
        def get(self, *_args, **_kwargs):
            raise ModuleNotFoundError("requests is required to call Binance APIs")

    requests = _RequestsFallback()

from src.binance.common import (
    get_binance_futures_base_url,
    interval_to_minutes,
    safe_float,
    to_binance_kline_interval,
)
from src.binance.binance_rate_limit import binance_api_call_with_retry
from src.infra.logger import get_logger

logger = get_logger("market_data")

MAX_BINANCE_KLINE_LIMIT = 1500


def _current_time_ms() -> int:
    return int(datetime.now().timestamp() * 1000)


def _resolve_as_of_ms(as_of_ms: Optional[int]) -> int:
    try:
        current_time_ms = int(as_of_ms) if as_of_ms is not None else _current_time_ms()
    except (TypeError, ValueError):
        current_time_ms = _current_time_ms()
    return current_time_ms if current_time_ms > 0 else _current_time_ms()


def _fetch_public_binance_json(
    path: str,
    *,
    params: Optional[Dict[str, Any]] = None,
    operation_name: str,
    pre_call_delay: float = 0.0,
) -> Any:
    """Fetch a public Binance endpoint with shared retry behavior."""
    url = f"{get_binance_futures_base_url()}{path}"

    def _make_api_call():
        return requests.get(url, params=params, timeout=30)

    response = binance_api_call_with_retry(
        _make_api_call,
        max_retries=5,
        initial_delay=0.5,
        pre_call_delay=pre_call_delay,
        operation_name=operation_name,
    )
    return response.json()


def fetch_klines(
    symbol: str,
    interval: Any,
    limit: int,
    *,
    as_of_ms: Optional[int] = None,
) -> list[Any]:
    """Fetch Binance klines inclusively, keeping the latest in-progress candle when available."""
    normalized_symbol = str(symbol or "").strip().upper()
    if not normalized_symbol:
        raise ValueError("symbol is required")

    current_time_ms = _resolve_as_of_ms(as_of_ms)
    binance_interval = to_binance_kline_interval(interval)
    requested_limit = max(1, int(limit))

    logger.debug(
        "Fetching klines for %s interval=%s limit=%s endTime=%s",
        normalized_symbol,
        binance_interval,
        requested_limit,
        current_time_ms,
    )

    klines_raw: list[Any] = []
    request_count = 0
    end_time_ms = current_time_ms

    while len(klines_raw) < requested_limit:
        remaining = requested_limit - len(klines_raw)
        batch_limit = min(MAX_BINANCE_KLINE_LIMIT, remaining)
        params = {
            "symbol": normalized_symbol,
            "interval": binance_interval,
            "limit": int(batch_limit),
            "endTime": int(end_time_ms),
        }

        data = _fetch_public_binance_json(
            "/fapi/v1/klines",
            params=params,
            operation_name=f"fetch_klines({normalized_symbol},page={request_count + 1})",
            pre_call_delay=0.1,
        )
        if not isinstance(data, list):
            raise Exception(f"Binance API error for {normalized_symbol}: {data}")
        if not data:
            if request_count == 0:
                logger.warning("Received 0 klines for %s (params=%s)", normalized_symbol, params)
            break

        try:
            data = sorted(data, key=lambda kline: int(kline[0]))
        except (TypeError, ValueError):
            logger.warning("Unable to sort kline page by timestamp for %s", normalized_symbol)

        klines_raw.extend(data)
        request_count += 1

        try:
            oldest_open_time_ms = int(data[0][0])
        except (TypeError, ValueError, IndexError):
            logger.warning("Unable to resolve oldest open time for %s; stopping pagination", normalized_symbol)
            break

        next_end_time_ms = oldest_open_time_ms - 1
        if next_end_time_ms <= 0 or next_end_time_ms >= end_time_ms:
            break
        end_time_ms = next_end_time_ms

        if len(data) < batch_limit:
            break

    klines_by_open_time: dict[int, Any] = {}
    for row in klines_raw:
        try:
            open_time_ms = int(row[0])
        except (TypeError, ValueError, IndexError):
            continue
        klines_by_open_time[open_time_ms] = row

    klines = [klines_by_open_time[timestamp] for timestamp in sorted(klines_by_open_time)]
    if len(klines) > requested_limit:
        klines = klines[-requested_limit:]

    if not klines:
        logger.warning("Received 0 klines for %s", normalized_symbol)
        return klines

    last_kline = klines[-1]
    last_open_time_ms = int(last_kline[0])
    try:
        last_close_time_ms = int(last_kline[6]) if len(last_kline) > 6 else last_open_time_ms
    except (TypeError, ValueError):
        last_close_time_ms = last_open_time_ms

    time_diff_seconds = max(0.0, (current_time_ms - last_close_time_ms) / 1000.0)
    interval_minutes = interval_to_minutes(binance_interval)
    if interval_minutes is None:
        return klines

    time_diff_minutes = time_diff_seconds / 60.0
    max_allowed_diff_minutes = interval_minutes * 2

    if time_diff_minutes >= max_allowed_diff_minutes:
        logger.error(
            "Last candle close timestamp is too old for %s: %.1f minutes difference (max allowed: %s minutes)",
            normalized_symbol,
            time_diff_minutes,
            max_allowed_diff_minutes,
        )
        raise Exception(
            f"Last candle close timestamp is too old for {normalized_symbol}: {time_diff_minutes:.1f} minutes difference"
        )

    if time_diff_minutes >= interval_minutes:
        logger.warning(
            "Last candle close timestamp is relatively old for %s: %.1f minutes difference (interval: %s minutes)",
            normalized_symbol,
            time_diff_minutes,
            interval_minutes,
        )

    return klines


def parse_klines(klines: Sequence[Any]) -> list[Dict[str, float]]:
    """Convert raw Binance kline rows into normalized candle dictionaries."""
    candles: list[Dict[str, float]] = []
    for kline in klines:
        try:
            volume_value = float(kline[7]) if len(kline) > 7 else float(kline[5])
        except (TypeError, ValueError, IndexError):
            volume_value = 0.0

        candles.append(
            {
                "timestamp": int(kline[0]),
                "open": float(kline[1]),
                "high": float(kline[2]),
                "low": float(kline[3]),
                "close": float(kline[4]),
                "volume": volume_value,
            }
        )

    logger.debug("Parsed %s candles", len(candles))
    return candles


def parse_closed_klines(raw_klines: Sequence[Any], *, now_ms: Optional[int] = None) -> list[Dict[str, float]]:
    """Compatibility shim that preserves the live inclusive candle set for callers using the legacy name."""
    del now_ms
    return parse_klines(raw_klines)
