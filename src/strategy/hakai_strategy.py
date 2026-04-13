"""Core HAK GEMINI BINANCE TRADER BTCUSDT-only trading cycle runtime."""

from __future__ import annotations

from bisect import bisect_left, bisect_right
import json
import math
import os
import shutil
import time
from datetime import datetime, timezone
from statistics import median
from typing import Any, Callable, Dict, Optional, Sequence

from src.ai.gemini_trader import evaluate_hakai_direction
from src.binance.common import interval_to_minutes
from src.binance.market_data import fetch_klines, parse_klines
from src.binance.trade_position import (
    adjust_qty_for_symbol,
    calculate_position_metrics,
    cancel_all_orders,
    close_position,
    decimal_to_str,
    evaluate_entry_order_notional,
    get_account_equity,
    get_position_snapshot,
    get_positions,
    get_reference_price,
    place_market_entry_order,
    place_reduce_only_market_order,
    safe_decimal,
    set_leverage,
    sync_existing_position_stop_loss,
    wait_for_close_propagation,
)
from src.infra.env_loader import get_binance_credentials
from src.infra.logger import format_log_details, get_logger
from src.strategy.runtime_config import DEFAULT_GEMINI_API_VERSION, load_runtime_config

logger = get_logger("hakai_strategy")

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
CONFIG_PATH = os.path.join(ROOT_DIR, "setting.yaml")
DB_DIR = os.path.join(ROOT_DIR, "db")
MAX_DB_CYCLE_DIRS = 20

DEFAULT_SYMBOL = "BTCUSDT"
DEFAULT_TIMEFRAMES: tuple[str, ...] = ("1d", "1h")
_SUPPORTED_GEMINI_THINKING_LEVELS = {"minimal", "low", "medium", "high"}
_DEFAULT_GEMINI_THINKING_LEVEL = "high"
_DEFAULT_GEMINI_API_VERSION = DEFAULT_GEMINI_API_VERSION
_TRIGGER_PRICE_DIGITS = 2
NotificationCallback = Optional[Callable[[str, Dict[str, Any]], None]]
_DIRECTIONAL_AI_DECISIONS = {"LONG", "SHORT"}
_VALID_AI_DECISIONS = set(_DIRECTIONAL_AI_DECISIONS)
_STATE_UNSET = object()

# Configuration and normalization helpers.

def _safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _normalize_positive_int(value: Any, default: int) -> int:
    parsed = _safe_int(value, default)
    return parsed if parsed > 0 else default


def _normalize_ratio(value: Any, default: float) -> float:
    parsed = _safe_float(value, default)
    if parsed is None or not math.isfinite(parsed) or parsed <= 0.0:
        parsed = default
    if 1.0 < parsed <= 100.0:
        parsed /= 100.0
    return float(parsed)


def _normalize_optional_ratio(value: Any) -> Optional[float]:
    parsed = _safe_float(value, None)
    if parsed is None or not math.isfinite(parsed) or parsed <= 0.0:
        return None
    if 1.0 < parsed <= 100.0:
        parsed /= 100.0
    return float(parsed)


def _normalize_positive_float(value: Any, default: float) -> float:
    parsed = _safe_float(value, default)
    if parsed is None or not math.isfinite(parsed) or parsed <= 0.0:
        parsed = default
    return float(parsed)


def _normalize_trigger_percent(value: Any, default: float) -> float:
    parsed_value = value
    if isinstance(value, str):
        parsed_value = value.strip()
        if parsed_value.endswith("%"):
            parsed_value = parsed_value[:-1]
    parsed = _safe_float(parsed_value, default)
    if parsed is None or not math.isfinite(parsed) or parsed <= 0.0 or parsed >= 100.0:
        parsed = default
    return float(parsed)


def _normalize_optional_trigger_percent(value: Any) -> Optional[float]:
    parsed_value = value
    if isinstance(value, str):
        parsed_value = value.strip()
        if parsed_value.endswith("%"):
            parsed_value = parsed_value[:-1]
    parsed = _safe_float(parsed_value, None)
    if parsed is None or not math.isfinite(parsed) or parsed <= 0.0 or parsed >= 100.0:
        return None
    return float(parsed)


def _normalize_gemini_thinking_level(value: Any) -> str:
    normalized = str(value or _DEFAULT_GEMINI_THINKING_LEVEL).strip().lower() or _DEFAULT_GEMINI_THINKING_LEVEL
    if normalized in _SUPPORTED_GEMINI_THINKING_LEVELS:
        return normalized
    logger.warning(
        "Unsupported gemini_thinking_level=%s for gemini-3-flash-preview; using %s",
        value,
        _DEFAULT_GEMINI_THINKING_LEVEL,
    )
    return _DEFAULT_GEMINI_THINKING_LEVEL


def _load_strategy_config() -> Dict[str, Any]:
    raw = load_runtime_config(CONFIG_PATH)

    symbol = str(raw.get("symbol") or DEFAULT_SYMBOL).strip().upper() or DEFAULT_SYMBOL
    if symbol != DEFAULT_SYMBOL:
        logger.warning("Forcing symbol=%s to BTCUSDT-only runtime", symbol)
        symbol = DEFAULT_SYMBOL

    position_size_ratio_min = _normalize_ratio(raw.get("position_size_ratio_min", 0.01), 0.01)
    position_size_ratio_max = _normalize_ratio(raw.get("position_size_ratio_max", 0.99), 0.99)
    if position_size_ratio_max <= position_size_ratio_min:
        logger.warning(
            "Invalid position size ratio bounds min=%s max=%s; using defaults",
            raw.get("position_size_ratio_min"),
            raw.get("position_size_ratio_max"),
        )
        position_size_ratio_min = 0.01
        position_size_ratio_max = 0.99

    return {
        "symbol": symbol,
        "cycle_interval_seconds": _normalize_positive_int(raw.get("cycle_interval_seconds", 60), 60),
        "trigger_pct_usdt": _normalize_trigger_percent(raw.get("trigger_pct_usdt", 0.4), 0.4),
        "fixed_leverage": _normalize_positive_int(raw.get("fixed_leverage", 10), 10),
        "stop_loss_pct": _normalize_ratio(raw.get("stop_loss_pct", 0.04), 0.04),
        "ai_candle_count_per_timeframe": _normalize_positive_int(raw.get("ai_candle_count_per_timeframe", 24), 24),
        "position_sizing_daily_sample_days": _normalize_positive_int(
            raw.get("position_sizing_daily_sample_days", 100),
            100,
        ),
        "position_sizing_live_window_hours": _normalize_positive_int(
            raw.get("position_sizing_live_window_hours", 24),
            24,
        ),
        "position_size_ratio_min": float(position_size_ratio_min),
        "position_size_ratio_max": float(position_size_ratio_max),
        "gemini_api_version": str(raw.get("gemini_api_version") or _DEFAULT_GEMINI_API_VERSION).strip()
        or _DEFAULT_GEMINI_API_VERSION,
        "gemini_thinking_level": _normalize_gemini_thinking_level(raw.get("gemini_thinking_level")),
    }


def _current_time_utc() -> datetime:
    return datetime.now(timezone.utc)


def _current_time_ms() -> int:
    return int(_current_time_utc().timestamp() * 1000)


def _resolve_as_of_ms(as_of_ms: Optional[int]) -> int:
    resolved = _safe_int(as_of_ms, 0)
    return resolved if resolved > 0 else _current_time_ms()


def _normalize_trigger_price(value: Any) -> Optional[float]:
    parsed = _safe_float(value, None)
    if parsed is None or not math.isfinite(parsed) or parsed <= 0.0:
        return None
    return float(round(float(parsed), _TRIGGER_PRICE_DIGITS))


def _format_price(value: Any) -> Optional[float]:
    parsed = _safe_float(value, None)
    if parsed is None or not math.isfinite(parsed) or parsed <= 0.0:
        return None
    return float(parsed)


def _calculate_log_high_low_ratio(*, high: Any, low: Any, context: str) -> float:
    high_value = _format_price(high)
    low_value = _format_price(low)
    if high_value is None or low_value is None:
        raise ValueError(f"{context} contains invalid high/low values")
    if high_value < low_value:
        raise ValueError(f"{context} high is below low")
    ratio = float(high_value) / float(low_value)
    if not math.isfinite(ratio) or ratio < 1.0:
        raise ValueError(f"{context} high/low ratio is invalid")
    log_ratio = math.log(ratio)
    if not math.isfinite(log_ratio) or log_ratio < 0.0:
        raise ValueError(f"{context} log(high/low) is invalid")
    return float(log_ratio)


def _normalize_ai_decision(value: Any) -> Optional[str]:
    normalized = str(value or "").strip().upper()
    if normalized in _VALID_AI_DECISIONS:
        return normalized
    return None


def _align_state_trigger_percent(
    previous_state: Optional[Dict[str, Any]],
    trigger_pct_usdt: float,
) -> tuple[Dict[str, Any], bool]:
    normalized_state = dict(previous_state or {})
    normalized_state.pop("last_ai_trigger_round_price", None)

    previous_trigger_pct = _normalize_optional_trigger_percent(normalized_state.get("trigger_pct_usdt"))
    trigger_pct_changed = previous_trigger_pct is None or not math.isclose(
        previous_trigger_pct,
        float(trigger_pct_usdt),
        rel_tol=0.0,
        abs_tol=1e-9,
    )
    if trigger_pct_changed:
        normalized_state["next_trigger_down"] = None
        normalized_state["next_trigger_up"] = None

    normalized_state["trigger_pct_usdt"] = float(trigger_pct_usdt)
    return normalized_state, trigger_pct_changed


def _build_trigger_levels(anchor_price: float, trigger_pct_usdt: float) -> Dict[str, float]:
    normalized_anchor_price = _normalize_trigger_price(anchor_price)
    if normalized_anchor_price is None:
        raise ValueError("anchor price must be positive")

    trigger_ratio = float(trigger_pct_usdt) / 100.0
    next_trigger_down = _normalize_trigger_price(normalized_anchor_price * (1.0 - trigger_ratio))
    next_trigger_up = _normalize_trigger_price(normalized_anchor_price * (1.0 + trigger_ratio))
    if next_trigger_down is None or next_trigger_up is None or next_trigger_down >= next_trigger_up:
        raise ValueError("trigger_pct_usdt produced invalid price levels")

    return {
        "trigger_price": normalized_anchor_price,
        "next_trigger_down": next_trigger_down,
        "next_trigger_up": next_trigger_up,
    }


# Cycle artifact persistence and notification hooks.
def _create_cycle_dir(base_dir: str = DB_DIR) -> str:
    os.makedirs(base_dir, exist_ok=True)
    timestamp = _current_time_utc().strftime("%Y%m%dT%H%M%S_%fZ")
    cycle_dir = os.path.join(base_dir, timestamp)
    os.makedirs(cycle_dir, exist_ok=True)
    _prune_old_cycle_dirs(base_dir)
    return cycle_dir


def _prune_old_cycle_dirs(base_dir: str, *, max_dirs: int = MAX_DB_CYCLE_DIRS) -> None:
    if max_dirs < 1:
        return

    try:
        with os.scandir(base_dir) as entries:
            child_dirs = []
            for entry in entries:
                if not entry.is_dir(follow_symlinks=False):
                    continue
                try:
                    stat_result = entry.stat(follow_symlinks=False)
                except FileNotFoundError:
                    continue
                child_dirs.append((stat_result.st_mtime, entry.name, entry.path))
    except FileNotFoundError:
        return
    except OSError as exc:
        logger.warning("Failed to scan cycle directories in %s: %s", base_dir, exc)
        return

    overflow_count = len(child_dirs) - int(max_dirs)
    if overflow_count <= 0:
        return

    child_dirs.sort(key=lambda item: (item[0], item[1]))
    for _, _, dir_path in child_dirs[:overflow_count]:
        try:
            shutil.rmtree(dir_path)
            logger.info("Removed old cycle directory: %s", dir_path)
        except FileNotFoundError:
            continue
        except OSError as exc:
            logger.warning("Failed to remove old cycle directory %s: %s", dir_path, exc)


def _write_json(path: str, payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, indent=2, ensure_ascii=False)


def _persist_cycle_output(result: Dict[str, Any]) -> None:
    cycle_dir = str(result.get("cycle_dir") or "").strip()
    if not cycle_dir:
        return
    try:
        _write_json(
            os.path.join(cycle_dir, "hakai_cycle_output.json"),
            result,
        )
    except Exception as exc:
        logger.warning("Failed to persist cycle output for %s: %s", cycle_dir, exc)


def _emit_notification(
    notification_callback: NotificationCallback,
    event_name: str,
    payload: Dict[str, Any],
) -> None:
    if not callable(notification_callback):
        return
    try:
        notification_callback(event_name, dict(payload or {}))
    except Exception as exc:
        logger.warning("Notification callback failed | event=%s error=%s", event_name, exc)


def _position_summary_for_log(position: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    payload = dict(position or {})
    return {
        "direction": payload.get("direction"),
        "size": payload.get("size"),
        "entry_price": payload.get("entry_price"),
        "entry_notional": payload.get("entry_notional"),
        "position_value": payload.get("position_value"),
        "position_margin": payload.get("position_margin"),
        "stop_loss": payload.get("stop_loss"),
        "stop_loss_distance_pct": payload.get("stop_loss_distance_pct"),
        "stop_loss_basis_effective_leverage": payload.get("stop_loss_basis_effective_leverage"),
    }


def _extract_managed_position(positions: Sequence[Dict[str, Any]], symbol: str) -> Optional[Dict[str, Any]]:
    normalized_symbol = str(symbol or "").strip().upper()
    for position in positions:
        if not isinstance(position, dict):
            continue
        if str(position.get("symbol") or "").strip().upper() == normalized_symbol:
            return dict(position)
    return None


def _validate_position_universe(positions: Sequence[Dict[str, Any]], symbol: str) -> Optional[str]:
    normalized_symbol = str(symbol or "").strip().upper()
    normalized_positions = [
        dict(position)
        for position in positions
        if isinstance(position, dict) and str(position.get("symbol") or "").strip()
    ]
    if len(normalized_positions) <= 1:
        if not normalized_positions:
            return None
        existing_symbol = str(normalized_positions[0].get("symbol") or "").strip().upper()
        return None if existing_symbol == normalized_symbol else f"unsupported_open_position:{existing_symbol}"

    symbols = sorted(
        {
            str(position.get("symbol") or "").strip().upper()
            for position in normalized_positions
            if str(position.get("symbol") or "").strip()
        }
    )
    return f"multiple_open_positions:{','.join(symbols)}"


def _serialize_ohlcv_rows(candles: Sequence[Dict[str, Any]], *, limit: int) -> list[list[float]]:
    visible = list(candles or [])[-limit:]
    rows: list[list[float]] = []
    for candle in visible:
        open_price = _format_price(candle.get("open"))
        high_price = _format_price(candle.get("high"))
        low_price = _format_price(candle.get("low"))
        close_price = _format_price(candle.get("close"))
        volume = _safe_float(candle.get("volume"), None)
        if None in (open_price, high_price, low_price, close_price, volume):
            raise ValueError("invalid candle found while serializing prompt OHLCV data")
        rows.append([open_price, high_price, low_price, close_price, float(volume)])
    return rows


# Prompt context and volatility sizing helpers.
def _fetch_prompt_market_context(
    *,
    symbol: str,
    candle_count: int,
    position_sizing_daily_sample_days: int,
    position_sizing_live_window_hours: int,
    as_of_ms: Optional[int],
) -> Dict[str, Any]:
    timeframe_payload: Dict[str, list[list[float]]] = {}
    daily_position_sizing_candles: Optional[list[Dict[str, Any]]] = None
    live_window_candles: Optional[list[Dict[str, Any]]] = None
    resolved_as_of_ms = _resolve_as_of_ms(as_of_ms)

    for timeframe in DEFAULT_TIMEFRAMES:
        fetch_limit = candle_count
        if timeframe == "1h":
            fetch_limit = max(candle_count, position_sizing_live_window_hours)
        elif timeframe == "1d":
            # Fetch a small buffer because Binance includes the current in-progress daily candle.
            fetch_limit = max(candle_count, position_sizing_daily_sample_days + 2)

        raw_klines = fetch_klines(symbol, timeframe, fetch_limit, as_of_ms=resolved_as_of_ms)
        candles = parse_klines(raw_klines)
        if len(candles) < fetch_limit:
            raise ValueError(
                f"not enough candles for {symbol} {timeframe}: have={len(candles)} need={fetch_limit}"
            )
        timeframe_payload[timeframe] = _serialize_ohlcv_rows(candles, limit=candle_count)
        if timeframe == "1h":
            live_window_candles = candles[-position_sizing_live_window_hours:]
        elif timeframe == "1d":
            daily_position_sizing_candles = _select_closed_candles(
                candles,
                interval=timeframe,
                limit=position_sizing_daily_sample_days,
                as_of_ms=resolved_as_of_ms,
            )

    if live_window_candles is None or len(live_window_candles) < position_sizing_live_window_hours:
        raise ValueError("1h candles unavailable for live window calculation")
    if (
        daily_position_sizing_candles is None
        or len(daily_position_sizing_candles) < position_sizing_daily_sample_days
    ):
        raise ValueError("1d candles unavailable for percentile position sizing")

    return {
        "timeframes": timeframe_payload,
        "daily_position_sizing_candles": daily_position_sizing_candles,
        "live_window_candles": live_window_candles,
    }


def _fetch_live_prompt_position_snapshot(
    *,
    api_key: str,
    api_secret: str,
    symbol: str,
) -> Optional[Dict[str, Any]]:
    snapshot = get_position_snapshot(api_key, api_secret, symbol, retries=1)
    if not isinstance(snapshot, dict):
        return None
    return calculate_position_metrics(snapshot)


def _select_closed_candles(
    candles: Sequence[Dict[str, Any]],
    *,
    interval: str,
    limit: int,
    as_of_ms: Optional[int],
) -> list[Dict[str, Any]]:
    resolved_limit = max(1, int(limit))
    interval_minutes = interval_to_minutes(interval)
    if interval_minutes is None:
        raise ValueError(f"unsupported interval for closed candle selection: {interval}")

    interval_ms = int(interval_minutes * 60 * 1000)
    resolved_as_of_ms = _resolve_as_of_ms(as_of_ms)
    closed_candles: list[Dict[str, Any]] = []
    for candle in candles:
        open_time_ms = _safe_int(candle.get("timestamp"), 0)
        if open_time_ms <= 0:
            continue
        if open_time_ms + interval_ms <= resolved_as_of_ms:
            closed_candles.append(dict(candle))

    if len(closed_candles) < resolved_limit:
        raise ValueError(
            f"not enough closed candles for {interval}: have={len(closed_candles)} need={resolved_limit}"
        )
    return closed_candles[-resolved_limit:]


def _estimate_percentile_position(sorted_samples: Sequence[float], value: float) -> Dict[str, Any]:
    samples = [float(sample) for sample in sorted_samples]
    sample_size = len(samples)
    if sample_size <= 0:
        raise ValueError("percentile estimation requires at least one sample")
    if not math.isfinite(value) or value < 0.0:
        raise ValueError("percentile estimation value is invalid")

    first_sample = float(samples[0])
    last_sample = float(samples[-1])

    if sample_size == 1:
        raw_percentile = 1.0 if value >= first_sample else 0.0
        return {
            "rank_estimate": 1.0 if value >= first_sample else 0.0,
            "raw_percentile": float(raw_percentile),
            "rank_floor": 1,
            "rank_ceiling": 1,
            "interpolation_weight": 0.0,
            "sample_lower_value": first_sample,
            "sample_upper_value": first_sample,
            "location": "single_sample",
        }

    if value < first_sample:
        return {
            "rank_estimate": 0.0,
            "raw_percentile": 0.0,
            "rank_floor": 1,
            "rank_ceiling": 1,
            "interpolation_weight": 0.0,
            "sample_lower_value": first_sample,
            "sample_upper_value": first_sample,
            "location": "below_sample_range",
        }

    if value > last_sample:
        return {
            "rank_estimate": float(sample_size),
            "raw_percentile": 1.0,
            "rank_floor": sample_size,
            "rank_ceiling": sample_size,
            "interpolation_weight": 0.0,
            "sample_lower_value": last_sample,
            "sample_upper_value": last_sample,
            "location": "above_sample_range",
        }

    left_index = bisect_left(samples, value)
    right_index = bisect_right(samples, value)
    if left_index != right_index:
        rank_floor = left_index + 1
        rank_ceiling = right_index
        rank_estimate = (rank_floor + rank_ceiling) / 2.0
        return {
            "rank_estimate": float(rank_estimate),
            "raw_percentile": float(rank_estimate / sample_size),
            "rank_floor": int(rank_floor),
            "rank_ceiling": int(rank_ceiling),
            "interpolation_weight": 0.0,
            "sample_lower_value": float(samples[left_index]),
            "sample_upper_value": float(samples[right_index - 1]),
            "location": "matched_sample_value",
        }

    lower_index = max(0, left_index - 1)
    upper_index = min(sample_size - 1, left_index)
    lower_value = float(samples[lower_index])
    upper_value = float(samples[upper_index])
    interpolation_weight = 0.0
    if upper_value > lower_value:
        interpolation_weight = (value - lower_value) / (upper_value - lower_value)
    interpolation_weight = min(max(float(interpolation_weight), 0.0), 1.0)
    rank_estimate = float(lower_index + 1) + interpolation_weight

    return {
        "rank_estimate": float(rank_estimate),
        "raw_percentile": float(rank_estimate / sample_size),
        "rank_floor": int(lower_index + 1),
        "rank_ceiling": int(upper_index + 1),
        "interpolation_weight": float(interpolation_weight),
        "sample_lower_value": lower_value,
        "sample_upper_value": upper_value,
        "location": "between_sample_values",
    }


def _calculate_volatility_snapshot(
    daily_candles: Sequence[Dict[str, Any]],
    *,
    daily_sample_days: int,
    live_window_candles: Sequence[Dict[str, Any]],
    live_window_hours: int,
    leverage: int,
    position_size_ratio_min: float,
    position_size_ratio_max: float,
) -> Dict[str, Any]:
    resolved_daily_sample_days = max(1, int(daily_sample_days))
    resolved_live_window_hours = max(1, int(live_window_hours))
    relevant_daily_candles = list(daily_candles or [])[-resolved_daily_sample_days:]
    relevant_live_window_candles = list(live_window_candles or [])[-resolved_live_window_hours:]
    if len(relevant_daily_candles) < resolved_daily_sample_days:
        raise ValueError("not enough daily candles for percentile position sizing")
    if len(relevant_live_window_candles) < resolved_live_window_hours:
        raise ValueError("not enough 1h candles for live window calculation")

    daily_log_samples = sorted(
        _calculate_log_high_low_ratio(
            high=candle.get("high"),
            low=candle.get("low"),
            context="daily position sizing sample",
        )
        for candle in relevant_daily_candles
    )

    live_highs: list[float] = []
    live_lows: list[float] = []
    for candle in relevant_live_window_candles:
        high_value = _format_price(candle.get("high"))
        low_value = _format_price(candle.get("low"))
        if high_value is None or low_value is None:
            raise ValueError("live window contains invalid high/low values")
        if high_value < low_value:
            raise ValueError("live window high is below low")
        live_highs.append(float(high_value))
        live_lows.append(float(low_value))

    live_range_high = max(live_highs)
    live_range_low = min(live_lows)
    live_range_log = _calculate_log_high_low_ratio(
        high=live_range_high,
        low=live_range_low,
        context="live window position sizing snapshot",
    )

    resolved_ratio_min = float(position_size_ratio_min)
    resolved_ratio_max = float(position_size_ratio_max)
    if (
        not math.isfinite(resolved_ratio_min)
        or not math.isfinite(resolved_ratio_max)
        or resolved_ratio_min <= 0.0
        or resolved_ratio_max <= 0.0
        or resolved_ratio_max <= resolved_ratio_min
    ):
        raise ValueError("position size ratio bounds are invalid")

    percentile_position = _estimate_percentile_position(daily_log_samples, live_range_log)
    raw_percentile = float(percentile_position["raw_percentile"])
    target_margin_ratio = min(max(raw_percentile, resolved_ratio_min), resolved_ratio_max)
    target_effective_leverage = target_margin_ratio * float(leverage)

    return {
        "daily_sample_days": int(resolved_daily_sample_days),
        "live_window_hours": int(resolved_live_window_hours),
        "sample_size": len(daily_log_samples),
        "daily_range_log_min": float(daily_log_samples[0]),
        "daily_range_log_median": float(median(daily_log_samples)),
        "daily_range_log_max": float(daily_log_samples[-1]),
        "live_range_high": float(live_range_high),
        "live_range_low": float(live_range_low),
        "live_range_log": float(live_range_log),
        "raw_percentile": raw_percentile,
        "percentile_rank_estimate": float(percentile_position["rank_estimate"]),
        "percentile_rank_floor": int(percentile_position["rank_floor"]),
        "percentile_rank_ceiling": int(percentile_position["rank_ceiling"]),
        "interpolation_weight": float(percentile_position["interpolation_weight"]),
        "sample_lower_value": float(percentile_position["sample_lower_value"]),
        "sample_upper_value": float(percentile_position["sample_upper_value"]),
        "percentile_location": str(percentile_position["location"]),
        "position_size_ratio_min": float(resolved_ratio_min),
        "position_size_ratio_max": float(resolved_ratio_max),
        "target_margin_ratio": float(target_margin_ratio),
        "target_effective_leverage": float(target_effective_leverage),
    }


def _calculate_target_notional(
    *,
    account_equity: float,
    target_margin_ratio: float,
    leverage: int,
) -> float:
    return float(account_equity * target_margin_ratio * float(leverage))


def _format_percentile_sizing_summary(snapshot: Optional[Dict[str, Any]]) -> str:
    payload = dict(snapshot or {})
    live_range_log = _safe_float(payload.get("live_range_log"), None)
    rank_estimate = _safe_float(payload.get("percentile_rank_estimate"), None)
    sample_size = _safe_int(payload.get("sample_size"), 0)
    target_margin_ratio = _safe_float(payload.get("target_margin_ratio"), None)
    location = str(payload.get("percentile_location") or "").strip().lower()
    if (
        live_range_log is None
        or rank_estimate is None
        or sample_size <= 0
        or target_margin_ratio is None
    ):
        return "-"

    if location == "below_sample_range":
        location_label = "below"
    elif location == "above_sample_range":
        location_label = "above"
    else:
        location_label = "in-range"

    return (
        f"24h ln={live_range_log:.4f} | "
        f"rank={rank_estimate:.1f}/{sample_size} | "
        f"{location_label} | "
        f"final={target_margin_ratio * 100.0:.2f}%"
    )


# Order planning and execution helpers.
def _build_entry_order_plan(
    *,
    symbol: str,
    desired_notional_usdt: float,
    reference_price: float,
    max_qty: Optional[float] = None,
) -> Dict[str, Any]:
    plan: Dict[str, Any] = {
        "qty": None,
        "order_notional_usdt": None,
        "min_notional_usdt": None,
        "meets_min_notional": True,
    }
    if desired_notional_usdt <= 0.0 or reference_price <= 0.0:
        return plan

    qty_value = desired_notional_usdt / reference_price
    if max_qty is not None and max_qty > 0.0:
        qty_value = min(qty_value, max_qty)
    if qty_value <= 0.0:
        return plan

    raw_qty = safe_decimal(str(qty_value))
    adjusted_qty = adjust_qty_for_symbol(symbol, raw_qty)
    if adjusted_qty is None or adjusted_qty <= 0:
        return plan

    notional_check = evaluate_entry_order_notional(symbol, adjusted_qty, reference_price)
    order_notional = _safe_float(notional_check.get("order_notional"), None)
    min_notional = _safe_float(notional_check.get("min_notional"), None)

    plan["qty"] = decimal_to_str(adjusted_qty)
    if order_notional is not None and order_notional > 0.0:
        plan["order_notional_usdt"] = float(order_notional)
    if min_notional is not None and min_notional > 0.0:
        plan["min_notional_usdt"] = float(min_notional)
    plan["meets_min_notional"] = bool(notional_check.get("meets_min_notional", True))
    return plan


def _calculate_qty_from_notional(
    *,
    symbol: str,
    notional_usdt: float,
    reference_price: float,
) -> Optional[str]:
    return _build_entry_order_plan(
        symbol=symbol,
        desired_notional_usdt=notional_usdt,
        reference_price=reference_price,
    ).get("qty")


def _calculate_qty_from_delta(
    *,
    symbol: str,
    delta_notional_usdt: float,
    reference_price: float,
    max_qty: Optional[float] = None,
) -> Optional[str]:
    return _build_entry_order_plan(
        symbol=symbol,
        desired_notional_usdt=delta_notional_usdt,
        reference_price=reference_price,
        max_qty=max_qty,
    ).get("qty")


def _build_min_notional_skip_result(
    *,
    action: str,
    qty: Optional[str],
    order_plan: Optional[Dict[str, Any]] = None,
    requested_notional_usdt: Optional[float] = None,
    order_error_code: Optional[int] = None,
    order_error_message: str = "",
) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "success": True,
        "action": action,
        "qty": qty,
    }
    if requested_notional_usdt is not None:
        result["requested_notional_usdt"] = float(requested_notional_usdt)
    if isinstance(order_plan, dict):
        if order_plan.get("order_notional_usdt") is not None:
            result["order_notional_usdt"] = order_plan.get("order_notional_usdt")
        if order_plan.get("min_notional_usdt") is not None:
            result["min_notional_usdt"] = order_plan.get("min_notional_usdt")
    if order_error_code is not None:
        result["order_error_code"] = order_error_code
        result["order_error_message"] = order_error_message
    return result


def _floats_close(
    left: Optional[float],
    right: Optional[float],
    *,
    rel_tol: float = 1e-6,
    abs_tol: float = 1e-9,
) -> bool:
    if left is None or right is None:
        return False
    return math.isclose(float(left), float(right), rel_tol=rel_tol, abs_tol=abs_tol)


def _resolve_position_stop_loss_price(
    *,
    direction: str,
    entry_price: float,
    stop_distance_pct: float,
) -> Optional[float]:
    if entry_price <= 0.0 or stop_distance_pct <= 0.0 or not math.isfinite(stop_distance_pct):
        return None
    normalized_direction = str(direction or "").strip().lower()
    if normalized_direction == "long":
        return float(entry_price * (1.0 - stop_distance_pct))
    if normalized_direction == "short":
        return float(entry_price * (1.0 + stop_distance_pct))
    return None


def _normalize_stop_risk_basis(value: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(value, dict):
        return None

    direction = str(value.get("direction") or "").strip().lower()
    if direction not in {"long", "short"}:
        return None

    entry_price = _format_price(value.get("entry_price"))
    size = _format_price(value.get("size"))
    basis_account_equity = _format_price(value.get("basis_account_equity"))
    basis_entry_notional = _format_price(value.get("basis_entry_notional") or value.get("entry_notional"))
    stop_loss_account_risk_pct = _normalize_optional_ratio(
        value.get("stop_loss_account_risk_pct") or value.get("account_risk_pct")
    )
    basis_effective_leverage = _safe_float(value.get("basis_effective_leverage"), None)
    stop_loss_distance_pct = _normalize_optional_ratio(
        value.get("stop_loss_distance_pct") or value.get("stop_distance_pct")
    )
    stop_loss = _format_price(value.get("stop_loss"))

    if entry_price is None or size is None or basis_account_equity is None:
        return None
    if basis_entry_notional is None:
        basis_entry_notional = float(entry_price * size)
    if basis_effective_leverage is None and basis_account_equity > 0.0 and basis_entry_notional > 0.0:
        basis_effective_leverage = float(basis_entry_notional / basis_account_equity)
    if (
        basis_effective_leverage is None
        or not math.isfinite(basis_effective_leverage)
        or basis_effective_leverage <= 0.0
    ):
        return None
    if stop_loss_account_risk_pct is None:
        return None
    if stop_loss_distance_pct is None:
        stop_loss_distance_pct = float(stop_loss_account_risk_pct / basis_effective_leverage)
    if stop_loss_distance_pct <= 0.0 or not math.isfinite(stop_loss_distance_pct):
        return None
    if stop_loss is None:
        stop_loss = _resolve_position_stop_loss_price(
            direction=direction,
            entry_price=entry_price,
            stop_distance_pct=stop_loss_distance_pct,
        )
    if stop_loss is None:
        return None

    return {
        "direction": direction,
        "size": float(size),
        "entry_price": float(entry_price),
        "basis_account_equity": float(basis_account_equity),
        "basis_entry_notional": float(basis_entry_notional),
        "basis_effective_leverage": float(basis_effective_leverage),
        "stop_loss_account_risk_pct": float(stop_loss_account_risk_pct),
        "stop_loss_distance_pct": float(stop_loss_distance_pct),
        "stop_loss": float(stop_loss),
        "updated_at": str(value.get("updated_at") or _current_time_utc().isoformat()),
    }


def _build_stop_risk_basis_from_position(
    *,
    position: Dict[str, Any],
    account_equity: float,
    stop_loss_account_risk_pct: float,
) -> Optional[Dict[str, Any]]:
    metrics = calculate_position_metrics(position)
    direction = str(metrics.get("direction") or "").strip().lower()
    entry_price = _format_price(metrics.get("entry_price"))
    size = _format_price(metrics.get("size"))
    basis_entry_notional = _format_price(metrics.get("entry_notional"))
    basis_account_equity = _format_price(account_equity)
    normalized_risk_pct = _normalize_optional_ratio(stop_loss_account_risk_pct)
    if (
        direction not in {"long", "short"}
        or entry_price is None
        or size is None
        or basis_entry_notional is None
        or basis_account_equity is None
        or normalized_risk_pct is None
    ):
        return None

    basis_effective_leverage = float(basis_entry_notional / basis_account_equity)
    if basis_effective_leverage <= 0.0 or not math.isfinite(basis_effective_leverage):
        return None

    stop_loss_distance_pct = float(normalized_risk_pct / basis_effective_leverage)
    stop_loss = _resolve_position_stop_loss_price(
        direction=direction,
        entry_price=entry_price,
        stop_distance_pct=stop_loss_distance_pct,
    )
    if stop_loss is None:
        return None

    return {
        "direction": direction,
        "size": float(size),
        "entry_price": float(entry_price),
        "basis_account_equity": float(basis_account_equity),
        "basis_entry_notional": float(basis_entry_notional),
        "basis_effective_leverage": float(basis_effective_leverage),
        "stop_loss_account_risk_pct": float(normalized_risk_pct),
        "stop_loss_distance_pct": float(stop_loss_distance_pct),
        "stop_loss": float(stop_loss),
        "updated_at": _current_time_utc().isoformat(),
    }


def _stop_risk_basis_matches_position(stop_risk_basis: Any, position: Optional[Dict[str, Any]]) -> bool:
    normalized_basis = _normalize_stop_risk_basis(stop_risk_basis)
    if normalized_basis is None or not isinstance(position, dict):
        return False

    metrics = calculate_position_metrics(position)
    direction = str(metrics.get("direction") or "").strip().lower()
    entry_price = _format_price(metrics.get("entry_price"))
    size = _format_price(metrics.get("size"))
    entry_notional = _format_price(metrics.get("entry_notional"))
    return (
        direction == normalized_basis["direction"]
        and _floats_close(entry_price, normalized_basis["entry_price"])
        and _floats_close(size, normalized_basis["size"])
        and _floats_close(entry_notional, normalized_basis["basis_entry_notional"], rel_tol=1e-5, abs_tol=1e-6)
    )


def _enrich_position_with_stop_risk(
    position: Optional[Dict[str, Any]],
    stop_risk_basis: Any,
) -> Optional[Dict[str, Any]]:
    if not isinstance(position, dict):
        return position

    normalized_basis = _normalize_stop_risk_basis(stop_risk_basis)
    if normalized_basis is None:
        return dict(position)

    updated_position = dict(position)
    updated_position["entry_notional"] = (
        updated_position.get("entry_notional") or normalized_basis["basis_entry_notional"]
    )
    updated_position["stop_loss_account_risk_pct"] = normalized_basis["stop_loss_account_risk_pct"]
    updated_position["stop_loss_distance_pct"] = normalized_basis["stop_loss_distance_pct"]
    updated_position["stop_loss_basis_account_equity"] = normalized_basis["basis_account_equity"]
    updated_position["stop_loss_basis_entry_notional"] = normalized_basis["basis_entry_notional"]
    updated_position["stop_loss_basis_effective_leverage"] = normalized_basis["basis_effective_leverage"]
    return updated_position


def _sync_account_risk_stop_loss(
    *,
    api_key: str,
    api_secret: str,
    symbol: str,
    position: Dict[str, Any],
    stop_risk_basis: Any,
) -> Dict[str, Any]:
    metrics = calculate_position_metrics(position)
    direction = str(metrics.get("direction") or "").strip().lower()
    side = str(metrics.get("side") or "").strip()
    current_stop = _format_price(metrics.get("stop_loss"))
    normalized_basis = _normalize_stop_risk_basis(stop_risk_basis)
    if direction not in ("long", "short") or not side:
        return {
            "success": False,
            "changed": False,
            "reason": "invalid_position_for_stop_sync",
        }
    if normalized_basis is None:
        return {
            "success": False,
            "changed": False,
            "reason": "stop_risk_basis_unavailable",
        }
    if normalized_basis["direction"] != direction:
        return {
            "success": False,
            "changed": False,
            "reason": "stop_risk_basis_mismatch",
            "stop_loss_account_risk_pct": normalized_basis["stop_loss_account_risk_pct"],
            "stop_loss_distance_pct": normalized_basis["stop_loss_distance_pct"],
            "basis_account_equity": normalized_basis["basis_account_equity"],
            "basis_entry_notional": normalized_basis["basis_entry_notional"],
            "basis_effective_leverage": normalized_basis["basis_effective_leverage"],
        }

    sync_result = sync_existing_position_stop_loss(
        api_key,
        api_secret,
        symbol,
        side,
        stop_loss=normalized_basis["stop_loss"],
        current_stop_loss=current_stop,
    )
    enriched_result = dict(sync_result)
    enriched_result.setdefault("stop_loss", normalized_basis["stop_loss"])
    enriched_result["stop_loss_account_risk_pct"] = normalized_basis["stop_loss_account_risk_pct"]
    enriched_result["stop_loss_distance_pct"] = normalized_basis["stop_loss_distance_pct"]
    enriched_result["basis_account_equity"] = normalized_basis["basis_account_equity"]
    enriched_result["basis_entry_notional"] = normalized_basis["basis_entry_notional"]
    enriched_result["basis_effective_leverage"] = normalized_basis["basis_effective_leverage"]
    return enriched_result


def _fetch_synced_position(
    *,
    api_key: str,
    api_secret: str,
    symbol: str,
    previous_position: Optional[Dict[str, Any]] = None,
    expected_action: Optional[str] = None,
    max_attempts: int = 8,
    sleep_seconds: float = 0.5,
) -> Optional[Dict[str, Any]]:
    normalized_symbol = str(symbol or "").strip().upper()
    last_snapshot: Optional[Dict[str, Any]] = None

    for attempt in range(max(1, int(max_attempts))):
        snapshot = get_position_snapshot(api_key, api_secret, normalized_symbol, retries=1)
        if isinstance(snapshot, dict) and str(snapshot.get("symbol") or "").strip().upper() == normalized_symbol:
            last_snapshot = snapshot
        else:
            positions = get_positions(api_key, api_secret) or []
            last_snapshot = _extract_managed_position(positions, normalized_symbol)

        if _position_sync_matches_expected(
            last_snapshot,
            previous_position=previous_position,
            expected_action=expected_action,
        ):
            return last_snapshot

        if attempt + 1 < max(1, int(max_attempts)):
            time.sleep(max(0.0, float(sleep_seconds)))

    return last_snapshot


def _position_sync_matches_expected(
    snapshot: Optional[Dict[str, Any]],
    *,
    previous_position: Optional[Dict[str, Any]],
    expected_action: Optional[str],
) -> bool:
    normalized_action = str(expected_action or "").strip()
    if not normalized_action:
        return isinstance(snapshot, dict)

    if not isinstance(snapshot, dict):
        return False

    snapshot_metrics = calculate_position_metrics(snapshot)
    snapshot_size = abs(_safe_float(snapshot_metrics.get("size"), 0.0) or 0.0)
    snapshot_direction = str(snapshot_metrics.get("direction") or "").strip().lower()
    if normalized_action == "opened_new_position":
        return snapshot_size > 0.0 and snapshot_direction in ("long", "short")

    if previous_position is None:
        return True

    previous_metrics = calculate_position_metrics(previous_position)
    previous_size = abs(_safe_float(previous_metrics.get("size"), 0.0) or 0.0)
    previous_direction = str(previous_metrics.get("direction") or "").strip().lower()

    if normalized_action == "scaled_in_position":
        return snapshot_size > previous_size
    if normalized_action == "scaled_out_position":
        return 0.0 <= snapshot_size < previous_size
    if normalized_action == "reversed_position":
        return (
            snapshot_size > 0.0
            and snapshot_direction in ("long", "short")
            and previous_direction in ("long", "short")
            and snapshot_direction != previous_direction
        )
    return True


def _apply_synced_stop_loss_to_position(
    position: Optional[Dict[str, Any]],
    stop_sync_result: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    if not isinstance(position, dict):
        return position
    if not isinstance(stop_sync_result, dict) or not bool(stop_sync_result.get("success")):
        return dict(position)

    synced_stop_loss = _format_price(stop_sync_result.get("stop_loss"))
    if synced_stop_loss is None:
        return dict(position)

    updated_position = dict(position)
    updated_position["stop_loss"] = synced_stop_loss
    return updated_position


def _build_state_update(
    *,
    previous_state: Optional[Dict[str, Any]],
    trigger_pct_usdt: float,
    ai_triggered: bool,
    trigger_price: Optional[float],
    ai_decision: Optional[str],
    next_trigger_down: Optional[float],
    next_trigger_up: Optional[float],
    stop_risk_basis: Any = _STATE_UNSET,
) -> Dict[str, Any]:
    state_update = dict(previous_state or {})
    state_update.pop("last_ai_trigger_round_price", None)
    state_update["trigger_pct_usdt"] = float(trigger_pct_usdt)
    if ai_triggered:
        state_update["last_ai_trigger_price"] = trigger_price
        state_update["last_ai_triggered_at"] = _current_time_utc().isoformat()
    normalized_decision = _normalize_ai_decision(ai_decision)
    if normalized_decision:
        state_update["last_ai_decision"] = normalized_decision
    if next_trigger_down is not None:
        state_update["next_trigger_down"] = next_trigger_down
    if next_trigger_up is not None:
        state_update["next_trigger_up"] = next_trigger_up
    if stop_risk_basis is not _STATE_UNSET:
        normalized_stop_risk_basis = _normalize_stop_risk_basis(stop_risk_basis)
        state_update["stop_risk_basis"] = (
            dict(normalized_stop_risk_basis) if normalized_stop_risk_basis is not None else None
        )
    return state_update


def _build_ai_state_update_from_reference_price(
    *,
    previous_state: Optional[Dict[str, Any]],
    trigger_pct_usdt: float,
    reference_price: float,
    ai_decision: Optional[str],
    stop_risk_basis: Any = _STATE_UNSET,
) -> Dict[str, Any]:
    next_levels = _build_trigger_levels(reference_price, trigger_pct_usdt)
    return _build_state_update(
        previous_state=previous_state,
        trigger_pct_usdt=trigger_pct_usdt,
        ai_triggered=True,
        trigger_price=next_levels["trigger_price"],
        ai_decision=ai_decision,
        next_trigger_down=next_levels["next_trigger_down"],
        next_trigger_up=next_levels["next_trigger_up"],
        stop_risk_basis=stop_risk_basis,
    )


def _place_new_direction_position(
    *,
    api_key: str,
    api_secret: str,
    symbol: str,
    decision: str,
    target_notional_usdt: float,
    reference_price: float,
    leverage: int,
) -> Dict[str, Any]:
    normalized_decision = _normalize_ai_decision(decision)
    if normalized_decision not in _DIRECTIONAL_AI_DECISIONS:
        return {
            "success": False,
            "action": "invalid_ai_decision",
        }

    side = "Buy" if normalized_decision == "LONG" else "Sell"
    order_plan = _build_entry_order_plan(
        symbol=symbol,
        desired_notional_usdt=target_notional_usdt,
        reference_price=reference_price,
    )
    qty = order_plan.get("qty")
    if not qty:
        return {
            "success": False,
            "action": "target_qty_below_min",
            "order": None,
        }
    if not bool(order_plan.get("meets_min_notional", True)):
        logger.info(
            "Skipping new entry below exchange minimum notional | %s",
            format_log_details(
                {
                    "symbol": symbol,
                    "decision": normalized_decision,
                    "qty": qty,
                    "requested_notional_usdt": target_notional_usdt,
                    "order_notional_usdt": order_plan.get("order_notional_usdt"),
                    "min_notional_usdt": order_plan.get("min_notional_usdt"),
                }
            ),
        )
        return _build_min_notional_skip_result(
            action="skipped_entry_below_min_notional",
            qty=qty,
            order_plan=order_plan,
            requested_notional_usdt=target_notional_usdt,
        )

    order, code, msg = place_market_entry_order(
        api_key,
        api_secret,
        symbol,
        side,
        qty,
        leverage=leverage,
    )
    if order is None:
        if code == -4164:
            logger.info(
                "New entry rejected by exchange minimum notional | %s",
                format_log_details(
                    {
                        "symbol": symbol,
                        "decision": normalized_decision,
                        "qty": qty,
                        "requested_notional_usdt": target_notional_usdt,
                        "order_notional_usdt": order_plan.get("order_notional_usdt"),
                        "min_notional_usdt": order_plan.get("min_notional_usdt"),
                        "error_code": code,
                        "error_message": msg,
                    }
                ),
            )
            return _build_min_notional_skip_result(
                action="skipped_entry_below_min_notional",
                qty=qty,
                order_plan=order_plan,
                requested_notional_usdt=target_notional_usdt,
                order_error_code=code,
                order_error_message=msg,
            )
        return {
            "success": False,
            "action": "entry_order_failed",
            "order_error_code": code,
            "order_error_message": msg,
        }

    return {
        "success": True,
        "action": "opened_new_position",
        "order": order,
        "qty": qty,
    }


def _rebalance_existing_position(
    *,
    api_key: str,
    api_secret: str,
    symbol: str,
    current_position: Dict[str, Any],
    decision: str,
    target_notional_usdt: float,
    reference_price: float,
    leverage: int,
) -> Dict[str, Any]:
    normalized_decision = _normalize_ai_decision(decision)
    if normalized_decision not in _DIRECTIONAL_AI_DECISIONS:
        return {
            "success": False,
            "action": "invalid_ai_decision",
        }

    current_metrics = calculate_position_metrics(current_position)
    current_direction = str(current_metrics.get("direction") or "").strip().lower()
    current_notional = abs(_safe_float(current_metrics.get("position_value"), 0.0) or 0.0)
    current_side = str(current_metrics.get("side") or "").strip()
    current_size = abs(_safe_float(current_metrics.get("size"), 0.0) or 0.0)
    desired_direction = "long" if normalized_decision == "LONG" else "short"

    if current_direction not in ("long", "short") or not current_side or current_size <= 0.0:
        return {
            "success": False,
            "action": "invalid_existing_position",
        }

    if current_direction != desired_direction:
        close_ok = close_position(
            api_key,
            api_secret,
            symbol,
            current_side,
            str(current_size),
        )
        if not close_ok:
            return {
                "success": False,
                "action": "reverse_close_failed",
            }

        wait_for_close_propagation(
            api_key,
            api_secret,
            [symbol],
            context="reverse_position",
        )
        cancel_all_orders(api_key, api_secret, symbol)
        reopen_result = _place_new_direction_position(
            api_key=api_key,
            api_secret=api_secret,
            symbol=symbol,
            decision=normalized_decision,
            target_notional_usdt=target_notional_usdt,
            reference_price=reference_price,
            leverage=leverage,
        )
        reopen_action = str(reopen_result.get("action") or "").strip()
        if bool(reopen_result.get("success")) and reopen_action == "opened_new_position":
            reopen_result["action"] = "reversed_position"
        elif not reopen_action:
            reopen_result["action"] = (
                "reversed_position"
                if reopen_result.get("success")
                else "reverse_reopen_failed"
            )
        return reopen_result

    delta_notional = target_notional_usdt - current_notional
    if abs(delta_notional) <= 0.0:
        return {
            "success": True,
            "action": "kept_position_size",
        }

    if delta_notional > 0.0:
        add_order_plan = _build_entry_order_plan(
            symbol=symbol,
            desired_notional_usdt=delta_notional,
            reference_price=reference_price,
        )
        add_qty = add_order_plan.get("qty")
        if not add_qty:
            return {
                "success": True,
                "action": "kept_position_size",
            }
        if not bool(add_order_plan.get("meets_min_notional", True)):
            logger.info(
                "Skipping scale-in below exchange minimum notional | %s",
                format_log_details(
                    {
                        "symbol": symbol,
                        "decision": normalized_decision,
                        "qty": add_qty,
                        "current_notional_usdt": current_notional,
                        "requested_notional_usdt": delta_notional,
                        "order_notional_usdt": add_order_plan.get("order_notional_usdt"),
                        "min_notional_usdt": add_order_plan.get("min_notional_usdt"),
                    }
                ),
            )
            return _build_min_notional_skip_result(
                action="skipped_scale_in_below_min_notional",
                qty=add_qty,
                order_plan=add_order_plan,
                requested_notional_usdt=delta_notional,
            )

        add_order, code, msg = place_market_entry_order(
            api_key,
            api_secret,
            symbol,
            current_side,
            add_qty,
            leverage=leverage,
        )
        if add_order is None:
            if code == -4164:
                logger.info(
                    "Scale-in rejected by exchange minimum notional | %s",
                    format_log_details(
                        {
                            "symbol": symbol,
                            "decision": normalized_decision,
                            "qty": add_qty,
                            "current_notional_usdt": current_notional,
                            "requested_notional_usdt": delta_notional,
                            "order_notional_usdt": add_order_plan.get("order_notional_usdt"),
                            "min_notional_usdt": add_order_plan.get("min_notional_usdt"),
                            "error_code": code,
                            "error_message": msg,
                        }
                    ),
                )
                return _build_min_notional_skip_result(
                    action="skipped_scale_in_below_min_notional",
                    qty=add_qty,
                    order_plan=add_order_plan,
                    requested_notional_usdt=delta_notional,
                    order_error_code=code,
                    order_error_message=msg,
                )
            return {
                "success": False,
                "action": "scale_in_failed",
                "order_error_code": code,
                "order_error_message": msg,
            }
        return {
            "success": True,
            "action": "scaled_in_position",
            "order": add_order,
            "qty": add_qty,
        }

    reduce_qty = _calculate_qty_from_delta(
        symbol=symbol,
        delta_notional_usdt=abs(delta_notional),
        reference_price=reference_price,
        max_qty=current_size,
    )
    if not reduce_qty:
        return {
            "success": True,
            "action": "kept_position_size",
        }

    reduce_side = "Sell" if current_direction == "long" else "Buy"
    reduce_order, code, msg = place_reduce_only_market_order(
        api_key,
        api_secret,
        symbol,
        reduce_side,
        reduce_qty,
    )
    if reduce_order is None:
        return {
            "success": False,
            "action": "scale_out_failed",
            "order_error_code": code,
            "order_error_message": msg,
        }
    return {
        "success": True,
        "action": "scaled_out_position",
        "order": reduce_order,
        "qty": reduce_qty,
    }


def _determine_ai_trigger(
    *,
    has_position: bool,
    current_price: float,
    last_ai_trigger_price: Optional[float],
    trigger_pct_usdt: float,
    next_trigger_down: Optional[float] = None,
    next_trigger_up: Optional[float] = None,
) -> Dict[str, Any]:
    current_trigger_price = _normalize_trigger_price(current_price)
    if current_trigger_price is None:
        raise ValueError("current_price must be positive")

    active_trigger_down = _normalize_trigger_price(next_trigger_down)
    active_trigger_up = _normalize_trigger_price(next_trigger_up)
    boundary_source = "state_boundaries"

    has_valid_trigger_window = not (
        active_trigger_down is None
        or active_trigger_up is None
        or active_trigger_down >= active_trigger_up
    )

    if not has_valid_trigger_window:
        if last_ai_trigger_price is not None:
            anchor_levels = _build_trigger_levels(last_ai_trigger_price, trigger_pct_usdt)
            active_trigger_down = anchor_levels["next_trigger_down"]
            active_trigger_up = anchor_levels["next_trigger_up"]
            boundary_source = "last_price_anchor"
            has_valid_trigger_window = True

    if not has_position:
        next_levels = _build_trigger_levels(current_price, trigger_pct_usdt)
        return {
            "should_trigger": True,
            "reason": "no_position",
            "current_trigger_price": current_trigger_price,
            "trigger_price": next_levels["trigger_price"],
            "next_trigger_down": next_levels["next_trigger_down"],
            "next_trigger_up": next_levels["next_trigger_up"],
            "boundary_source": "no_position",
        }

    elif not has_valid_trigger_window:
        next_levels = _build_trigger_levels(current_price, trigger_pct_usdt)
        boundary_source = "live_price_initialization"
        return {
            "should_trigger": False,
            "reason": "waiting_for_next_price_trigger",
            "current_trigger_price": current_trigger_price,
            "trigger_price": None,
            "next_trigger_down": next_levels["next_trigger_down"],
            "next_trigger_up": next_levels["next_trigger_up"],
            "boundary_source": boundary_source,
        }

    if current_price >= float(active_trigger_up):
        next_levels = _build_trigger_levels(current_price, trigger_pct_usdt)
        return {
            "should_trigger": True,
            "reason": "price_distance_reached",
            "current_trigger_price": current_trigger_price,
            "trigger_price": next_levels["trigger_price"],
            "next_trigger_down": next_levels["next_trigger_down"],
            "next_trigger_up": next_levels["next_trigger_up"],
            "boundary_source": boundary_source,
            "trigger_direction": "up",
        }

    if current_price <= float(active_trigger_down):
        next_levels = _build_trigger_levels(current_price, trigger_pct_usdt)
        return {
            "should_trigger": True,
            "reason": "price_distance_reached",
            "current_trigger_price": current_trigger_price,
            "trigger_price": next_levels["trigger_price"],
            "next_trigger_down": next_levels["next_trigger_down"],
            "next_trigger_up": next_levels["next_trigger_up"],
            "boundary_source": boundary_source,
            "trigger_direction": "down",
        }

    return {
        "should_trigger": False,
        "reason": "waiting_for_next_price_trigger",
        "current_trigger_price": current_trigger_price,
        "trigger_price": None,
        "next_trigger_down": active_trigger_down,
        "next_trigger_up": active_trigger_up,
        "boundary_source": boundary_source,
    }


# Cycle entrypoints.
def run_hakai_cycle(
    *,
    state: Optional[Dict[str, Any]] = None,
    as_of_ms: Optional[int] = None,
    notification_callback: NotificationCallback = None,
) -> Dict[str, Any]:
    """Run one full HAK GEMINI BINANCE TRADER cycle and return a serializable result payload."""
    config = _load_strategy_config()
    symbol = str(config["symbol"])
    leverage = int(config["fixed_leverage"])
    trigger_pct_usdt = float(config["trigger_pct_usdt"])
    stop_loss_account_risk_pct = float(config["stop_loss_pct"])
    resolved_as_of_ms = _safe_int(as_of_ms, 0) or _current_time_ms()
    raw_previous_state = dict(state or {})
    previous_state, trigger_pct_state_refreshed = _align_state_trigger_percent(
        raw_previous_state,
        trigger_pct_usdt,
    )
    state_stop_risk_basis = _normalize_stop_risk_basis(previous_state.get("stop_risk_basis"))

    result: Dict[str, Any] = {
        "success": False,
        "symbol": symbol,
        "action": "init",
        "state_update": _build_state_update(
            previous_state=previous_state,
            trigger_pct_usdt=trigger_pct_usdt,
            ai_triggered=False,
            trigger_price=None,
            ai_decision=None,
            next_trigger_down=None,
            next_trigger_up=None,
            stop_risk_basis=state_stop_risk_basis,
        ),
        "ai_triggered": False,
        "ai_decision": None,
        "current_price": None,
        "next_trigger_down": None,
        "next_trigger_up": None,
        "trigger_reason": None,
        "trigger_price": None,
        "cycle_dir": None,
        "position": None,
        "position_before": None,
        "volatility_snapshot": None,
        "stop_risk_basis": dict(state_stop_risk_basis) if state_stop_risk_basis is not None else None,
    }
    logger.debug(
        "HAK GEMINI BINANCE TRADER cycle started | %s",
        format_log_details(
            {
                "symbol": symbol,
                "as_of_ms": resolved_as_of_ms,
                "fixed_leverage": leverage,
                "trigger_pct_usdt": trigger_pct_usdt,
                "previous_trigger_pct_usdt": raw_previous_state.get("trigger_pct_usdt"),
                "previous_last_ai_decision": previous_state.get("last_ai_decision"),
                "previous_last_ai_trigger_price": previous_state.get("last_ai_trigger_price"),
                "previous_next_trigger_down": previous_state.get("next_trigger_down"),
                "previous_next_trigger_up": previous_state.get("next_trigger_up"),
            }
        ),
    )
    if trigger_pct_state_refreshed and any(
        raw_previous_state.get(key) is not None
        for key in ("last_ai_trigger_price", "next_trigger_down", "next_trigger_up", "trigger_pct_usdt")
    ):
        logger.info(
            "Trigger window refreshed for updated trigger percent | %s",
            format_log_details(
                {
                    "symbol": symbol,
                    "previous_trigger_pct_usdt": raw_previous_state.get("trigger_pct_usdt"),
                    "active_trigger_pct_usdt": trigger_pct_usdt,
                    "anchor_price": previous_state.get("last_ai_trigger_price"),
                }
            ),
        )

    try:
        api_key, api_secret = get_binance_credentials()
    except ValueError as exc:
        result["action"] = "credentials_error"
        result["error"] = str(exc)
        return result

    positions = get_positions(api_key, api_secret)
    if positions is None:
        result["action"] = "positions_fetch_failed"
        return result

    universe_error = _validate_position_universe(positions, symbol)
    if universe_error:
        result["action"] = universe_error
        return result

    current_position = _extract_managed_position(positions, symbol)
    has_position = isinstance(current_position, dict)
    if has_position and current_position is not None:
        current_position_metrics = calculate_position_metrics(current_position)
        result["position"] = dict(current_position_metrics)
        result["position_before"] = dict(current_position_metrics)
        if not _stop_risk_basis_matches_position(state_stop_risk_basis, current_position):
            state_stop_risk_basis = None
            result["stop_risk_basis"] = None
        else:
            result["position"] = _enrich_position_with_stop_risk(result.get("position"), state_stop_risk_basis)
            result["position_before"] = _enrich_position_with_stop_risk(result.get("position_before"), state_stop_risk_basis)
    else:
        state_stop_risk_basis = None
        result["stop_risk_basis"] = None
    logger.debug(
        "Current managed position snapshot | %s",
        format_log_details(
            {
                "symbol": symbol,
                "has_position": has_position,
                "position": _position_summary_for_log(result.get("position_before")),
            }
        ),
    )

    reference_price_payload = get_reference_price(symbol)
    reference_price = _format_price((reference_price_payload or {}).get("price"))
    if reference_price is None:
        result["action"] = "reference_price_unavailable"
        return result

    result["current_price"] = reference_price

    stop_sync_result: Optional[Dict[str, Any]] = None
    prefetched_account_equity: Optional[float] = None
    if has_position and current_position is not None:
        if state_stop_risk_basis is None:
            prefetched_account_equity = _format_price(get_account_equity(api_key, api_secret))
            if prefetched_account_equity is not None:
                state_stop_risk_basis = _build_stop_risk_basis_from_position(
                    position=current_position,
                    account_equity=prefetched_account_equity,
                    stop_loss_account_risk_pct=stop_loss_account_risk_pct,
                )
                result["stop_risk_basis"] = (
                    dict(state_stop_risk_basis) if state_stop_risk_basis is not None else None
                )
            else:
                logger.warning(
                    "Account equity unavailable for stop risk basis refresh | %s",
                    format_log_details({"symbol": symbol}),
                )
        if state_stop_risk_basis is not None:
            result["position"] = _enrich_position_with_stop_risk(result.get("position"), state_stop_risk_basis)
            result["position_before"] = _enrich_position_with_stop_risk(
                result.get("position_before"),
                state_stop_risk_basis,
            )
            stop_sync_result = _sync_account_risk_stop_loss(
                api_key=api_key,
                api_secret=api_secret,
                symbol=symbol,
                position=current_position,
                stop_risk_basis=state_stop_risk_basis,
            )
            result["stop_sync"] = stop_sync_result
            result["position"] = _apply_synced_stop_loss_to_position(result.get("position"), stop_sync_result)
            result["position"] = _enrich_position_with_stop_risk(result.get("position"), state_stop_risk_basis)
            logger.debug(
                "Pre-AI stop sync completed | %s",
                format_log_details(
                    {
                        "symbol": symbol,
                        "stop_sync": stop_sync_result,
                    }
                ),
            )

    trigger_info = _determine_ai_trigger(
        has_position=has_position,
        current_price=reference_price,
        last_ai_trigger_price=_normalize_trigger_price(previous_state.get("last_ai_trigger_price")),
        trigger_pct_usdt=trigger_pct_usdt,
        next_trigger_down=_normalize_trigger_price(previous_state.get("next_trigger_down")),
        next_trigger_up=_normalize_trigger_price(previous_state.get("next_trigger_up")),
    )
    trigger_price = _normalize_trigger_price(trigger_info.get("trigger_price"))
    result["trigger_reason"] = trigger_info.get("reason")
    result["trigger_price"] = trigger_price
    result["next_trigger_down"] = trigger_info.get("next_trigger_down")
    result["next_trigger_up"] = trigger_info.get("next_trigger_up")
    logger.debug(
        "AI trigger evaluated | %s",
        format_log_details(
            {
                "symbol": symbol,
                "current_price": reference_price,
                "has_position": has_position,
                "trigger_info": trigger_info,
            }
        ),
    )

    if not bool(trigger_info.get("should_trigger")):
        result["success"] = True
        result["action"] = (
            "hold_stop_updated"
            if stop_sync_result and bool(stop_sync_result.get("changed"))
            else "hold_waiting_price_trigger"
        )
        result["state_update"] = _build_state_update(
            previous_state=previous_state,
            trigger_pct_usdt=trigger_pct_usdt,
            ai_triggered=False,
            trigger_price=None,
            ai_decision=None,
            next_trigger_down=_normalize_trigger_price(trigger_info.get("next_trigger_down")),
            next_trigger_up=_normalize_trigger_price(trigger_info.get("next_trigger_up")),
            stop_risk_basis=state_stop_risk_basis,
        )
        logger.debug(
            "HAK GEMINI BINANCE TRADER cycle completed without AI trigger | %s",
            format_log_details(
                {
                    "symbol": symbol,
                    "action": result["action"],
                    "current_price": reference_price,
                    "next_trigger_down": result["next_trigger_down"],
                    "next_trigger_up": result["next_trigger_up"],
                    "stop_sync": stop_sync_result,
                }
            ),
        )
        return result

    cycle_dir = _create_cycle_dir()
    result["cycle_dir"] = cycle_dir
    logger.info(
        "AI trigger activated | %s",
        format_log_details(
            {
                "symbol": symbol,
                "cycle_dir": cycle_dir,
                "trigger_reason": trigger_info.get("reason"),
                "trigger_price": trigger_price,
                "current_price": reference_price,
            }
        ),
    )

    market_context = _fetch_prompt_market_context(
        symbol=symbol,
        candle_count=int(config["ai_candle_count_per_timeframe"]),
        position_sizing_daily_sample_days=int(config["position_sizing_daily_sample_days"]),
        position_sizing_live_window_hours=int(config["position_sizing_live_window_hours"]),
        as_of_ms=resolved_as_of_ms,
    )
    volatility_snapshot = _calculate_volatility_snapshot(
        market_context["daily_position_sizing_candles"],
        daily_sample_days=int(config["position_sizing_daily_sample_days"]),
        live_window_candles=market_context["live_window_candles"],
        live_window_hours=int(config["position_sizing_live_window_hours"]),
        leverage=leverage,
        position_size_ratio_min=float(config["position_size_ratio_min"]),
        position_size_ratio_max=float(config["position_size_ratio_max"]),
    )
    result["volatility_snapshot"] = volatility_snapshot
    logger.info(
        "Market context prepared for AI | %s",
        format_log_details(
            {
                "symbol": symbol,
                "cycle_dir": cycle_dir,
                "timeframes": {key: len(value) for key, value in market_context["timeframes"].items()},
                "volatility_snapshot": volatility_snapshot,
            }
        ),
    )

    prompt_position_snapshot = _fetch_live_prompt_position_snapshot(
        api_key=api_key,
        api_secret=api_secret,
        symbol=symbol,
    )
    logger.info(
        "Live prompt position snapshot prepared | %s",
        format_log_details(
            {
                "symbol": symbol,
                "cycle_dir": cycle_dir,
                "prompt_position": _position_summary_for_log(prompt_position_snapshot),
            }
        ),
    )

    ai_analysis: Dict[str, Any] = {}
    _emit_notification(
        notification_callback,
        "ai_cycle_before",
        {
            "timestamp": _current_time_utc().isoformat(),
            "symbol": symbol,
            "cycle_dir": cycle_dir,
            "current_price": reference_price,
            "trigger_reason": trigger_info.get("reason"),
            "trigger_price": trigger_price,
            "next_trigger_down": result.get("next_trigger_down"),
            "next_trigger_up": result.get("next_trigger_up"),
            "position": result.get("position_before"),
            "previous_ai_decision": previous_state.get("last_ai_decision"),
            "volatility_snapshot": volatility_snapshot,
        },
    )

    ai_decision = evaluate_hakai_direction(
        cycle_dir=cycle_dir,
        symbol=symbol,
        reference_price=reference_price,
        timeframe_ohlcv=market_context["timeframes"],
        api_version=str(config["gemini_api_version"]),
        thinking_level=str(config["gemini_thinking_level"]),
        analysis_sink=ai_analysis,
        current_position_snapshot=prompt_position_snapshot,
    )
    if ai_decision is None:
        result["action"] = "ai_decision_failed"
        if ai_analysis:
            result["ai_analysis"] = dict(ai_analysis)
        result["state_update"] = _build_state_update(
            previous_state=previous_state,
            trigger_pct_usdt=trigger_pct_usdt,
            ai_triggered=False,
            trigger_price=None,
            ai_decision=None,
            next_trigger_down=_normalize_trigger_price(trigger_info.get("next_trigger_down")),
            next_trigger_up=_normalize_trigger_price(trigger_info.get("next_trigger_up")),
            stop_risk_basis=state_stop_risk_basis,
        )
        _emit_notification(
            notification_callback,
            "ai_cycle_after",
            {
                "timestamp": _current_time_utc().isoformat(),
                "success": False,
                "symbol": symbol,
                "cycle_dir": cycle_dir,
                "current_price": reference_price,
                "trigger_reason": trigger_info.get("reason"),
                "decision": None,
                "analysis": dict(ai_analysis),
                "volatility_snapshot": volatility_snapshot,
            },
        )
        _persist_cycle_output(result)
        logger.error(
            "AI decision failed | %s",
            format_log_details(
                {
                    "symbol": symbol,
                    "cycle_dir": cycle_dir,
                    "current_price": reference_price,
                    "trigger_reason": trigger_info.get("reason"),
                    "ai_analysis": ai_analysis,
                }
            ),
        )
        return result

    result["ai_triggered"] = True
    result["ai_decision"] = ai_decision.decision
    result["ai_analysis"] = dict(ai_analysis)
    logger.info(
        "AI decision received | %s",
        format_log_details(
            {
                "symbol": symbol,
                "cycle_dir": cycle_dir,
                "decision": ai_decision.decision,
                "thought_signatures": len(ai_analysis.get("thought_signatures") or []),
                "thought_summary": ai_analysis.get("thought_summary"),
                "usage_metadata": ai_analysis.get("usage_metadata"),
            }
        ),
    )
    _emit_notification(
        notification_callback,
        "ai_cycle_after",
        {
            "timestamp": _current_time_utc().isoformat(),
            "success": True,
            "symbol": symbol,
            "cycle_dir": cycle_dir,
            "current_price": reference_price,
            "trigger_reason": trigger_info.get("reason"),
            "decision": ai_decision.decision,
            "analysis": dict(ai_analysis),
            "volatility_snapshot": volatility_snapshot,
            "position": result.get("position_before"),
        },
    )

    account_equity = prefetched_account_equity
    if account_equity is None:
        account_equity = _format_price(get_account_equity(api_key, api_secret))
    if account_equity is None:
        result["action"] = "account_equity_unavailable"
        result["state_update"] = _build_state_update(
            previous_state=previous_state,
            trigger_pct_usdt=trigger_pct_usdt,
            ai_triggered=True,
            trigger_price=trigger_price,
            ai_decision=ai_decision.decision,
            next_trigger_down=_normalize_trigger_price(trigger_info.get("next_trigger_down")),
            next_trigger_up=_normalize_trigger_price(trigger_info.get("next_trigger_up")),
            stop_risk_basis=state_stop_risk_basis,
        )
        _persist_cycle_output(result)
        return result

    target_notional_usdt = _calculate_target_notional(
        account_equity=account_equity,
        target_margin_ratio=float(volatility_snapshot["target_margin_ratio"]),
        leverage=leverage,
    )
    result["account_equity"] = account_equity
    result["target_notional_usdt"] = target_notional_usdt
    logger.info(
        "Position sizing computed | %s",
        format_log_details(
            {
                "symbol": symbol,
                "account_equity": account_equity,
                "target_margin_ratio": volatility_snapshot.get("target_margin_ratio"),
                "target_effective_leverage": volatility_snapshot.get("target_effective_leverage"),
                "target_notional_usdt": target_notional_usdt,
                "sizing_summary": _format_percentile_sizing_summary(volatility_snapshot),
            }
        ),
    )

    applied_leverage = set_leverage(api_key, api_secret, symbol, leverage)
    if applied_leverage is None:
        result["action"] = "set_leverage_failed"
        result["state_update"] = _build_state_update(
            previous_state=previous_state,
            trigger_pct_usdt=trigger_pct_usdt,
            ai_triggered=True,
            trigger_price=trigger_price,
            ai_decision=ai_decision.decision,
            next_trigger_down=_normalize_trigger_price(trigger_info.get("next_trigger_down")),
            next_trigger_up=_normalize_trigger_price(trigger_info.get("next_trigger_up")),
            stop_risk_basis=state_stop_risk_basis,
        )
        _persist_cycle_output(result)
        return result

    leverage = int(applied_leverage)
    result["applied_leverage"] = leverage
    logger.info(
        "Leverage confirmed | %s",
        format_log_details(
            {
                "symbol": symbol,
                "applied_leverage": leverage,
            }
        ),
    )

    if current_position is None:
        execution_result = _place_new_direction_position(
            api_key=api_key,
            api_secret=api_secret,
            symbol=symbol,
            decision=ai_decision.decision,
            target_notional_usdt=target_notional_usdt,
            reference_price=reference_price,
            leverage=leverage,
        )
    else:
        execution_result = _rebalance_existing_position(
            api_key=api_key,
            api_secret=api_secret,
            symbol=symbol,
            current_position=current_position,
            decision=ai_decision.decision,
            target_notional_usdt=target_notional_usdt,
            reference_price=reference_price,
            leverage=leverage,
        )

    result["execution"] = execution_result
    logger.info(
        "Execution result captured | %s",
        format_log_details(
            {
                "symbol": symbol,
                "decision": ai_decision.decision,
                "execution": execution_result,
            }
        ),
    )
    if not bool(execution_result.get("success")):
        result["action"] = str(execution_result.get("action") or "execution_failed")
        result["state_update"] = _build_state_update(
            previous_state=previous_state,
            trigger_pct_usdt=trigger_pct_usdt,
            ai_triggered=True,
            trigger_price=trigger_price,
            ai_decision=ai_decision.decision,
            next_trigger_down=_normalize_trigger_price(trigger_info.get("next_trigger_down")),
            next_trigger_up=_normalize_trigger_price(trigger_info.get("next_trigger_up")),
            stop_risk_basis=state_stop_risk_basis,
        )
        _persist_cycle_output(result)
        logger.error(
            "Execution failed after AI decision | %s",
            format_log_details(
                {
                    "symbol": symbol,
                    "cycle_dir": cycle_dir,
                    "decision": ai_decision.decision,
                    "execution": execution_result,
                }
            ),
        )
        return result

    execution_action = str(execution_result.get("action") or "executed")
    synced_position = _fetch_synced_position(
        api_key=api_key,
        api_secret=api_secret,
        symbol=symbol,
        previous_position=current_position,
        expected_action=execution_action,
    )
    if synced_position is not None:
        state_stop_risk_basis = _build_stop_risk_basis_from_position(
            position=synced_position,
            account_equity=account_equity,
            stop_loss_account_risk_pct=stop_loss_account_risk_pct,
        )
        result["stop_risk_basis"] = dict(state_stop_risk_basis) if state_stop_risk_basis is not None else None
        synced_stop_result = _sync_account_risk_stop_loss(
            api_key=api_key,
            api_secret=api_secret,
            symbol=symbol,
            position=synced_position,
            stop_risk_basis=state_stop_risk_basis,
        )
        result["post_trade_stop_sync"] = synced_stop_result
        result["position"] = calculate_position_metrics(synced_position)
        result["position"] = _apply_synced_stop_loss_to_position(result.get("position"), synced_stop_result)
        result["position"] = _enrich_position_with_stop_risk(result.get("position"), state_stop_risk_basis)
        logger.info(
            "Post-trade position synchronized | %s",
            format_log_details(
                {
                    "symbol": symbol,
                    "position": _position_summary_for_log(result.get("position")),
                    "post_trade_stop_sync": synced_stop_result,
                }
            ),
        )
    else:
        state_stop_risk_basis = None
        result["stop_risk_basis"] = None

    result["success"] = True
    result["action"] = execution_action
    result["state_update"] = _build_state_update(
        previous_state=previous_state,
        trigger_pct_usdt=trigger_pct_usdt,
        ai_triggered=True,
        trigger_price=trigger_price,
        ai_decision=ai_decision.decision,
        next_trigger_down=_normalize_trigger_price(trigger_info.get("next_trigger_down")),
        next_trigger_up=_normalize_trigger_price(trigger_info.get("next_trigger_up")),
        stop_risk_basis=state_stop_risk_basis,
    )

    _persist_cycle_output(result)
    logger.info(
        "HAK GEMINI BINANCE TRADER cycle completed | %s",
        format_log_details(
            {
                "symbol": symbol,
                "cycle_dir": cycle_dir,
                "success": result["success"],
                "action": result["action"],
                "ai_decision": result.get("ai_decision"),
                "current_price": result.get("current_price"),
                "position_before": _position_summary_for_log(result.get("position_before")),
                "position_after": _position_summary_for_log(result.get("position")),
                "next_trigger_down": result.get("next_trigger_down"),
                "next_trigger_up": result.get("next_trigger_up"),
            }
        ),
    )
    return result


def run_hourly_volatility_resize(
    *,
    as_of_ms: Optional[int] = None,
    state: Optional[Dict[str, Any]] = None,
    notification_callback: NotificationCallback = None,
) -> Dict[str, Any]:
    """Re-evaluate open-position sizing on the hourly reporting boundary."""
    config = _load_strategy_config()
    symbol = str(config["symbol"])
    leverage = int(config["fixed_leverage"])
    trigger_pct_usdt = float(config["trigger_pct_usdt"])
    stop_loss_account_risk_pct = float(config["stop_loss_pct"])
    resolved_as_of_ms = _safe_int(as_of_ms, 0) or _current_time_ms()
    previous_state = dict(state or {})
    state_stop_risk_basis = _normalize_stop_risk_basis(previous_state.get("stop_risk_basis"))

    result: Dict[str, Any] = {
        "success": False,
        "symbol": symbol,
        "action": "init",
        "ai_triggered": False,
        "ai_decision": None,
        "ai_analysis": None,
        "cycle_dir": None,
        "current_price": None,
        "next_trigger_down": None,
        "next_trigger_up": None,
        "decision": None,
        "position": None,
        "position_before": None,
        "volatility_snapshot": None,
        "account_equity": None,
        "target_notional_usdt": None,
        "execution": None,
        "state_update": _build_state_update(
            previous_state=previous_state,
            trigger_pct_usdt=trigger_pct_usdt,
            ai_triggered=False,
            trigger_price=None,
            ai_decision=None,
            next_trigger_down=None,
            next_trigger_up=None,
            stop_risk_basis=state_stop_risk_basis,
        ),
        "stop_risk_basis": dict(state_stop_risk_basis) if state_stop_risk_basis is not None else None,
    }
    logger.info(
        "Hourly volatility resize started | %s",
        format_log_details(
            {
                "symbol": symbol,
                "as_of_ms": resolved_as_of_ms,
                "fixed_leverage": leverage,
                "previous_last_ai_decision": previous_state.get("last_ai_decision"),
                "previous_last_ai_trigger_price": previous_state.get("last_ai_trigger_price"),
            }
        ),
    )

    try:
        api_key, api_secret = get_binance_credentials()
    except ValueError as exc:
        result["action"] = "credentials_error"
        result["error"] = str(exc)
        return result

    positions = get_positions(api_key, api_secret)
    if positions is None:
        result["action"] = "positions_fetch_failed"
        return result

    universe_error = _validate_position_universe(positions, symbol)
    if universe_error:
        result["action"] = universe_error
        return result

    current_position = _extract_managed_position(positions, symbol)
    if current_position is not None:
        current_position_metrics = calculate_position_metrics(current_position)
        result["position_before"] = dict(current_position_metrics)
        result["position"] = dict(current_position_metrics)
        if not _stop_risk_basis_matches_position(state_stop_risk_basis, current_position):
            state_stop_risk_basis = None
            result["stop_risk_basis"] = None
        else:
            result["position_before"] = _enrich_position_with_stop_risk(
                result.get("position_before"),
                state_stop_risk_basis,
            )
            result["position"] = _enrich_position_with_stop_risk(result.get("position"), state_stop_risk_basis)
    else:
        state_stop_risk_basis = None
        result["stop_risk_basis"] = None

    reference_price_payload = get_reference_price(symbol)
    reference_price = _format_price((reference_price_payload or {}).get("price"))
    if reference_price is None:
        result["action"] = "reference_price_unavailable"
        return result
    result["current_price"] = reference_price

    market_context = _fetch_prompt_market_context(
        symbol=symbol,
        candle_count=int(config["ai_candle_count_per_timeframe"]),
        position_sizing_daily_sample_days=int(config["position_sizing_daily_sample_days"]),
        position_sizing_live_window_hours=int(config["position_sizing_live_window_hours"]),
        as_of_ms=resolved_as_of_ms,
    )
    volatility_snapshot = _calculate_volatility_snapshot(
        market_context["daily_position_sizing_candles"],
        daily_sample_days=int(config["position_sizing_daily_sample_days"]),
        live_window_candles=market_context["live_window_candles"],
        live_window_hours=int(config["position_sizing_live_window_hours"]),
        leverage=leverage,
        position_size_ratio_min=float(config["position_size_ratio_min"]),
        position_size_ratio_max=float(config["position_size_ratio_max"]),
    )
    result["volatility_snapshot"] = volatility_snapshot
    hourly_trigger_window = _build_trigger_levels(reference_price, trigger_pct_usdt)
    result["next_trigger_down"] = hourly_trigger_window["next_trigger_down"]
    result["next_trigger_up"] = hourly_trigger_window["next_trigger_up"]

    cycle_dir = _create_cycle_dir()
    result["cycle_dir"] = cycle_dir
    prompt_position_snapshot = _fetch_live_prompt_position_snapshot(
        api_key=api_key,
        api_secret=api_secret,
        symbol=symbol,
    )
    logger.info(
        "Hourly AI prompt context prepared | %s",
        format_log_details(
            {
                "symbol": symbol,
                "cycle_dir": cycle_dir,
                "timeframes": {key: len(value) for key, value in market_context["timeframes"].items()},
                "volatility_snapshot": volatility_snapshot,
                "prompt_position": _position_summary_for_log(prompt_position_snapshot),
            }
        ),
    )

    ai_analysis: Dict[str, Any] = {}
    _emit_notification(
        notification_callback,
        "ai_cycle_before",
        {
            "timestamp": _current_time_utc().isoformat(),
            "symbol": symbol,
            "cycle_dir": cycle_dir,
            "current_price": reference_price,
            "trigger_reason": "hourly_time_trigger",
            "trigger_price": reference_price,
            "next_trigger_down": result.get("next_trigger_down"),
            "next_trigger_up": result.get("next_trigger_up"),
            "position": result.get("position_before"),
            "previous_ai_decision": previous_state.get("last_ai_decision"),
            "volatility_snapshot": volatility_snapshot,
        },
    )
    ai_decision = evaluate_hakai_direction(
        cycle_dir=cycle_dir,
        symbol=symbol,
        reference_price=reference_price,
        timeframe_ohlcv=market_context["timeframes"],
        api_version=str(config["gemini_api_version"]),
        thinking_level=str(config["gemini_thinking_level"]),
        analysis_sink=ai_analysis,
        current_position_snapshot=prompt_position_snapshot,
    )
    if ai_decision is None:
        result["action"] = "ai_decision_failed"
        if ai_analysis:
            result["ai_analysis"] = dict(ai_analysis)
        _emit_notification(
            notification_callback,
            "ai_cycle_after",
            {
                "timestamp": _current_time_utc().isoformat(),
                "success": False,
                "symbol": symbol,
                "cycle_dir": cycle_dir,
                "current_price": reference_price,
                "trigger_reason": "hourly_time_trigger",
                "decision": None,
                "analysis": dict(ai_analysis),
                "volatility_snapshot": volatility_snapshot,
                "position": result.get("position_before"),
            },
        )
        _persist_cycle_output(result)
        return result

    ai_state_update = _build_ai_state_update_from_reference_price(
        previous_state=previous_state,
        trigger_pct_usdt=trigger_pct_usdt,
        reference_price=reference_price,
        ai_decision=ai_decision.decision,
        stop_risk_basis=state_stop_risk_basis,
    )
    result["ai_triggered"] = True
    result["ai_decision"] = ai_decision.decision
    result["ai_analysis"] = dict(ai_analysis)
    result["decision"] = ai_decision.decision
    result["state_update"] = ai_state_update
    _emit_notification(
        notification_callback,
        "ai_cycle_after",
        {
            "timestamp": _current_time_utc().isoformat(),
            "success": True,
            "symbol": symbol,
            "cycle_dir": cycle_dir,
            "current_price": reference_price,
            "trigger_reason": "hourly_time_trigger",
            "decision": ai_decision.decision,
            "analysis": dict(ai_analysis),
            "volatility_snapshot": volatility_snapshot,
            "position": result.get("position_before"),
        },
    )

    account_equity = _format_price(get_account_equity(api_key, api_secret))
    if account_equity is None:
        result["action"] = "account_equity_unavailable"
        _persist_cycle_output(result)
        return result
    result["account_equity"] = account_equity

    target_notional_usdt = _calculate_target_notional(
        account_equity=account_equity,
        target_margin_ratio=float(volatility_snapshot["target_margin_ratio"]),
        leverage=leverage,
    )
    result["target_notional_usdt"] = target_notional_usdt
    logger.info(
        "Hourly volatility position sizing computed | %s",
        format_log_details(
            {
                "symbol": symbol,
                "decision": ai_decision.decision,
                "account_equity": account_equity,
                "target_margin_ratio": volatility_snapshot.get("target_margin_ratio"),
                "target_effective_leverage": volatility_snapshot.get("target_effective_leverage"),
                "target_notional_usdt": target_notional_usdt,
                "sizing_summary": _format_percentile_sizing_summary(volatility_snapshot),
            }
        ),
    )

    applied_leverage = set_leverage(api_key, api_secret, symbol, leverage)
    if applied_leverage is None:
        result["action"] = "set_leverage_failed"
        _persist_cycle_output(result)
        return result

    leverage = int(applied_leverage)
    result["applied_leverage"] = leverage

    if current_position is None:
        execution_result = _place_new_direction_position(
            api_key=api_key,
            api_secret=api_secret,
            symbol=symbol,
            decision=ai_decision.decision,
            target_notional_usdt=target_notional_usdt,
            reference_price=reference_price,
            leverage=leverage,
        )
    else:
        execution_result = _rebalance_existing_position(
            api_key=api_key,
            api_secret=api_secret,
            symbol=symbol,
            current_position=current_position,
            decision=ai_decision.decision,
            target_notional_usdt=target_notional_usdt,
            reference_price=reference_price,
            leverage=leverage,
        )
    result["execution"] = execution_result
    logger.info(
        "Hourly volatility execution result captured | %s",
        format_log_details(
            {
                "symbol": symbol,
                "decision": ai_decision.decision,
                "execution": execution_result,
            }
        ),
    )
    if not bool(execution_result.get("success")):
        result["action"] = str(execution_result.get("action") or "execution_failed")
        _persist_cycle_output(result)
        return result

    result["success"] = True
    result["action"] = str(execution_result.get("action") or "executed")

    did_resize = result["action"] in {
        "scaled_in_position",
        "scaled_out_position",
        "reversed_position",
        "opened_new_position",
    }
    if did_resize:
        synced_position = _fetch_synced_position(
            api_key=api_key,
            api_secret=api_secret,
            symbol=symbol,
            previous_position=current_position,
            expected_action=result.get("action"),
        )
        result["position"] = calculate_position_metrics(synced_position) if synced_position is not None else None
        if synced_position is not None:
            state_stop_risk_basis = _build_stop_risk_basis_from_position(
                position=synced_position,
                account_equity=account_equity,
                stop_loss_account_risk_pct=stop_loss_account_risk_pct,
            )
            result["stop_risk_basis"] = dict(state_stop_risk_basis) if state_stop_risk_basis is not None else None
            synced_stop_result = _sync_account_risk_stop_loss(
                api_key=api_key,
                api_secret=api_secret,
                symbol=symbol,
                position=synced_position,
                stop_risk_basis=state_stop_risk_basis,
            )
            result["post_trade_stop_sync"] = synced_stop_result
            result["position"] = _apply_synced_stop_loss_to_position(result.get("position"), synced_stop_result)
            result["position"] = _enrich_position_with_stop_risk(result.get("position"), state_stop_risk_basis)
        else:
            state_stop_risk_basis = None
            result["stop_risk_basis"] = None
    result["state_update"] = _build_ai_state_update_from_reference_price(
        previous_state=previous_state,
        trigger_pct_usdt=trigger_pct_usdt,
        reference_price=reference_price,
        ai_decision=ai_decision.decision,
        stop_risk_basis=state_stop_risk_basis,
    )
    _persist_cycle_output(result)
    logger.info(
        "Hourly volatility resize completed | %s",
        format_log_details(
            {
                "symbol": symbol,
                "success": result["success"],
                "action": result["action"],
                "decision": result.get("decision"),
                "cycle_dir": result.get("cycle_dir"),
                "current_price": result.get("current_price"),
                "position_before": _position_summary_for_log(result.get("position_before")),
                "position_after": _position_summary_for_log(result.get("position")),
            }
        ),
    )
    return result


__all__ = [
    "DEFAULT_SYMBOL",
    "DEFAULT_TIMEFRAMES",
    "run_hourly_volatility_resize",
    "run_hakai_cycle",
]
