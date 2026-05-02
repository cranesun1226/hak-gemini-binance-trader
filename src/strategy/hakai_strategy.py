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

from src.ai.gemini_trader import (
    evaluate_hakai_entry_direction,
    evaluate_hakai_position_management,
)
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
from src.strategy.runtime_config import (
    DEFAULT_AI_PROMPT_CANDLE_COUNT,
    DEFAULT_AI_PROMPT_TIMEFRAME,
    DEFAULT_ENABLE_AUTO_POSITION,
    DEFAULT_GEMINI_API_VERSION,
    DEFAULT_INITIAL_POSITION_SIZE_RATIO,
    DEFAULT_POSITION_SIZING_DAILY_SAMPLE_DAYS,
    DEFAULT_POSITION_SIZING_LIVE_WINDOW_HOURS,
    DEFAULT_POSITION_SIZE_RATIO_MAX,
    DEFAULT_POSITION_SIZE_RATIO_MIN,
    DEFAULT_PROFIT_ACTIVATION_PCT,
    DEFAULT_TRIGGER_PCT_USDT,
    load_runtime_config,
)

logger = get_logger("hakai_strategy")

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
CONFIG_PATH = os.path.join(ROOT_DIR, "setting.yaml")
DB_DIR = os.path.join(ROOT_DIR, "db")
MAX_DB_CYCLE_DIRS = 20

DEFAULT_SYMBOL = "BTCUSDT"
_SUPPORTED_GEMINI_THINKING_LEVELS = {"low", "medium", "high"}
_DEFAULT_GEMINI_THINKING_LEVEL = "high"
_DEFAULT_GEMINI_API_VERSION = DEFAULT_GEMINI_API_VERSION
_TRIGGER_PRICE_DIGITS = 2
_INITIAL_ENTRY_VOLATILITY_COOLDOWN_THRESHOLD_PCT = 0.01
_INITIAL_ENTRY_VOLATILITY_COOLDOWN_CANDLE_COUNT = 2
NotificationCallback = Optional[Callable[[str, Dict[str, Any]], None]]
_DIRECTIONAL_AI_DECISIONS = {"LONG", "SHORT"}
_ENTRY_AI_DECISIONS = set(_DIRECTIONAL_AI_DECISIONS) | {"FLAT"}
_POSITION_MANAGEMENT_AI_DECISIONS = {"KEEP", "CLOSE"}
_VALID_AI_DECISIONS = _ENTRY_AI_DECISIONS | _POSITION_MANAGEMENT_AI_DECISIONS
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


def _normalize_ai_prompt_timeframe(value: Any) -> str:
    normalized = str(value or DEFAULT_AI_PROMPT_TIMEFRAME).strip().lower() or DEFAULT_AI_PROMPT_TIMEFRAME
    if normalized != DEFAULT_AI_PROMPT_TIMEFRAME:
        logger.warning(
            "Unsupported ai_prompt_timeframe=%s; forcing %s",
            value,
            DEFAULT_AI_PROMPT_TIMEFRAME,
        )
        return DEFAULT_AI_PROMPT_TIMEFRAME
    return DEFAULT_AI_PROMPT_TIMEFRAME


def _normalize_ratio(value: Any, default: float) -> float:
    parsed = _safe_float(value, default)
    if parsed is None or not math.isfinite(parsed) or parsed <= 0.0:
        parsed = default
    if 1.0 < parsed <= 100.0:
        parsed /= 100.0
    return float(parsed)


def _normalize_activation_ratio(value: Any, default: float) -> float:
    parsed_value = value
    if isinstance(value, str):
        parsed_value = value.strip()
        if parsed_value.endswith("%"):
            parsed_value = parsed_value[:-1]

    parsed = _safe_float(parsed_value, default)
    if parsed is None or not math.isfinite(parsed) or parsed <= 0.0:
        parsed = default
    if parsed >= 1.0 and parsed <= 100.0:
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
    if normalized == "minimal":
        logger.warning(
            "gemini_thinking_level=minimal is not supported by gemini-3.1-pro-preview; using low",
        )
        return "low"
    if normalized in _SUPPORTED_GEMINI_THINKING_LEVELS:
        return normalized
    logger.warning(
        "Unsupported gemini_thinking_level=%s for gemini-3.1-pro-preview; using %s",
        value,
        _DEFAULT_GEMINI_THINKING_LEVEL,
    )
    return _DEFAULT_GEMINI_THINKING_LEVEL


def _normalize_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return bool(value)

    normalized = str(value or "").strip().lower()
    if normalized in {"true", "1", "yes", "y", "on"}:
        return True
    if normalized in {"false", "0", "no", "n", "off", "flase"}:
        return False
    if value is not None:
        logger.warning("Unsupported boolean value=%s; using default=%s", value, default)
    return bool(default)


def _load_strategy_config() -> Dict[str, Any]:
    raw = load_runtime_config(CONFIG_PATH)

    symbol = str(raw.get("symbol") or DEFAULT_SYMBOL).strip().upper() or DEFAULT_SYMBOL
    if symbol != DEFAULT_SYMBOL:
        logger.warning("Forcing symbol=%s to BTCUSDT-only runtime", symbol)
        symbol = DEFAULT_SYMBOL

    initial_position_size_ratio = _normalize_ratio(
        raw.get("initial_position_size_ratio", DEFAULT_INITIAL_POSITION_SIZE_RATIO),
        DEFAULT_INITIAL_POSITION_SIZE_RATIO,
    )

    ai_prompt_candle_count = _normalize_positive_int(
        raw.get("ai_prompt_candle_count", raw.get("ai_candle_count_per_timeframe", DEFAULT_AI_PROMPT_CANDLE_COUNT)),
        DEFAULT_AI_PROMPT_CANDLE_COUNT,
    )

    return {
        "symbol": symbol,
        "cycle_interval_seconds": _normalize_positive_int(raw.get("cycle_interval_seconds", 60), 60),
        "trigger_pct_usdt": _normalize_trigger_percent(
            raw.get("trigger_pct_usdt", DEFAULT_TRIGGER_PCT_USDT),
            DEFAULT_TRIGGER_PCT_USDT,
        ),
        "fixed_leverage": _normalize_positive_int(raw.get("fixed_leverage", 10), 10),
        "stop_loss_pct": _normalize_ratio(raw.get("stop_loss_pct", 0.04), 0.04),
        "ai_prompt_timeframe": _normalize_ai_prompt_timeframe(
            raw.get("ai_prompt_timeframe", DEFAULT_AI_PROMPT_TIMEFRAME)
        ),
        "ai_prompt_candle_count": ai_prompt_candle_count,
        "initial_position_size_ratio": float(initial_position_size_ratio),
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


def _calculate_high_low_midpoint_range_pct(*, high: Any, low: Any, context: str) -> float:
    high_value = _format_price(high)
    low_value = _format_price(low)
    if high_value is None or low_value is None:
        raise ValueError(f"{context} contains invalid high/low values")
    if high_value < low_value:
        raise ValueError(f"{context} high is below low")

    midpoint = (float(high_value) + float(low_value)) / 2.0
    if not math.isfinite(midpoint) or midpoint <= 0.0:
        raise ValueError(f"{context} midpoint(high, low) is invalid")

    range_pct = (float(high_value) - float(low_value)) / midpoint
    if not math.isfinite(range_pct) or range_pct < 0.0:
        raise ValueError(f"{context} midpoint range pct is invalid")
    return float(range_pct)


def _calculate_max_single_candle_range_pct(
    candles: Sequence[Dict[str, Any]],
    *,
    context: str,
) -> float:
    relevant_candles = list(candles or [])
    if not relevant_candles:
        raise ValueError(f"{context} requires at least one candle")

    max_range_pct = 0.0
    for index, candle in enumerate(relevant_candles):
        candle_range_pct = _calculate_high_low_midpoint_range_pct(
            high=candle.get("high"),
            low=candle.get("low"),
            context=f"{context} candle[{index}]",
        )
        max_range_pct = max(float(max_range_pct), float(candle_range_pct))

    return float(max_range_pct)


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
    ai_prompt_timeframe: str,
    ai_prompt_candle_count: int,
    as_of_ms: Optional[int],
) -> Dict[str, Any]:
    resolved_ai_prompt_timeframe = _normalize_ai_prompt_timeframe(ai_prompt_timeframe)
    resolved_ai_prompt_candle_count = max(1, int(ai_prompt_candle_count))
    resolved_as_of_ms = _resolve_as_of_ms(as_of_ms)
    prompt_fetch_limit = resolved_ai_prompt_candle_count + 2
    raw_prompt_klines = fetch_klines(
        symbol,
        resolved_ai_prompt_timeframe,
        prompt_fetch_limit,
        as_of_ms=resolved_as_of_ms,
    )
    prompt_candles = parse_klines(raw_prompt_klines)
    closed_prompt_candles = _select_closed_candles(
        prompt_candles,
        interval=resolved_ai_prompt_timeframe,
        limit=resolved_ai_prompt_candle_count,
        as_of_ms=resolved_as_of_ms,
    )

    timeframe_payload: Dict[str, list[list[float]]] = {
        resolved_ai_prompt_timeframe: _serialize_ohlcv_rows(
            closed_prompt_candles,
            limit=resolved_ai_prompt_candle_count,
        )
    }

    return {
        "timeframes": timeframe_payload,
        "ai_prompt_timeframe": resolved_ai_prompt_timeframe,
        "ai_prompt_candle_count": resolved_ai_prompt_candle_count,
    }


def _fetch_position_sizing_live_window_candles(
    *,
    symbol: str,
    live_window_hours: int,
    as_of_ms: Optional[int],
) -> list[Dict[str, Any]]:
    resolved_live_window_hours = max(1, int(live_window_hours))
    resolved_as_of_ms = _resolve_as_of_ms(as_of_ms)
    raw_klines = fetch_klines(symbol, "1h", resolved_live_window_hours, as_of_ms=resolved_as_of_ms)
    candles = parse_klines(raw_klines)
    if len(candles) < resolved_live_window_hours:
        raise ValueError(
            f"not enough candles for {symbol} 1h lock threshold: have={len(candles)} need={resolved_live_window_hours}"
        )
    return candles[-resolved_live_window_hours:]


def _evaluate_initial_entry_volatility_cooldown(
    *,
    symbol: str,
    as_of_ms: Optional[int],
    threshold_pct: float = _INITIAL_ENTRY_VOLATILITY_COOLDOWN_THRESHOLD_PCT,
) -> Dict[str, Any]:
    resolved_as_of_ms = _resolve_as_of_ms(as_of_ms)
    raw_klines = fetch_klines(
        symbol,
        "1h",
        _INITIAL_ENTRY_VOLATILITY_COOLDOWN_CANDLE_COUNT,
        as_of_ms=resolved_as_of_ms,
    )
    candles = parse_klines(raw_klines)
    if len(candles) < _INITIAL_ENTRY_VOLATILITY_COOLDOWN_CANDLE_COUNT:
        raise ValueError(
            f"not enough 1h candles for initial-entry volatility cooldown: have={len(candles)} "
            f"need={_INITIAL_ENTRY_VOLATILITY_COOLDOWN_CANDLE_COUNT}"
        )

    relevant_candles = candles[-_INITIAL_ENTRY_VOLATILITY_COOLDOWN_CANDLE_COUNT:]
    candle_labels = ("previous_1h", "current_1h")
    cooldown_candles: list[Dict[str, Any]] = []
    breached_labels: list[str] = []
    max_range_pct = 0.0

    for candle_label, candle in zip(candle_labels, relevant_candles):
        range_pct = _calculate_high_low_midpoint_range_pct(
            high=candle.get("high"),
            low=candle.get("low"),
            context=f"initial entry volatility cooldown {candle_label}",
        )
        max_range_pct = max(float(max_range_pct), float(range_pct))
        if float(range_pct) > float(threshold_pct):
            breached_labels.append(candle_label)

        cooldown_candles.append(
            {
                "label": candle_label,
                "timestamp": _safe_int(candle.get("timestamp"), 0),
                "open": _format_price(candle.get("open")),
                "high": _format_price(candle.get("high")),
                "low": _format_price(candle.get("low")),
                "close": _format_price(candle.get("close")),
                "range_pct": float(range_pct),
            }
        )

    return {
        "threshold_pct": float(threshold_pct),
        "cooldown_active": bool(breached_labels),
        "breached_candle_labels": breached_labels,
        "max_range_pct": float(max_range_pct),
        "candles": cooldown_candles,
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


def _interpolate_rank_to_margin_ratio(
    *,
    rank_estimate: float,
    sample_size: int,
    ratio_min: float,
    ratio_max: float,
) -> float:
    resolved_sample_size = max(1, int(sample_size))
    clamped_rank = min(max(float(rank_estimate), 1.0), float(resolved_sample_size))
    if resolved_sample_size == 1:
        return float(ratio_max if float(rank_estimate) >= 1.0 else ratio_min)

    normalized_rank = (clamped_rank - 1.0) / float(resolved_sample_size - 1)
    return float(ratio_min + (normalized_rank * (ratio_max - ratio_min)))


def _calculate_volatility_snapshot(
    daily_candles: Sequence[Dict[str, Any]],
    *,
    daily_sample_days: int,
    live_window_candles: Sequence[Dict[str, Any]],
    live_window_hours: int,
    leverage: int,
    position_size_ratio_max: float,
) -> Dict[str, Any]:
    resolved_daily_sample_days = max(1, int(daily_sample_days))
    resolved_live_window_hours = max(1, int(live_window_hours))
    relevant_daily_candles = list(daily_candles or [])[-resolved_daily_sample_days:]
    relevant_live_window_candles = list(live_window_candles or [])[-resolved_live_window_hours:]
    if len(relevant_daily_candles) < resolved_daily_sample_days:
        raise ValueError("not enough daily candles for rank-based position sizing")
    if len(relevant_live_window_candles) < resolved_live_window_hours:
        raise ValueError("not enough 1h candles for live window calculation")

    max_single_candle_range_pct = _calculate_max_single_candle_range_pct(
        relevant_live_window_candles,
        context="live window activation threshold",
    )

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

    resolved_ratio_min = float(DEFAULT_POSITION_SIZE_RATIO_MIN)
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
    rank_estimate = float(percentile_position["rank_estimate"])
    # Keep the requested 25-rank sizing rule by default, while preserving linear interpolation
    # between configured bounds when operators choose different sample sizes or ratio limits.
    unclamped_margin_ratio = (
        ((rank_estimate * 4.0) - 2.0) * 0.01
        if len(daily_log_samples) == DEFAULT_POSITION_SIZING_DAILY_SAMPLE_DAYS
        and math.isclose(resolved_ratio_min, DEFAULT_POSITION_SIZE_RATIO_MIN, rel_tol=0.0, abs_tol=1e-9)
        and math.isclose(resolved_ratio_max, DEFAULT_POSITION_SIZE_RATIO_MAX, rel_tol=0.0, abs_tol=1e-9)
        else _interpolate_rank_to_margin_ratio(
            rank_estimate=rank_estimate,
            sample_size=len(daily_log_samples),
            ratio_min=resolved_ratio_min,
            ratio_max=resolved_ratio_max,
        )
    )
    target_margin_ratio = min(max(unclamped_margin_ratio, resolved_ratio_min), resolved_ratio_max)
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
        "live_window_max_single_candle_range_pct": float(max_single_candle_range_pct),
        "raw_percentile": raw_percentile,
        "percentile_rank_estimate": rank_estimate,
        "percentile_rank_floor": int(percentile_position["rank_floor"]),
        "percentile_rank_ceiling": int(percentile_position["rank_ceiling"]),
        "interpolation_weight": float(percentile_position["interpolation_weight"]),
        "sample_lower_value": float(percentile_position["sample_lower_value"]),
        "sample_upper_value": float(percentile_position["sample_upper_value"]),
        "percentile_location": str(percentile_position["location"]),
        "volatility_position_size_ratio_min": float(resolved_ratio_min),
        "position_size_ratio_max": float(resolved_ratio_max),
        "rank_interpolated_margin_ratio": float(unclamped_margin_ratio),
        "volatility_target_margin_ratio": float(target_margin_ratio),
        "volatility_target_effective_leverage": float(target_effective_leverage),
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


def _build_fixed_position_sizing(
    *,
    initial_position_size_ratio: float,
    leverage: int,
) -> Dict[str, float | str]:
    target_margin_ratio = float(initial_position_size_ratio)
    return {
        "position_sizing_mode": "fixed_ratio",
        "applied_target_margin_ratio": target_margin_ratio,
        "target_margin_ratio": target_margin_ratio,
        "applied_target_effective_leverage": float(target_margin_ratio * float(leverage)),
        "target_effective_leverage": float(target_margin_ratio * float(leverage)),
        "initial_position_size_ratio": target_margin_ratio,
    }


def _format_percentile_sizing_summary(snapshot: Optional[Dict[str, Any]]) -> str:
    payload = dict(snapshot or {})
    live_range_log = _safe_float(payload.get("live_range_log"), None)
    rank_estimate = _safe_float(payload.get("percentile_rank_estimate"), None)
    sample_size = _safe_int(payload.get("sample_size"), 0)
    volatility_margin_ratio = _safe_float(
        payload.get("volatility_target_margin_ratio"),
        _safe_float(payload.get("target_margin_ratio"), None),
    )
    applied_margin_ratio = _safe_float(payload.get("applied_target_margin_ratio"), volatility_margin_ratio)
    location = str(payload.get("percentile_location") or "").strip().lower()
    if (
        live_range_log is None
        or rank_estimate is None
        or sample_size <= 0
        or volatility_margin_ratio is None
    ):
        return "-"

    if location == "below_sample_range":
        location_label = "below"
    elif location == "above_sample_range":
        location_label = "above"
    else:
        location_label = "in-range"

    keep_current_position_size = bool(payload.get("keep_current_position_size"))
    enable_auto_position = _normalize_bool(payload.get("enable_auto_position", True), True)
    if keep_current_position_size:
        final_text = "bootstrap-hold"
    elif not enable_auto_position:
        final_text = f"fixed={applied_margin_ratio * 100.0:.2f}% (auto off)"
    else:
        final_text = f"final={applied_margin_ratio * 100.0:.2f}%"

    return (
        f"24h ln={live_range_log:.4f} | "
        f"rank={rank_estimate:.1f}/{sample_size} | "
        f"{location_label} | "
        f"vol={volatility_margin_ratio * 100.0:.2f}% | "
        f"{final_text}"
    )


def _empty_position_episode_state() -> Dict[str, Any]:
    return {
        "initial_entry_price": None,
        "initial_entry_direction": None,
        "position_sizing_activation_pct": None,
        "position_sizing_activation_price": None,
        "position_sizing_unlocked": False,
        "position_sizing_activated_at": None,
    }


def _normalize_position_episode_direction(value: Any) -> Optional[str]:
    normalized = str(value or "").strip().lower()
    return normalized if normalized in {"long", "short"} else None


def _normalize_position_episode_state(value: Any) -> Dict[str, Any]:
    payload = value if isinstance(value, dict) else {}
    initial_entry_price = _format_price(payload.get("initial_entry_price"))
    initial_entry_direction = _normalize_position_episode_direction(payload.get("initial_entry_direction"))
    if initial_entry_price is None or initial_entry_direction is None:
        return _empty_position_episode_state()

    activation_pct = _normalize_optional_ratio(payload.get("position_sizing_activation_pct"))
    activation_price = _format_price(payload.get("position_sizing_activation_price"))
    if activation_pct is None and activation_price is not None:
        derived_activation_pct = _calculate_directional_return_pct(
            direction=initial_entry_direction,
            entry_price=initial_entry_price,
            current_price=activation_price,
        )
        if derived_activation_pct is not None and derived_activation_pct > 0.0:
            activation_pct = float(derived_activation_pct)
    if activation_price is None and activation_pct is not None:
        activation_price = _resolve_profit_activation_price(
            direction=initial_entry_direction,
            entry_price=initial_entry_price,
            profit_activation_pct=activation_pct,
        )

    activated_at = str(payload.get("position_sizing_activated_at") or "").strip() or None
    position_sizing_unlocked = bool(payload.get("position_sizing_unlocked"))
    if not position_sizing_unlocked:
        activated_at = None

    return {
        "initial_entry_price": float(initial_entry_price),
        "initial_entry_direction": initial_entry_direction,
        "position_sizing_activation_pct": float(activation_pct) if activation_pct is not None else None,
        "position_sizing_activation_price": float(activation_price) if activation_price is not None else None,
        "position_sizing_unlocked": position_sizing_unlocked,
        "position_sizing_activated_at": activated_at,
    }


def _clear_position_episode_sizing_state(value: Any) -> Dict[str, Any]:
    normalized_state = _normalize_position_episode_state(value)
    if normalized_state["initial_entry_price"] is None:
        return normalized_state

    cleared_state = dict(normalized_state)
    cleared_state["position_sizing_activation_pct"] = None
    cleared_state["position_sizing_activation_price"] = None
    cleared_state["position_sizing_unlocked"] = False
    cleared_state["position_sizing_activated_at"] = None
    return cleared_state


def _build_position_episode_state_from_position(position: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    metrics = calculate_position_metrics(position)
    direction = _normalize_position_episode_direction(metrics.get("direction"))
    entry_price = _format_price(metrics.get("entry_price"))
    if direction is None or entry_price is None:
        return _empty_position_episode_state()
    return {
        "initial_entry_price": float(entry_price),
        "initial_entry_direction": direction,
        "position_sizing_activation_pct": None,
        "position_sizing_activation_price": None,
        "position_sizing_unlocked": False,
        "position_sizing_activated_at": None,
    }


def _reconcile_position_episode_state(
    *,
    previous_state: Optional[Dict[str, Any]],
    current_position: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    live_episode_state = _build_position_episode_state_from_position(current_position)
    if live_episode_state["initial_entry_price"] is None:
        return {
            "position_episode_state": _empty_position_episode_state(),
            "bootstrapped": False,
        }

    previous_episode_state = _normalize_position_episode_state(previous_state)
    if (
        previous_episode_state["initial_entry_price"] is None
        or previous_episode_state["initial_entry_direction"] != live_episode_state["initial_entry_direction"]
    ):
        return {
            "position_episode_state": live_episode_state,
            "bootstrapped": True,
        }

    return {
        "position_episode_state": dict(previous_episode_state),
        "bootstrapped": False,
    }


def _calculate_directional_return_pct(
    *,
    direction: str,
    entry_price: float,
    current_price: float,
) -> Optional[float]:
    normalized_direction = _normalize_position_episode_direction(direction)
    normalized_entry_price = _format_price(entry_price)
    normalized_current_price = _format_price(current_price)
    if (
        normalized_direction is None
        or normalized_entry_price is None
        or normalized_current_price is None
    ):
        return None

    if normalized_direction == "long":
        return float((normalized_current_price / normalized_entry_price) - 1.0)
    return float((normalized_entry_price - normalized_current_price) / normalized_entry_price)


def _resolve_profit_activation_price(
    *,
    direction: str,
    entry_price: float,
    profit_activation_pct: float,
) -> Optional[float]:
    normalized_direction = _normalize_position_episode_direction(direction)
    normalized_entry_price = _format_price(entry_price)
    normalized_profit_activation_pct = _normalize_optional_ratio(profit_activation_pct)
    if (
        normalized_direction is None
        or normalized_entry_price is None
        or normalized_profit_activation_pct is None
    ):
        return None

    if normalized_direction == "long":
        return float(normalized_entry_price * (1.0 + normalized_profit_activation_pct))
    return float(normalized_entry_price * (1.0 - normalized_profit_activation_pct))


def _resolve_position_sizing_activation_pct(
    *,
    position_episode_state: Any,
    fallback_profit_activation_pct: float,
) -> Optional[float]:
    normalized_state = _normalize_position_episode_state(position_episode_state)
    stored_activation_pct = _normalize_optional_ratio(normalized_state.get("position_sizing_activation_pct"))
    if stored_activation_pct is not None:
        return float(stored_activation_pct)

    fallback_activation_pct = _normalize_optional_ratio(fallback_profit_activation_pct)
    return float(fallback_activation_pct) if fallback_activation_pct is not None else None


def _resolve_position_sizing_activation_price(
    *,
    position_episode_state: Any,
    fallback_profit_activation_pct: float,
) -> Optional[float]:
    normalized_state = _normalize_position_episode_state(position_episode_state)
    stored_activation_price = _format_price(normalized_state.get("position_sizing_activation_price"))
    if stored_activation_price is not None:
        return float(stored_activation_price)

    activation_pct = _resolve_position_sizing_activation_pct(
        position_episode_state=normalized_state,
        fallback_profit_activation_pct=fallback_profit_activation_pct,
    )
    if (
        normalized_state["initial_entry_price"] is None
        or normalized_state["initial_entry_direction"] not in {"long", "short"}
        or activation_pct is None
    ):
        return None

    return _resolve_profit_activation_price(
        direction=normalized_state["initial_entry_direction"],
        entry_price=normalized_state["initial_entry_price"],
        profit_activation_pct=activation_pct,
    )


def _refresh_position_episode_lock_threshold(
    *,
    position_episode_state: Any,
    profit_activation_pct: float,
    live_window_candles: Optional[Sequence[Dict[str, Any]]] = None,
    live_window_hours: Optional[int] = None,
) -> Dict[str, Any]:
    normalized_state = _normalize_position_episode_state(position_episode_state)
    if normalized_state["initial_entry_price"] is None or normalized_state["position_sizing_unlocked"]:
        return normalized_state

    configured_activation_floor_pct = _normalize_optional_ratio(profit_activation_pct)
    stored_activation_pct = _normalize_optional_ratio(normalized_state.get("position_sizing_activation_pct"))
    resolved_activation_pct = (
        float(configured_activation_floor_pct)
        if configured_activation_floor_pct is not None
        else float(stored_activation_pct)
        if stored_activation_pct is not None
        else None
    )
    if resolved_activation_pct is None:
        return normalized_state

    if live_window_candles is not None:
        resolved_live_window_hours = max(1, int(live_window_hours or len(live_window_candles)))
        relevant_live_window_candles = list(live_window_candles or [])[-resolved_live_window_hours:]
        if len(relevant_live_window_candles) < resolved_live_window_hours:
            raise ValueError("not enough 1h candles for profit activation threshold refresh")
        live_window_max_single_candle_range_pct = _calculate_max_single_candle_range_pct(
            relevant_live_window_candles,
            context="profit activation threshold refresh",
        )
        resolved_activation_pct = max(
            float(configured_activation_floor_pct)
            if configured_activation_floor_pct is not None
            else float(resolved_activation_pct),
            float(live_window_max_single_candle_range_pct),
        )
    elif stored_activation_pct is not None:
        resolved_activation_pct = float(stored_activation_pct)

    refreshed_state = dict(normalized_state)
    refreshed_state["position_sizing_activation_pct"] = float(resolved_activation_pct)
    refreshed_state["position_sizing_activation_price"] = _resolve_profit_activation_price(
        direction=normalized_state["initial_entry_direction"],
        entry_price=normalized_state["initial_entry_price"],
        profit_activation_pct=resolved_activation_pct,
    )
    return refreshed_state


def _update_position_episode_unlock_state(
    *,
    position_episode_state: Any,
    current_position: Optional[Dict[str, Any]],
    reference_price: float,
    profit_activation_pct: float,
) -> Dict[str, Any]:
    normalized_state = _normalize_position_episode_state(position_episode_state)
    if normalized_state["initial_entry_price"] is None or normalized_state["position_sizing_unlocked"]:
        return normalized_state

    metrics = calculate_position_metrics(current_position)
    current_direction = _normalize_position_episode_direction(metrics.get("direction"))
    if current_direction != normalized_state["initial_entry_direction"]:
        return normalized_state

    activation_pct = _resolve_position_sizing_activation_pct(
        position_episode_state=normalized_state,
        fallback_profit_activation_pct=profit_activation_pct,
    )
    if activation_pct is None:
        return normalized_state

    current_return_pct = _calculate_directional_return_pct(
        direction=current_direction,
        entry_price=normalized_state["initial_entry_price"],
        current_price=reference_price,
    )
    if current_return_pct is None or current_return_pct < float(activation_pct):
        return normalized_state

    unlocked_state = dict(normalized_state)
    unlocked_state["position_sizing_activation_pct"] = float(activation_pct)
    unlocked_state["position_sizing_activation_price"] = _resolve_position_sizing_activation_price(
        position_episode_state=normalized_state,
        fallback_profit_activation_pct=profit_activation_pct,
    )
    unlocked_state["position_sizing_unlocked"] = True
    unlocked_state["position_sizing_activated_at"] = _current_time_utc().isoformat()
    return unlocked_state

def _build_position_sizing_plan(
    *,
    volatility_snapshot: Dict[str, Any],
    current_position: Optional[Dict[str, Any]],
    decision: str,
    reference_price: float,
    leverage: int,
    initial_position_size_ratio: float,
    position_size_ratio_max: float,
    enable_auto_position: bool,
    profit_activation_pct: float,
    position_episode_state: Any,
    bootstrap_protected: bool,
) -> Dict[str, Any]:
    normalized_decision = _normalize_ai_decision(decision)
    if normalized_decision not in _DIRECTIONAL_AI_DECISIONS:
        raise ValueError("decision must be LONG or SHORT for sizing")

    volatility_margin_ratio = _safe_float(
        volatility_snapshot.get("volatility_target_margin_ratio"),
        _safe_float(volatility_snapshot.get("target_margin_ratio"), None),
    )
    if volatility_margin_ratio is None:
        raise ValueError("volatility snapshot is missing target margin ratio")

    desired_direction = "long" if normalized_decision == "LONG" else "short"
    current_metrics = calculate_position_metrics(current_position)
    current_direction = _normalize_position_episode_direction(current_metrics.get("direction"))
    current_notional_usdt = abs(_safe_float(current_metrics.get("position_value"), 0.0) or 0.0)
    normalized_episode_state = _normalize_position_episode_state(position_episode_state)
    auto_position_enabled = bool(enable_auto_position)

    activation_price = None
    activation_pct = None
    current_return_pct = None
    if (
        auto_position_enabled
        and normalized_episode_state["initial_entry_price"] is not None
        and normalized_episode_state["initial_entry_direction"] == desired_direction
    ):
        activation_pct = _resolve_position_sizing_activation_pct(
            position_episode_state=normalized_episode_state,
            fallback_profit_activation_pct=profit_activation_pct,
        )
        activation_price = _resolve_position_sizing_activation_price(
            position_episode_state=normalized_episode_state,
            fallback_profit_activation_pct=profit_activation_pct,
        )
        current_return_pct = _calculate_directional_return_pct(
            direction=desired_direction,
            entry_price=normalized_episode_state["initial_entry_price"],
            current_price=reference_price,
        )

    keep_current_position_size = bool(
        auto_position_enabled
        and bootstrap_protected
        and current_direction in {"long", "short"}
        and current_direction == desired_direction
    )
    is_same_direction_position = current_direction in {"long", "short"} and current_direction == desired_direction
    unlocked = bool(normalized_episode_state["position_sizing_unlocked"]) if auto_position_enabled else False

    if not is_same_direction_position:
        applied_margin_ratio = float(initial_position_size_ratio)
        position_sizing_mode = (
            "reversal_initial_fixed" if current_direction in {"long", "short"} else "initial_entry_fixed"
        )
    elif keep_current_position_size:
        applied_margin_ratio = float(initial_position_size_ratio)
        position_sizing_mode = "bootstrap_hold"
    elif not auto_position_enabled:
        applied_margin_ratio = float(initial_position_size_ratio)
        position_sizing_mode = "auto_position_disabled"
    elif unlocked:
        applied_margin_ratio = max(float(initial_position_size_ratio), float(volatility_margin_ratio))
        position_sizing_mode = (
            "volatility_unlocked"
            if applied_margin_ratio > float(initial_position_size_ratio)
            else "volatility_unlocked_floor"
        )
    else:
        applied_margin_ratio = float(initial_position_size_ratio)
        position_sizing_mode = "profit_gate_locked"

    applied_margin_ratio = min(
        max(float(applied_margin_ratio), float(initial_position_size_ratio)),
        float(position_size_ratio_max),
    )
    return {
        "desired_direction": desired_direction,
        "current_direction": current_direction,
        "current_notional_usdt": float(current_notional_usdt),
        "volatility_target_margin_ratio": float(volatility_margin_ratio),
        "applied_target_margin_ratio": float(applied_margin_ratio),
        "applied_target_effective_leverage": float(applied_margin_ratio * float(leverage)),
        "initial_position_size_ratio": float(initial_position_size_ratio),
        "position_size_ratio_max": float(position_size_ratio_max),
        "enable_auto_position": auto_position_enabled,
        "profit_activation_pct": float(profit_activation_pct),
        "position_sizing_activation_pct": float(activation_pct) if activation_pct is not None else None,
        "position_sizing_mode": position_sizing_mode,
        "position_sizing_unlocked": unlocked,
        "position_sizing_activation_price": activation_price,
        "position_sizing_current_return_pct": current_return_pct,
        "keep_current_position_size": keep_current_position_size,
        "bootstrap_protected": bool(bootstrap_protected),
    }


def _annotate_volatility_snapshot_with_position_sizing(
    *,
    volatility_snapshot: Dict[str, Any],
    position_sizing_plan: Dict[str, Any],
) -> Dict[str, Any]:
    annotated_snapshot = dict(volatility_snapshot or {})
    volatility_margin_ratio = _safe_float(
        annotated_snapshot.get("volatility_target_margin_ratio"),
        _safe_float(annotated_snapshot.get("target_margin_ratio"), None),
    )
    volatility_effective_leverage = _safe_float(
        annotated_snapshot.get("volatility_target_effective_leverage"),
        _safe_float(annotated_snapshot.get("target_effective_leverage"), None),
    )
    if volatility_margin_ratio is not None:
        annotated_snapshot["volatility_target_margin_ratio"] = float(volatility_margin_ratio)
    if volatility_effective_leverage is not None:
        annotated_snapshot["volatility_target_effective_leverage"] = float(volatility_effective_leverage)

    applied_margin_ratio = _safe_float(position_sizing_plan.get("applied_target_margin_ratio"), None)
    applied_effective_leverage = _safe_float(position_sizing_plan.get("applied_target_effective_leverage"), None)
    if applied_margin_ratio is not None:
        annotated_snapshot["applied_target_margin_ratio"] = float(applied_margin_ratio)
        annotated_snapshot["target_margin_ratio"] = float(applied_margin_ratio)
    if applied_effective_leverage is not None:
        annotated_snapshot["applied_target_effective_leverage"] = float(applied_effective_leverage)
        annotated_snapshot["target_effective_leverage"] = float(applied_effective_leverage)

    for key in (
        "initial_position_size_ratio",
        "position_size_ratio_max",
        "enable_auto_position",
        "profit_activation_pct",
        "position_sizing_activation_pct",
        "position_sizing_mode",
        "position_sizing_unlocked",
        "position_sizing_activation_price",
        "position_sizing_current_return_pct",
        "keep_current_position_size",
        "bootstrap_protected",
        "current_notional_usdt",
    ):
        if key in position_sizing_plan:
            annotated_snapshot[key] = position_sizing_plan.get(key)
    return annotated_snapshot


def _build_pre_ai_display_volatility_snapshot(
    *,
    volatility_snapshot: Dict[str, Any],
    current_position: Optional[Dict[str, Any]],
    position_episode_state: Any,
    initial_position_size_ratio: float,
    enable_auto_position: bool,
    profit_activation_pct: float,
    bootstrap_protected: bool,
) -> Dict[str, Any]:
    display_snapshot = dict(volatility_snapshot or {})
    normalized_episode_state = _normalize_position_episode_state(position_episode_state)
    current_metrics = calculate_position_metrics(current_position)
    current_direction = _normalize_position_episode_direction(current_metrics.get("direction"))
    auto_position_enabled = bool(enable_auto_position)

    keep_current_position_size = bool(
        auto_position_enabled
        and bootstrap_protected
        and current_direction in {"long", "short"}
    )
    unlocked = bool(normalized_episode_state["position_sizing_unlocked"]) if auto_position_enabled else False
    activation_price = None
    activation_pct = None
    if (
        auto_position_enabled
        and normalized_episode_state["initial_entry_price"] is not None
        and normalized_episode_state["initial_entry_direction"] in {"long", "short"}
    ):
        activation_pct = _resolve_position_sizing_activation_pct(
            position_episode_state=normalized_episode_state,
            fallback_profit_activation_pct=profit_activation_pct,
        )
        activation_price = _resolve_position_sizing_activation_price(
            position_episode_state=normalized_episode_state,
            fallback_profit_activation_pct=profit_activation_pct,
        )

    display_snapshot["initial_position_size_ratio"] = float(initial_position_size_ratio)
    display_snapshot["enable_auto_position"] = auto_position_enabled
    display_snapshot["profit_activation_pct"] = float(profit_activation_pct)
    display_snapshot["position_sizing_unlocked"] = unlocked
    display_snapshot["keep_current_position_size"] = keep_current_position_size
    if activation_pct is not None:
        display_snapshot["position_sizing_activation_pct"] = float(activation_pct)
    if activation_price is not None:
        display_snapshot["position_sizing_activation_price"] = activation_price

    if current_position is None or keep_current_position_size or not unlocked:
        applied_margin_ratio = float(initial_position_size_ratio)
        display_snapshot["applied_target_margin_ratio"] = applied_margin_ratio
        display_snapshot["target_margin_ratio"] = applied_margin_ratio

    return display_snapshot


def _resolve_target_notional_usdt(
    *,
    account_equity: float,
    leverage: int,
    current_position: Optional[Dict[str, Any]],
    position_sizing_plan: Dict[str, Any],
) -> float:
    if bool(position_sizing_plan.get("keep_current_position_size")):
        current_notional_usdt = abs(_safe_float(position_sizing_plan.get("current_notional_usdt"), 0.0) or 0.0)
        if current_notional_usdt > 0.0:
            return float(current_notional_usdt)

    return _calculate_target_notional(
        account_equity=account_equity,
        target_margin_ratio=float(position_sizing_plan["applied_target_margin_ratio"]),
        leverage=leverage,
    )


def _resolve_post_trade_position_episode_state(
    *,
    previous_position_episode_state: Any,
    previous_position: Optional[Dict[str, Any]],
    current_position: Optional[Dict[str, Any]],
    execution_action: Optional[str],
) -> Dict[str, Any]:
    current_episode_state = _build_position_episode_state_from_position(current_position)
    if current_episode_state["initial_entry_price"] is None:
        return _empty_position_episode_state()

    previous_metrics = calculate_position_metrics(previous_position)
    previous_direction = _normalize_position_episode_direction(previous_metrics.get("direction"))
    current_direction = current_episode_state["initial_entry_direction"]
    normalized_action = str(execution_action or "").strip()
    is_new_episode = (
        normalized_action in {"opened_new_position", "reversed_position"}
        or previous_direction not in {"long", "short"}
        or previous_direction != current_direction
    )
    if is_new_episode:
        return current_episode_state

    normalized_previous_state = _normalize_position_episode_state(previous_position_episode_state)
    if normalized_previous_state["initial_entry_direction"] == current_direction:
        return normalized_previous_state
    return current_episode_state


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
    if normalized_action in {"opened_new_position", "closed_and_opened_position"}:
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
    position_episode_state: Any = _STATE_UNSET,
    stop_risk_basis: Any = _STATE_UNSET,
) -> Dict[str, Any]:
    state_update = dict(previous_state or {})
    state_update.pop("last_ai_trigger_round_price", None)
    state_update.pop("initial_entry_price", None)
    state_update.pop("initial_entry_direction", None)
    state_update.pop("position_sizing_activation_pct", None)
    state_update.pop("position_sizing_activation_price", None)
    state_update.pop("position_sizing_unlocked", None)
    state_update.pop("position_sizing_activated_at", None)
    state_update.pop("last_hourly_resize_slot", None)
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
    del position_episode_state
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
    position_episode_state: Any = _STATE_UNSET,
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
        position_episode_state=position_episode_state,
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


def _close_existing_position_for_ai_close(
    *,
    api_key: str,
    api_secret: str,
    symbol: str,
    current_position: Dict[str, Any],
) -> Dict[str, Any]:
    current_metrics = calculate_position_metrics(current_position)
    current_direction = str(current_metrics.get("direction") or "").strip().lower()
    current_side = str(current_metrics.get("side") or "").strip()
    current_size = abs(_safe_float(current_metrics.get("size"), 0.0) or 0.0)

    if current_direction not in ("long", "short") or not current_side or current_size <= 0.0:
        return {
            "success": False,
            "action": "invalid_existing_position",
        }

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
            "action": "close_position_failed",
        }

    close_propagated = wait_for_close_propagation(
        api_key,
        api_secret,
        [symbol],
        context="ai_close_position",
    )
    cancel_all_orders(api_key, api_secret, symbol)
    if not close_propagated:
        return {
            "success": False,
            "action": "close_position_failed",
            "closed_direction": current_direction,
            "closed_qty": current_size,
            "close_propagated": False,
        }
    return {
        "success": True,
        "action": "closed_position",
        "closed_direction": current_direction,
        "closed_qty": current_size,
        "close_propagated": close_propagated,
    }


def _determine_ai_trigger(
    *,
    has_position: bool,
    current_price: float,
    last_ai_trigger_price: Optional[float],
    trigger_pct_usdt: float,
    last_ai_decision: Optional[str] = None,
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

    normalized_last_ai_decision = _normalize_ai_decision(last_ai_decision)

    if not has_position and normalized_last_ai_decision == "FLAT" and has_valid_trigger_window:
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
    initial_position_size_ratio = float(config["initial_position_size_ratio"])
    stop_loss_account_risk_pct = float(config["stop_loss_pct"])
    resolved_as_of_ms = _safe_int(as_of_ms, 0) or _current_time_ms()
    raw_previous_state = dict(state or {})
    previous_state, trigger_pct_state_refreshed = _align_state_trigger_percent(
        raw_previous_state,
        trigger_pct_usdt,
    )
    state_stop_risk_basis = _normalize_stop_risk_basis(previous_state.get("stop_risk_basis"))
    fixed_position_sizing = _build_fixed_position_sizing(
        initial_position_size_ratio=initial_position_size_ratio,
        leverage=leverage,
    )

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
        "position_sizing": dict(fixed_position_sizing),
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
                "initial_position_size_ratio": initial_position_size_ratio,
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
    result["state_update"] = _build_state_update(
        previous_state=previous_state,
        trigger_pct_usdt=trigger_pct_usdt,
        ai_triggered=False,
        trigger_price=None,
        ai_decision=None,
        next_trigger_down=None,
        next_trigger_up=None,
        stop_risk_basis=state_stop_risk_basis,
    )
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
        last_ai_decision=previous_state.get("last_ai_decision"),
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
        ai_prompt_timeframe=str(config["ai_prompt_timeframe"]),
        ai_prompt_candle_count=int(config["ai_prompt_candle_count"]),
        as_of_ms=resolved_as_of_ms,
    )
    logger.info(
        "Market context prepared for AI | %s",
        format_log_details(
            {
                "symbol": symbol,
                "cycle_dir": cycle_dir,
                "ai_prompt_timeframes": {key: len(value) for key, value in market_context["timeframes"].items()},
                "ai_prompt_timeframe": market_context.get("ai_prompt_timeframe"),
                "ai_prompt_candle_count": market_context.get("ai_prompt_candle_count"),
                "position_sizing": fixed_position_sizing,
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

    decision_mode = "position" if current_position is not None else "entry"
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
            "position_sizing": dict(fixed_position_sizing),
            "decision_mode": decision_mode,
        },
    )

    entry_decision_value: Optional[str] = None
    execution_previous_position = current_position
    closed_before_entry = False

    if current_position is None:
        ai_decision = evaluate_hakai_entry_direction(
            cycle_dir=cycle_dir,
            symbol=symbol,
            reference_price=reference_price,
            timeframe_ohlcv=market_context["timeframes"],
            api_version=str(config["gemini_api_version"]),
            thinking_level=str(config["gemini_thinking_level"]),
            analysis_sink=ai_analysis,
            current_position_snapshot=None,
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
                    "position_sizing": dict(fixed_position_sizing),
                    "decision_mode": "entry",
                },
            )
            _persist_cycle_output(result)
            logger.error(
                "AI entry decision failed | %s",
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
            "AI entry decision received | %s",
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
                "position_sizing": dict(fixed_position_sizing),
                "position": result.get("position_before"),
                "decision_mode": "entry",
            },
        )

        if ai_decision.decision == "FLAT":
            result["success"] = True
            result["action"] = "flat_no_entry"
            result["state_update"] = _build_state_update(
                previous_state=previous_state,
                trigger_pct_usdt=trigger_pct_usdt,
                ai_triggered=True,
                trigger_price=trigger_price,
                ai_decision=ai_decision.decision,
                next_trigger_down=_normalize_trigger_price(trigger_info.get("next_trigger_down")),
                next_trigger_up=_normalize_trigger_price(trigger_info.get("next_trigger_up")),
                stop_risk_basis=None,
            )
            _persist_cycle_output(result)
            logger.info(
                "AI chose FLAT; no entry order submitted | %s",
                format_log_details(
                    {
                        "symbol": symbol,
                        "cycle_dir": cycle_dir,
                        "current_price": reference_price,
                        "next_trigger_down": result.get("next_trigger_down"),
                        "next_trigger_up": result.get("next_trigger_up"),
                    }
                ),
            )
            return result

        entry_decision_value = ai_decision.decision

    else:
        position_decision = evaluate_hakai_position_management(
            cycle_dir=cycle_dir,
            symbol=symbol,
            reference_price=reference_price,
            timeframe_ohlcv=market_context["timeframes"],
            api_version=str(config["gemini_api_version"]),
            thinking_level=str(config["gemini_thinking_level"]),
            analysis_sink=ai_analysis,
            current_position_snapshot=prompt_position_snapshot or result.get("position_before"),
        )
        if position_decision is None:
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
                    "position_sizing": dict(fixed_position_sizing),
                    "decision_mode": "position",
                },
            )
            _persist_cycle_output(result)
            logger.error(
                "AI position-management decision failed | %s",
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
        result["ai_decision"] = position_decision.decision
        result["position_ai_decision"] = position_decision.decision
        result["ai_analysis"] = dict(ai_analysis)
        logger.info(
            "AI position-management decision received | %s",
            format_log_details(
                {
                    "symbol": symbol,
                    "cycle_dir": cycle_dir,
                    "decision": position_decision.decision,
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
                "decision": position_decision.decision,
                "analysis": dict(ai_analysis),
                "position_sizing": dict(fixed_position_sizing),
                "position": result.get("position_before"),
                "decision_mode": "position",
            },
        )

        if position_decision.decision == "KEEP":
            result["success"] = True
            result["action"] = "kept_position_by_ai"
            result["state_update"] = _build_state_update(
                previous_state=previous_state,
                trigger_pct_usdt=trigger_pct_usdt,
                ai_triggered=True,
                trigger_price=trigger_price,
                ai_decision=position_decision.decision,
                next_trigger_down=_normalize_trigger_price(trigger_info.get("next_trigger_down")),
                next_trigger_up=_normalize_trigger_price(trigger_info.get("next_trigger_up")),
                stop_risk_basis=state_stop_risk_basis,
            )
            _persist_cycle_output(result)
            logger.info(
                "AI chose KEEP; existing position maintained | %s",
                format_log_details(
                    {
                        "symbol": symbol,
                        "cycle_dir": cycle_dir,
                        "position": _position_summary_for_log(result.get("position")),
                    }
                ),
            )
            return result

        close_result = _close_existing_position_for_ai_close(
            api_key=api_key,
            api_secret=api_secret,
            symbol=symbol,
            current_position=current_position,
        )
        result["close_execution"] = close_result
        if not bool(close_result.get("success")):
            result["action"] = str(close_result.get("action") or "close_position_failed")
            result["state_update"] = _build_state_update(
                previous_state=previous_state,
                trigger_pct_usdt=trigger_pct_usdt,
                ai_triggered=True,
                trigger_price=trigger_price,
                ai_decision=position_decision.decision,
                next_trigger_down=_normalize_trigger_price(trigger_info.get("next_trigger_down")),
                next_trigger_up=_normalize_trigger_price(trigger_info.get("next_trigger_up")),
                stop_risk_basis=state_stop_risk_basis,
            )
            _persist_cycle_output(result)
            logger.error(
                "Failed to close position after AI CLOSE decision | %s",
                format_log_details(
                    {
                        "symbol": symbol,
                        "cycle_dir": cycle_dir,
                        "close_execution": close_result,
                    }
                ),
            )
            return result

        closed_before_entry = True
        current_position = None
        state_stop_risk_basis = None
        result["position"] = None
        result["stop_risk_basis"] = None

        entry_analysis: Dict[str, Any] = {}
        entry_decision = evaluate_hakai_entry_direction(
            cycle_dir=cycle_dir,
            symbol=symbol,
            reference_price=reference_price,
            timeframe_ohlcv=market_context["timeframes"],
            api_version=str(config["gemini_api_version"]),
            thinking_level=str(config["gemini_thinking_level"]),
            analysis_sink=entry_analysis,
            current_position_snapshot=None,
        )
        result["entry_ai_analysis"] = dict(entry_analysis)
        if entry_decision is None:
            result["action"] = "ai_decision_failed"
            result["state_update"] = _build_state_update(
                previous_state=previous_state,
                trigger_pct_usdt=trigger_pct_usdt,
                ai_triggered=True,
                trigger_price=trigger_price,
                ai_decision=position_decision.decision,
                next_trigger_down=_normalize_trigger_price(trigger_info.get("next_trigger_down")),
                next_trigger_up=_normalize_trigger_price(trigger_info.get("next_trigger_up")),
                stop_risk_basis=None,
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
                    "analysis": dict(entry_analysis),
                    "position_sizing": dict(fixed_position_sizing),
                    "position": None,
                    "decision_mode": "entry_after_close",
                },
            )
            _persist_cycle_output(result)
            logger.error(
                "AI entry decision failed after CLOSE decision | %s",
                format_log_details(
                    {
                        "symbol": symbol,
                        "cycle_dir": cycle_dir,
                        "current_price": reference_price,
                        "trigger_reason": trigger_info.get("reason"),
                        "entry_ai_analysis": entry_analysis,
                    }
                ),
            )
            return result

        result["entry_ai_decision"] = entry_decision.decision
        result["ai_decision"] = entry_decision.decision
        result["ai_analysis"] = dict(entry_analysis)
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
                "decision": entry_decision.decision,
                "analysis": dict(entry_analysis),
                "position_sizing": dict(fixed_position_sizing),
                "position": None,
                "decision_mode": "entry_after_close",
            },
        )

        if entry_decision.decision == "FLAT":
            result["success"] = True
            result["action"] = "closed_to_flat"
            result["state_update"] = _build_state_update(
                previous_state=previous_state,
                trigger_pct_usdt=trigger_pct_usdt,
                ai_triggered=True,
                trigger_price=trigger_price,
                ai_decision=entry_decision.decision,
                next_trigger_down=_normalize_trigger_price(trigger_info.get("next_trigger_down")),
                next_trigger_up=_normalize_trigger_price(trigger_info.get("next_trigger_up")),
                stop_risk_basis=None,
            )
            _persist_cycle_output(result)
            logger.info(
                "AI chose FLAT after CLOSE decision; no new entry submitted | %s",
                format_log_details(
                    {
                        "symbol": symbol,
                        "cycle_dir": cycle_dir,
                        "close_execution": close_result,
                    }
                ),
            )
            return result

        entry_decision_value = entry_decision.decision

    if entry_decision_value not in _DIRECTIONAL_AI_DECISIONS:
        result["action"] = "invalid_ai_decision"
        result["state_update"] = _build_state_update(
            previous_state=previous_state,
            trigger_pct_usdt=trigger_pct_usdt,
            ai_triggered=True,
            trigger_price=trigger_price,
            ai_decision=entry_decision_value,
            next_trigger_down=_normalize_trigger_price(trigger_info.get("next_trigger_down")),
            next_trigger_up=_normalize_trigger_price(trigger_info.get("next_trigger_up")),
            stop_risk_basis=state_stop_risk_basis,
        )
        _persist_cycle_output(result)
        return result

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
            ai_decision=entry_decision_value,
            next_trigger_down=_normalize_trigger_price(trigger_info.get("next_trigger_down")),
            next_trigger_up=_normalize_trigger_price(trigger_info.get("next_trigger_up")),
            stop_risk_basis=state_stop_risk_basis,
        )
        _persist_cycle_output(result)
        return result

    target_notional_usdt = _calculate_target_notional(
        account_equity=account_equity,
        target_margin_ratio=initial_position_size_ratio,
        leverage=leverage,
    )
    result["account_equity"] = account_equity
    result["target_notional_usdt"] = target_notional_usdt
    result["position_sizing"] = dict(fixed_position_sizing)
    logger.info(
        "Fixed position sizing computed | %s",
        format_log_details(
            {
                "symbol": symbol,
                "account_equity": account_equity,
                "target_margin_ratio": fixed_position_sizing.get("target_margin_ratio"),
                "target_effective_leverage": fixed_position_sizing.get("target_effective_leverage"),
                "target_notional_usdt": target_notional_usdt,
                "position_sizing": fixed_position_sizing,
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
            ai_decision=entry_decision_value,
            next_trigger_down=_normalize_trigger_price(trigger_info.get("next_trigger_down")),
            next_trigger_up=_normalize_trigger_price(trigger_info.get("next_trigger_up")),
            stop_risk_basis=state_stop_risk_basis,
        )
        _persist_cycle_output(result)
        return result

    leverage = int(applied_leverage)
    result["applied_leverage"] = leverage
    result["position_sizing"] = _build_fixed_position_sizing(
        initial_position_size_ratio=initial_position_size_ratio,
        leverage=leverage,
    )
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
            decision=str(entry_decision_value),
            target_notional_usdt=target_notional_usdt,
            reference_price=reference_price,
            leverage=leverage,
        )
        if (
            closed_before_entry
            and bool(execution_result.get("success"))
            and str(execution_result.get("action") or "") == "opened_new_position"
        ):
            execution_result["action"] = "closed_and_opened_position"
    else:
        execution_result = _rebalance_existing_position(
            api_key=api_key,
            api_secret=api_secret,
            symbol=symbol,
            current_position=current_position,
            decision=str(entry_decision_value),
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
                "decision": entry_decision_value,
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
            ai_decision=entry_decision_value,
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
                    "decision": entry_decision_value,
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
        previous_position=execution_previous_position,
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
        ai_decision=entry_decision_value,
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
__all__ = [
    "DEFAULT_SYMBOL",
    "DEFAULT_AI_PROMPT_TIMEFRAME",
    "run_hakai_cycle",
]
