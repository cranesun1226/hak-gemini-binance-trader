"""Binance Futures execution, position, and protection helpers."""

from __future__ import annotations

import hashlib
import hmac
import math
import time
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation, ROUND_DOWN, ROUND_UP
from typing import Any, Dict, List, Optional, Sequence, Tuple
from urllib.parse import urlencode

try:
    import requests
except ModuleNotFoundError:  # pragma: no cover - exercised in lightweight test environments
    class _RequestsFallback:
        def get(self, *_args, **_kwargs):
            raise ModuleNotFoundError("requests is required to call Binance APIs")

        def post(self, *_args, **_kwargs):
            raise ModuleNotFoundError("requests is required to call Binance APIs")

        def delete(self, *_args, **_kwargs):
            raise ModuleNotFoundError("requests is required to call Binance APIs")

    requests = _RequestsFallback()

from src.binance.common import get_binance_futures_base_url, get_recv_window_ms, safe_float as _safe_float_common
from src.binance.binance_rate_limit import BinanceExecutionStatusUnknown, binance_api_call_with_retry
from src.infra.logger import format_log_details, get_logger

logger = get_logger("trade_position")

BINANCE_RECV_WINDOW = int(get_recv_window_ms())
INSTRUMENT_FILTER_CACHE: Dict[str, Dict[str, Decimal]] = {}
_EXCHANGE_INFO_BY_SYMBOL: Dict[str, Dict[str, Any]] = {}
ENTRY_ORDER_RETRY_QTY_FACTOR = Decimal("0.99")
ENTRY_ORDER_MAX_ATTEMPTS = 25
NON_RETRYABLE_ENTRY_ERROR_CODES = {-1111, -4164}

# Numeric helpers and exchange metadata.

def safe_float(value: Optional[str], default: float = 0.0) -> float:
    return _safe_float_common(value, default)


def safe_decimal(value: Optional[str], default: Decimal = Decimal("0")) -> Decimal:
    try:
        return Decimal(str(value))
    except (TypeError, InvalidOperation):
        return default


def decimal_to_str(value: Decimal) -> str:
    if value is None:
        return "0"
    normalized = value.normalize()
    text = format(normalized, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text or "0"


def _base_url() -> str:
    return get_binance_futures_base_url()


def _load_exchange_info_once() -> None:
    global _EXCHANGE_INFO_BY_SYMBOL
    if _EXCHANGE_INFO_BY_SYMBOL:
        return

    url = f"{_base_url()}/fapi/v1/exchangeInfo"

    def _make_api_call():
        return requests.get(url, timeout=30)

    response = binance_api_call_with_retry(
        _make_api_call,
        max_retries=5,
        initial_delay=0.5,
        operation_name="exchangeInfo",
    )
    payload = response.json()
    if not isinstance(payload, dict):
        raise Exception(f"Binance exchangeInfo unexpected response: {payload}")

    symbols = payload.get("symbols", []) or []
    by_symbol: Dict[str, Dict[str, Any]] = {}
    for symbol_info in symbols:
        if not isinstance(symbol_info, dict):
            continue
        symbol = str(symbol_info.get("symbol") or "").upper()
        if symbol:
            by_symbol[symbol] = symbol_info

    _EXCHANGE_INFO_BY_SYMBOL = by_symbol
    logger.info("Loaded exchangeInfo for %s symbols", len(_EXCHANGE_INFO_BY_SYMBOL))


def _find_filter(symbol_info: Dict[str, Any], filter_type: str) -> Dict[str, Any]:
    for item in symbol_info.get("filters", []) or []:
        if isinstance(item, dict) and item.get("filterType") == filter_type:
            return item
    return {}


def get_instrument_filters(symbol: str) -> Optional[Dict[str, Decimal]]:
    normalized_symbol = str(symbol or "").strip().upper()
    if not normalized_symbol:
        return None

    cached = INSTRUMENT_FILTER_CACHE.get(normalized_symbol)
    if cached:
        return cached

    try:
        _load_exchange_info_once()
    except Exception as exc:
        logger.debug("Failed to load exchange info for %s: %s", normalized_symbol, exc)
        return None

    symbol_info = _EXCHANGE_INFO_BY_SYMBOL.get(normalized_symbol)
    if not symbol_info:
        return None

    market_lot = _find_filter(symbol_info, "MARKET_LOT_SIZE") or _find_filter(symbol_info, "LOT_SIZE")
    lot = _find_filter(symbol_info, "LOT_SIZE")
    price_filter = _find_filter(symbol_info, "PRICE_FILTER")
    min_notional_filter = _find_filter(symbol_info, "MIN_NOTIONAL")

    filters = {
        "min_qty": safe_decimal(market_lot.get("minQty")) or safe_decimal(lot.get("minQty")),
        "max_qty": safe_decimal(market_lot.get("maxQty")) or safe_decimal(lot.get("maxQty")),
        "step": safe_decimal(market_lot.get("stepSize")) or safe_decimal(lot.get("stepSize")),
        "tick_size": safe_decimal(price_filter.get("tickSize")),
        "min_price": safe_decimal(price_filter.get("minPrice")),
        "max_price": safe_decimal(price_filter.get("maxPrice")),
        "min_notional": safe_decimal(min_notional_filter.get("notional")),
    }
    INSTRUMENT_FILTER_CACHE[normalized_symbol] = filters
    return filters


def adjust_qty_for_symbol(symbol: str, qty: Decimal) -> Optional[Decimal]:
    if qty <= 0:
        return None

    filters = get_instrument_filters(symbol)
    if not filters:
        return qty

    adjusted = qty
    min_qty = filters.get("min_qty", Decimal("0"))
    max_qty = filters.get("max_qty", Decimal("0"))
    step = filters.get("step", Decimal("0"))

    if min_qty > 0 and adjusted < min_qty:
        adjusted = min_qty

    if step > 0:
        adjusted = (adjusted / step).to_integral_value(rounding=ROUND_DOWN) * step

    if max_qty > 0 and adjusted > max_qty:
        adjusted = (max_qty / step).to_integral_value(rounding=ROUND_DOWN) * step if step > 0 else max_qty

    return adjusted if adjusted > 0 else None


def _adjust_close_qty_for_symbol(symbol: str, qty: Decimal) -> Optional[Decimal]:
    if qty <= 0:
        return None

    filters = get_instrument_filters(symbol)
    if not filters:
        return qty

    adjusted = qty
    max_qty = filters.get("max_qty", Decimal("0"))
    step = filters.get("step", Decimal("0"))

    if max_qty > 0 and adjusted > max_qty:
        adjusted = max_qty

    if step > 0:
        adjusted = (adjusted / step).to_integral_value(rounding=ROUND_DOWN) * step

    return adjusted if adjusted > 0 else None


def _reduce_entry_qty_for_retry(symbol: str, qty: Decimal) -> Optional[Decimal]:
    if qty <= 0:
        return None

    reduced_qty = adjust_qty_for_symbol(symbol, qty * ENTRY_ORDER_RETRY_QTY_FACTOR)
    if reduced_qty is not None and Decimal("0") < reduced_qty < qty:
        return reduced_qty

    filters = get_instrument_filters(symbol) or {}
    step = safe_decimal(filters.get("step"))
    if step > 0:
        reduced_qty = adjust_qty_for_symbol(symbol, qty - step)
        if reduced_qty is not None and Decimal("0") < reduced_qty < qty:
            return reduced_qty
    return None


def evaluate_entry_order_notional(symbol: str, qty: Decimal, reference_price: float) -> Dict[str, Any]:
    resolved_qty = safe_decimal(str(qty))
    resolved_price = safe_decimal(str(reference_price))
    filters = get_instrument_filters(symbol) or {}
    min_notional = safe_decimal(filters.get("min_notional"))
    order_notional = Decimal("0")
    if resolved_qty > 0 and resolved_price > 0:
        order_notional = resolved_qty * resolved_price
    return {
        "order_notional": order_notional,
        "min_notional": min_notional,
        "meets_min_notional": min_notional <= 0 or order_notional >= min_notional,
    }


def adjust_price_for_symbol(symbol: str, price: float, rounding: str = "down") -> Optional[float]:
    if price is None:
        return None

    price_decimal = safe_decimal(str(price))
    if price_decimal <= 0:
        return None

    filters = get_instrument_filters(symbol)
    if not filters:
        return float(price_decimal)

    tick_size = filters.get("tick_size", Decimal("0"))
    min_price = filters.get("min_price", Decimal("0"))
    max_price = filters.get("max_price", Decimal("0"))

    adjusted = price_decimal
    if tick_size > 0:
        rounding_mode = ROUND_DOWN if rounding == "down" else ROUND_UP
        adjusted = (adjusted / tick_size).to_integral_value(rounding=rounding_mode) * tick_size

    if min_price > 0 and adjusted < min_price:
        adjusted = min_price
    if max_price > 0 and adjusted > max_price:
        adjusted = max_price

    return float(adjusted) if adjusted > 0 else None


def _coerce_positive_price(value: Any) -> Optional[float]:
    parsed = safe_float(value, 0.0)
    if parsed <= 0.0 or not math.isfinite(parsed):
        return None
    return float(parsed)


def _extract_order_type(order: Dict[str, Any]) -> str:
    return str(order.get("type") or order.get("orderType") or "").strip().upper()


def _extract_order_status(order: Dict[str, Any]) -> str:
    return str(order.get("status") or order.get("algoStatus") or "").strip().upper()


def _is_binance_error_payload(payload: Any) -> bool:
    if not isinstance(payload, dict):
        return False
    code, _msg = _parse_binance_error(payload)
    return code is not None and code < 0


# Signed REST request helpers.
def _build_signed_params(api_secret: str, params: Sequence[Tuple[str, Any]]) -> List[Tuple[str, str]]:
    timestamp = str(int(time.time() * 1000))
    items: List[Tuple[str, str]] = []
    for key, value in params:
        if value is None:
            continue
        items.append((str(key), str(value)))

    items.append(("recvWindow", str(BINANCE_RECV_WINDOW)))
    items.append(("timestamp", timestamp))

    query_string = urlencode(items, doseq=True)
    signature = hmac.new(api_secret.encode("utf-8"), query_string.encode("utf-8"), hashlib.sha256).hexdigest()
    items.append(("signature", signature))
    return items


def _signed_request(
    method: str,
    path: str,
    *,
    api_key: str,
    api_secret: str,
    params: Sequence[Tuple[str, Any]],
    operation_name: str,
    pre_call_delay: float = 0.0,
) -> Any:
    url = f"{_base_url()}{path}"
    signed_params = _build_signed_params(api_secret, params)
    headers = {"X-MBX-APIKEY": api_key}

    def _make_api_call():
        if method == "GET":
            return requests.get(url, params=signed_params, headers=headers, timeout=30)
        if method == "POST":
            return requests.post(url, params=signed_params, headers=headers, timeout=30)
        if method == "DELETE":
            return requests.delete(url, params=signed_params, headers=headers, timeout=30)
        raise ValueError(f"Unsupported method: {method}")

    return binance_api_call_with_retry(
        _make_api_call,
        max_retries=5,
        initial_delay=0.5,
        pre_call_delay=pre_call_delay,
        operation_name=operation_name,
    )


def _parse_binance_error(payload: Any) -> tuple[Optional[int], str]:
    if not isinstance(payload, dict):
        return None, str(payload)
    code = payload.get("code")
    try:
        normalized_code = int(code) if code is not None else None
    except (TypeError, ValueError):
        normalized_code = None
    return normalized_code, str(payload.get("msg") or "")


def _signed_post_expect_key(
    path: str,
    *,
    api_key: str,
    api_secret: str,
    params: Sequence[Tuple[str, Any]],
    operation_name: str,
    pre_call_delay: float = 0.0,
) -> tuple[Optional[Dict[str, Any]], Optional[int], str]:
    try:
        response = _signed_request(
            "POST",
            path,
            api_key=api_key,
            api_secret=api_secret,
            params=params,
            operation_name=operation_name,
            pre_call_delay=pre_call_delay,
        )
        payload = response.json()
    except BinanceExecutionStatusUnknown as exc:
        return None, None, str(exc)
    except Exception as exc:
        return None, None, str(exc)

    if isinstance(payload, dict) and payload.get("code") is not None:
        code, msg = _parse_binance_error(payload)
        if code is not None and code < 0:
            return None, code, msg
    if not isinstance(payload, dict):
        return None, None, str(payload)
    return payload, None, ""


def _signed_delete_json(
    path: str,
    *,
    api_key: str,
    api_secret: str,
    params: Sequence[Tuple[str, Any]],
    operation_name: str,
) -> Optional[Any]:
    try:
        response = _signed_request(
            "DELETE",
            path,
            api_key=api_key,
            api_secret=api_secret,
            params=params,
            operation_name=operation_name,
        )
        return response.json()
    except Exception as exc:
        logger.error("%s failed: %s", operation_name, exc)
        return None


def _signed_get_json(
    path: str,
    *,
    api_key: str,
    api_secret: str,
    params: Sequence[Tuple[str, Any]],
    operation_name: str,
) -> Optional[Any]:
    try:
        response = _signed_request(
            "GET",
            path,
            api_key=api_key,
            api_secret=api_secret,
            params=params,
            operation_name=operation_name,
        )
        return response.json()
    except Exception as exc:
        logger.error("%s failed: %s", operation_name, exc)
        return None


def _binance_side_from_side(side: str) -> Optional[str]:
    normalized = str(side or "").strip().upper()
    if normalized in {"BUY", "LONG"}:
        return "BUY"
    if normalized in {"SELL", "SHORT"}:
        return "SELL"
    if normalized == "BUY":
        return "BUY"
    if normalized == "SELL":
        return "SELL"
    return None


def _opposite_binance_side(side: str) -> Optional[str]:
    normalized = _binance_side_from_side(side)
    if normalized == "BUY":
        return "SELL"
    if normalized == "SELL":
        return "BUY"
    return None


def _first_valid_float(*values: Any, positive_only: bool = False) -> Optional[float]:
    for value in values:
        parsed = safe_float(value, None)
        if parsed is None or not math.isfinite(parsed):
            continue
        if positive_only and parsed <= 0.0:
            continue
        return float(parsed)
    return None


def _normalize_position_side(position: Optional[Dict[str, Any]]) -> Optional[str]:
    payload = dict(position or {})
    raw_side = str(payload.get("side") or "").strip().lower()
    if raw_side in {"buy", "long"}:
        return "Buy"
    if raw_side in {"sell", "short"}:
        return "Sell"

    position_amt = safe_float(payload.get("positionAmt"), 0.0)
    if position_amt > 0.0:
        return "Buy"
    if position_amt < 0.0:
        return "Sell"
    return None


def _normalize_position_direction(side: Optional[str]) -> Optional[str]:
    normalized = str(side or "").strip().lower()
    if normalized == "buy":
        return "long"
    if normalized == "sell":
        return "short"
    return None


def calculate_position_metrics(position: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    payload = dict(position or {})
    side = _normalize_position_side(payload)
    direction = _normalize_position_direction(side)

    position_amt = safe_float(payload.get("positionAmt"), 0.0)
    size = _first_valid_float(payload.get("size"), abs(position_amt), positive_only=True)
    entry_price = _first_valid_float(
        payload.get("entryPrice"),
        payload.get("avgPrice"),
        payload.get("breakEvenPrice"),
        positive_only=True,
    )
    mark_price = _first_valid_float(payload.get("markPrice"), payload.get("mark_price"), positive_only=True)
    leverage = _first_valid_float(payload.get("leverage"), positive_only=True)
    position_value = _first_valid_float(
        payload.get("positionValue"),
        payload.get("notional"),
        positive_only=True,
    )
    if position_value is None and mark_price is not None and size is not None:
        position_value = abs(mark_price * size)
    if position_value is None and entry_price is not None and size is not None:
        position_value = abs(entry_price * size)

    entry_notional = None
    if entry_price is not None and size is not None:
        entry_notional = abs(entry_price * size)

    position_margin = _first_valid_float(
        payload.get("positionMargin"),
        payload.get("isolatedMargin"),
        payload.get("initialMargin"),
        positive_only=True,
    )
    if position_margin is None and leverage and leverage > 0 and position_value is not None:
        position_margin = position_value / leverage

    return {
        "symbol": payload.get("symbol"),
        "side": side,
        "direction": direction,
        "size": size,
        "entry_price": entry_price,
        "entry_notional": entry_notional,
        "mark_price": mark_price,
        "leverage": leverage,
        "position_value": position_value,
        "position_margin": position_margin,
        "stop_loss": _first_valid_float(payload.get("stop_loss"), payload.get("stopLoss"), positive_only=True),
        "take_profit": _first_valid_float(payload.get("take_profit"), payload.get("takeProfit"), positive_only=True),
    }


# Position inspection and market reference helpers.
def get_account_equity(api_key: str, api_secret: str) -> Optional[float]:
    payload = _signed_get_json(
        "/fapi/v3/account",
        api_key=api_key,
        api_secret=api_secret,
        params=[],
        operation_name="account_equity",
    )
    if not isinstance(payload, dict):
        return None

    equity = safe_float(payload.get("totalMarginBalance"), 0.0)
    if equity <= 0.0:
        equity = safe_float(payload.get("totalWalletBalance"), 0.0)
    return equity if equity > 0.0 else None


def get_last_price(symbol: str) -> Optional[float]:
    normalized_symbol = str(symbol or "").strip().upper()
    if not normalized_symbol:
        return None

    url = f"{_base_url()}/fapi/v1/ticker/24hr"
    params = {"symbol": normalized_symbol}

    def _make_api_call():
        return requests.get(url, params=params, timeout=30)

    try:
        response = binance_api_call_with_retry(
            _make_api_call,
            max_retries=5,
            initial_delay=0.5,
            operation_name=f"ticker.24hr({normalized_symbol})",
        )
        payload = response.json()
    except Exception as exc:
        logger.error("Failed to fetch last price for %s: %s", normalized_symbol, exc)
        return None

    if not isinstance(payload, dict):
        return None

    last_price = safe_float(payload.get("lastPrice"), 0.0)
    return last_price if last_price > 0.0 else None


def get_book_ticker_mid_price(symbol: str) -> Optional[float]:
    normalized_symbol = str(symbol or "").strip().upper()
    if not normalized_symbol:
        return None

    url = f"{_base_url()}/fapi/v1/ticker/bookTicker"
    params = {"symbol": normalized_symbol}

    def _make_api_call():
        return requests.get(url, params=params, timeout=30)

    try:
        response = binance_api_call_with_retry(
            _make_api_call,
            max_retries=5,
            initial_delay=0.5,
            operation_name=f"ticker.bookTicker({normalized_symbol})",
        )
        payload = response.json()
    except Exception as exc:
        logger.debug("Failed to fetch book ticker for %s: %s", normalized_symbol, exc)
        return None

    if not isinstance(payload, dict):
        return None

    bid = safe_float(payload.get("bidPrice"), 0.0)
    ask = safe_float(payload.get("askPrice"), 0.0)
    if bid <= 0.0 or ask <= 0.0:
        return None
    return (bid + ask) / 2.0


def get_reference_price(symbol: str) -> Optional[Dict[str, Any]]:
    """Return the best available live price reference for a symbol."""
    normalized_symbol = str(symbol or "").strip().upper()
    if not normalized_symbol:
        return None

    mid_price = get_book_ticker_mid_price(normalized_symbol)
    if mid_price is not None:
        return {
            "symbol": normalized_symbol,
            "price": mid_price,
            "source": "book_ticker_mid",
            "captured_at": datetime.now(timezone.utc).isoformat(),
        }

    last_price = get_last_price(normalized_symbol)
    if last_price is not None:
        return {
            "symbol": normalized_symbol,
            "price": last_price,
            "source": "last_price",
            "captured_at": datetime.now(timezone.utc).isoformat(),
        }

    return None


def set_leverage(api_key: str, api_secret: str, symbol: str, leverage: int) -> Optional[int]:
    resolved_leverage = max(1, int(leverage))
    order, code, msg = _signed_post_expect_key(
        "/fapi/v1/leverage",
        api_key=api_key,
        api_secret=api_secret,
        params=[
            ("symbol", str(symbol or "").strip().upper()),
            ("leverage", resolved_leverage),
        ],
        operation_name=f"set_leverage({symbol})",
    )
    if order is not None:
        actual_leverage = safe_float(order.get("leverage"), 0.0) if isinstance(order, dict) else 0.0
        if actual_leverage > 0.0:
            return int(actual_leverage)
        return resolved_leverage
    logger.error("set_leverage(%s) failed: code=%s msg=%s", symbol, code, msg)
    return None


def _fetch_position_risk_rows(api_key: str, api_secret: str) -> Optional[list[Dict[str, Any]]]:
    payload = _signed_get_json(
        "/fapi/v3/positionRisk",
        api_key=api_key,
        api_secret=api_secret,
        params=[],
        operation_name="position_risk",
    )
    if not isinstance(payload, list):
        return None
    return [row for row in payload if isinstance(row, dict)]


def _get_open_orders(
    api_key: str,
    api_secret: str,
    *,
    symbol: Optional[str] = None,
) -> list[Dict[str, Any]]:
    params: list[Tuple[str, Any]] = []
    if symbol:
        params.append(("symbol", str(symbol).strip().upper()))

    payload = _signed_get_json(
        "/fapi/v1/openOrders",
        api_key=api_key,
        api_secret=api_secret,
        params=params,
        operation_name=f"open_orders({symbol or 'all'})",
    )
    if not isinstance(payload, list):
        return []
    return [item for item in payload if isinstance(item, dict)]


def _get_open_algo_orders(
    api_key: str,
    api_secret: str,
    *,
    symbol: Optional[str] = None,
) -> list[Dict[str, Any]]:
    params: list[Tuple[str, Any]] = [("algoType", "CONDITIONAL")]
    if symbol:
        params.append(("symbol", str(symbol).strip().upper()))

    payload = _signed_get_json(
        "/fapi/v1/openAlgoOrders",
        api_key=api_key,
        api_secret=api_secret,
        params=params,
        operation_name=f"open_algo_orders({symbol or 'all'})",
    )
    if not isinstance(payload, list):
        return []
    return [item for item in payload if isinstance(item, dict)]


def _attach_position_protection(
    position: Dict[str, Any],
    open_orders: Sequence[Dict[str, Any]],
    algo_orders: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    enriched = dict(position)
    normalized_symbol = str(position.get("symbol") or "").strip().upper()
    for order in list(open_orders) + list(algo_orders):
        if str(order.get("symbol") or "").strip().upper() != normalized_symbol:
            continue
        order_type = _extract_order_type(order)
        stop_price = _coerce_positive_price(order.get("stopPrice"))
        if stop_price is None:
            stop_price = _coerce_positive_price(order.get("triggerPrice"))
        if stop_price is None:
            continue
        if order_type in {"STOP", "STOP_MARKET"}:
            enriched["stop_loss"] = stop_price
        elif order_type in {"TAKE_PROFIT", "TAKE_PROFIT_MARKET"}:
            enriched["take_profit"] = stop_price
    return enriched


def _normalize_position_risk_payload(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    symbol = str(row.get("symbol") or "").strip().upper()
    position_amt = safe_float(row.get("positionAmt"), 0.0)
    if not symbol or abs(position_amt) <= 0.0:
        return None

    payload = dict(row)
    payload["symbol"] = symbol
    payload["positionAmt"] = position_amt
    payload["size"] = abs(position_amt)
    payload["positionValue"] = abs(
        _first_valid_float(
            row.get("notional"),
            row.get("positionValue"),
            positive_only=True,
        )
        or 0.0
    )
    payload["side"] = "Buy" if position_amt > 0.0 else "Sell"
    return payload


def get_positions(api_key: str, api_secret: str) -> Optional[List[Dict[str, Any]]]:
    """Fetch open futures positions enriched with attached protection orders."""
    rows = _fetch_position_risk_rows(api_key, api_secret)
    if rows is None:
        return None

    open_orders = _get_open_orders(api_key, api_secret)
    algo_orders = _get_open_algo_orders(api_key, api_secret)
    positions: list[Dict[str, Any]] = []
    for row in rows:
        normalized = _normalize_position_risk_payload(row)
        if normalized is None:
            continue
        positions.append(_attach_position_protection(normalized, open_orders, algo_orders))
    return positions


def get_position_snapshot(
    api_key: str,
    api_secret: str,
    symbol: str,
    *,
    retries: int = 1,
    sleep_seconds: float = 0.35,
) -> Optional[Dict[str, Any]]:
    normalized_symbol = str(symbol or "").strip().upper()
    for attempt in range(max(1, int(retries))):
        positions = get_positions(api_key, api_secret)
        if positions is None:
            if attempt + 1 < max(1, int(retries)):
                time.sleep(max(0.0, float(sleep_seconds)))
            continue
        for position in positions:
            if str(position.get("symbol") or "").strip().upper() == normalized_symbol:
                return position
        if attempt + 1 < max(1, int(retries)):
            time.sleep(max(0.0, float(sleep_seconds)))
    return None


def _query_position_amt(api_key: str, api_secret: str, symbol: str) -> float:
    snapshot = get_position_snapshot(api_key, api_secret, symbol, retries=1)
    if not isinstance(snapshot, dict):
        return 0.0
    return safe_float(snapshot.get("positionAmt"), 0.0)


def _get_position_amt(api_key: str, api_secret: str, symbol: str) -> float:
    return _query_position_amt(api_key, api_secret, symbol)


# Order execution helpers.
def place_market_entry_order(
    api_key: str,
    api_secret: str,
    symbol: str,
    side: str,
    qty: str,
    *,
    leverage: Optional[int] = None,
) -> tuple[Optional[Dict[str, Any]], Optional[int], str]:
    """Submit a market entry order and shrink quantity on retryable exchange errors."""
    normalized_symbol = str(symbol or "").strip().upper()
    binance_side = _binance_side_from_side(side)
    if not normalized_symbol or binance_side is None:
        return None, None, "invalid_symbol_or_side"

    adjusted_qty = adjust_qty_for_symbol(normalized_symbol, safe_decimal(qty))
    if adjusted_qty is None or adjusted_qty <= 0:
        return None, None, "invalid_quantity"

    if leverage is not None and not set_leverage(api_key, api_secret, normalized_symbol, leverage):
        return None, None, "set_leverage_failed"

    current_qty = adjusted_qty
    last_code: Optional[int] = None
    last_msg = ""
    for attempt in range(1, ENTRY_ORDER_MAX_ATTEMPTS + 1):
        current_qty_text = decimal_to_str(current_qty)
        logger.info(
            "Placing market entry order | %s",
            format_log_details(
                {
                    "symbol": normalized_symbol,
                    "side": binance_side,
                    "requested_qty": qty,
                    "adjusted_qty": current_qty_text,
                    "leverage": leverage,
                    "attempt": attempt,
                    "max_attempts": ENTRY_ORDER_MAX_ATTEMPTS,
                }
            ),
        )
        order, code, msg = _signed_post_expect_key(
            "/fapi/v1/order",
            api_key=api_key,
            api_secret=api_secret,
            params=[
                ("symbol", normalized_symbol),
                ("side", binance_side),
                ("type", "MARKET"),
                ("quantity", current_qty_text),
            ],
            operation_name=f"place_market_entry_order({normalized_symbol})",
        )
        if order is not None:
            logger.info(
                "Market entry order accepted | %s",
                format_log_details(
                    {
                        "symbol": normalized_symbol,
                        "side": binance_side,
                        "qty": current_qty_text,
                        "attempt": attempt,
                        "order_id": order.get("orderId"),
                        "status": order.get("status"),
                    }
                ),
            )
            return order, code, msg

        last_code = code
        last_msg = msg
        can_retry = (
            code is not None
            and code not in NON_RETRYABLE_ENTRY_ERROR_CODES
            and attempt < ENTRY_ORDER_MAX_ATTEMPTS
        )
        if not can_retry:
            break

        next_qty = _reduce_entry_qty_for_retry(normalized_symbol, current_qty)
        if next_qty is None or next_qty <= 0 or next_qty >= current_qty:
            break

        logger.warning(
            "Market entry order failed; retrying with reduced quantity | %s",
            format_log_details(
                {
                    "symbol": normalized_symbol,
                    "side": binance_side,
                    "attempt": attempt,
                    "failed_qty": current_qty_text,
                    "next_qty": decimal_to_str(next_qty),
                    "error_code": code,
                    "error_message": msg,
                }
            ),
        )
        current_qty = next_qty

    logger.error(
        "Market entry order failed | %s",
        format_log_details(
            {
                "symbol": normalized_symbol,
                "side": binance_side,
                "qty": decimal_to_str(current_qty),
                "error_code": last_code,
                "error_message": last_msg,
            }
        ),
    )
    return None, last_code, last_msg


def place_reduce_only_market_order(
    api_key: str,
    api_secret: str,
    symbol: str,
    side: str,
    qty: str,
) -> tuple[Optional[Dict[str, Any]], Optional[int], str]:
    """Submit a reduce-only market order for closing or reducing a position."""
    normalized_symbol = str(symbol or "").strip().upper()
    binance_side = _binance_side_from_side(side)
    if not normalized_symbol or binance_side is None:
        return None, None, "invalid_symbol_or_side"

    adjusted_qty = _adjust_close_qty_for_symbol(normalized_symbol, safe_decimal(qty))
    if adjusted_qty is None or adjusted_qty <= 0:
        return None, None, "invalid_quantity"

    logger.info(
        "Placing reduce-only market order | %s",
        format_log_details(
            {
                "symbol": normalized_symbol,
                "side": binance_side,
                "requested_qty": qty,
                "adjusted_qty": decimal_to_str(adjusted_qty),
            }
        ),
    )
    order, code, msg = _signed_post_expect_key(
        "/fapi/v1/order",
        api_key=api_key,
        api_secret=api_secret,
        params=[
            ("symbol", normalized_symbol),
            ("side", binance_side),
            ("type", "MARKET"),
            ("quantity", decimal_to_str(adjusted_qty)),
            ("reduceOnly", "true"),
        ],
        operation_name=f"place_reduce_only_market_order({normalized_symbol})",
    )
    if order is not None:
        logger.info(
            "Reduce-only market order accepted | %s",
            format_log_details(
                {
                    "symbol": normalized_symbol,
                    "side": binance_side,
                    "qty": decimal_to_str(adjusted_qty),
                    "order_id": order.get("orderId"),
                    "status": order.get("status"),
                }
            ),
        )
    else:
        logger.error(
            "Reduce-only market order failed | %s",
            format_log_details(
                {
                    "symbol": normalized_symbol,
                    "side": binance_side,
                    "qty": decimal_to_str(adjusted_qty),
                    "error_code": code,
                    "error_message": msg,
                }
            ),
        )
    return order, code, msg


def close_position(api_key: str, api_secret: str, symbol: str, side: str, qty: str) -> bool:
    """Best-effort close a position with exchange-aware quantity normalization."""
    normalized_symbol = str(symbol or "").strip().upper()
    close_side = _opposite_binance_side(side)
    if not normalized_symbol or close_side is None:
        return False

    live_position_amt = abs(_get_position_amt(api_key, api_secret, normalized_symbol))
    requested_qty = abs(safe_float(qty, 0.0))
    close_qty_value = min(value for value in (live_position_amt, requested_qty) if value > 0.0) if requested_qty > 0.0 and live_position_amt > 0.0 else max(live_position_amt, requested_qty)
    if close_qty_value <= 0.0:
        return True

    close_qty = _adjust_close_qty_for_symbol(normalized_symbol, safe_decimal(str(close_qty_value)))
    if close_qty is None or close_qty <= 0:
        return True

    logger.info(
        "Closing position with reduce-only market order | %s",
        format_log_details(
            {
                "symbol": normalized_symbol,
                "close_side": close_side,
                "live_position_amt": live_position_amt,
                "requested_qty": requested_qty,
                "close_qty": decimal_to_str(close_qty),
            }
        ),
    )
    _order, code, msg = _signed_post_expect_key(
        "/fapi/v1/order",
        api_key=api_key,
        api_secret=api_secret,
        params=[
            ("symbol", normalized_symbol),
            ("side", close_side),
            ("type", "MARKET"),
            ("quantity", decimal_to_str(close_qty)),
            ("reduceOnly", "true"),
        ],
        operation_name=f"close_position({normalized_symbol})",
    )
    if _order is not None:
        return True

    normalized_msg = str(msg or "").lower()
    if code == -2022 and "reduceonly order is rejected" in normalized_msg:
        time.sleep(0.25)
        return abs(_get_position_amt(api_key, api_secret, normalized_symbol)) <= 0.0

    logger.error("close_position(%s) failed: code=%s msg=%s", normalized_symbol, code, msg)
    return False


def _prices_match_with_symbol_tolerance(symbol: str, left: float, right: float) -> bool:
    filters = get_instrument_filters(symbol) or {}
    tick_size = safe_float(filters.get("tick_size"), 0.0)
    tolerance = tick_size if tick_size > 0.0 else 1e-9
    return math.isclose(float(left), float(right), abs_tol=tolerance, rel_tol=0.0)


def _cancel_stop_orders(api_key: str, api_secret: str, symbol: str) -> bool:
    open_orders = _get_open_orders(api_key, api_secret, symbol=symbol)
    open_algo_orders = _get_open_algo_orders(api_key, api_secret, symbol=symbol)
    success = True
    cancelled_standard = 0
    cancelled_algo = 0
    for order in open_orders:
        order_type = _extract_order_type(order)
        if order_type not in {"STOP", "STOP_MARKET"}:
            continue
        order_id = order.get("orderId")
        if order_id is None:
            continue
        payload = _signed_delete_json(
            "/fapi/v1/order",
            api_key=api_key,
            api_secret=api_secret,
            params=[
                ("symbol", str(symbol).strip().upper()),
                ("orderId", order_id),
            ],
            operation_name=f"cancel_stop_order({symbol},{order_id})",
        )
        if payload is None or _is_binance_error_payload(payload):
            success = False
        else:
            cancelled_standard += 1
    for order in open_algo_orders:
        order_type = _extract_order_type(order)
        if order_type not in {"STOP", "STOP_MARKET"}:
            continue
        algo_id = order.get("algoId")
        client_algo_id = order.get("clientAlgoId")
        params: list[Tuple[str, Any]] = [("symbol", str(symbol).strip().upper())]
        if algo_id is not None:
            params.append(("algoId", algo_id))
        elif client_algo_id:
            params.append(("clientAlgoId", client_algo_id))
        if len(params) == 1:
            continue
        payload = _signed_delete_json(
            "/fapi/v1/algoOrder",
            api_key=api_key,
            api_secret=api_secret,
            params=params,
            operation_name=f"cancel_stop_algo_order({symbol},{algo_id or client_algo_id})",
        )
        if payload is None or _is_binance_error_payload(payload):
            success = False
        else:
            cancelled_algo += 1
    logger.info(
        "Cancelled existing stop orders | %s",
        format_log_details(
            {
                "symbol": symbol,
                "standard_orders": cancelled_standard,
                "algo_orders": cancelled_algo,
                "success": success,
            }
        ),
    )
    return success


# Stop-loss and cleanup helpers.
def sync_existing_position_stop_loss(
    api_key: str,
    api_secret: str,
    symbol: str,
    side: str,
    *,
    stop_loss: float,
    current_stop_loss: Optional[float] = None,
) -> Dict[str, Any]:
    """Ensure a position has a synchronized exchange-native stop-loss order."""
    normalized_symbol = str(symbol or "").strip().upper()
    position_side = _binance_side_from_side(side)
    if not normalized_symbol or position_side is None:
        return {"success": False, "changed": False, "reason": "invalid_symbol_or_side"}

    rounding = "down" if position_side == "BUY" else "up"
    adjusted_stop = adjust_price_for_symbol(normalized_symbol, float(stop_loss), rounding=rounding)
    if adjusted_stop is None:
        return {"success": False, "changed": False, "reason": "invalid_stop_loss"}

    if current_stop_loss is not None:
        existing_stop = adjust_price_for_symbol(normalized_symbol, float(current_stop_loss), rounding=rounding)
        if existing_stop is not None and _prices_match_with_symbol_tolerance(normalized_symbol, existing_stop, adjusted_stop):
            logger.debug(
                "Stop loss already synchronized | %s",
                format_log_details(
                    {
                        "symbol": normalized_symbol,
                        "side": position_side,
                        "existing_stop": existing_stop,
                        "adjusted_stop": adjusted_stop,
                    }
                ),
            )
            return {
                "success": True,
                "changed": False,
                "reason": "stop_loss_already_synced",
                "stop_loss": adjusted_stop,
            }

    logger.info(
        "Syncing stop loss via algo order | %s",
        format_log_details(
            {
                "symbol": normalized_symbol,
                "side": position_side,
                "requested_stop_loss": stop_loss,
                "current_stop_loss": current_stop_loss,
                "adjusted_stop": adjusted_stop,
            }
        ),
    )
    _cancel_stop_orders(api_key, api_secret, normalized_symbol)

    exit_side = _opposite_binance_side(position_side)
    order, code, msg = _signed_post_expect_key(
        "/fapi/v1/algoOrder",
        api_key=api_key,
        api_secret=api_secret,
        params=[
            ("algoType", "CONDITIONAL"),
            ("symbol", normalized_symbol),
            ("side", exit_side),
            ("type", "STOP_MARKET"),
            ("triggerPrice", adjusted_stop),
            ("closePosition", "true"),
            ("workingType", "CONTRACT_PRICE"),
            ("priceProtect", "TRUE"),
        ],
        operation_name=f"sync_stop_loss({normalized_symbol})",
    )
    if order is None:
        logger.error(
            "Stop loss algo order failed | %s",
            format_log_details(
                {
                    "symbol": normalized_symbol,
                    "side": exit_side,
                    "trigger_price": adjusted_stop,
                    "error_code": code,
                    "error_message": msg,
                }
            ),
        )
        return {
            "success": False,
            "changed": False,
            "reason": "place_stop_loss_failed",
            "error_code": code,
            "error_message": msg,
        }

    logger.info(
        "Stop loss algo order synchronized | %s",
        format_log_details(
            {
                "symbol": normalized_symbol,
                "side": exit_side,
                "trigger_price": adjusted_stop,
                "algo_id": order.get("algoId"),
                "status": _extract_order_status(order),
            }
        ),
    )
    return {
        "success": True,
        "changed": True,
        "stop_loss": adjusted_stop,
        "order": order,
    }


def cancel_all_orders(api_key: str, api_secret: str, symbol: str) -> bool:
    """Cancel all open standard orders for a symbol."""
    normalized_symbol = str(symbol or "").strip().upper()
    if not normalized_symbol:
        return False

    payload = _signed_delete_json(
        "/fapi/v1/allOpenOrders",
        api_key=api_key,
        api_secret=api_secret,
        params=[("symbol", normalized_symbol)],
        operation_name=f"cancel_all_orders({normalized_symbol})",
    )
    return payload is not None


def wait_for_close_propagation(
    api_key: str,
    api_secret: str,
    symbols: Sequence[str],
    *,
    context: str = "",
    max_attempts: int = 10,
    sleep_seconds: float = 0.5,
) -> bool:
    """Poll until the exchange no longer reports positions for the given symbols."""
    normalized_symbols = {
        str(symbol or "").strip().upper()
        for symbol in symbols
        if str(symbol or "").strip()
    }
    if not normalized_symbols:
        return True

    for attempt in range(max(1, int(max_attempts))):
        positions = get_positions(api_key, api_secret)
        if positions is None:
            if attempt + 1 < max(1, int(max_attempts)):
                time.sleep(max(0.0, float(sleep_seconds)))
            continue
        open_symbols = {
            str(position.get("symbol") or "").strip().upper()
            for position in positions
            if str(position.get("symbol") or "").strip()
        }
        if not (open_symbols & normalized_symbols):
            return True
        if attempt + 1 < max(1, int(max_attempts)):
            time.sleep(max(0.0, float(sleep_seconds)))

    logger.warning(
        "wait_for_close_propagation timeout | context=%s symbols=%s",
        context,
        sorted(normalized_symbols),
    )
    return False
