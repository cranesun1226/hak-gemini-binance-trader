"""Retry helpers for Binance APIs with transient failure handling."""

from __future__ import annotations

import time
from typing import Any, Callable, Optional

from src.infra.logger import get_logger

logger = get_logger("binance_rate_limit")


class BinanceExecutionStatusUnknown(RuntimeError):
    """
    Raised when Binance returns HTTP 503 with an "Unknown error" style message where the
    execution status is unknown and callers must verify state before retrying.
    """

    def __init__(
        self,
        message: str,
        *,
        status_code: Optional[int] = None,
        response_text: Optional[str] = None,
        payload: Optional[dict] = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.response_text = response_text
        self.payload = payload or {}


def _safe_json(response_obj: Any) -> dict:
    """Best-effort JSON parsing for error inspection."""
    try:
        data = response_obj.json()
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _is_rate_limit_error(response_obj: Any, payload: dict) -> bool:
    if getattr(response_obj, "status_code", None) == 429:
        return True

    code = payload.get("code")
    if code in (-1003, -1015):
        return True

    msg = str(payload.get("msg", "")).lower()
    if "too many requests" in msg:
        return True
    if "rate" in msg and "limit" in msg:
        return True
    return False


def _is_system_overload_throttle(payload: dict) -> bool:
    # -1008: Request throttled by system-level protection.
    code = payload.get("code")
    if code == -1008:
        return True
    msg = str(payload.get("msg", "")).lower()
    return "throttled" in msg and "system" in msg


def _is_service_unavailable(payload: dict) -> bool:
    msg = str(payload.get("msg", "")).lower()
    if "service unavailable" in msg:
        return True
    if "internal error" in msg and "try again" in msg:
        return True
    return False


def _is_execution_status_unknown(response_obj: Any, payload: dict) -> bool:
    if getattr(response_obj, "status_code", None) != 503:
        return False
    msg = str(payload.get("msg", "")).lower()
    return "unknown error" in msg


def binance_api_call_with_retry(
    api_call_func: Callable[[], Any],
    *,
    max_retries: int = 5,
    initial_delay: float = 0.5,
    backoff_factor: float = 2.0,
    pre_call_delay: float = 0.0,
    operation_name: str = "binance_api_call",
) -> Any:
    """
    Execute a Binance API call with exponential backoff retry on rate limits / transient overload.

    Retries:
    - HTTP 429 / code -1003 / -1015
    - code -1008 (system overload throttling)
    - HTTP 503 with messages indicating temporary service unavailability

    Does NOT auto-retry execution-unknown 503 ("Unknown error...") because callers must
    confirm execution (e.g., via order queries) to avoid duplicates.
    """
    attempt = 0

    while attempt <= max_retries:
        try:
            if pre_call_delay > 0 and attempt == 0:
                time.sleep(pre_call_delay)

            if attempt == 0:
                logger.debug(f"{operation_name}: Making API call")
            else:
                logger.info(f"{operation_name}: Retry attempt {attempt}/{max_retries}")

            response = api_call_func()
            payload = _safe_json(response)

            if getattr(response, "status_code", None) == 418:
                msg = payload.get("msg") or getattr(response, "text", "")
                raise Exception(f"{operation_name}: IP banned (HTTP 418): {msg}")

            if _is_execution_status_unknown(response, payload):
                raise BinanceExecutionStatusUnknown(
                    f"{operation_name}: Execution status unknown (HTTP 503): {payload.get('msg')}",
                    status_code=getattr(response, "status_code", None),
                    response_text=getattr(response, "text", None),
                    payload=payload,
                )

            retryable = False
            reason = None
            if _is_rate_limit_error(response, payload):
                retryable = True
                reason = f"rate_limit HTTP {getattr(response, 'status_code', None)} code={payload.get('code')}"
            elif _is_system_overload_throttle(payload):
                retryable = True
                reason = f"system_overload code={payload.get('code')}"
            elif getattr(response, "status_code", None) == 503 and _is_service_unavailable(payload):
                retryable = True
                reason = "service_unavailable"

            if retryable:
                if attempt < max_retries:
                    delay = initial_delay * (backoff_factor ** attempt)
                    logger.warning(
                        f"{operation_name}: Retryable error ({reason}). "
                        f"Retrying in {delay:.2f}s (attempt {attempt + 1}/{max_retries})"
                    )
                    time.sleep(delay)
                    attempt += 1
                    continue
                raise Exception(
                    f"{operation_name}: Retryable error but max retries exceeded "
                    f"(HTTP {getattr(response, 'status_code', None)} payload={payload})"
                )

            return response
        except BinanceExecutionStatusUnknown:
            raise
        except Exception as exc:
            if attempt < max_retries:
                delay = initial_delay * (backoff_factor ** attempt)
                logger.warning(f"{operation_name}: Exception: {exc} | retry in {delay:.2f}s")
                time.sleep(delay)
                attempt += 1
                continue
            logger.error(f"{operation_name}: Failed after retries: {exc}")
            raise

    raise Exception(f"{operation_name}: Unexpected retry loop termination")


__all__ = ["BinanceExecutionStatusUnknown", "binance_api_call_with_retry"]
