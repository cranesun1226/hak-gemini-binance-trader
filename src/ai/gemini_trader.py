"""Gemini direction-decision helper for HAK GEMINI BINANCE TRADER."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Generic, Literal, Optional, TypeVar

try:
    from pydantic import BaseModel, Field
except ModuleNotFoundError:
    class BaseModel:
        def __init__(self, **kwargs: Any) -> None:
            for field_name in self.__annotations__:
                setattr(self, field_name, kwargs.get(field_name))

        @classmethod
        def model_validate_json(cls, raw_json: str):
            return cls(**json.loads(raw_json))

        @classmethod
        def model_json_schema(cls) -> dict[str, Any]:
            return {
                "type": "object",
                "properties": {
                    field_name: {"title": field_name}
                    for field_name in getattr(cls, "__annotations__", {})
                },
            }

        def model_dump(self) -> dict[str, Any]:
            return {
                field_name: getattr(self, field_name, None)
                for field_name in getattr(self, "__annotations__", {})
            }

        def model_dump_json(self, indent: Optional[int] = None) -> str:
            return json.dumps(self.model_dump(), indent=indent, ensure_ascii=False)

    def Field(default: Any = None, **_kwargs: Any) -> Any:
        return default

try:
    from google import genai
    from google.genai import errors
    from google.genai import types
except ModuleNotFoundError:
    genai = None
    errors = None
    types = None

from src.infra.env_loader import get_gemini_api_key
from src.infra.logger import format_log_details, get_logger

logger = get_logger("gemini_trader")

DecisionT = TypeVar("DecisionT")

GEMINI_GENERATE_MAX_RETRIES = 3
GEMINI_DIRECTION_MODEL = "gemini-3-flash-preview"
# Backward-compatible alias for existing imports.
GEMINI_PRO_ONLY_MODEL = GEMINI_DIRECTION_MODEL

_ONE_MILLION = 1_000_000
_GEMINI_MODEL_PRICING_USD_PER_MILLION: dict[str, dict[str, float]] = {
    GEMINI_DIRECTION_MODEL: {
        "input_text_image_video": 0.5,
        "input_audio": 1.0,
        "output_including_thinking": 3.0,
    },
}


class HakaiTradeDirectionDecision(BaseModel):
    """Structured Gemini response containing the required directional decision."""

    decision: Literal["LONG", "SHORT"] = Field(
        description="Return exactly one BTCUSDT directional decision (LONG or SHORT) based only on the supplied market and position context."
    )


@dataclass
class GeminiStructuredResponse(Generic[DecisionT]):
    """Container for a parsed Gemini decision plus raw diagnostics."""

    decision: DecisionT
    raw_response: str
    usage_metadata: dict[str, Any]
    response_payload: dict[str, Any] = field(default_factory=dict)
    thought_summary: str = ""
    thought_signatures: list[str] = field(default_factory=list)


def _to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}

    if isinstance(value, (list, tuple, set)):
        return [_to_jsonable(item) for item in value]

    for method_name in ("to_json_dict", "model_dump", "dict"):
        method = getattr(value, method_name, None)
        if not callable(method):
            continue
        try:
            return _to_jsonable(method())
        except TypeError:
            try:
                return _to_jsonable(method(mode="json", exclude_none=True))
            except TypeError:
                continue

    value_dict = getattr(value, "__dict__", None)
    if isinstance(value_dict, dict):
        return {
            str(key): _to_jsonable(item)
            for key, item in value_dict.items()
            if not str(key).startswith("_")
        }

    return str(value)


def extract_usage_metadata(response: Any) -> dict[str, Any]:
    serialized = _to_jsonable(getattr(response, "usage_metadata", None))
    return serialized if isinstance(serialized, dict) else {}


def _extract_response_payload(response: Any) -> dict[str, Any]:
    serialized = _to_jsonable(response)
    return serialized if isinstance(serialized, dict) else {"response": serialized}


def _iter_response_parts(response_payload: Optional[Dict[str, Any]]) -> list[dict[str, Any]]:
    payload = response_payload if isinstance(response_payload, dict) else {}
    parts: list[dict[str, Any]] = []

    for candidate in payload.get("candidates") or []:
        if not isinstance(candidate, dict):
            continue
        content = candidate.get("content")
        if not isinstance(content, dict):
            continue
        candidate_parts = content.get("parts") or []
        for part in candidate_parts:
            if isinstance(part, dict):
                parts.append(part)

    return parts


def extract_thought_summary(response_payload: Optional[Dict[str, Any]]) -> str:
    thought_summaries: list[str] = []
    for part in _iter_response_parts(response_payload):
        if not bool(part.get("thought")):
            continue
        text = str(part.get("text") or "").strip()
        if text:
            thought_summaries.append(text)
    return "\n".join(thought_summaries).strip()


def extract_thought_signatures(response_payload: Optional[Dict[str, Any]]) -> list[str]:
    signatures: list[str] = []
    seen: set[str] = set()

    for part in _iter_response_parts(response_payload):
        raw_signature = part.get("thought_signature")
        if raw_signature is None:
            raw_signature = part.get("thoughtSignature")
        signature = str(raw_signature or "").strip()
        if not signature or signature in seen:
            continue
        signatures.append(signature)
        seen.add(signature)

    return signatures


def _safe_non_negative_int(value: Any) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return 0
    return max(0, parsed)


def _summarize_timeframe_ohlcv(timeframe_ohlcv: Dict[str, Any]) -> Dict[str, int]:
    summary: dict[str, int] = {}
    for timeframe, candles in (timeframe_ohlcv or {}).items():
        if isinstance(candles, list):
            summary[str(timeframe)] = len(candles)
    return summary


def estimate_gemini_cost(
    usage_metadata: Optional[dict[str, Any]],
    *,
    model: str = GEMINI_DIRECTION_MODEL,
) -> Optional[dict[str, Any]]:
    """Estimate Gemini API cost from usage metadata when pricing is known."""
    usage = usage_metadata if isinstance(usage_metadata, dict) else {}
    pricing = _GEMINI_MODEL_PRICING_USD_PER_MILLION.get(str(model or "").strip())
    if not usage or not pricing:
        return None

    prompt_token_count = _safe_non_negative_int(usage.get("prompt_token_count"))
    candidates_token_count = _safe_non_negative_int(usage.get("candidates_token_count"))
    thoughts_token_count = _safe_non_negative_int(usage.get("thoughts_token_count"))
    output_token_count = candidates_token_count + thoughts_token_count
    prompt_tokens_details = usage.get("prompt_tokens_details")
    audio_input_token_count = 0
    if isinstance(prompt_tokens_details, list):
        audio_input_token_count = sum(
            _safe_non_negative_int(item.get("token_count"))
            for item in prompt_tokens_details
            if isinstance(item, dict) and str(item.get("modality") or "").upper() == "AUDIO"
        )
    audio_input_token_count = min(audio_input_token_count, prompt_token_count)
    text_image_video_input_token_count = max(prompt_token_count - audio_input_token_count, 0)
    text_image_video_input_rate = float(
        pricing.get("input_text_image_video", pricing.get("input_standard", 0.0))
    )
    audio_input_rate = float(pricing.get("input_audio", text_image_video_input_rate))
    output_rate = float(
        pricing.get("output_including_thinking", pricing.get("output_standard_including_thinking", 0.0))
    )

    input_cost_usd = (
        text_image_video_input_token_count * text_image_video_input_rate / _ONE_MILLION
        + audio_input_token_count * audio_input_rate / _ONE_MILLION
    )
    output_cost_usd = output_token_count * output_rate / _ONE_MILLION
    total_cost_usd = input_cost_usd + output_cost_usd

    return {
        "currency": "USD",
        "model": model,
        "pricing_tier": "standard",
        "pricing_prompt_token_threshold": None,
        "input_rate_usd_per_million": text_image_video_input_rate,
        "input_text_image_video_rate_usd_per_million": text_image_video_input_rate,
        "input_audio_rate_usd_per_million": audio_input_rate,
        "output_rate_usd_per_million": output_rate,
        "input_token_count": prompt_token_count,
        "text_image_video_input_token_count": text_image_video_input_token_count,
        "audio_input_token_count": audio_input_token_count,
        "candidate_token_count": candidates_token_count,
        "thoughts_token_count": thoughts_token_count,
        "output_token_count": output_token_count,
        "input_cost_usd": round(input_cost_usd, 12),
        "output_cost_usd": round(output_cost_usd, 12),
        "total_cost_usd": round(total_cost_usd, 12),
    }


def log_usage_metadata(
    logger: Any,
    *,
    context: str,
    usage_metadata: Optional[dict[str, Any]],
    model: str = GEMINI_DIRECTION_MODEL,
) -> None:
    usage = usage_metadata if isinstance(usage_metadata, dict) else {}
    if not usage:
        return

    estimated_cost = estimate_gemini_cost(usage, model=model)
    estimated_cost_usd = estimated_cost.get("total_cost_usd") if isinstance(estimated_cost, dict) else None

    logger.info(
        "%s token usage | model=%s prompt=%s candidates=%s thoughts=%s total=%s estimated_cost_usd=%s",
        context,
        model,
        usage.get("prompt_token_count"),
        usage.get("candidates_token_count"),
        usage.get("thoughts_token_count"),
        usage.get("total_token_count"),
        estimated_cost_usd,
    )


def _is_retryable_gemini_error(exc: Exception) -> bool:
    if errors is not None and isinstance(exc, errors.ServerError):
        return True
    if errors is not None and isinstance(exc, errors.ClientError):
        code = getattr(exc, "code", None)
        status = str(getattr(exc, "status", "") or "").lower()
        if code == 429:
            return True
        if code in (500, 502, 503, 504):
            return True
        if status in {"internal", "unavailable", "resource_exhausted", "too_many_requests"}:
            return True
        return False
    error_msg = str(exc).lower()
    return any(
        marker in error_msg
        for marker in (
            "timeout",
            "timed out",
            "connection",
            "temporarily",
            "try again",
        )
    )


def _build_direction_input_payload(
    *,
    symbol: str,
    reference_price: float,
    timeframe_ohlcv: Dict[str, Any],
    current_position_snapshot: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    normalized_position_snapshot = dict(current_position_snapshot or {})
    raw_position = str(
        normalized_position_snapshot.get("direction") or normalized_position_snapshot.get("decision") or ""
    ).strip().upper()
    current_position = raw_position if raw_position in {"LONG", "SHORT"} else "NONE"
    return {
        "symbol": symbol,
        "current_price": reference_price,
        "current_position": current_position,
        "ohlcv": timeframe_ohlcv,
    }


def _build_direction_prompt(
    *,
    symbol: str,
    reference_price: float,
    timeframe_ohlcv: Dict[str, Any],
    current_position_snapshot: Optional[Dict[str, Any]] = None,
) -> str:
    payload = _build_direction_input_payload(
        symbol=symbol,
        reference_price=reference_price,
        timeframe_ohlcv=timeframe_ohlcv,
        current_position_snapshot=current_position_snapshot,
    )
    current_position = str(payload.get("current_position") or "NONE")
    return (
        "You are a disciplined BTCUSDT futures trader.\n"
        f"Current Price:{reference_price}, Current Position: {current_position}\n"
        "As a trader, rationally choose to hold or switch LONG/SHORT position. If Current Position is NONE, choose the higher-conviction directional setup between LONG and SHORT.\n"
        "Do not get shaken by ordinary and usual candles. Identify the dominant and important candles within the full market structure, and let them drive your directional decision.\n"
        "Use multi-timeframe OHLCV candles to judge market regime, structure, trend quality, timeframe alignment, continuation vs reversal, impulse vs correction, healthy pullback vs structural damage, exhaustion vs re-acceleration, breakout/breakdown acceptance vs failure, retest hold vs rejection, balance vs imbalance, volatility expansion vs compression, momentum and volume confirmation vs divergence, liquidity sweeps, and location relative to key swings and range boundaries.\n"
        "Since my assets are in your hands, please act responsibly.\n"
        "Schema: {\"decision\":\"LONG\"} or {\"decision\":\"SHORT\"}.\n"
        "Return JSON only.\n"
        "Within each timeframe array, candle rows are ordered from oldest to most recent.\n"
        "Input OHLCV candles' structure: {\"symbol\": str, \"current_price\": number, \"current_position\": \"LONG\"|\"SHORT\"|\"NONE\", \"ohlcv\": {timeframe: [[open, high, low, close, volume], ...]}}.\n\n"
        f"{json.dumps(payload, ensure_ascii=False)}"
    )


def _save_direction_analysis_data(
    cycle_dir: str,
    *,
    prompt: str,
    prompt_payload: Dict[str, Any],
    raw_response: str,
    decision: HakaiTradeDirectionDecision,
    usage_metadata: Optional[Dict[str, Any]],
    response_payload: Optional[Dict[str, Any]],
    thought_summary: str,
    thought_signatures: list[str],
    model: str,
    thinking_level: str,
    api_version: str,
) -> None:
    try:
        input_path = os.path.join(cycle_dir, "hakai_ai_input.json")
        with open(input_path, "w", encoding="utf-8") as file_obj:
            json.dump(
                {
                    "model": model,
                    "api_version": api_version,
                    "thinking_level": thinking_level,
                    "prompt": prompt,
                    "payload": prompt_payload,
                },
                file_obj,
                indent=2,
                ensure_ascii=False,
            )

        output_path = os.path.join(cycle_dir, "hakai_ai_output.json")
        with open(output_path, "w", encoding="utf-8") as file_obj:
            json.dump(
                {
                    "decision": decision.model_dump(),
                    "raw_response": raw_response,
                    "thought_summary": thought_summary,
                    "thought_signatures": thought_signatures,
                    "usage_metadata": usage_metadata or {},
                    "response_payload": response_payload or {},
                },
                file_obj,
                indent=2,
                ensure_ascii=False,
            )
        logger.info(
            "Gemini analysis persisted | %s",
            format_log_details(
                {
                    "cycle_dir": cycle_dir,
                    "input_path": input_path,
                    "output_path": output_path,
                    "decision": decision.model_dump(),
                    "thought_signatures": len(thought_signatures),
                    "thought_summary_chars": len(thought_summary or ""),
                }
            ),
        )
    except Exception as exc:
        logger.warning("Failed to save HAK GEMINI BINANCE TRADER AI analysis data: %s", exc)


def _call_gemini_direction_decision(
    *,
    prompt: str,
    api_version: str,
    thinking_level: str,
) -> Optional[GeminiStructuredResponse[HakaiTradeDirectionDecision]]:
    if genai is None or types is None:
        logger.error("google-genai dependency is unavailable")
        return None

    api_key = get_gemini_api_key()
    client = genai.Client(
        api_key=api_key,
        http_options={"api_version": api_version},
    )

    last_error: Optional[Exception] = None
    for attempt in range(1, GEMINI_GENERATE_MAX_RETRIES + 1):
        try:
            logger.info(
                "Gemini BTC direction call starting | %s",
                format_log_details(
                    {
                        "attempt": attempt,
                        "max_retries": GEMINI_GENERATE_MAX_RETRIES,
                        "model": GEMINI_DIRECTION_MODEL,
                        "api_version": api_version,
                        "thinking_level": thinking_level,
                        "prompt_chars": len(prompt or ""),
                    }
                ),
            )
            response = client.models.generate_content(
                model=GEMINI_DIRECTION_MODEL,
                contents=[prompt],
                config=types.GenerateContentConfig(
                    thinking_config=types.ThinkingConfig(
                        thinking_level=thinking_level,
                        include_thoughts=True,
                    ),
                    response_mime_type="application/json",
                    response_json_schema=HakaiTradeDirectionDecision.model_json_schema(),
                ),
            )
            decision = HakaiTradeDirectionDecision.model_validate_json(response.text)
            usage_metadata = extract_usage_metadata(response)
            response_payload = _extract_response_payload(response)
            log_usage_metadata(
                logger,
                context="Gemini BTC direction",
                usage_metadata=usage_metadata,
                model=GEMINI_DIRECTION_MODEL,
            )
            logger.info(
                "Gemini BTC direction call succeeded | %s",
                format_log_details(
                    {
                        "model": GEMINI_DIRECTION_MODEL,
                        "raw_response": str(getattr(response, "text", "") or ""),
                        "thought_summary_chars": len(extract_thought_summary(response_payload)),
                        "thought_signatures": len(extract_thought_signatures(response_payload)),
                    }
                ),
            )
            return GeminiStructuredResponse(
                decision=decision,
                raw_response=str(getattr(response, "text", "") or ""),
                usage_metadata=usage_metadata,
                response_payload=response_payload,
                thought_summary=extract_thought_summary(response_payload),
                thought_signatures=extract_thought_signatures(response_payload),
            )
        except Exception as exc:
            last_error = exc
            if not _is_retryable_gemini_error(exc) or attempt >= GEMINI_GENERATE_MAX_RETRIES:
                break
            retry_delay = 2 ** (attempt - 1)
            logger.warning(
                "Gemini BTC direction call failed (attempt %s/%s): %s. Retrying in %ss.",
                attempt,
                GEMINI_GENERATE_MAX_RETRIES,
                exc,
                retry_delay,
            )
            time.sleep(retry_delay)

    if last_error is not None:
        logger.error("Gemini BTC direction call failed: %s", last_error, exc_info=True)
    return None


def evaluate_hakai_direction(
    *,
    cycle_dir: str,
    symbol: str,
    reference_price: float,
    timeframe_ohlcv: Dict[str, Any],
    api_version: str,
    thinking_level: str,
    analysis_sink: Optional[Dict[str, Any]] = None,
    current_position_snapshot: Optional[Dict[str, Any]] = None,
) -> Optional[HakaiTradeDirectionDecision]:
    """Request a BTCUSDT LONG/SHORT decision and persist the analysis artifacts."""
    normalized_symbol = str(symbol or "").strip().upper()
    if normalized_symbol != "BTCUSDT":
        logger.error("evaluate_hakai_direction only supports BTCUSDT | received=%s", symbol)
        return None

    prompt_payload = _build_direction_input_payload(
        symbol=normalized_symbol,
        reference_price=reference_price,
        timeframe_ohlcv=timeframe_ohlcv,
        current_position_snapshot=current_position_snapshot,
    )
    prompt = _build_direction_prompt(
        symbol=normalized_symbol,
        reference_price=reference_price,
        timeframe_ohlcv=timeframe_ohlcv,
        current_position_snapshot=current_position_snapshot,
    )
    logger.info(
        "Evaluating HAK GEMINI BINANCE TRADER AI direction | %s",
        format_log_details(
            {
                "symbol": normalized_symbol,
                "reference_price": reference_price,
                "api_version": api_version,
                "thinking_level": thinking_level,
                "cycle_dir": cycle_dir,
                "timeframes": _summarize_timeframe_ohlcv(timeframe_ohlcv),
            }
        ),
    )

    try:
        call_result = _call_gemini_direction_decision(
            prompt=prompt,
            api_version=api_version,
            thinking_level=thinking_level,
        )
    except ValueError as exc:
        logger.error(str(exc))
        return None

    if call_result is None:
        return None

    decision = call_result.decision
    raw_response = call_result.raw_response or decision.model_dump_json(indent=2)
    normalized_value = str(getattr(decision, "decision", "") or "").strip().upper()
    if normalized_value not in {"LONG", "SHORT"}:
        logger.error("Gemini returned invalid BTC direction decision=%s", normalized_value)
        return None

    normalized_decision = HakaiTradeDirectionDecision(decision=normalized_value)
    _save_direction_analysis_data(
        cycle_dir,
        prompt=prompt,
        prompt_payload=prompt_payload,
        raw_response=raw_response,
        decision=normalized_decision,
        usage_metadata=call_result.usage_metadata,
        response_payload=call_result.response_payload,
        thought_summary=call_result.thought_summary,
        thought_signatures=call_result.thought_signatures,
        model=GEMINI_DIRECTION_MODEL,
        thinking_level=thinking_level,
        api_version=api_version,
    )
    logger.info(
        "HAK GEMINI BINANCE TRADER AI direction finalized | %s",
        format_log_details(
            {
                "symbol": normalized_symbol,
                "decision": normalized_value,
                "thought_signatures": len(call_result.thought_signatures),
                "thought_summary_chars": len(call_result.thought_summary or ""),
                "usage_metadata": call_result.usage_metadata,
                "cycle_dir": cycle_dir,
            }
        ),
    )
    if isinstance(analysis_sink, dict):
        analysis_sink.clear()
        analysis_sink.update(
            {
                "model": GEMINI_DIRECTION_MODEL,
                "api_version": api_version,
                "thinking_level": thinking_level,
                "decision": normalized_decision.model_dump(),
                "raw_response": raw_response,
                "thought_summary": call_result.thought_summary,
                "thought_signatures": list(call_result.thought_signatures),
                "usage_metadata": dict(call_result.usage_metadata or {}),
                "response_payload": dict(call_result.response_payload or {}),
                "input_path": os.path.join(cycle_dir, "hakai_ai_input.json"),
                "output_path": os.path.join(cycle_dir, "hakai_ai_output.json"),
            }
        )
    return normalized_decision


__all__ = [
    "GEMINI_DIRECTION_MODEL",
    "GEMINI_PRO_ONLY_MODEL",
    "GeminiStructuredResponse",
    "HakaiTradeDirectionDecision",
    "estimate_gemini_cost",
    "evaluate_hakai_direction",
    "extract_thought_signatures",
    "extract_thought_summary",
    "extract_usage_metadata",
    "log_usage_metadata",
]
