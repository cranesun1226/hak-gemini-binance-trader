"""Scheduling and notification orchestration for HAK GEMINI BINANCE TRADER."""

import json
import os
import re
import signal
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional, Sequence

from src.infra.logger import format_log_details, get_logger
from src.strategy.runtime_config import load_runtime_config
from src.infra.telegram import escape_telegram_html, sanitize_telegram_html, send_telegram_message
from src.strategy.hakai_strategy import run_hakai_cycle

logger = get_logger("scheduler")

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
STATE_FILE = os.path.join(ROOT_DIR, "scheduler_state.json")
CONFIG_PATH = os.path.join(ROOT_DIR, "setting.yaml")
STATE_VERSION = "8.0-hakai-fixed-15m"
SCHEDULE_SECOND_OFFSET = 10
HOURLY_REPORT_DELAY_SECONDS = 10

# Human-readable labels for logs and Telegram notifications.
TRIGGER_REASON_LABELS = {
    "no_position": "No open position",
    "missing_last_round_anchor": "Missing round anchor data",
    "round_distance_reached": "Next round reached",
    "waiting_for_next_round_trigger": "Waiting for next round",
    "price_distance_reached": "Next price level reached",
    "waiting_for_next_price_trigger": "Waiting for next price level",
}

ACTION_LABELS = {
    "init": "Initialized",
    "hold_waiting_round_trigger": "Waiting for next round",
    "hold_waiting_price_trigger": "Waiting for next price level",
    "hold_stop_updated": "Position kept, stop updated",
    "flat_no_entry": "Flat, no entry",
    "kept_position_by_ai": "Position kept by AI",
    "flipped_position_closed": "Position closed by flip",
    "flipped_to_flat": "Flipped to flat",
    "flipped_and_opened_position": "Flipped and opened new position",
    "opened_new_position": "Opened new position",
    "scaled_in_position": "Scaled in position",
    "scaled_out_position": "Scaled out position",
    "kept_position_size": "Kept position size",
    "reversed_position": "Reversed position",
    "close_position_failed": "Failed to close position",
    "entry_order_failed": "Entry order failed",
    "scale_in_failed": "Scale-in order failed",
    "scale_out_failed": "Scale-out order failed",
    "reverse_close_failed": "Failed to close opposite position",
    "reverse_reopen_failed": "Failed to reopen in opposite direction",
    "target_qty_below_min": "Order quantity below minimum",
    "skipped_entry_below_min_notional": "Skipped new entry below exchange min notional",
    "skipped_scale_in_below_min_notional": "Skipped scale-in below exchange min notional",
    "invalid_existing_position": "Invalid existing position data",
    "invalid_ai_decision": "Invalid AI decision",
    "credentials_error": "API credentials error",
    "positions_fetch_failed": "Failed to fetch positions",
    "reference_price_unavailable": "Failed to fetch current price",
    "ai_decision_failed": "AI decision failed",
    "account_equity_unavailable": "Failed to fetch account equity",
    "set_leverage_failed": "Failed to set leverage",
    "execution_failed": "Order execution failed",
    "executed": "Order executed",
}

STOP_SYNC_REASON_LABELS = {
    "stop_loss_already_synced": "Existing stop kept",
    "invalid_symbol_or_side": "Invalid symbol or side",
    "invalid_position_for_stop_sync": "Invalid position snapshot",
    "invalid_stop_loss": "Failed to calculate stop price",
    "stop_risk_basis_unavailable": "Missing stop-risk basis",
    "stop_risk_basis_mismatch": "Stop-risk basis mismatch",
    "place_stop_loss_failed": "Failed to place stop-loss order",
}

MARKUP_TAG_PATTERN = re.compile(
    r"</?(?:a|b|blockquote|code|em|i|ins|pre|s|span|strike|strong|tg-spoiler|u)(?:\s+[^>]*)?>",
    flags=re.IGNORECASE,
)


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _compact_log_details(details: Dict[str, Any]) -> Dict[str, Any]:
    return {
        key: value
        for key, value in details.items()
        if value is not None
    }


def _build_cycle_completion_log_details(result: Dict[str, Any]) -> Dict[str, Any]:
    position = result.get("position") if isinstance(result.get("position"), dict) else {}
    return _compact_log_details(
        {
            "success": bool(result.get("success")),
            "symbol": result.get("symbol"),
            "action": result.get("action"),
            "ai_triggered": bool(result.get("ai_triggered")),
            "ai_decision": result.get("ai_decision"),
            "trigger_reason": result.get("trigger_reason"),
            "current_price": result.get("current_price"),
            "position_direction": position.get("direction"),
            "position_size": position.get("size"),
            "entry_price": position.get("entry_price"),
            "stop_loss": position.get("stop_loss"),
            "next_trigger_down": result.get("next_trigger_down"),
            "next_trigger_up": result.get("next_trigger_up"),
            "cycle_dir": result.get("cycle_dir"),
        }
    )


class TradingScheduler:
    """Persist scheduler state and run HAK GEMINI BINANCE TRADER cycles on a fixed cadence."""

    def __init__(self) -> None:
        self.is_running = False
        self._shutdown_requested = False
        self.state_file_path = os.path.abspath(STATE_FILE)
        self.state = self.load_state()

        logger.info("Scheduler state file path: %s", self.state_file_path)
        logger.info("TradingScheduler initialized in HAK GEMINI BINANCE TRADER mode")

    def _default_state(self) -> Dict[str, Any]:
        return {
            "version": STATE_VERSION,
            "last_cycle_time": None,
            "last_minute_slot": None,
            "trigger_pct_usdt": None,
            "last_ai_trigger_price": None,
            "last_ai_triggered_at": None,
            "last_ai_decision": None,
            "next_trigger_down": None,
            "next_trigger_up": None,
            "stop_risk_basis": None,
            "last_cycle_result": None,
        }

    def _load_config(self) -> Dict[str, Any]:
        return load_runtime_config(CONFIG_PATH)

    def _get_cycle_interval_seconds(self) -> int:
        config = self._load_config()
        return max(1, _safe_int(config.get("cycle_interval_seconds", 60), 60))

    def load_state(self) -> Dict[str, Any]:
        default_state = self._default_state()
        try:
            if os.path.exists(self.state_file_path):
                with open(self.state_file_path, "r", encoding="utf-8") as file_obj:
                    loaded = json.load(file_obj)
                if isinstance(loaded, dict):
                    merged = dict(default_state)
                    merged.update(loaded)
                    has_legacy_trigger_state = "last_ai_trigger_round_price" in loaded
                    if str(loaded.get("version") or "").strip() != STATE_VERSION or has_legacy_trigger_state:
                        merged["last_ai_trigger_price"] = None
                        merged["next_trigger_down"] = None
                        merged["next_trigger_up"] = None
                    merged.pop("last_ai_trigger_round_price", None)
                    merged.pop("initial_entry_price", None)
                    merged.pop("initial_entry_direction", None)
                    merged.pop("position_sizing_activation_pct", None)
                    merged.pop("position_sizing_activation_price", None)
                    merged.pop("position_sizing_unlocked", None)
                    merged.pop("position_sizing_activated_at", None)
                    merged.pop("last_hourly_report_slot", None)
                    merged.pop("last_hourly_resize_slot", None)
                    merged["version"] = STATE_VERSION
                    return merged
        except json.JSONDecodeError as exc:
            logger.error("Corrupted state file: %s", exc)
        except Exception as exc:
            logger.error("Error loading scheduler state: %s", exc)
        return default_state

    def save_state(self) -> None:
        try:
            self.state.pop("initial_entry_price", None)
            self.state.pop("initial_entry_direction", None)
            self.state.pop("position_sizing_activation_pct", None)
            self.state.pop("position_sizing_activation_price", None)
            self.state.pop("position_sizing_unlocked", None)
            self.state.pop("position_sizing_activated_at", None)
            self.state.pop("last_hourly_report_slot", None)
            self.state.pop("last_hourly_resize_slot", None)
            with open(self.state_file_path, "w", encoding="utf-8") as file_obj:
                json.dump(self.state, file_obj, indent=2, ensure_ascii=False)
        except Exception as exc:
            logger.error("Error saving scheduler state: %s", exc)

    def _merge_state_update(self, update: Optional[Dict[str, Any]]) -> None:
        if not isinstance(update, dict):
            return
        for key, value in update.items():
            self.state[key] = value
        self.state.pop("last_ai_trigger_round_price", None)
        self.state.pop("initial_entry_price", None)
        self.state.pop("initial_entry_direction", None)
        self.state.pop("position_sizing_activation_pct", None)
        self.state.pop("position_sizing_activation_price", None)
        self.state.pop("position_sizing_unlocked", None)
        self.state.pop("position_sizing_activated_at", None)
        self.state.pop("last_hourly_report_slot", None)
        self.state.pop("last_hourly_resize_slot", None)

    def _summarize_cycle_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "time": datetime.now(timezone.utc).isoformat(),
            "success": bool(result.get("success")),
            "action": result.get("action"),
            "symbol": result.get("symbol"),
            "current_price": result.get("current_price"),
            "ai_triggered": bool(result.get("ai_triggered")),
            "ai_decision": result.get("ai_decision"),
            "trigger_reason": result.get("trigger_reason"),
            "trigger_price": result.get("trigger_price"),
            "next_trigger_down": result.get("next_trigger_down"),
            "next_trigger_up": result.get("next_trigger_up"),
            "cycle_dir": result.get("cycle_dir"),
        }

    def _format_timestamp(self, timestamp_value: Optional[str]) -> str:
        try:
            if timestamp_value:
                parsed = datetime.fromisoformat(timestamp_value)
            else:
                parsed = datetime.now(timezone.utc)
        except Exception:
            parsed = datetime.now(timezone.utc)

        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")

    def _format_float(self, value: Any, digits: int = 2) -> str:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return "-"
        return f"{parsed:,.{digits}f}"

    def _format_pct(self, value: Any) -> str:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return "-"
        return f"{parsed * 100.0:.2f}%"

    def _get_display_target_margin_ratio(self, snapshot: Any) -> Any:
        payload = snapshot if isinstance(snapshot, dict) else {}
        return payload.get("applied_target_margin_ratio", payload.get("target_margin_ratio"))

    def _get_display_target_effective_leverage(self, snapshot: Any) -> Any:
        payload = snapshot if isinstance(snapshot, dict) else {}
        return payload.get("applied_target_effective_leverage", payload.get("target_effective_leverage"))

    def _format_percentile_sizing_summary(self, snapshot: Any) -> str:
        payload = snapshot if isinstance(snapshot, dict) else {}
        try:
            target_margin_ratio = float(
                payload.get("applied_target_margin_ratio", payload.get("target_margin_ratio"))
            )
        except (TypeError, ValueError):
            target_margin_ratio = None
        if str(payload.get("position_sizing_mode") or "").strip().lower() == "fixed_ratio" and target_margin_ratio is not None:
            return f"Fixed {target_margin_ratio * 100.0:.2f}%"

        try:
            live_range_log = float(payload.get("live_range_log"))
            rank_estimate = float(payload.get("percentile_rank_estimate"))
            sample_size = int(payload.get("sample_size"))
            volatility_margin_ratio = float(
                payload.get("volatility_target_margin_ratio", payload.get("target_margin_ratio"))
            )
            target_margin_ratio = float(
                payload.get("applied_target_margin_ratio", payload.get("target_margin_ratio"))
            )
        except (TypeError, ValueError):
            return "-"

        location = str(payload.get("percentile_location") or "").strip().lower()
        if location == "below_sample_range":
            location_label = "below"
        elif location == "above_sample_range":
            location_label = "above"
        else:
            location_label = "in-range"

        activation_price = payload.get("position_sizing_activation_price")
        activation_pct = payload.get("position_sizing_activation_pct")
        is_unlocked = bool(payload.get("position_sizing_unlocked"))
        keep_current_position_size = bool(payload.get("keep_current_position_size"))
        initial_position_size_ratio = payload.get("initial_position_size_ratio")
        if keep_current_position_size:
            status_text = "Bootstrap hold"
        elif not is_unlocked and activation_price is not None:
            locked_until_text = f"Locked until {self._format_usdt(activation_price, digits=0)}"
            try:
                if activation_pct is not None:
                    locked_until_text = (
                        f"Locked {float(activation_pct) * 100.0:.2f}% until "
                        f"{self._format_usdt(activation_price, digits=0)}"
                    )
            except (TypeError, ValueError):
                locked_until_text = f"Locked until {self._format_usdt(activation_price, digits=0)}"
            status_text = (
                f"Final {target_margin_ratio * 100.0:.2f}% "
                f"({locked_until_text})"
            )
        else:
            floor_kept = False
            try:
                floor_kept = float(target_margin_ratio) <= float(initial_position_size_ratio) + 1e-9
            except (TypeError, ValueError):
                floor_kept = False
            status_text = (
                f"Unlocked, floor {target_margin_ratio * 100.0:.2f}%"
                if is_unlocked and floor_kept
                else f"Final {target_margin_ratio * 100.0:.2f}%"
            )

        return (
            f"24h ln {live_range_log:.4f} | "
            f"Rank {rank_estimate:.1f}/{sample_size} | "
            f"{location_label} | "
            f"Vol {volatility_margin_ratio * 100.0:.2f}% | "
            f"{status_text}"
        )

    def _clip_text(self, value: Any, *, limit: int) -> str:
        text = self._strip_markup(value)
        if not text:
            return "-"
        return text if len(text) <= limit else f"{text[:limit - 3]}..."

    def _format_html_title(self, title: str, *, emoji: str) -> str:
        return f"{emoji} <b>{escape_telegram_html(title)}</b>"

    def _format_html_value(
        self,
        value: Any,
        *,
        code: bool = False,
        bold: bool = False,
        preserve_html: bool = False,
    ) -> str:
        text = str(value or "").strip()
        if not text or text in {"-", "없음"}:
            return "<i>None</i>"

        safe_text = sanitize_telegram_html(text) if preserve_html else escape_telegram_html(text)
        if code:
            return f"<code>{safe_text}</code>"
        if bold:
            return f"<b>{safe_text}</b>"
        return safe_text

    def _format_html_line(
        self,
        label: str,
        value: Any,
        *,
        code: bool = False,
        bold: bool = False,
        preserve_html: bool = False,
    ) -> str:
        return (
            f"<b>{escape_telegram_html(label)}:</b> "
            f"{self._format_html_value(value, code=code, bold=bold, preserve_html=preserve_html)}"
        )

    def _format_html_summary(self, value: Any) -> str:
        return sanitize_telegram_html(value)

    def _format_cycle_dir(self, cycle_dir: Any) -> str:
        path = str(cycle_dir or "").strip()
        if not path:
            return "-"
        return os.path.basename(path.rstrip("/")) or path

    def _strip_markup(self, value: Any) -> str:
        text = str(value or "").strip()
        if not text:
            return ""
        cleaned = MARKUP_TAG_PATTERN.sub("", text)
        cleaned = cleaned.replace("**", "").replace("__", "").replace("`", "")
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    def _humanize_code_label(self, value: Any) -> str:
        text = self._strip_markup(value)
        if not text:
            return "None"
        return text.replace("_", " ")

    def _translate_trigger_reason(self, value: Any) -> str:
        key = str(value or "").strip()
        if key in ("", "-"):
            return "None"
        return TRIGGER_REASON_LABELS.get(key, self._humanize_code_label(key))

    def _translate_action(self, value: Any) -> str:
        key = str(value or "").strip()
        if key in ("", "-"):
            return "None"
        if key.startswith("unsupported_open_position:"):
            symbol = key.split(":", 1)[1] or "-"
            return f"Unsupported open position ({symbol})"
        if key.startswith("multiple_open_positions:"):
            symbols = key.split(":", 1)[1] or "-"
            return f"Multiple open positions detected ({symbols})"
        return ACTION_LABELS.get(key, self._humanize_code_label(key))

    def _translate_stop_sync_reason(self, value: Any) -> str:
        key = str(value or "").strip()
        if key in ("", "-"):
            return "None"
        return STOP_SYNC_REASON_LABELS.get(key, self._humanize_code_label(key))

    def _format_usdt(self, value: Any, *, digits: int = 2) -> str:
        formatted = self._format_float(value, digits=digits)
        return f"{formatted} USDT" if formatted != "-" else "-"

    def _format_trigger_window(self, down_price: Any, up_price: Any) -> str:
        down_text = self._format_usdt(down_price, digits=0)
        up_text = self._format_usdt(up_price, digits=0)
        if down_text == "-" and up_text == "-":
            return "-"
        if down_text == "-":
            return up_text
        if up_text == "-":
            return down_text
        return f"{down_text} ~ {up_text}"

    def _format_position_summary(self, position: Any) -> str:
        if not isinstance(position, dict):
            return "None"

        direction = str(position.get("direction") or "").strip().upper()
        size = self._format_float(position.get("size"), 3)
        entry_price = self._format_float(position.get("entry_price"))
        stop_loss = self._format_float(position.get("stop_loss"))

        segments = []
        if direction:
            segments.append(direction)
        if size != "-":
            segments.append(f"{size} BTC")
        if entry_price != "-":
            segments.append(f"Entry {entry_price}")
        if stop_loss != "-":
            segments.append(f"Stop {stop_loss}")

        return " / ".join(segments) if segments else "None"

    def _format_position_line(self, position: Any, *, label: str) -> str:
        return f"{label}: {self._format_position_summary(position)}"

    def _format_execution_summary(self, execution: Any) -> str:
        if not isinstance(execution, dict):
            return "No details"
        action = self._translate_action(execution.get("action"))
        qty = execution.get("qty")
        qty_text = self._format_float(qty, 6) if qty is not None else "-"
        order_notional_text = self._format_usdt(execution.get("order_notional_usdt"))
        min_notional_text = self._format_usdt(execution.get("min_notional_usdt"))

        segments = [action]
        if qty_text != "-":
            segments.append(f"Qty {qty_text} BTC")
        if order_notional_text != "-":
            segments.append(f"Order notional {order_notional_text}")
        if min_notional_text != "-":
            segments.append(f"Min notional {min_notional_text}")
        return " / ".join(segments)

    def _format_stop_sync_summary(self, result: Dict[str, Any]) -> str:
        stop_sync = result.get("post_trade_stop_sync") or result.get("stop_sync")
        if not isinstance(stop_sync, dict):
            return "No details"

        reason = self._translate_stop_sync_reason(stop_sync.get("reason"))
        if bool(stop_sync.get("success")):
            stop_price = self._format_float(stop_sync.get("stop_loss"))
            segments = [reason if reason and reason != "None" else "Synced", f"Stop {stop_price}"]
            risk_pct = self._format_pct(stop_sync.get("stop_loss_account_risk_pct"))
            if risk_pct != "-":
                segments.append(f"Risk {risk_pct}")
            distance_pct = self._format_pct(stop_sync.get("stop_loss_distance_pct"))
            if distance_pct != "-":
                segments.append(f"Dist {distance_pct}")
            basis_leverage = self._format_float(stop_sync.get("basis_effective_leverage"))
            if basis_leverage != "-":
                segments.append(f"Basis {basis_leverage}x")
            return " / ".join(segments)

        error_code = stop_sync.get("error_code")
        error_message = self._clip_text(stop_sync.get("error_message"), limit=140)
        if error_code not in (None, ""):
            return f"Failed / code {error_code} / {error_message}"
        if error_message != "-":
            return f"Failed / {error_message}"
        if reason and reason != "None":
            return f"Failed / {reason}"
        return "Failed"

    def _hourly_slot_start(self, cycle_time: datetime) -> datetime:
        if cycle_time.tzinfo is None:
            cycle_time = cycle_time.replace(tzinfo=timezone.utc)
        return cycle_time.replace(minute=0, second=0, microsecond=0)

    def _hourly_slot(self, cycle_time: datetime) -> str:
        return self._hourly_slot_start(cycle_time).isoformat()

    def _is_hourly_report_cycle(self, cycle_time: datetime) -> bool:
        return cycle_time >= (
            self._hourly_slot_start(cycle_time) + timedelta(seconds=HOURLY_REPORT_DELAY_SECONDS)
        )

    def _cycle_bucket_start(self, cycle_time: datetime, *, interval_seconds: int) -> datetime:
        if cycle_time.tzinfo is None:
            cycle_time = cycle_time.replace(tzinfo=timezone.utc)
        resolved_interval_seconds = max(1, int(interval_seconds))
        epoch_seconds = int(cycle_time.timestamp())
        bucket_seconds = (epoch_seconds // resolved_interval_seconds) * resolved_interval_seconds
        return datetime.fromtimestamp(bucket_seconds, tz=timezone.utc)

    def _cycle_bucket_slot(self, cycle_time: datetime, *, interval_seconds: int) -> str:
        return self._cycle_bucket_start(cycle_time, interval_seconds=interval_seconds).isoformat()

    def _cycle_due_time(
        self,
        cycle_time: datetime,
        *,
        interval_seconds: int,
        offset_seconds: int,
    ) -> datetime:
        resolved_interval_seconds = max(1, int(interval_seconds))
        resolved_offset_seconds = int(offset_seconds) % resolved_interval_seconds
        return self._cycle_bucket_start(
            cycle_time,
            interval_seconds=resolved_interval_seconds,
        ) + timedelta(seconds=resolved_offset_seconds)

    def _last_cycle_bucket_slot(self, *, interval_seconds: int) -> Optional[str]:
        last_cycle_time = str(self.state.get("last_cycle_time") or "").strip()
        if not last_cycle_time:
            return None
        try:
            parsed = datetime.fromisoformat(last_cycle_time)
        except ValueError:
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return self._cycle_bucket_slot(parsed, interval_seconds=interval_seconds)

    def _should_run_immediate_cycle(
        self,
        now_utc: datetime,
        *,
        interval_seconds: int,
        offset_seconds: int,
    ) -> bool:
        if now_utc.tzinfo is None:
            now_utc = now_utc.replace(tzinfo=timezone.utc)
        if now_utc < self._cycle_due_time(
            now_utc,
            interval_seconds=interval_seconds,
            offset_seconds=offset_seconds,
        ):
            return False
        current_bucket_slot = self._cycle_bucket_slot(now_utc, interval_seconds=interval_seconds)
        return self._last_cycle_bucket_slot(interval_seconds=interval_seconds) != current_bucket_slot

    def _build_message(
        self,
        *,
        title: str,
        summary_lines: Sequence[str],
        sections: Sequence[tuple[str, Sequence[str], bool]],
    ) -> str:
        blocks = [title]
        normalized_summary = [line for line in summary_lines if str(line or "").strip()]
        if normalized_summary:
            blocks.append("\n".join(normalized_summary))

        for section_title, lines, bulleted in sections:
            visible_lines = [
                f"• {line}" if bulleted else str(line)
                for line in lines
                if str(line or "").strip()
            ]
            if not visible_lines:
                continue
            blocks.append("\n".join([section_title, *visible_lines]))

        return "\n\n".join(blocks)

    # Telegram notification rendering.
    def _build_ai_cycle_before_message(self, payload: Dict[str, Any]) -> str:
        position_sizing = (
            payload.get("position_sizing")
            if isinstance(payload.get("position_sizing"), dict)
            else payload.get("volatility_snapshot")
            if isinstance(payload.get("volatility_snapshot"), dict)
            else {}
        )
        return self._build_message(
            title=self._format_html_title("HAK GEMINI BINANCE TRADER | AI Cycle Start", emoji="🚦"),
            summary_lines=[
                self._format_html_line("Symbol", payload.get("symbol"), code=True),
                self._format_html_line("Price", self._format_usdt(payload.get("current_price"))),
            ],
            sections=[
                (
                    self._format_html_title("Trigger", emoji="🎯"),
                    [
                        self._format_html_line(
                            "Reason",
                            self._translate_trigger_reason(payload.get("trigger_reason")),
                        ),
                        self._format_html_line(
                            "Trigger Price",
                            self._format_usdt(payload.get("trigger_price")),
                        ),
                        self._format_html_line(
                            "Next Level",
                            self._format_trigger_window(
                                payload.get("next_trigger_down"),
                                payload.get("next_trigger_up"),
                            ),
                        ),
                    ],
                    True,
                ),
                (
                    self._format_html_title("Position", emoji="💼"),
                    [
                        self._format_html_line("Now", self._format_position_summary(payload.get("position"))),
                        self._format_html_line(
                            "Target",
                            self._format_pct(self._get_display_target_margin_ratio(position_sizing)),
                        ),
                        self._format_html_line(
                            "Sizing",
                            self._format_percentile_sizing_summary(position_sizing),
                        ),
                    ],
                    True,
                ),
            ],
        )

    def _build_ai_cycle_after_message(self, payload: Dict[str, Any]) -> str:
        analysis = payload.get("analysis") if isinstance(payload.get("analysis"), dict) else {}
        thought_summary = self._format_html_summary(analysis.get("thought_summary"))

        sections: list[tuple[str, Sequence[str], bool]] = [
            (
                self._format_html_title("Standard", emoji="🧭"),
                [
                    self._format_html_line("Position", self._format_position_summary(payload.get("position"))),
                ],
                True,
            ),
        ]
        if thought_summary:
            sections.append(
                (
                    self._format_html_title("AI Thinking", emoji="📝"),
                    [thought_summary],
                    False,
                )
            )

        return self._build_message(
            title=self._format_html_title("AI Decision", emoji="🧠"),
            summary_lines=[
                self._format_html_line("Symbol", payload.get("symbol"), code=True),
                self._format_html_line("Decision", payload.get("decision") or "None", bold=True),
            ],
            sections=sections,
        )

    def _build_cycle_completed_message(self, payload: Dict[str, Any]) -> str:
        position_sizing = (
            payload.get("position_sizing")
            if isinstance(payload.get("position_sizing"), dict)
            else payload.get("volatility_snapshot")
            if isinstance(payload.get("volatility_snapshot"), dict)
            else {}
        )
        is_ai_cycle = bool(payload.get("ai_triggered"))
        title = (
            "HAK GEMINI BINANCE TRADER | AI Cycle Done"
            if is_ai_cycle
            else "HAK GEMINI BINANCE TRADER | Cycle Update"
        )
        direction = payload.get("ai_decision") or payload.get("last_ai_decision") or "None"
        stop_sync = payload.get("post_trade_stop_sync") or payload.get("stop_sync")

        position_lines = [
            self._format_html_line("Before", self._format_position_summary(payload.get("position_before"))),
            self._format_html_line("After", self._format_position_summary(payload.get("position"))),
        ]
        if isinstance(stop_sync, dict):
            position_lines.append(self._format_html_line("Stop Sync", self._format_stop_sync_summary(payload)))
        position_lines.append(
            self._format_html_line(
                "Next Level",
                self._format_trigger_window(
                    payload.get("next_trigger_down"),
                    payload.get("next_trigger_up"),
                ),
            )
        )

        sections: list[tuple[str, Sequence[str], bool]] = [
            (
                self._format_html_title("Position", emoji="💼"),
                position_lines,
                True,
            ),
        ]

        has_account_data = bool(position_sizing) or any(
            payload.get(key) is not None for key in ("account_equity", "target_notional_usdt")
        )
        if has_account_data:
            sections.append(
                (
                    self._format_html_title("Account", emoji="💰"),
                    [
                        self._format_html_line("Asset", self._format_usdt(payload.get("account_equity"))),
                        self._format_html_line(
                            "Position Pct",
                            self._format_pct(self._get_display_target_margin_ratio(position_sizing)),
                        ),
                        self._format_html_line(
                            "Actual Leverage",
                            f"{self._format_float(self._get_display_target_effective_leverage(position_sizing))}x",
                        ),
                        self._format_html_line("Target Amount", self._format_usdt(payload.get("target_notional_usdt"))),
                        self._format_html_line(
                            "Sizing",
                            self._format_percentile_sizing_summary(position_sizing),
                        ),
                    ],
                    True,
                )
            )

        return self._build_message(
            title=self._format_html_title(title, emoji="📈"),
            summary_lines=[
                self._format_html_line("Symbol", payload.get("symbol"), code=True),
                self._format_html_line("Price", self._format_usdt(payload.get("current_price"))),
                self._format_html_line("Action", self._translate_action(payload.get("action"))),
                self._format_html_line("Trigger", self._translate_trigger_reason(payload.get("trigger_reason"))),
                self._format_html_line("Decision", direction, bold=True),
            ],
            sections=sections,
        )

    def _build_hourly_status_message(self, payload: Dict[str, Any]) -> str:
        return self._build_message(
            title=self._format_html_title("HAK GEMINI BINANCE TRADER | 1 Hour Report", emoji="⏰"),
            summary_lines=[
                self._format_html_line("Symbol", payload.get("symbol"), code=True),
                self._format_html_line("Price", self._format_usdt(payload.get("current_price"))),
            ],
            sections=[
                (
                    self._format_html_title("Current Account", emoji="📌"),
                    [
                        self._format_html_line("Position", self._format_position_summary(payload.get("position"))),
                        self._format_html_line("Status", self._translate_action(payload.get("action"))),
                        self._format_html_line(
                            "AI Decision",
                            payload.get("last_ai_decision") or payload.get("ai_decision") or "None",
                            bold=True,
                        ),
                        self._format_html_line(
                            "Next Level",
                            self._format_trigger_window(
                                payload.get("next_trigger_down"),
                                payload.get("next_trigger_up"),
                            ),
                        ),
                    ],
                    True,
                ),
            ],
        )

    def _build_exception_message(self, payload: Dict[str, Any]) -> str:
        return self._build_message(
            title=self._format_html_title("HAK GEMINI BINANCE TRADER | Scheduler Error", emoji="⚠️"),
            summary_lines=[
                self._format_html_line("Time", self._format_timestamp(payload.get("timestamp"))),
                self._format_html_line("Error", self._clip_text(payload.get("error"), limit=600)),
            ],
            sections=[],
        )

    def _emit_telegram_text(self, message: str) -> bool:
        if not str(message or "").strip():
            return False
        return bool(send_telegram_message(message))

    def _notify_telegram_event(self, event_name: str, payload: Dict[str, Any]) -> None:
        if event_name == "ai_cycle_before":
            sent = self._emit_telegram_text(self._build_ai_cycle_before_message(payload))
            logger.info(
                "Telegram event dispatched | %s",
                format_log_details(
                    {
                        "event": event_name,
                        "sent": sent,
                        "symbol": payload.get("symbol"),
                        "cycle_dir": payload.get("cycle_dir"),
                    }
                ),
            )
            return
        if event_name == "ai_cycle_after":
            sent = self._emit_telegram_text(self._build_ai_cycle_after_message(payload))
            logger.info(
                "Telegram event dispatched | %s",
                format_log_details(
                    {
                        "event": event_name,
                        "sent": sent,
                        "symbol": payload.get("symbol"),
                        "cycle_dir": payload.get("cycle_dir"),
                        "decision": payload.get("decision"),
                    }
                ),
            )
            return

    def _maybe_send_cycle_notifications(self, cycle_time: datetime, result: Dict[str, Any]) -> None:
        if bool(result.get("ai_triggered")):
            cycle_payload = dict(result)
            cycle_payload["timestamp"] = cycle_time.isoformat()
            cycle_payload["last_ai_decision"] = self.state.get("last_ai_decision")
            self._emit_telegram_text(self._build_cycle_completed_message(cycle_payload))

    def _next_cycle_boundary(
        self,
        now_utc: datetime,
        *,
        interval_seconds: int,
        offset_seconds: int = 0,
        include_current: bool = False,
    ) -> datetime:
        if now_utc.tzinfo is None:
            now_utc = now_utc.replace(tzinfo=timezone.utc)

        interval_seconds = max(1, int(interval_seconds))
        offset_seconds = int(offset_seconds) % interval_seconds
        epoch_seconds = int(now_utc.timestamp())
        boundary_seconds = (
            ((epoch_seconds - offset_seconds) // interval_seconds) * interval_seconds
        ) + offset_seconds
        boundary = datetime.fromtimestamp(boundary_seconds, tz=timezone.utc)
        if include_current and now_utc == boundary:
            return boundary
        if now_utc < boundary:
            return boundary
        return datetime.fromtimestamp(boundary_seconds + interval_seconds, tz=timezone.utc)

    # Scheduler execution entrypoints.
    def run_cycle_once(self, now_utc: Optional[datetime] = None) -> Dict[str, Any]:
        """Execute one scheduler-controlled HAK GEMINI BINANCE TRADER cycle and persist its state."""
        cycle_time = now_utc or datetime.now(timezone.utc)
        if cycle_time.tzinfo is None:
            cycle_time = cycle_time.replace(tzinfo=timezone.utc)
        logger.debug(
            "Scheduler cycle starting | %s",
            format_log_details(
                {
                    "cycle_time": cycle_time.isoformat(),
                    "last_ai_decision": self.state.get("last_ai_decision"),
                    "last_ai_trigger_price": self.state.get("last_ai_trigger_price"),
                    "next_trigger_down": self.state.get("next_trigger_down"),
                    "next_trigger_up": self.state.get("next_trigger_up"),
                }
            ),
        )

        result = run_hakai_cycle(
            state=dict(self.state),
            as_of_ms=int(cycle_time.timestamp() * 1000),
            notification_callback=self._notify_telegram_event,
        )
        self._merge_state_update(result.get("state_update"))
        self.state["last_cycle_time"] = cycle_time.isoformat()
        self.state["last_minute_slot"] = cycle_time.replace(second=0, microsecond=0).isoformat()
        self.state["last_cycle_result"] = self._summarize_cycle_result(result)
        self.save_state()
        self._maybe_send_cycle_notifications(cycle_time, result)
        logger.debug(
            "Scheduler cycle state persisted | %s",
            format_log_details(
                {
                    "cycle_time": cycle_time.isoformat(),
                    "state_last_ai_decision": self.state.get("last_ai_decision"),
                    "state_last_ai_trigger_price": self.state.get("last_ai_trigger_price"),
                    "state_next_trigger_down": self.state.get("next_trigger_down"),
                    "state_next_trigger_up": self.state.get("next_trigger_up"),
                    "result_action": result.get("action"),
                    "result_cycle_dir": result.get("cycle_dir"),
                }
            ),
        )
        return result

    def minute_mechanical_check(self, now_utc: datetime) -> None:
        """Run one protected cycle and convert unexpected failures into alerts."""
        try:
            result = self.run_cycle_once(now_utc)
            logger.info(
                "Cycle completed | %s",
                format_log_details(_build_cycle_completion_log_details(result)),
            )
        except Exception as exc:
            logger.error("Error in minute_mechanical_check: %s", exc, exc_info=True)
            self._emit_telegram_text(
                self._build_exception_message(
                    {
                        "timestamp": now_utc.isoformat(),
                        "error": str(exc),
                    }
                )
            )

    def _signal_handler(self, signum: int, _frame: Any) -> None:
        signal_name = signal.Signals(signum).name
        logger.info("Received %s, initiating graceful shutdown...", signal_name)
        self._shutdown_requested = True

    def run_forever(self) -> None:
        """Run the scheduler loop until shutdown is requested."""
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        self.is_running = True
        interval_seconds = self._get_cycle_interval_seconds()

        logger.info("=" * 60)
        logger.info("=== HAK GEMINI BINANCE TRADER Trading Scheduler Started ===")
        logger.info("=" * 60)
        logger.info("Cycle interval: %s seconds", interval_seconds)
        logger.info("Scheduled second offset: %s seconds", SCHEDULE_SECOND_OFFSET)
        logger.info("Press Ctrl+C to stop")
        logger.info("=" * 60)

        try:
            startup_now = datetime.now(timezone.utc)
            if self._should_run_immediate_cycle(
                startup_now,
                interval_seconds=interval_seconds,
                offset_seconds=SCHEDULE_SECOND_OFFSET,
            ):
                logger.info(
                    "Running immediate catch-up cycle | now=%s interval_seconds=%s",
                    startup_now.isoformat(),
                    interval_seconds,
                )
                self.minute_mechanical_check(startup_now)

            next_cycle = self._next_cycle_boundary(
                datetime.now(timezone.utc),
                interval_seconds=interval_seconds,
                offset_seconds=SCHEDULE_SECOND_OFFSET,
                include_current=False,
            )

            while self.is_running and not self._shutdown_requested:
                now_utc = datetime.now(timezone.utc)
                if now_utc >= next_cycle:
                    self.minute_mechanical_check(now_utc)
                    next_cycle = self._next_cycle_boundary(
                        datetime.now(timezone.utc),
                        interval_seconds=interval_seconds,
                        offset_seconds=SCHEDULE_SECOND_OFFSET,
                        include_current=False,
                    )

                sleep_seconds = (next_cycle - datetime.now(timezone.utc)).total_seconds()
                if sleep_seconds < 0.25:
                    sleep_seconds = 0.25
                time.sleep(sleep_seconds)
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")
        except Exception as exc:
            logger.error("Scheduler loop error: %s", exc, exc_info=True)
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        """Persist final state and stop the scheduler loop."""
        self.is_running = False
        self._shutdown_requested = True
        self.save_state()
        logger.info("Trading scheduler stopped")
