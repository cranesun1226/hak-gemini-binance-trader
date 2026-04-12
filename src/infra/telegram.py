"""Telegram message formatting and delivery helpers."""

from __future__ import annotations

import html
import re
from typing import Iterable

try:
    import requests
except ModuleNotFoundError:  # pragma: no cover - exercised in lightweight test environments
    class _RequestsFallback:
        def post(self, *_args, **_kwargs):
            raise ModuleNotFoundError("requests is required to call Telegram APIs")

    requests = _RequestsFallback()

from src.infra.env_loader import load_env_var
from src.infra.logger import format_log_details, get_logger

logger = get_logger("telegram")

_TELEGRAM_API_BASE_URL = "https://api.telegram.org"
_MAX_TELEGRAM_MESSAGE_LENGTH = 3500
_HTML_TOKEN_PATTERN = re.compile(r"(<[^>]+>)")
_HTML_BOLD_PATTERN = re.compile(r"(\*\*|__)(.+?)\1", flags=re.DOTALL)
_HTML_CODE_PATTERN = re.compile(r"`([^`\n]+)`")
_ALLOWED_HTML_TAG_PATTERN = re.compile(
    r"</?(?:b|strong|i|em|code|pre|u|ins|s|strike|blockquote|tg-spoiler)\s*>",
    flags=re.IGNORECASE,
)
_HTML_TAG_NAME_PATTERN = re.compile(r"</?\s*([a-zA-Z0-9-]+)")
_HTML_TAG_ALIASES = {
    "strong": "b",
    "em": "i",
    "ins": "u",
    "strike": "s",
}
_DUPLICATED_TAG_PATTERNS = (
    (re.compile(r"<b>\s*<b>(.*?)</b>\s*</b>", flags=re.DOTALL), "b"),
    (re.compile(r"<i>\s*<i>(.*?)</i>\s*</i>", flags=re.DOTALL), "i"),
    (re.compile(r"<code>\s*<code>(.*?)</code>\s*</code>", flags=re.DOTALL), "code"),
)


def escape_telegram_html(value: object) -> str:
    return html.escape(str(value or "").strip(), quote=False)


def _convert_basic_markdown_to_html(text: str) -> str:
    converted = _HTML_CODE_PATTERN.sub(r"<code>\1</code>", text)
    converted = _HTML_BOLD_PATTERN.sub(r"<b>\2</b>", converted)
    return converted


def sanitize_telegram_html(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return ""

    converted = _convert_basic_markdown_to_html(text)
    placeholders: dict[str, str] = {}

    def _replace_allowed_tag(match: re.Match[str]) -> str:
        raw_tag = match.group(0)
        name_match = _HTML_TAG_NAME_PATTERN.match(raw_tag)
        if name_match is None:
            return raw_tag

        tag_name = name_match.group(1).lower()
        canonical_name = _HTML_TAG_ALIASES.get(tag_name, tag_name)
        safe_tag = f"</{canonical_name}>" if raw_tag.startswith("</") else f"<{canonical_name}>"
        placeholder = f"__TG_HTML_TAG_{len(placeholders)}__"
        placeholders[placeholder] = safe_tag
        return placeholder

    masked = _ALLOWED_HTML_TAG_PATTERN.sub(_replace_allowed_tag, converted)
    escaped = html.escape(masked, quote=False)
    for placeholder, safe_tag in placeholders.items():
        escaped = escaped.replace(placeholder, safe_tag)
    for pattern, tag_name in _DUPLICATED_TAG_PATTERNS:
        while True:
            collapsed = pattern.sub(rf"<{tag_name}>\1</{tag_name}>", escaped)
            if collapsed == escaped:
                break
            escaped = collapsed
    return escaped


def _tokenize_html(text: str) -> list[str]:
    parts = _HTML_TOKEN_PATTERN.split(text)
    return [part for part in parts if part]


def _parse_html_tag(token: str) -> tuple[bool, str] | None:
    name_match = _HTML_TAG_NAME_PATTERN.match(token)
    if name_match is None:
        return None

    tag_name = _HTML_TAG_ALIASES.get(name_match.group(1).lower(), name_match.group(1).lower())
    if tag_name not in {"b", "i", "code", "pre", "u", "s", "blockquote", "tg-spoiler"}:
        return None
    return (token.startswith("</"), tag_name)


def _closing_suffix(open_tags: list[str]) -> str:
    return "".join(f"</{tag_name}>" for tag_name in reversed(open_tags))


def _opening_prefix(open_tags: list[str]) -> str:
    return "".join(f"<{tag_name}>" for tag_name in open_tags)


def _take_text_piece(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text

    newline_idx = text.rfind("\n", 0, limit + 1)
    if newline_idx >= max(limit // 2, 1):
        return text[: newline_idx + 1]

    space_idx = text.rfind(" ", 0, limit + 1)
    if space_idx >= max(limit // 2, 1):
        return text[: space_idx + 1]

    return text[:limit]


def _split_message(text: str, *, max_length: int = _MAX_TELEGRAM_MESSAGE_LENGTH) -> Iterable[str]:
    normalized = str(text or "").strip()
    if not normalized:
        return []

    chunks: list[str] = []
    open_tags: list[str] = []
    current = ""
    current_visible_chars = 0

    def _finalize_chunk() -> None:
        nonlocal current, current_visible_chars
        chunk = f"{current}{_closing_suffix(open_tags)}".strip()
        if re.sub(_HTML_TOKEN_PATTERN, "", chunk).strip():
            chunks.append(chunk)
        current = _opening_prefix(open_tags)
        current_visible_chars = 0

    current = _opening_prefix(open_tags)

    for token in _tokenize_html(normalized):
        tag_info = _parse_html_tag(token)
        if tag_info is not None:
            is_closing, tag_name = tag_info
            next_open_tags = list(open_tags)
            if is_closing:
                if next_open_tags and next_open_tags[-1] == tag_name:
                    next_open_tags.pop()
            else:
                next_open_tags.append(tag_name)

            candidate = f"{current}{token}{_closing_suffix(next_open_tags)}"
            if len(candidate) > max_length and current_visible_chars > 0:
                _finalize_chunk()
            current += token
            open_tags = next_open_tags
            continue

        remaining = token
        while remaining:
            available = max_length - len(current) - len(_closing_suffix(open_tags))
            if available <= 0:
                _finalize_chunk()
                continue

            piece = _take_text_piece(remaining, available)
            current += piece
            current_visible_chars += len(piece)
            remaining = remaining[len(piece) :]
            if remaining:
                _finalize_chunk()

    if current_visible_chars > 0:
        chunks.append(f"{current}{_closing_suffix(open_tags)}")

    return chunks


def send_telegram_message(text: str) -> bool:
    """Send a Telegram message, splitting long HTML-safe payloads when needed."""
    bot_token = load_env_var("TELEGRAM_BOT_TOKEN")
    chat_id = load_env_var("TELEGRAM_CHAT_ID")
    if not bot_token or not chat_id:
        logger.info("Telegram is not configured; skipping notification")
        return False

    message_chunks = list(_split_message(text))
    if not message_chunks:
        return False

    url = f"{_TELEGRAM_API_BASE_URL}/bot{bot_token}/sendMessage"
    success = True
    logger.info(
        "Sending Telegram message | %s",
        format_log_details(
            {
                "chunks": len(message_chunks),
                "chat_id": chat_id,
                "total_chars": len(text or ""),
            }
        ),
    )

    for idx, chunk in enumerate(message_chunks, start=1):
        try:
            response = requests.post(
                url,
                data={
                    "chat_id": chat_id,
                    "parse_mode": "HTML",
                    "text": chunk,
                    "disable_web_page_preview": "true",
                },
                timeout=15,
            )
            payload = response.json()
        except Exception as exc:
            logger.warning("Failed to send Telegram message: %s", exc)
            success = False
            continue

        if response.status_code >= 400 or not isinstance(payload, dict) or not bool(payload.get("ok")):
            logger.warning(
                "Telegram API rejected message | status=%s payload=%s",
                response.status_code,
                payload,
            )
            success = False
        else:
            logger.info(
                "Telegram message chunk delivered | %s",
                format_log_details(
                    {
                        "chunk_index": idx,
                        "chunk_chars": len(chunk),
                        "status": response.status_code,
                    }
                ),
            )

    return success


__all__ = ["escape_telegram_html", "sanitize_telegram_html", "send_telegram_message"]
