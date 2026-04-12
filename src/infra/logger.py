"""Shared logging setup for the trading runtime."""

from __future__ import annotations

import json
import logging
import os
from logging.handlers import RotatingFileHandler
from threading import Lock
from typing import Any, Dict, Optional

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
LOG_DIR = os.path.join(ROOT_DIR, "log")
LOG_FILE = os.path.join(LOG_DIR, "ai_trader.log")

os.makedirs(LOG_DIR, exist_ok=True)

_CONFIG_LOCK = Lock()
_CONFIGURED = False
_FILE_HANDLER_NAME = "ai_trader_file_handler"
_CONSOLE_HANDLER_NAME = "ai_trader_console_handler"


def _has_named_handler(logger: logging.Logger, handler_name: str) -> bool:
    return any(handler.get_name() == handler_name for handler in logger.handlers)


def _configure_root_logger() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return

    with _CONFIG_LOCK:
        if _CONFIGURED:
            return

        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)

        log_format = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(filename)s:%(lineno)d:%(funcName)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        if not _has_named_handler(root_logger, _FILE_HANDLER_NAME):
            file_handler = RotatingFileHandler(
                LOG_FILE,
                maxBytes=10 * 1024 * 1024,
                backupCount=5,
                encoding="utf-8",
            )
            file_handler.set_name(_FILE_HANDLER_NAME)
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(log_format)
            root_logger.addHandler(file_handler)

        if not _has_named_handler(root_logger, _CONSOLE_HANDLER_NAME):
            console_handler = logging.StreamHandler()
            console_handler.set_name(_CONSOLE_HANDLER_NAME)
            console_handler.setLevel(logging.INFO)
            console_handler.setFormatter(log_format)
            root_logger.addHandler(console_handler)

        _CONFIGURED = True


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return a logger configured to use the shared root handlers."""
    _configure_root_logger()

    logger = logging.getLogger(name)
    root_logger = logging.getLogger()

    # Ensure module loggers use only shared root handlers.
    if logger is not root_logger and logger.handlers:
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            try:
                handler.close()
            except Exception:
                pass
    logger.propagate = True

    return logger


def format_log_details(details: Dict[str, Any]) -> str:
    """Render structured metadata into a compact key=value log string."""
    items: list[str] = []
    for key, value in details.items():
        if isinstance(value, float):
            rendered = f"{value:.8f}"
        elif isinstance(value, (dict, list, tuple, set)):
            rendered = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
        else:
            rendered = str(value)
        items.append(f"{key}={rendered}")
    return " ".join(items)


__all__ = ["format_log_details", "get_logger"]
