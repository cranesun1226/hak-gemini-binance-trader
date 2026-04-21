"""Runtime configuration defaults and loader helpers."""

import os
from copy import deepcopy
from typing import Any, Dict

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - exercised in lightweight test environments
    class _YamlFallback:
        @staticmethod
        def safe_load(*_args, **_kwargs):
            return {}

    yaml = _YamlFallback()


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
CONFIG_PATH = os.path.join(ROOT_DIR, "setting.yaml")

# Runtime defaults live here so optional keys can be omitted from setting.yaml.
DEFAULT_GEMINI_API_VERSION = "v1beta"
DEFAULT_GEMINI_THINKING_LEVEL = "high"
DEFAULT_AI_PROMPT_TIMEFRAME = "1h"
DEFAULT_AI_PROMPT_CANDLE_COUNT = 100
DEFAULT_POSITION_SIZING_DAILY_SAMPLE_DAYS = 25
DEFAULT_POSITION_SIZING_LIVE_WINDOW_HOURS = 24
DEFAULT_INITIAL_POSITION_SIZE_RATIO = 0.4
DEFAULT_ENABLE_AUTO_POSITION = True
DEFAULT_PROFIT_ACTIVATION_PCT = 0.01
DEFAULT_POSITION_SIZE_RATIO_MIN = 0.02
DEFAULT_POSITION_SIZE_RATIO_MAX = 0.98

DEFAULT_CONFIG: Dict[str, Any] = {
    "symbol": "BTCUSDT",
    "cycle_interval_seconds": 60,
    "trigger_pct_usdt": 0.4,
    "fixed_leverage": 10,
    "stop_loss_pct": 0.04,
    "ai_prompt_timeframe": DEFAULT_AI_PROMPT_TIMEFRAME,
    "ai_prompt_candle_count": DEFAULT_AI_PROMPT_CANDLE_COUNT,
    "position_sizing_daily_sample_days": DEFAULT_POSITION_SIZING_DAILY_SAMPLE_DAYS,
    "position_sizing_live_window_hours": DEFAULT_POSITION_SIZING_LIVE_WINDOW_HOURS,
    "initial_position_size_ratio": DEFAULT_INITIAL_POSITION_SIZE_RATIO,
    "enable_auto_position": DEFAULT_ENABLE_AUTO_POSITION,
    "profit_activation_pct": DEFAULT_PROFIT_ACTIVATION_PCT,
    "position_size_ratio_max": DEFAULT_POSITION_SIZE_RATIO_MAX,
    "gemini_api_version": DEFAULT_GEMINI_API_VERSION,
    "gemini_thinking_level": DEFAULT_GEMINI_THINKING_LEVEL,
}


def get_default_config() -> Dict[str, Any]:
    """Return a deep-copied default configuration payload."""
    return deepcopy(DEFAULT_CONFIG)


def get_default_config_value(key: str, default: Any = None) -> Any:
    """Return one default config value without exposing shared mutable state."""
    return deepcopy(DEFAULT_CONFIG.get(key, default))


def load_runtime_config(config_path: str = CONFIG_PATH) -> Dict[str, Any]:
    """Load runtime config from disk and merge it on top of the defaults."""
    config = get_default_config()
    try:
        with open(config_path, "r", encoding="utf-8") as file_obj:
            loaded = yaml.safe_load(file_obj) or {}
    except Exception:
        return config

    if isinstance(loaded, dict):
        config.update(loaded)
    return config
