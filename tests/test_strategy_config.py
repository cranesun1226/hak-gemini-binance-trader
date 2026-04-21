import unittest
from unittest.mock import patch

from src.strategy import hakai_strategy


class StrategyConfigTests(unittest.TestCase):
    @patch("src.strategy.hakai_strategy.load_runtime_config", return_value={})
    def test_defaults_use_single_hour_prompt(self, _mocked_load_runtime_config):
        config = hakai_strategy._load_strategy_config()

        self.assertEqual(config["ai_prompt_timeframe"], "1h")
        self.assertEqual(config["ai_prompt_candle_count"], 100)
        self.assertEqual(config["trigger_pct_usdt"], 1.0)
        self.assertTrue(config["enable_auto_position"])

    @patch("src.strategy.hakai_strategy.load_runtime_config", return_value={"ai_candle_count_per_timeframe": 48})
    def test_legacy_prompt_candle_count_key_is_supported(self, _mocked_load_runtime_config):
        config = hakai_strategy._load_strategy_config()

        self.assertEqual(config["ai_prompt_timeframe"], "1h")
        self.assertEqual(config["ai_prompt_candle_count"], 48)

    @patch("src.strategy.hakai_strategy.load_runtime_config", return_value={"ai_prompt_timeframe": "4h"})
    def test_invalid_prompt_timeframe_is_forced_back_to_one_hour(self, _mocked_load_runtime_config):
        config = hakai_strategy._load_strategy_config()

        self.assertEqual(config["ai_prompt_timeframe"], "1h")

    @patch("src.strategy.hakai_strategy.load_runtime_config", return_value={"enable_auto_position": "false"})
    def test_string_false_disables_auto_position(self, _mocked_load_runtime_config):
        config = hakai_strategy._load_strategy_config()

        self.assertFalse(config["enable_auto_position"])

    @patch("src.strategy.hakai_strategy.load_runtime_config", return_value={"gemini_thinking_level": "minimal"})
    def test_legacy_minimal_thinking_level_is_coerced_to_low_for_pro_model(self, _mocked_load_runtime_config):
        config = hakai_strategy._load_strategy_config()

        self.assertEqual(config["gemini_thinking_level"], "low")

    @patch(
        "src.strategy.hakai_strategy.load_runtime_config",
        return_value={
            "enable_auto_position": False,
            "initial_position_size_ratio": 0.4,
            "position_size_ratio_max": 0.2,
        },
    )
    def test_fixed_mode_clamps_ratio_max_to_initial_ratio(self, _mocked_load_runtime_config):
        config = hakai_strategy._load_strategy_config()

        self.assertFalse(config["enable_auto_position"])
        self.assertEqual(config["initial_position_size_ratio"], 0.4)
        self.assertEqual(config["position_size_ratio_max"], 0.4)


if __name__ == "__main__":
    unittest.main()
