import unittest
from unittest.mock import patch

from src.strategy import hakai_strategy


class StrategyConfigTests(unittest.TestCase):
    @patch("src.strategy.hakai_strategy.load_runtime_config", return_value={})
    def test_defaults_use_single_hour_prompt(self, _mocked_load_runtime_config):
        config = hakai_strategy._load_strategy_config()

        self.assertEqual(config["ai_prompt_timeframe"], "1h")
        self.assertEqual(config["ai_prompt_candle_count"], 100)

    @patch("src.strategy.hakai_strategy.load_runtime_config", return_value={"ai_candle_count_per_timeframe": 48})
    def test_legacy_prompt_candle_count_key_is_supported(self, _mocked_load_runtime_config):
        config = hakai_strategy._load_strategy_config()

        self.assertEqual(config["ai_prompt_timeframe"], "1h")
        self.assertEqual(config["ai_prompt_candle_count"], 48)

    @patch("src.strategy.hakai_strategy.load_runtime_config", return_value={"ai_prompt_timeframe": "4h"})
    def test_invalid_prompt_timeframe_is_forced_back_to_one_hour(self, _mocked_load_runtime_config):
        config = hakai_strategy._load_strategy_config()

        self.assertEqual(config["ai_prompt_timeframe"], "1h")


if __name__ == "__main__":
    unittest.main()
