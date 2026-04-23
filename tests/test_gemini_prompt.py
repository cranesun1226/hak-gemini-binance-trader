import unittest

from src.ai.gemini_trader import _build_direction_prompt


class GeminiPromptTests(unittest.TestCase):
    def test_direction_prompt_keeps_original_style_with_single_fifteen_minute_context(self):
        prompt = _build_direction_prompt(
            symbol="BTCUSDT",
            reference_price=100000.0,
            timeframe_ohlcv={"15m": [[1.0, 2.0, 0.5, 1.5, 100.0] for _ in range(100)]},
            current_position_snapshot={"direction": "LONG"},
        )

        self.assertIn(
            "As a trader, rationally choose to hold or switch LONG/SHORT position.",
            prompt,
        )
        self.assertIn(
            "Do not get shaken by ordinary and usual candles.",
            prompt,
        )
        self.assertIn("Use the provided 15m OHLCV candles", prompt)
        self.assertIn('"15m"', prompt)
        self.assertNotIn("multi-timeframe", prompt.lower())


if __name__ == "__main__":
    unittest.main()
