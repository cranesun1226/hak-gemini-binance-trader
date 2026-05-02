import unittest

from src.ai.gemini_trader import _build_direction_prompt, _build_position_management_prompt


class GeminiPromptTests(unittest.TestCase):
    def test_entry_prompt_allows_flat_with_single_fifteen_minute_context(self):
        prompt = _build_direction_prompt(
            symbol="BTCUSDT",
            reference_price=100000.0,
            timeframe_ohlcv={"15m": [[1.0, 2.0, 0.5, 1.5, 100.0] for _ in range(100)]},
            current_position_snapshot=None,
        )

        self.assertIn(
            "Initial-entry task: choose LONG, SHORT, or FLAT.",
            prompt,
        )
        self.assertIn("reasonable directional edge", prompt)
        self.assertIn("Not every factor must be perfect.", prompt)
        self.assertIn("Choose FLAT when directional evidence is genuinely balanced", prompt)
        self.assertIn(
            "Do not get shaken by ordinary and usual candles.",
            prompt,
        )
        self.assertIn("Use the provided 15m OHLCV candles", prompt)
        self.assertIn('"15m"', prompt)
        self.assertNotIn("multi-timeframe", prompt.lower())

    def test_position_management_prompt_uses_keep_or_close_only(self):
        prompt = _build_position_management_prompt(
            symbol="BTCUSDT",
            reference_price=100000.0,
            timeframe_ohlcv={"15m": [[1.0, 2.0, 0.5, 1.5, 100.0] for _ in range(100)]},
            current_position_snapshot={"direction": "LONG", "size": 0.1},
        )

        self.assertIn("Existing-position task: rationally choose KEEP or CLOSE", prompt)
        self.assertIn("Do not choose a new direction in this response.", prompt)
        self.assertIn("Default to KEEP when the current position thesis remains reasonably valid.", prompt)
        self.assertIn("Choose CLOSE only when objective evidence shows", prompt)
        self.assertIn('Schema: {"decision":"KEEP"} or {"decision":"CLOSE"}.', prompt)
        self.assertIn('"current_position": "LONG"', prompt)


if __name__ == "__main__":
    unittest.main()
