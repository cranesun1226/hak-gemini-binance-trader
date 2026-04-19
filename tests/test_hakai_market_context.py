import unittest
from unittest.mock import patch

from src.strategy import hakai_strategy


def _build_candles(count: int, interval_ms: int) -> list[dict[str, float]]:
    candles: list[dict[str, float]] = []
    for index in range(count):
        base_price = 10000.0 + float(index)
        candles.append(
            {
                "timestamp": index * interval_ms,
                "open": base_price,
                "high": base_price + 10.0,
                "low": base_price - 10.0,
                "close": base_price + 5.0,
                "volume": 1000.0 + float(index),
            }
        )
    return candles


class PromptMarketContextTests(unittest.TestCase):
    def test_fetch_prompt_market_context_keeps_ai_payload_hourly_only(self):
        hourly_candles = _build_candles(100, 60 * 60 * 1000)
        daily_candles = _build_candles(27, 24 * 60 * 60 * 1000)

        def _fake_fetch_klines(_symbol: str, timeframe: str, limit: int, *, as_of_ms=None):
            source = hourly_candles if timeframe == "1h" else daily_candles
            return source[-limit:]

        with patch("src.strategy.hakai_strategy.fetch_klines", side_effect=_fake_fetch_klines), patch(
            "src.strategy.hakai_strategy.parse_klines",
            side_effect=lambda rows: list(rows),
        ):
            context = hakai_strategy._fetch_prompt_market_context(
                symbol="BTCUSDT",
                ai_prompt_timeframe="1h",
                ai_prompt_candle_count=100,
                position_sizing_daily_sample_days=25,
                position_sizing_live_window_hours=24,
                as_of_ms=daily_candles[-1]["timestamp"] + (2 * 24 * 60 * 60 * 1000),
            )

        self.assertEqual(context["ai_prompt_timeframe"], "1h")
        self.assertEqual(context["ai_prompt_candle_count"], 100)
        self.assertEqual(list(context["timeframes"].keys()), ["1h"])
        self.assertEqual(len(context["timeframes"]["1h"]), 100)
        self.assertEqual(len(context["live_window_candles"]), 24)
        self.assertEqual(len(context["daily_position_sizing_candles"]), 25)


if __name__ == "__main__":
    unittest.main()
