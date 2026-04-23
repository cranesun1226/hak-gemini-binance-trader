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
    def test_fetch_prompt_market_context_keeps_ai_payload_closed_fifteen_minute_only(self):
        fifteen_minute_candles = _build_candles(102, 15 * 60 * 1000)

        def _fake_fetch_klines(_symbol: str, timeframe: str, limit: int, *, as_of_ms=None):
            del as_of_ms
            self.assertEqual(timeframe, "15m")
            source = fifteen_minute_candles
            return source[-limit:]

        with patch("src.strategy.hakai_strategy.fetch_klines", side_effect=_fake_fetch_klines), patch(
            "src.strategy.hakai_strategy.parse_klines",
            side_effect=lambda rows: list(rows),
        ):
            context = hakai_strategy._fetch_prompt_market_context(
                symbol="BTCUSDT",
                ai_prompt_timeframe="15m",
                ai_prompt_candle_count=100,
                as_of_ms=fifteen_minute_candles[-1]["timestamp"] + (5 * 60 * 1000),
            )

        self.assertEqual(context["ai_prompt_timeframe"], "15m")
        self.assertEqual(context["ai_prompt_candle_count"], 100)
        self.assertEqual(list(context["timeframes"].keys()), ["15m"])
        self.assertEqual(len(context["timeframes"]["15m"]), 100)
        self.assertEqual(context["timeframes"]["15m"][0][0], fifteen_minute_candles[1]["open"])
        self.assertEqual(context["timeframes"]["15m"][-1][0], fifteen_minute_candles[100]["open"])


if __name__ == "__main__":
    unittest.main()
