import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from src.strategy import hakai_strategy


def _config():
    return {
        "symbol": "BTCUSDT",
        "fixed_leverage": 10,
        "trigger_pct_usdt": 0.75,
        "initial_position_size_ratio": 0.4,
        "stop_loss_pct": 0.04,
        "ai_prompt_timeframe": "15m",
        "ai_prompt_candle_count": 100,
        "gemini_api_version": "v1alpha",
        "gemini_thinking_level": "high",
    }


def _market_context():
    return {
        "ai_prompt_timeframe": "15m",
        "ai_prompt_candle_count": 100,
        "timeframes": {"15m": [[1.0, 2.0, 0.5, 1.5, 100.0] for _ in range(100)]},
    }


def _long_position():
    return {
        "symbol": "BTCUSDT",
        "positionAmt": "0.1",
        "side": "Buy",
        "entryPrice": "100000",
        "markPrice": "100750",
        "leverage": "10",
        "positionValue": "10075",
    }


class HakaiDecisionFlowTests(unittest.TestCase):
    def test_flat_entry_skips_account_sizing_and_order_execution(self):
        with tempfile.TemporaryDirectory() as temp_dir, patch(
            "src.strategy.hakai_strategy._load_strategy_config", return_value=_config()
        ), patch("src.strategy.hakai_strategy.get_binance_credentials", return_value=("key", "secret")), patch(
            "src.strategy.hakai_strategy.get_positions", return_value=[]
        ), patch(
            "src.strategy.hakai_strategy.get_reference_price", return_value={"price": 100000.0}
        ), patch(
            "src.strategy.hakai_strategy._create_cycle_dir", return_value=temp_dir
        ), patch(
            "src.strategy.hakai_strategy._fetch_prompt_market_context", return_value=_market_context()
        ), patch(
            "src.strategy.hakai_strategy._fetch_live_prompt_position_snapshot", return_value=None
        ), patch(
            "src.strategy.hakai_strategy.evaluate_hakai_entry_direction",
            return_value=SimpleNamespace(decision="FLAT"),
        ), patch(
            "src.strategy.hakai_strategy.get_account_equity"
        ) as mocked_equity, patch(
            "src.strategy.hakai_strategy.set_leverage"
        ) as mocked_set_leverage, patch(
            "src.strategy.hakai_strategy._place_new_direction_position"
        ) as mocked_place, patch(
            "src.strategy.hakai_strategy._persist_cycle_output"
        ):
            result = hakai_strategy.run_hakai_cycle(state={})

        self.assertTrue(result["success"])
        self.assertEqual(result["action"], "flat_no_entry")
        self.assertEqual(result["ai_decision"], "FLAT")
        self.assertEqual(result["state_update"]["last_ai_decision"], "FLAT")
        mocked_equity.assert_not_called()
        mocked_set_leverage.assert_not_called()
        mocked_place.assert_not_called()

    def test_keep_existing_position_skips_close_and_entry(self):
        state = {"last_ai_trigger_price": 100000.0, "trigger_pct_usdt": 0.75}
        with tempfile.TemporaryDirectory() as temp_dir, patch(
            "src.strategy.hakai_strategy._load_strategy_config", return_value=_config()
        ), patch("src.strategy.hakai_strategy.get_binance_credentials", return_value=("key", "secret")), patch(
            "src.strategy.hakai_strategy.get_positions", return_value=[_long_position()]
        ), patch(
            "src.strategy.hakai_strategy.get_reference_price", return_value={"price": 100750.0}
        ), patch(
            "src.strategy.hakai_strategy.get_account_equity", return_value=1000.0
        ), patch(
            "src.strategy.hakai_strategy._sync_account_risk_stop_loss",
            return_value={"success": True, "changed": False},
        ), patch(
            "src.strategy.hakai_strategy._create_cycle_dir", return_value=temp_dir
        ), patch(
            "src.strategy.hakai_strategy._fetch_prompt_market_context", return_value=_market_context()
        ), patch(
            "src.strategy.hakai_strategy._fetch_live_prompt_position_snapshot",
            return_value=hakai_strategy.calculate_position_metrics(_long_position()),
        ), patch(
            "src.strategy.hakai_strategy.evaluate_hakai_position_management",
            return_value=SimpleNamespace(decision="KEEP"),
        ), patch(
            "src.strategy.hakai_strategy.evaluate_hakai_entry_direction"
        ) as mocked_entry, patch(
            "src.strategy.hakai_strategy._close_existing_position_for_flip"
        ) as mocked_close, patch(
            "src.strategy.hakai_strategy.set_leverage"
        ) as mocked_set_leverage, patch(
            "src.strategy.hakai_strategy._persist_cycle_output"
        ):
            result = hakai_strategy.run_hakai_cycle(state=state)

        self.assertTrue(result["success"])
        self.assertEqual(result["action"], "kept_position_by_ai")
        self.assertEqual(result["ai_decision"], "KEEP")
        self.assertEqual(result["state_update"]["last_ai_decision"], "KEEP")
        mocked_entry.assert_not_called()
        mocked_close.assert_not_called()
        mocked_set_leverage.assert_not_called()

    def test_flip_then_flat_closes_position_without_reentry(self):
        state = {"last_ai_trigger_price": 100000.0, "trigger_pct_usdt": 0.75}
        with tempfile.TemporaryDirectory() as temp_dir, patch(
            "src.strategy.hakai_strategy._load_strategy_config", return_value=_config()
        ), patch("src.strategy.hakai_strategy.get_binance_credentials", return_value=("key", "secret")), patch(
            "src.strategy.hakai_strategy.get_positions", return_value=[_long_position()]
        ), patch(
            "src.strategy.hakai_strategy.get_reference_price", return_value={"price": 100750.0}
        ), patch(
            "src.strategy.hakai_strategy.get_account_equity", return_value=1000.0
        ), patch(
            "src.strategy.hakai_strategy._sync_account_risk_stop_loss",
            return_value={"success": True, "changed": False},
        ), patch(
            "src.strategy.hakai_strategy._create_cycle_dir", return_value=temp_dir
        ), patch(
            "src.strategy.hakai_strategy._fetch_prompt_market_context", return_value=_market_context()
        ), patch(
            "src.strategy.hakai_strategy._fetch_live_prompt_position_snapshot",
            return_value=hakai_strategy.calculate_position_metrics(_long_position()),
        ), patch(
            "src.strategy.hakai_strategy.evaluate_hakai_position_management",
            return_value=SimpleNamespace(decision="FLIP"),
        ), patch(
            "src.strategy.hakai_strategy.evaluate_hakai_entry_direction",
            return_value=SimpleNamespace(decision="FLAT"),
        ), patch(
            "src.strategy.hakai_strategy._close_existing_position_for_flip",
            return_value={"success": True, "action": "flipped_position_closed"},
        ), patch(
            "src.strategy.hakai_strategy.set_leverage"
        ) as mocked_set_leverage, patch(
            "src.strategy.hakai_strategy._place_new_direction_position"
        ) as mocked_place, patch(
            "src.strategy.hakai_strategy._persist_cycle_output"
        ):
            result = hakai_strategy.run_hakai_cycle(state=state)

        self.assertTrue(result["success"])
        self.assertEqual(result["action"], "flipped_to_flat")
        self.assertEqual(result["position_ai_decision"], "FLIP")
        self.assertEqual(result["entry_ai_decision"], "FLAT")
        self.assertEqual(result["state_update"]["last_ai_decision"], "FLAT")
        mocked_set_leverage.assert_not_called()
        mocked_place.assert_not_called()


if __name__ == "__main__":
    unittest.main()
