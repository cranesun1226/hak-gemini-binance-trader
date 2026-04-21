import unittest

from src.strategy import hakai_strategy


def _build_long_position() -> dict:
    return {
        "symbol": "BTCUSDT",
        "side": "Buy",
        "positionAmt": 0.04,
        "entryPrice": 100000.0,
        "markPrice": 100000.0,
        "leverage": 10,
        "positionValue": 4000.0,
    }


class PositionSizingTests(unittest.TestCase):
    def test_auto_position_disabled_keeps_initial_ratio_for_same_direction_position(self):
        plan = hakai_strategy._build_position_sizing_plan(
            volatility_snapshot={
                "volatility_target_margin_ratio": 0.98,
                "target_margin_ratio": 0.98,
            },
            current_position=_build_long_position(),
            decision="LONG",
            reference_price=102000.0,
            leverage=10,
            initial_position_size_ratio=0.4,
            position_size_ratio_max=0.98,
            enable_auto_position=False,
            profit_activation_pct=0.01,
            position_episode_state={
                "initial_entry_price": 100000.0,
                "initial_entry_direction": "long",
                "position_sizing_activation_pct": 0.01,
                "position_sizing_activation_price": 101000.0,
                "position_sizing_unlocked": True,
                "position_sizing_activated_at": "2026-04-21T00:00:00+00:00",
            },
            bootstrap_protected=True,
        )

        self.assertEqual(plan["position_sizing_mode"], "auto_position_disabled")
        self.assertFalse(plan["enable_auto_position"])
        self.assertFalse(plan["position_sizing_unlocked"])
        self.assertIsNone(plan["position_sizing_activation_pct"])
        self.assertIsNone(plan["position_sizing_activation_price"])
        self.assertFalse(plan["keep_current_position_size"])
        self.assertEqual(plan["applied_target_margin_ratio"], 0.4)

    def test_clear_position_episode_sizing_state_preserves_entry_anchor(self):
        cleared = hakai_strategy._clear_position_episode_sizing_state(
            {
                "initial_entry_price": 100000.0,
                "initial_entry_direction": "long",
                "position_sizing_activation_pct": 0.02,
                "position_sizing_activation_price": 102000.0,
                "position_sizing_unlocked": True,
                "position_sizing_activated_at": "2026-04-21T00:00:00+00:00",
            }
        )

        self.assertEqual(cleared["initial_entry_price"], 100000.0)
        self.assertEqual(cleared["initial_entry_direction"], "long")
        self.assertIsNone(cleared["position_sizing_activation_pct"])
        self.assertIsNone(cleared["position_sizing_activation_price"])
        self.assertFalse(cleared["position_sizing_unlocked"])
        self.assertIsNone(cleared["position_sizing_activated_at"])

    def test_fixed_ratio_target_notional_still_tracks_account_equity(self):
        target_notional = hakai_strategy._resolve_target_notional_usdt(
            account_equity=1000.0,
            leverage=10,
            current_position=_build_long_position(),
            position_sizing_plan={
                "keep_current_position_size": False,
                "applied_target_margin_ratio": 0.4,
                "current_notional_usdt": 4000.0,
            },
        )

        self.assertEqual(target_notional, 4000.0)


if __name__ == "__main__":
    unittest.main()
