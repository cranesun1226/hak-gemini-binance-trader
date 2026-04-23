import unittest

from src.strategy import hakai_strategy


class FixedPositionSizingTests(unittest.TestCase):
    def test_build_fixed_position_sizing_uses_initial_ratio_only(self):
        plan = hakai_strategy._build_fixed_position_sizing(
            initial_position_size_ratio=0.4,
            leverage=10,
        )

        self.assertEqual(plan["position_sizing_mode"], "fixed_ratio")
        self.assertEqual(plan["applied_target_margin_ratio"], 0.4)
        self.assertEqual(plan["target_margin_ratio"], 0.4)
        self.assertEqual(plan["applied_target_effective_leverage"], 4.0)

    def test_state_update_clears_legacy_auto_position_fields(self):
        state_update = hakai_strategy._build_state_update(
            previous_state={
                "initial_entry_price": 100000.0,
                "initial_entry_direction": "long",
                "position_sizing_activation_pct": 0.01,
                "position_sizing_activation_price": 101000.0,
                "position_sizing_unlocked": True,
                "position_sizing_activated_at": "2026-04-21T00:00:00+00:00",
            },
            trigger_pct_usdt=0.75,
            ai_triggered=False,
            trigger_price=None,
            ai_decision=None,
            next_trigger_down=99000.0,
            next_trigger_up=101000.0,
        )

        self.assertNotIn("initial_entry_price", state_update)
        self.assertNotIn("initial_entry_direction", state_update)
        self.assertNotIn("position_sizing_activation_pct", state_update)
        self.assertNotIn("position_sizing_activation_price", state_update)
        self.assertNotIn("position_sizing_unlocked", state_update)
        self.assertNotIn("position_sizing_activated_at", state_update)


if __name__ == "__main__":
    unittest.main()
