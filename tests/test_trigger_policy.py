import unittest

from src.strategy import hakai_strategy


class TriggerPolicyTests(unittest.TestCase):
    def test_price_trigger_waits_until_half_percent_move(self):
        trigger = hakai_strategy._determine_ai_trigger(
            has_position=True,
            current_price=100400.0,
            last_ai_trigger_price=100000.0,
            trigger_pct_usdt=0.5,
        )

        self.assertFalse(trigger["should_trigger"])
        self.assertEqual(trigger["reason"], "waiting_for_next_price_trigger")
        self.assertEqual(trigger["next_trigger_down"], 99500.0)
        self.assertEqual(trigger["next_trigger_up"], 100500.0)

    def test_price_trigger_fires_once_half_percent_boundary_is_reached(self):
        trigger = hakai_strategy._determine_ai_trigger(
            has_position=True,
            current_price=100500.0,
            last_ai_trigger_price=100000.0,
            trigger_pct_usdt=0.5,
        )

        self.assertTrue(trigger["should_trigger"])
        self.assertEqual(trigger["reason"], "price_distance_reached")
        self.assertEqual(trigger["trigger_price"], 100500.0)

    def test_flat_no_position_waits_until_price_trigger_window_is_reached(self):
        trigger = hakai_strategy._determine_ai_trigger(
            has_position=False,
            current_price=100400.0,
            last_ai_trigger_price=100000.0,
            last_ai_decision="FLAT",
            trigger_pct_usdt=0.5,
        )

        self.assertFalse(trigger["should_trigger"])
        self.assertEqual(trigger["reason"], "waiting_for_next_price_trigger")
        self.assertEqual(trigger["next_trigger_down"], 99500.0)
        self.assertEqual(trigger["next_trigger_up"], 100500.0)

    def test_flat_no_position_rechecks_after_price_trigger_window_is_reached(self):
        trigger = hakai_strategy._determine_ai_trigger(
            has_position=False,
            current_price=100500.0,
            last_ai_trigger_price=100000.0,
            last_ai_decision="FLAT",
            trigger_pct_usdt=0.5,
        )

        self.assertTrue(trigger["should_trigger"])
        self.assertEqual(trigger["reason"], "price_distance_reached")
        self.assertEqual(trigger["trigger_price"], 100500.0)

    def test_no_position_without_flat_still_triggers_immediately(self):
        trigger = hakai_strategy._determine_ai_trigger(
            has_position=False,
            current_price=100400.0,
            last_ai_trigger_price=100000.0,
            last_ai_decision="LONG",
            trigger_pct_usdt=0.5,
        )

        self.assertTrue(trigger["should_trigger"])
        self.assertEqual(trigger["reason"], "no_position")


if __name__ == "__main__":
    unittest.main()
