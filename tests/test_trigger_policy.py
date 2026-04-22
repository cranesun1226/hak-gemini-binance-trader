import unittest

from src.strategy import hakai_strategy


class TriggerPolicyTests(unittest.TestCase):
    def test_price_trigger_waits_until_three_quarter_percent_move(self):
        trigger = hakai_strategy._determine_ai_trigger(
            has_position=True,
            current_price=100500.0,
            last_ai_trigger_price=100000.0,
            trigger_pct_usdt=0.75,
        )

        self.assertFalse(trigger["should_trigger"])
        self.assertEqual(trigger["reason"], "waiting_for_next_price_trigger")
        self.assertEqual(trigger["next_trigger_down"], 99250.0)
        self.assertEqual(trigger["next_trigger_up"], 100750.0)

    def test_price_trigger_fires_once_three_quarter_percent_boundary_is_reached(self):
        trigger = hakai_strategy._determine_ai_trigger(
            has_position=True,
            current_price=100750.0,
            last_ai_trigger_price=100000.0,
            trigger_pct_usdt=0.75,
        )

        self.assertTrue(trigger["should_trigger"])
        self.assertEqual(trigger["reason"], "price_distance_reached")
        self.assertEqual(trigger["trigger_price"], 100750.0)


if __name__ == "__main__":
    unittest.main()
