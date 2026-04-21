import os
import tempfile
import unittest
from datetime import datetime, timezone
from unittest.mock import patch

from src.strategy.scheduler import TradingScheduler


class SchedulerTests(unittest.TestCase):
    def test_run_cycle_once_only_executes_one_ai_cycle_even_on_hour_boundary(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            state_file = os.path.join(temp_dir, "scheduler_state.json")
            with patch("src.strategy.scheduler.STATE_FILE", state_file), patch(
                "src.strategy.scheduler.run_hakai_cycle",
                return_value={
                    "success": True,
                    "symbol": "BTCUSDT",
                    "action": "hold_waiting_price_trigger",
                    "ai_triggered": False,
                    "ai_decision": None,
                    "trigger_reason": "waiting_for_next_price_trigger",
                    "trigger_price": None,
                    "current_price": 100000.0,
                    "next_trigger_down": 99000.0,
                    "next_trigger_up": 101000.0,
                    "cycle_dir": None,
                    "position": None,
                    "position_before": None,
                    "state_update": {},
                },
            ) as mocked_run_hakai_cycle:
                scheduler = TradingScheduler()
                with patch.object(scheduler, "_maybe_send_cycle_notifications", return_value=None):
                    scheduler.run_cycle_once(now_utc=datetime(2026, 4, 22, 10, 0, 10, tzinfo=timezone.utc))

        self.assertEqual(mocked_run_hakai_cycle.call_count, 1)


if __name__ == "__main__":
    unittest.main()
