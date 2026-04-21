import unittest

from src.ai import gemini_trader


class GeminiModelTests(unittest.TestCase):
    def test_direction_model_uses_gemini_3_1_pro_preview(self):
        self.assertEqual(gemini_trader.GEMINI_DIRECTION_MODEL, "gemini-3.1-pro-preview")

    def test_cost_estimate_uses_lower_prompt_tier_for_small_requests(self):
        estimate = gemini_trader.estimate_gemini_cost(
            {
                "prompt_token_count": 1000,
                "candidates_token_count": 200,
                "thoughts_token_count": 300,
            }
        )

        self.assertIsNotNone(estimate)
        self.assertEqual(estimate["pricing_prompt_token_threshold"], 200000)
        self.assertEqual(estimate["input_rate_usd_per_million"], 2.0)
        self.assertEqual(estimate["output_rate_usd_per_million"], 12.0)

    def test_cost_estimate_uses_higher_prompt_tier_for_large_requests(self):
        estimate = gemini_trader.estimate_gemini_cost(
            {
                "prompt_token_count": 250000,
                "candidates_token_count": 200,
                "thoughts_token_count": 300,
            }
        )

        self.assertIsNotNone(estimate)
        self.assertIsNone(estimate["pricing_prompt_token_threshold"])
        self.assertEqual(estimate["input_rate_usd_per_million"], 4.0)
        self.assertEqual(estimate["output_rate_usd_per_million"], 18.0)


if __name__ == "__main__":
    unittest.main()
