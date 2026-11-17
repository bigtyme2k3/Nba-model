import unittest

import game_predictions_report as g


class GamePredictionReportTests(unittest.TestCase):
    def test_market_home_line_from_either_side(self):
        rows = [{"market": "SPREAD", "date": "2026-11-01", "game": "A @ B", "side": "AWAY", "line": 4.5},
                {"market": "SPREAD", "date": "2026-11-01", "game": "C @ D", "side": "HOME", "line": -3.0}]
        m = g.market_home_lines(rows)
        self.assertEqual(m[g._key("2026-11-01", "A", "B")], -4.5)   # away +4.5 -> home -4.5
        self.assertEqual(m[g._key("2026-11-01", "C", "D")], -3.0)

    def test_follow_and_fade_at_pair_prices(self):
        rows = [{"outcome": "WIN", "odds": -110, "market_probability": 0.5, "decision": "PASS"},
                {"outcome": "LOSS", "odds": -110, "market_probability": 0.5, "decision": "BET"}]
        r = g.follow_fade(rows)
        self.assertEqual((r["model_wins"], r["model_losses"]), (1, 1))
        self.assertAlmostEqual(r["follow_units"], round(100 / 110 - 1, 2))
        self.assertAlmostEqual(r["fade_units"], round(-1 + 100 / 110, 2))
        self.assertEqual(r["qualified_bets"], "0-1")


if __name__ == "__main__":
    unittest.main()
