import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import daily_runner
import forward_ledger
import kelly_sizing
import merge_data


class _SumModel:
    def predict(self, X):
        return np.array([X.sum(axis=1).iloc[0]])


class PredictRowTests(unittest.TestCase):
    def test_missing_feature_returns_none_instead_of_defaulting_to_zero(self):
        bundle = {"model": _SumModel(), "feature_names": ["a", "b"]}
        self.assertIsNone(daily_runner.predict_row(bundle, {"a": 200}, "Totals", (170, 280)))

    def test_out_of_range_prediction_is_rejected(self):
        bundle = {"model": _SumModel(), "feature_names": ["a"]}
        self.assertIsNone(daily_runner.predict_row(bundle, {"a": 125}, "Totals", (170, 280)))
        self.assertEqual(daily_runner.predict_row(bundle, {"a": 225}, "Totals", (170, 280)), 225.0)


class LedgerDecisionTests(unittest.TestCase):
    def _data(self, verdict):
        return {"date": "2026-10-20", "generated": "g", "games": [{
            "home": {"name": "H"}, "away": {"name": "A"},
            "spread": {"pred": 3, "posted_line": -1, "play": "H -1", "stars": 3, "verdict": verdict},
            "totals": {"pred": 230, "line": 220, "play": "OVER", "stars": 3},
            "props": [],
        }]}

    def test_decision_follows_sizing_verdict_not_stars(self):
        rows = {r["market"]: r["decision"] for r in forward_ledger.flatten(self._data("PASS"))}
        self.assertEqual(rows, {"SPREAD": "PASS", "TOTAL": "PASS"})  # missing verdict = PASS
        rows = {r["market"]: r["decision"] for r in forward_ledger.flatten(self._data("STRONG"))}
        self.assertEqual(rows["SPREAD"], "BET")


class KellySignTests(unittest.TestCase):
    def test_under_and_over_edges_size_the_same(self):
        self.assertEqual(kelly_sizing.size_bet(-4, "totals"), {**kelly_sizing.size_bet(4, "totals"), "edge_pts": -4})


class MergeDataTests(unittest.TestCase):
    def test_filter_drops_exhibitions_preseason_and_duplicates(self):
        df = pd.DataFrame([
            {"game_id": 1, "home_team": "Celtics", "away_team": "Knicks", "season_type": 2},
            {"game_id": 1, "home_team": "Celtics", "away_team": "Knicks", "season_type": 2},
            {"game_id": 2, "home_team": "Team Shaq", "away_team": "Team Chuck", "season_type": 2},
            {"game_id": 3, "home_team": "Celtics", "away_team": "Knicks", "season_type": 1},
            {"game_id": 4, "home_team": "Celtics", "away_team": "Knicks", "season_type": 3},
        ])
        out = merge_data.filter_model_games(df)
        self.assertEqual(sorted(out["game_id"]), [1, 4])
        self.assertEqual(out["home_team"].iloc[0], "Boston Celtics")

    def test_load_odds_uses_eastern_game_date_and_model_column_names(self):
        with tempfile.TemporaryDirectory() as temp:
            pd.DataFrame([
                # scraped 10-20 for a late game that tips 01:30Z on 10-21 = 9:30 PM ET on 10-20
                {"game_date": "2026-10-20", "commence_time": "2026-10-21T01:30:00Z", "game_id": "x",
                 "home_team": "San Antonio Spurs", "away_team": "Oklahoma City Thunder",
                 "spread_home": 1.0, "total": 230.5, "scraped_at": "2026-10-20T10:00:00"},
                {"game_date": "2026-10-20", "commence_time": "2026-10-21T01:30:00Z", "game_id": "x",
                 "home_team": "San Antonio Spurs", "away_team": "Oklahoma City Thunder",
                 "spread_home": 1.5, "total": 231.0, "scraped_at": "2026-10-20T17:00:00"},
            ]).to_csv(Path(temp) / "odds_historical.csv", index=False)
            old = merge_data.RAW_DIR
            merge_data.RAW_DIR = temp
            try:
                odds = merge_data.load_odds(2027)
            finally:
                merge_data.RAW_DIR = old
        self.assertEqual(len(odds), 1)
        row = odds.iloc[0]
        self.assertEqual(str(row["game_date"].date()), "2026-10-20")
        self.assertEqual(row["avg_spread_home_line"], 1.5)  # latest scrape wins
        self.assertEqual(row["avg_total_point"], 231.0)


if __name__ == "__main__":
    unittest.main()
