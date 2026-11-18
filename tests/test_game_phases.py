import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

import game_predictions_report as gpr
import model_picks as mp
import totals_model as tm

NOW = datetime(2026, 11, 18, 18, 0, tzinfo=timezone.utc)


def _data(margin=3.0, total=220.0, tip="2026-11-19T00:00Z"):
    return {"date": "2026-11-18", "generated": "x", "injury_feed": {"healthy": True},
            "games": [{"home": {"name": "Home"}, "away": {"name": "Away"}, "tip": tip,
                       "spread": {"pred": margin, "posted_line": -2.5}, "totals": {"pred": total, "line": 218.5},
                       "ctx": {"home_vac_minutes": 30.0, "away_vac_minutes": 0.0, "home_out": ["X"], "away_out": []}}]}


class GameRowTests(unittest.TestCase):
    def test_row_has_winner_lines_and_absences_and_skips_started(self):
        r = mp.game_rows(_data(), "EARLY", [], now=NOW)[0]
        self.assertEqual((r["pred_winner"], r["market_home_line"], r["market_total"]), ("Home", -2.5, 218.5))
        self.assertEqual(r["home_out"], ["X"])
        self.assertEqual(mp.game_rows(_data(tip="2026-11-18T17:00Z"), "EARLY", [], now=NOW), [])
        self.assertEqual(mp.game_rows(_data(), "EARLY", [r], now=NOW), [])          # immutable per phase

    def test_phase_grading_and_changes(self):
        with tempfile.TemporaryDirectory() as t:
            path = Path(t, "g.jsonl")
            early = mp.game_rows(_data(margin=3.0, total=220.0), "EARLY", [], now=NOW)[0]
            adj = mp.game_rows(_data(margin=-1.0, total=214.0), "ADJUSTED", [], now=NOW)[0]
            path.write_text(json.dumps(early) + "\n" + json.dumps(adj) + "\n")
            fin = {gpr._key("2026-11-18", "Away", "Home"): (100.0, 105.0)}   # away wins by 5, total 205
            p = gpr.phase_records(fin, path)
        self.assertEqual(p["EARLY"]["winner"]["losses"], 1)                  # early picked Home
        self.assertEqual(p["ADJUSTED"]["winner"]["wins"], 1)                 # adjusted flipped to Away
        self.assertEqual(p["ADJUSTED"]["ats"]["wins"], 1)                    # -1 - 2.5 < 0 -> away; -5 - 2.5 < 0
        self.assertEqual(p["EARLY"]["ou"]["losses"], 1)                      # 220 over 218.5, actual 205
        self.assertEqual(p["changes"]["winner_flipped"], 1)
        self.assertAlmostEqual(p["changes"]["mean_spread_move"], 4.0)


class TotalsAvailabilityTests(unittest.TestCase):
    def test_engineered_totals_include_absence_sums(self):
        df = pd.DataFrame({"home_vac_value": [1.0, None], "away_vac_value": [2.0, 3.0],
                           "home_vac_minutes": [10.0, 0.0], "away_vac_minutes": [5.0, None]})
        out = tm.add_availability_totals(df.copy())
        self.assertEqual(out["vac_value_sum"].tolist(), [3.0, 3.0])
        self.assertEqual(out["vac_minutes_sum"].tolist(), [15.0, 0.0])
        for f in ("home_vac_value", "away_vac_minutes", "vac_value_sum", "vac_minutes_sum"):
            self.assertIn(f, tm.TOTALS_FEATURES)


if __name__ == "__main__":
    unittest.main()
