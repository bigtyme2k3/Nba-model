import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from v2.simulation import shadow as S

DATE = "2026-11-20"
GAME = "Away Team @ Home Team"


def _predictions():
    def player(name, team, opp, pts, v2=None):
        props = {"pts": {"pred": pts, "line": 20.5, "odds": -110, "offer_side": "OVER"},
                 "reb": {"pred": 6.0}, "ast": {"pred": 4.0}, "threes": {"pred": 2.0}, "pra": {"pred": pts + 10}}
        if v2 is not None:
            props["pts"]["v2_pred"] = v2
        return {"player": name, "team": team, "opp": opp, "proj_min": 32.0, "props": props}
    return {"date": DATE, "generated": "2026-11-20T18:00:00+00:00",
            "games": [{"tip": "2026-11-21T00:00:00Z", "home": {"name": "Home Team"}, "away": {"name": "Away Team"},
                       "props": [player("Star Guard", "Home Team", "Away Team", 24.0, v2=26.0),
                                 player("Wing Player", "Away Team", "Home Team", 14.0)]}]}


def _ledger(outcome="PENDING"):
    return [{"market": "PROP", "date": DATE, "game": GAME, "player": "Star Guard", "stat": "PTS", "side": "OVER",
             "line": 20.5, "odds": -110, "sportsbook_key": "fanduel", "market_probability": 0.5,
             "raw_model_probability": 0.6, "projection": 24.0, "v2_projection": 26.0,
             "commence_time": "2099-01-01T00:00:00Z", "outcome": outcome, "actual": 25.0,
             "ranking_key": f"{DATE}|away team @ home team|prop|star guard|pts", "prediction_id": "abc"}]


class ShadowTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        t = Path(self.tmp.name)
        (t / "predictions_2026-11-20.json").write_text(json.dumps(_predictions()))
        cal = {"version": "TEST", "params": {"minutes_sd_scale": 1.0, "early_exit_prob": 0.0, "game_sd": 0.0,
                                             "player_sd": 0.0, "dispersion": {"pts": 1000.0}}}
        (t / "cal.json").write_text(json.dumps(cal))
        self.patches = [patch.object(S, "PRED_DIR", t), patch.object(S, "SHADOW", t / "shadow.jsonl"),
                        patch.object(S, "LAB", t / "lab.json"), patch.object(S, "REPORT", t / "report.json"),
                        patch.object(S, "CALIBRATION", t / "cal.json"), patch.object(S, "RAW_DIR", t),
                        patch.object(S, "LOGS", t / "missing.csv"), patch.object(S, "V2_FEATURES", t / "none.csv")]
        for p in self.patches:
            p.start()
        self.t = t

    def tearDown(self):
        for p in self.patches:
            p.stop()
        self.tmp.cleanup()

    def test_publish_links_to_the_ledger_once_and_never_rewrites(self):
        with patch.object(S, "read_ledger", return_value=_ledger()):
            first = S.publish(DATE, n_sims=2000)
            again = S.publish(DATE, n_sims=2000)
        self.assertEqual(first["shadow_rows_appended"], 1)
        self.assertEqual(again["shadow_rows_appended"], 0)
        rows = [json.loads(l) for l in (self.t / "shadow.jsonl").read_text().splitlines()]
        self.assertEqual(len(rows), 1)
        r = rows[0]
        self.assertEqual(r["prediction_id"], "abc")
        self.assertEqual(r["evidence"], "SHADOW_ONLY")
        self.assertGreater(r["p_mc_v2"], r["p_mc_v1"])          # V2 projects 26 vs V1 24 over 20.5
        self.assertAlmostEqual(r["mean_mc_v2"], 26.0, delta=0.6)

    def test_v2_arm_is_missing_without_a_v2_projection(self):
        with patch.object(S, "read_ledger", return_value=_ledger()):
            S.publish(DATE, n_sims=1000)
        lab = json.loads((self.t / "lab.json").read_text())
        wing = [c for c in lab["cards"] if c["player"] == "Wing Player" and c["stat"] == "PTS"][0]
        self.assertEqual(wing["mc_arm"], "mc_v1")
        self.assertEqual(lab["v2_players"], 1)
        self.assertIn("SHADOW_ONLY", lab["policy"])

    def test_grade_scores_only_rows_published_before_tip(self):
        with patch.object(S, "read_ledger", return_value=_ledger()):
            S.publish(DATE, n_sims=1000)
        rows = [json.loads(l) for l in (self.t / "shadow.jsonl").read_text().splitlines()]
        rows[0]["published_at_utc"] = "2099-02-01T00:00:00+00:00"   # after the tip
        (self.t / "shadow.jsonl").write_text(json.dumps(rows[0]) + "\n")
        with patch.object(S, "read_ledger", return_value=_ledger("WIN")):
            report = S.grade()
        self.assertEqual(report["excluded_published_after_tip"], 1)
        self.assertEqual(report["graded_rows"], 0)


if __name__ == "__main__":
    unittest.main()
