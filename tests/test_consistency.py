import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import consistency as cs

COLS = ["game_id", "season", "game_date", "team_name", "team_location", "team_abbr", "home_away", "player_id",
        "player_name", "position", "starter", "did_not_play", "minutes", "pts", "reb", "ast", "stl", "blk",
        "threes_made"]


def _box(path, players, days=14, dnp_day=None, extra=None):
    rows = []
    for d in range(days):
        date = f"2026-11-{d + 1:02d}"
        for i, (pid, name, pts) in enumerate(players):
            dnp = dnp_day == d and i == 0
            rows.append([1000 + d, 2026, date, "Celtics", "Boston", "BOS", "home", pid, name, "F", True, dnp,
                         0 if dnp else 34, 0 if dnp else pts[d % len(pts)], 7, 5, 1, 0, 2])
        rows.append([1000 + d, 2026, date, "Knicks", "New York", "NY", "away", 99, "Opp Guy", "G", True, False,
                     30, 10, 3, 3, 0, 0, 1])
    rows += extra or []
    pd.DataFrame(rows, columns=COLS).to_csv(path, index=False)


class ProfileTests(unittest.TestCase):
    def test_threshold_is_highest_meaningful_with_80_percent(self):
        v = np.array([20, 22, 25, 18, 30, 21, 24, 26, 19, 23])   # 20+ in 8/10
        p = cs.category_profile(v, "PTS")
        self.assertEqual(p["threshold"], 20)
        self.assertTrue(p["qualifies"])
        self.assertEqual((p["season_hits"], p["gp"]), (8, 10))

    def test_floor_blocks_trivial_thresholds_and_small_samples_do_not_qualify(self):
        p = cs.category_profile(np.array([1, 2, 1, 2, 1, 2, 1, 2, 1, 2]), "REB")   # never reaches 4
        self.assertEqual(p["threshold"], 4)
        self.assertFalse(p["qualifies"])
        self.assertFalse(cs.category_profile(np.array([30] * 9), "PTS")["qualifies"])   # 9 games < 10

    def test_streak_counts_back_from_latest(self):
        p = cs.category_profile(np.array([5, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30]), "PTS")
        self.assertEqual(p["streak"], 10)

    def test_wilson_shrinks_small_samples(self):
        self.assertLess(cs.wilson_lower(5, 5), cs.wilson_lower(50, 50))


class BuildTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.box = self.dir / "box.csv"
        for attr, name in (("OUT_DIR", "out"), ("SIGNALS", "signals.jsonl"), ("RANKINGS", "rank.jsonl"),
                           ("REPORT", "report.json"), ("OVERRIDES", "overrides.json"), ("PRED_DIR", "pred"),
                           ("RAW", "raw"), ("CALIBRATION", "cal.json")):
            self.patch(attr, self.dir / name)
        (self.dir / "raw").mkdir()
        (self.dir / "pred").mkdir()

    def patch(self, attr, value):
        old = getattr(cs, attr)
        setattr(cs, attr, value)
        self.addCleanup(setattr, cs, attr, old)

    def tearDown(self):
        self.tmp.cleanup()

    def test_dnp_excluded_duplicates_dropped_and_no_future_games(self):
        _box(self.box, [(1, "Steady Star", [25, 26, 27])], dnp_day=3,
             extra=[[1000, 2026, "2026-11-01", "Celtics", "Boston", "BOS", "home", 1, "Steady Star", "F", True,
                     False, 34, 25, 7, 5, 1, 0, 2],                      # duplicate player-game
                    [2000, 2026, "2026-11-20", "Celtics", "Boston", "BOS", "home", 1, "Steady Star", "F", True,
                     False, 34, 99, 7, 5, 1, 0, 2]])                     # target date: never used
        g = cs.load_games("2026-11-20", self.box)
        self.assertEqual(len(g[g["player_id"] == 1]), 14)
        prof = cs.player_profile(g[g["player_id"] == 1])
        self.assertEqual((prof["gp"], prof["dnp"]), (13, 1))
        self.assertLess(prof["categories"]["PTS"]["mean"], 30)

    def test_rank_never_fills_with_unqualified_and_marks_overrides(self):
        late = [[2000 + d, 2026, f"2026-11-{d + 1:02d}", "Celtics", "Boston", "BOS", "home", 2, "Bench Guy", "G",
                 False, False, 30, 30, 7, 5, 1, 0, 2] for d in range(9)]        # 9 games: never qualifies
        _box(self.box, [(1, "Steady Star", [25, 26, 27])], days=12, extra=late)
        snap = cs.build("2026-11-20", record=False, box=self.box, now=datetime(2026, 11, 20, tzinfo=timezone.utc))
        listed = [p["player"] for p in snap["active"] + snap["reserve"]]
        self.assertIn("Steady Star", listed)
        self.assertNotIn("Bench Guy", listed)
        self.assertLess(len(snap["active"]), 20)                         # empty slots stay empty
        (self.dir / "overrides.json").write_text(json.dumps({"include": [{"player": "Bench Guy", "reason": "test"}],
                                                             "exclude": [{"player": "Opp Guy", "reason": "no"}]}))
        snap = cs.build("2026-11-20", record=False, box=self.box, now=datetime(2026, 11, 20, tzinfo=timezone.utc))
        manual = [p for p in snap["active"] if p.get("override")]
        self.assertEqual([(p["player"], p["override"]) for p in manual], [("Bench Guy", "MANUAL_INCLUDE")])
        self.assertEqual([r["player"] for r in snap["removed_by_override"]], ["Opp Guy"])


class LineTests(unittest.TestCase):
    def quotes(self, line=24.5, over=-110, under=-110):
        rows = []
        for book in ("fanduel", "draftkings"):
            for side, price in (("OVER", over), ("UNDER", under)):
                rows.append({"player_key": "steady star", "stat": "pts", "book_key": book, "side": side,
                             "line": line, "price": price, "scraped_at": "2026-11-20T15:00:00Z"})
        return pd.DataFrame(rows)

    player = {"player_key": "steady star"}
    values = np.array([25, 26, 27, 24, 28, 30, 25, 26, 29, 31, 22, 27])   # over 24.5 in 10/12

    def prop(self, cal=0.70):
        return {"proj_min": 34, "props": {"pts": {"pred": 27.0, "line": 24.5, "offer_side": "OVER",
                                                  "raw_model_probability": 0.72, "model_probability": cal}}}

    def test_exact_line_counts_and_no_bet_without_calibration(self):
        r = cs.evaluate_line(self.player, "PTS", self.values, self.quotes(), self.prop(), None,
                             {"calibrated": False}, stale=False)
        self.assertEqual(r["hits"]["season"], {"n": 12, "over": 10, "under": 2, "push": 0})
        self.assertIsNone(r["edge"])
        self.assertIsNone(r["calibrated_probability"])
        self.assertEqual(r["recommendation"], "LEAN")

    def test_bet_only_with_calibration_and_margin(self):
        r = cs.evaluate_line(self.player, "PTS", self.values, self.quotes(), self.prop(0.70), None,
                             {"calibrated": True}, stale=False)
        self.assertEqual(r["recommendation"], "BET")
        self.assertAlmostEqual(r["edge"], round(0.70 - 110 / 210, 4))
        r = cs.evaluate_line(self.player, "PTS", self.values, self.quotes(), self.prop(0.53), None,
                             {"calibrated": True}, stale=False)
        self.assertNotEqual(r["recommendation"], "BET")

    def test_out_player_and_stale_data_pass(self):
        r = cs.evaluate_line(self.player, "PTS", self.values, self.quotes(), self.prop(), {"status": "Out"},
                             {"calibrated": True}, stale=False)
        self.assertEqual(r["recommendation"], "PASS")
        r = cs.evaluate_line(self.player, "PTS", self.values, self.quotes(), self.prop(), None,
                             {"calibrated": True}, stale=True)
        self.assertEqual(r["recommendation"], "PASS")

    def test_no_line_for_categories_without_markets(self):
        self.assertIsNone(cs.evaluate_line(self.player, "STL", self.values, self.quotes(), None, None,
                                           {"calibrated": False}, stale=False))

    def test_integer_line_pushes(self):
        r = cs.evaluate_line(self.player, "PTS", np.array([25, 25, 26, 24]), self.quotes(line=25),
                             None, None, {"calibrated": False}, stale=False)
        self.assertEqual(r["hits"]["season"]["push"], 2)


class GradeTests(unittest.TestCase):
    def test_trend_line_push_dnp_and_latest_phase(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            _box(t / "box.csv", [(1, "Steady Star", [25])], days=2, dnp_day=1)
            base = {"player_id": 1, "player": "Steady Star", "team": "BOS", "combined": False, "version": "v"}
            rows = [
                {**base, "signal_key": "a", "date": "2026-11-01", "phase": "EARLY", "kind": "TREND", "category": "PTS", "threshold": 20},
                {**base, "signal_key": "b", "date": "2026-11-01", "phase": "EARLY", "kind": "LINE", "category": "PTS",
                 "line": 30.5, "side": "OVER", "best_odds": -110, "recommendation": "LEAN"},
                {**base, "signal_key": "c", "date": "2026-11-01", "phase": "PRETIP", "kind": "LINE", "category": "PTS",
                 "line": 25.0, "side": "OVER", "best_odds": -110, "recommendation": "WATCH"},
                {**base, "signal_key": "d", "date": "2026-11-02", "phase": "EARLY", "kind": "TREND", "category": "PTS", "threshold": 20},
            ]
            (t / "s.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
            rep = cs.grade(box=t / "box.csv", path=t / "s.jsonl", out=t / "r.json")
            out = {(h["date"], h["kind"]): h["outcome"] for h in rep["history"]}
            self.assertEqual(out[("2026-11-01", "TREND")], "HIT")
            self.assertEqual(out[("2026-11-01", "LINE")], "PUSH")          # PRETIP row graded, not EARLY
            self.assertEqual(out[("2026-11-02", "TREND")], "VOID_DNP")
            self.assertEqual(rep["alternate_thresholds"]["n"], 1)
            self.assertNotIn("units_if_followed", rep["alternate_thresholds"])   # trends are never wagers


if __name__ == "__main__":
    unittest.main()
