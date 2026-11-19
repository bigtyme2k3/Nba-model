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


    def test_include_override_promotes_reserve_player(self):
        players = [(i, f"Player {i}", [20 + i % 3]) for i in range(1, 26)]   # 25 qualified, 5 in reserve
        _box(self.box, players, days=12)
        snap = cs.build("2026-11-20", record=False, box=self.box, now=datetime(2026, 11, 20, tzinfo=timezone.utc))
        reserve_player = snap["reserve"][0]["player"]
        (self.dir / "overrides.json").write_text(json.dumps({"include": [{"player": reserve_player, "reason": "t"}]}))
        snap = cs.build("2026-11-20", record=False, box=self.box, now=datetime(2026, 11, 20, tzinfo=timezone.utc))
        self.assertNotIn(reserve_player, [p["player"] for p in snap["reserve"]])
        promoted = [p for p in snap["active"] if p["player"] == reserve_player]
        self.assertEqual(promoted[0]["override"], "MANUAL_INCLUDE")

    def test_pretip_without_window_file_records_no_signals(self):
        _box(self.box, [(1, "Steady Star", [25, 26, 27])], days=12)
        (self.dir / "pred" / "predictions_2026-11-20.json").write_text(json.dumps({"games": [{
            "tip": "2026-11-21T00:00Z", "home": {"name": "Boston Celtics", "abbr": "BOS"},
            "away": {"name": "New York Knicks", "abbr": "NY"}, "props": []}]}))
        now = datetime(2026, 11, 20, 15, tzinfo=timezone.utc)
        snap = cs.build("2026-11-20", "PRETIP", predictions=str(self.dir / "pred" / "missing.json"), box=self.box, now=now)
        self.assertEqual(snap["signals_recorded"], 0)
        self.assertTrue(snap["active"][0]["playing_tonight"])            # display still refreshed
        snap = cs.build("2026-11-20", "EARLY", box=self.box, now=now)
        self.assertGreater(snap["signals_recorded"], 0)

    def test_missing_stat_never_breaks_the_snapshot(self):
        _box(self.box, [(1, "Steady Star", [25, 26, 27])], days=12)
        b = pd.read_csv(self.box)
        b.loc[0, "reb"] = None
        b.loc[0, "position"] = None
        b.to_csv(self.box, index=False)
        snap = cs.build("2026-11-20", box=self.box, now=datetime(2026, 11, 20, tzinfo=timezone.utc))
        json.dumps(snap, allow_nan=False)
        star = [p for p in snap["active"] + snap["reserve"] if p["player"] == "Steady Star"][0]
        self.assertEqual(star["categories"]["REB"]["gp"], 11)              # that game left out of REB only
        self.assertEqual(star["categories"]["PTS"]["gp"], 12)

    def test_staleness_follows_the_schedule(self):
        sched = self.dir / "sched.csv"
        pd.DataFrame({"game_id": [1, 2], "season": 2026, "game_date": ["2026-02-10", "2026-02-19"],
                      "season_type": 2, "season_slug": "regular-season",
                      "status": ["STATUS_FINAL", "STATUS_SCHEDULED"]}).to_csv(sched, index=False)
        self.assertFalse(cs.box_is_stale("2026-02-19", "2026-02-10", sched))   # All-Star break, nothing missing
        self.assertTrue(cs.box_is_stale("2026-02-19", "2026-02-08", sched))    # a played game is missing


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



class SafeguardTests(unittest.TestCase):
    def test_calibration_requires_real_holdout_and_matching_family(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "calibration.json"
            data = {
                "evidence_epoch": cs.EVIDENCE_EPOCH, "min_graded": 200,
                "markets": {"prop_price": {
                    "status": "VALIDATED", "validation": "PASSED", "shrink": 0.25,
                    "model_family_id": "current-family", "dated_evidence_dates": 5,
                    "training_rows": 220, "holdout_rows": 60,
                    "candidate_scores": {"brier": .21, "log_loss": .60},
                    "market_scores": {"brier": .22, "log_loss": .61},
                }},
            }
            def check():
                path.write_text(json.dumps(data))
                return cs.calibration_state(path=path, expected_family="current-family")
            self.assertTrue(check()["calibrated"])
            data["markets"]["prop_price"]["model_family_id"] = "stale-family"
            self.assertFalse(check()["calibrated"])
            data["markets"]["prop_price"]["model_family_id"] = "current-family"
            data["markets"]["prop_price"]["holdout_rows"] = 8
            self.assertFalse(check()["calibrated"])
            data["markets"]["prop_price"]["holdout_rows"] = 60
            data["markets"]["prop_price"]["validation"] = "INSUFFICIENT_DATED_EVIDENCE"
            self.assertFalse(check()["calibrated"])
            data["markets"]["prop_price"]["validation"] = "PASSED"
            data["markets"]["prop_price"]["candidate_scores"]["brier"] = .23
            self.assertFalse(check()["calibrated"])
            data["markets"]["prop_price"]["candidate_scores"]["brier"] = .21
            data["evidence_epoch"] = "OLD_EPOCH"
            self.assertFalse(check()["calibrated"])

    def test_latest_quotes_excludes_stale_wrong_date_future_and_posttip(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "sportsbook.csv"
            template = {"game_date": "2026-11-20", "event_id": "game-1",
                        "commence_time": "2026-11-20T20:00:00Z",
                        "home_team": "Boston Celtics", "away_team": "New York Knicks",
                        "player": "Steady Star", "stat": "pts", "side": "OVER",
                        "line": 24.5, "price": -110, "book_key": "fanduel",
                        "scraped_at": "2026-11-20T14:50:00Z",
                        "last_update": "2026-11-20T14:49:00Z"}
            rows = [template]
            rows += [{**template, "book_key": "draftkings", "scraped_at": "2026-11-20T11:00:00Z"}]
            rows += [{**template, "book_key": "fanatics", "last_update": "2026-11-20T11:00:00Z"}]
            rows += [{**template, "event_id": "future-quote", "scraped_at": "2026-11-20T15:10:00Z",
                      "last_update": "2026-11-20T15:10:00Z"}]
            rows += [{**template, "event_id": "after-tip", "scraped_at": "2026-11-20T20:01:00Z",
                      "last_update": "2026-11-20T20:01:00Z"}]
            rows += [{**template, "game_date": "2026-11-19", "event_id": "previous-game"}]
            pd.DataFrame(rows).to_csv(path, index=False)
            now = datetime(2026, 11, 20, 15, tzinfo=timezone.utc)
            q = cs.latest_quotes("2026-11-20", now, path=path)
            self.assertEqual(len(q), 1)
            self.assertEqual(q.iloc[0]["event_id"], "game-1")
            self.assertEqual(q.iloc[0]["book_key"], "fanduel")

    def test_exact_game_matching_prevents_same_player_wrong_event(self):
        def item(event, home, away, line, price, book="fanduel"):
            return [{"player_key": "steady star", "stat": "pts", "book_key": book,
                     "event_id": event, "home_team": home, "away_team": away,
                     "commence_time": "2026-11-20T20:10:00Z",
                     "scraped_at": "2026-11-20T19:00:00Z",
                     "side": side, "line": line, "price": price}
                    for side in ("OVER", "UNDER")]
        q = pd.DataFrame(item("game-1", "Boston Celtics", "New York Knicks", 24.5, -110)
                         + item("game-2", "Miami Heat", "Orlando Magic", 20.5, 120, "draftkings"))
        now = datetime(2026, 11, 20, 19, 30, tzinfo=timezone.utc)
        player = {"player_key": "steady star"}
        vals = np.array([25, 26, 27, 24, 28, 30, 25, 26, 29, 31, 22, 27])
        game = {"event_id": "game-1", "tip": "2026-11-20T20:00:00Z",
                "home_team": "Boston Celtics", "away_team": "New York Knicks"}
        r = cs.evaluate_line(player, "PTS", vals, q, None, None, {"calibrated": False},
                             stale=False, game=game, now=now)
        self.assertEqual(r["line"], 24.5)
        self.assertEqual(r["best_book"], "fanduel")
        self.assertEqual(r["event_id"], "game-1")
        self.assertIsNone(r["books"]["draftkings"]["over"])
        game["event_id"] = None
        r = cs.evaluate_line(player, "PTS", vals, q, None, None, {"calibrated": False},
                             stale=False, game=game, now=now)
        self.assertEqual(r["line"], 24.5)
        game["home_team"] = "Miami Heat"
        self.assertIsNone(cs.evaluate_line(player, "PTS", vals, q, None, None,
                                            {"calibrated": False}, stale=False, game=game, now=now))

    def test_incomplete_bookmaker_sides_never_mix_lines(self):
        base = {"player_key": "steady star", "stat": "pts", "event_id": "game-1"}
        q = pd.DataFrame([
            {**base, "book_key": "fanduel", "side": "OVER", "line": 24.5, "price": -110},
            {**base, "book_key": "fanduel", "side": "UNDER", "line": 25.5, "price": -115},
            {**base, "book_key": "draftkings", "side": "OVER", "line": 24.5, "price": -120},
            {**base, "book_key": "draftkings", "side": "UNDER", "line": 24.5, "price": +100},
        ])
        r = cs.evaluate_line({"player_key": "steady star"}, "PTS", np.array([25] * 12),
                             q, None, None, {"calibrated": False}, stale=False)
        self.assertEqual(r["best_book"], "draftkings")
        self.assertIsNone(r["books"]["fanduel"]["over"])
        self.assertIsNone(cs.evaluate_line({"player_key": "steady star"}, "PTS",
                          np.array([25] * 12), q[q["book_key"] == "fanduel"], None,
                          None, {"calibrated": False}, stale=False))

    def test_same_phase_updates_remain_immutable_and_latest_is_graded(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            path = folder / "signals.jsonl"
            box = folder / "box.csv"
            out = folder / "report.json"
            _box(box, [(1, "Steady Star", [25])], days=1)
            player = {"player_id": 1, "player": "Steady Star", "team": "BOS", "rank": 1,
                      "injury": None}
            game = {"event_id": "game-1", "tip": "2026-11-01T23:00:00Z", "opp": "NY"}
            a = cs._signal("2026-11-01", "PRETIP", "LINE", player, "PTS", game,
                           datetime(2026, 11, 1, 20, 0, tzinfo=timezone.utc),
                           {"line": 25, "side": "OVER", "best_odds": -110, "recommendation": "WATCH"},
                           "snapshot-early")
            b = cs._signal("2026-11-01", "PRETIP", "LINE", player, "PTS", game,
                           datetime(2026, 11, 1, 21, 0, tzinfo=timezone.utc),
                           {"line": 24.5, "side": "OVER", "best_odds": -105, "recommendation": "LEAN"},
                           "snapshot-later")
            self.assertEqual(cs.append_signals([a], path=path), 1)
            self.assertEqual(cs.append_signals([b], path=path), 1)
            self.assertEqual(cs.append_signals([b], path=path), 0)
            self.assertEqual(len(cs.read_jsonl(path)), 2)
            result = cs.grade(box=box, path=path, out=out)
            self.assertEqual(result["exact_lines"]["n"], 1)
            rows = [r for r in result["history"] if r["kind"] == "LINE"]
            self.assertEqual(rows[0]["outcome"], "WIN")
            self.assertEqual(rows[0]["line"], 24.5)
            self.assertEqual(rows[0]["snapshot_id"], "snapshot-later")


if __name__ == "__main__":
    unittest.main()
