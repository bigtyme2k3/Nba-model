import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

import backfill_historical_lines as bh
import backtest_vs_close as bt
import odds_quota
import scrape_period_odds as spo


def _event(eid, tip, home_spread=-4.5, total=221.5):
    def book(key):
        return {"key": key, "title": key, "markets": [
            {"key": "spreads", "last_update": tip, "outcomes": [
                {"name": "Home", "point": home_spread, "price": -110},
                {"name": "Away", "point": -home_spread, "price": -110}]},
            {"key": "totals", "last_update": tip, "outcomes": [
                {"name": "Over", "point": total, "price": -110, "link": "https://book.example/slip"},
                {"name": "Under", "point": total, "price": -110}]}]}
    return {"id": eid, "commence_time": tip, "home_team": "Home", "away_team": "Away",
            "bookmakers": [book("fanduel"), book("draftkings")]}


class QuotaTests(unittest.TestCase):
    def test_allow_respects_reserve_and_unknown(self):
        self.assertTrue(odds_quota.allow(100, None, "x")[0])
        self.assertTrue(odds_quota.allow(100, odds_quota.reserve() + 100, "x")[0])
        self.assertFalse(odds_quota.allow(101, odds_quota.reserve() + 100, "x")[0])

    def test_last_remaining_uses_most_recent_status(self):
        with tempfile.TemporaryDirectory() as d:
            Path(d, "odds_api_status.json").write_text(json.dumps(
                {"checked_at_utc": "2026-11-13T20:00:00+00:00", "api_requests_remaining": "15203"}))
            Path(d, "sportsbook_props_status.json").write_text(json.dumps(
                {"generated_at_utc": "2026-11-13T20:05:00+00:00", "api_requests_remaining": "15158"}))
            self.assertEqual(odds_quota.last_remaining(d), 15158)


class PeriodOddsTests(unittest.TestCase):
    def test_parses_two_way_first_quarter_markets_with_no_vig(self):
        data = {"id": "e1", "commence_time": "2026-11-14T00:00:00Z", "home_team": "Home", "away_team": "Away",
                "bookmakers": [{"key": "fanduel", "title": "FanDuel", "markets": [
                    {"key": "spreads_q1", "outcomes": [{"name": "Home", "point": -1.5, "price": -115},
                                                       {"name": "Away", "point": 1.5, "price": -105}]},
                    {"key": "h2h_q1", "outcomes": [{"name": "Home", "price": -130},
                                                   {"name": "Away", "price": 110}]},
                    {"key": "totals_q1", "outcomes": [{"name": "Over", "point": 56.5, "price": -110}]}]}]}
        rows = spo.parse_event(data, "2026-11-13", "now")
        kinds = sorted({r["market"] for r in rows})
        self.assertEqual(kinds, ["ML_Q1", "SPREAD_Q1"])   # one-sided total is dropped
        spread = [r for r in rows if r["market"] == "SPREAD_Q1"]
        self.assertAlmostEqual(sum(r["no_vig_probability"] for r in spread), 1.0)


class BackfillTests(unittest.TestCase):
    def test_plan_adds_evening_snapshot_only_for_early_days(self):
        tips = pd.Series(pd.to_datetime(["2026-01-03 15:30", "2026-01-03 22:00", "2026-01-05 19:00"])
                         ).dt.tz_localize(bh.ET)
        one = bh.snapshot_plan(tips, second_pass=False)
        two = bh.snapshot_plan(tips, second_pass=True)
        self.assertEqual(len(one), 2)
        self.assertEqual(len(two), 3)
        self.assertEqual(two[-1].hour, 18)

    def test_consolidate_keeps_latest_pregame_quote_and_drops_in_play(self):
        with tempfile.TemporaryDirectory() as d:
            cache = Path(d)
            early = {"timestamp": "2026-01-05T20:00:00Z",
                     "data": [_event("g1", "2026-01-06T00:00:00Z", -3.5), _event("g0", "2026-01-05T19:00:00Z")]}
            late = {"timestamp": "2026-01-05T23:55:00Z", "data": [_event("g1", "2026-01-06T00:00:00Z", -5.0)]}
            (cache / "snap_a.json").write_text(json.dumps(early))
            (cache / "snap_b.json").write_text(json.dumps(late))
            lines = bh.consolidate(cache, 2025)
        self.assertEqual(list(lines["event_id"]), ["g1"])           # g0 had tipped: in-play
        self.assertEqual(lines.loc[0, "close_spread_home"], -5.0)   # latest pregame snapshot
        self.assertAlmostEqual(lines.loc[0, "hours_before_tip"], 0.08, places=2)


class BacktestSignTests(unittest.TestCase):
    def test_model_side_record(self):
        edge = pd.Series([3.0, -3.0, 1.0])
        result = pd.Series([2.0, 1.0, 0.0])   # home covered, home covered, push
        rec = {r["min_edge"]: r for r in bt.side_record(edge, result)}
        self.assertEqual((rec[0]["wins"], rec[0]["losses"], rec[0]["pushes"]), (1, 1, 1))
        self.assertEqual(rec[2]["bets"], 2)


if __name__ == "__main__":
    unittest.main()
