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


class NovigTests(unittest.TestCase):
    def test_novig_quotes_are_kept_with_the_sportsbooks(self):
        from game_markets import offers_from_event
        event = _event("g", "2026-01-06T00:00:00Z")
        event["bookmakers"][1]["key"] = "novig"
        books = {o["book_key"] for o in offers_from_event(event, "2026-01-05T23:00:00Z")}
        self.assertEqual(books, {"fanduel", "novig"})

    def test_v1_never_selects_novig_and_novig_is_not_v1_evidence(self):
        from datetime import datetime, timezone
        from game_markets import select_offer, priced_evidence
        tip, now = "2026-01-06T00:00:00Z", datetime(2026, 1, 5, 23, 10, tzinfo=timezone.utc)
        offer = lambda book, price: {"market": "TOTAL", "side": "OVER", "line": 221.5, "price": price,
                                     "book_key": book, "book": book, "no_vig_probability": 0.5,
                                     "captured_at_utc": "2026-01-05T23:00:00Z", "commence_time": tip,
                                     "last_update": "2026-01-05T22:59:00Z", "quote_policy": "EXACT_BOOK_SIDE_V1"}
        # Novig's +105 beats the books' -110 on EV, but V1 must still pick a sportsbook.
        pick = select_offer(230.0, "TOTAL", [offer("novig", 105), offer("draftkings", -110)], now=now,
                            require_value=False)
        self.assertEqual(pick["book_key"], "draftkings")
        self.assertIsNone(select_offer(230.0, "TOTAL", [offer("novig", 105)], now=now, require_value=False))
        row = {"market": "PROP", "side": "OVER", "odds": -110}
        self.assertTrue(priced_evidence({**row, "sportsbook_key": "fanduel"}))
        self.assertFalse(priced_evidence({**row, "sportsbook_key": "novig"}))

    def test_prop_selection_uses_sportsbooks_and_keeps_novig_as_reference(self):
        import daily_runner as dr
        offers = [{"book_key": "novig", "side": "OVER", "line": 34.5, "price": 1438, "no_vig_probability": 0.065},
                  {"book_key": "novig", "side": "OVER", "line": 18.5, "price": 110, "no_vig_probability": 0.47},
                  {"book_key": "fanduel", "side": "OVER", "line": 18.5, "price": -110, "no_vig_probability": 0.49},
                  {"book_key": "fanduel", "side": "UNDER", "line": 18.5, "price": -120, "no_vig_probability": 0.51}]
        books = [o for o in offers if o["book_key"] in dr.V1_BOOKS]
        best = dr._best_prop_offer(22.5, {"residual_std": 5.8}, books, 2.0)
        self.assertEqual(best["book_key"], "fanduel")
        self.assertEqual(dr.novig_reference(offers, best["line"], best["side"])["odds"], 110)
        self.assertIsNone(dr.novig_reference(offers, 19.5, "OVER"))


class BacktestSignTests(unittest.TestCase):
    def test_model_side_record(self):
        edge = pd.Series([3.0, -3.0, 1.0])
        result = pd.Series([2.0, 1.0, 0.0])   # home covered, home covered, push
        rec = {r["min_edge"]: r for r in bt.side_record(edge, result)}
        self.assertEqual((rec[0]["wins"], rec[0]["losses"], rec[0]["pushes"]), (1, 1, 1))
        self.assertEqual(rec[2]["bets"], 2)


if __name__ == "__main__":
    unittest.main()
