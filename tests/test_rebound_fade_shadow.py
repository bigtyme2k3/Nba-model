"""Synthetic and source-replay checks for the read-only rebound fade."""
import unittest
from copy import deepcopy
from pathlib import Path

from research import rebound_fade_shadow as f


def signal(**changes):
    row = {
        "date": "2026-11-20",
        "signal_key": "2026-11-20|PRETIP|LINE|42|REB",
        "player_id": 42,
        "player": "Test Player",
        "kind": "LINE",
        "phase": "PRETIP",
        "category": "REB",
        "side": "OVER",
        "line": 5.5,
        "recommendation": "PASS",
        "version": "consistency-v2.0",
        "recorded_at_utc": "2026-11-21T00:00:30Z",
        "quote_scraped_at": "2026-11-21T00:00:00Z",
        "tip": "2026-11-21T01:00:00Z",
        "books": {
            "fanduel": {"line": 5.5, "over": -110, "under": -115},
            "draftkings": {"line": 5.5, "over": -115, "under": -105},
            "fanatics": {"line": 6.5, "over": 105, "under": -105},
        },
    }
    row.update(changes)
    return row


def box(reb, date="2026-11-20", minutes="32.0", dnp="False"):
    return {
        (date, "42"): {
            "game_date": date, "player_id": "42", "reb": str(reb),
            "minutes": minutes, "did_not_play": dnp,
        }
    }


class QuoteTests(unittest.TestCase):
    def test_best_exact_line_under_quote(self):
        q, error = f.quote_selection(signal())
        self.assertIsNone(error)
        self.assertEqual((q["book"], q["odds"]), ("draftkings", -105))
        self.assertAlmostEqual(q["no_vig_under"], f.implied(-105) / (f.implied(-115) + f.implied(-105)))
        self.assertEqual(q["minutes_to_tip"], 60)

    def test_no_cross_line_substitution(self):
        s = signal(books={"fanduel": {"line": 6.5, "under": 125}})
        q, error = f.quote_selection(s)
        self.assertIsNone(q)
        self.assertEqual(error, "NO_SAME_LINE_UNDER_QUOTE")

    def test_posttip_and_stale_quotes_rejected(self):
        for row in (
            signal(quote_scraped_at="2026-11-21T01:01:00Z"),
            signal(quote_scraped_at="2026-11-20T23:00:00Z"),
            signal(recorded_at_utc="2026-11-21T01:01:00Z"),
        ):
            with self.subTest(row=row):
                self.assertEqual(f.quote_selection(row)[1], "STALE_OR_POST_TIP_QUOTE")

    def test_requires_explicit_verifiable_timestamp(self):
        q, reason = f.quote_selection(signal(tip=None))
        self.assertIsNone(q)
        self.assertEqual(reason, "NO_VERIFIABLE_TIMESTAMPS")

    def test_dedupe_latest_pretip_without_early_contamination(self):
        early = signal(phase="EARLY", recorded_at_utc="2026-11-21T00:01:00Z")
        older = signal(recorded_at_utc="2026-11-20T23:55:00Z")
        newer = signal(recorded_at_utc="2026-11-21T00:00:30Z")
        rows = f.latest_pretip_signals([newer, early, older])
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["recorded_at_utc"], newer["recorded_at_utc"])

    def test_missing_same_line_quote_does_not_impute(self):
        bad = signal(books={"fanduel": {"line": 6.5, "under": -110}})
        rows = f.build_rows([bad], box(3))
        self.assertEqual(rows[0]["price_status"], "NO_SAME_LINE_UNDER_QUOTE")
        self.assertIsNone(rows[0]["profit_units"])


class GradingTests(unittest.TestCase):
    def test_under_fade_at_same_line_grades_both_directions(self):
        self.assertEqual(f.build_rows([signal()], box(4))[0]["outcome"], "WIN")
        self.assertEqual(f.build_rows([signal()], box(9))[0]["outcome"], "LOSS")

    def test_push_void_and_pending_do_not_count_as_losses(self):
        self.assertEqual(f.build_rows([signal(line=5.0)], box(5))[0]["outcome"], "PUSH")
        self.assertEqual(f.build_rows([signal()], box(0, minutes="0", dnp="True"))[0]["outcome"], "VOID_DNP")
        self.assertEqual(f.build_rows([signal()], {})[0]["outcome"], "PENDING")

    def test_flat_stake_correct_and_no_bets(self):
        s1 = signal()
        s2 = signal(date="2026-11-21", player_id=13, line=5.5)
        items = f.build_rows(
            [s1, s2],
            {**box(4), ("2026-11-21", "13"): {"reb": "7", "minutes": "30", "did_not_play": "False"}},
        )
        self.assertEqual([r["outcome"] for r in items], ["WIN", "LOSS"])
        self.assertAlmostEqual(items[0]["profit_units"], 100 / 105, places=7)
        self.assertEqual(items[1]["profit_units"], -1)
        self.assertTrue(all(r["is_real_wager"] is False for r in items))
        report = f.make_report(items)
        self.assertEqual(report["discovery"]["wins"], 1)
        self.assertEqual(report["prospective_shadow"]["losses"], 1)
        self.assertEqual(report["prospective_shadow"]["bet_qualified"], 0)

    def test_american_price_math(self):
        self.assertAlmostEqual(f.american_profit(-110), 100 / 110)
        self.assertAlmostEqual(f.american_profit(120), 1.2)
        with self.assertRaises(ValueError):
            f.american_profit(50)


class ReproducibilityTests(unittest.TestCase):
    def test_existing_discovery_quotes_match_frozen_report(self):
        if not f.SIGNALS.exists() or not f.PLAYER_BOX.exists():
            self.skipTest("Repository data not available")
        rows = f.build_rows(f.read_signals(f.SIGNALS), f.read_player_box(f.PLAYER_BOX))
        discovery = [r for r in rows if r["period"] == "DISCOVERY"]
        dates = [r for r in discovery if r["date"] in {"2026-11-19", "2026-11-20"}]
        stats = f.summarize(dates)
        self.assertEqual((stats["graded"], stats["wins"], stats["losses"]), (15, 13, 2))
        self.assertEqual(stats["priced"], 15)
        self.assertAlmostEqual(stats["profit_units"], 10.9584, places=3)
        self.assertEqual(stats["bet_qualified"], 0)


if __name__ == "__main__":
    unittest.main()
