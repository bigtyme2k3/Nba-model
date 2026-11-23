import json
import tempfile
import unittest
from pathlib import Path

import top10_calibration as tc
from forward_ledger import EVIDENCE_EPOCH

FAM = "fam1"


def _row(date, player, raw, mkt, win, book="draftkings", fam=FAM, stat="PTS", side="OVER"):
    return {"market": "PROP", "date": date, "player": player, "stat": stat, "side": side, "line": 20.5,
            "odds": -110, "raw_model_probability": raw, "market_probability": mkt,
            "sportsbook_key": book, "model_family_id": fam, "outcome": "WIN" if win else "LOSS",
            "evidence_epoch": EVIDENCE_EPOCH, "decision": "PASS"}


class Top10Tests(unittest.TestCase):
    def test_selection_top_gaps_one_per_player_v1_books_family_first_observed(self):
        rows = [tc.eligible([_row("2026-11-01", f"P{i}", 0.5 + i / 100, 0.5, True)], FAM)[0] for i in range(15)]
        rows += tc.eligible([_row("2026-11-01", "P14", 0.99, 0.5, True, stat="REB")], FAM)   # same player again
        picks = tc.top_per_day(rows)
        self.assertEqual(len(picks), 10)
        self.assertEqual(picks[0]["player"], "P14")
        self.assertEqual(len({p["player"] for p in picks}), 10)
        self.assertEqual(tc.eligible([_row("d", "X", .7, .5, True, book="novig")], FAM), [])
        self.assertEqual(tc.eligible([_row("d", "X", .7, .5, True, fam="old")], FAM), [])
        self.assertEqual(tc.eligible([_row("d", "X", .7, .5, True)], None), [])            # never mixes families
        dup = tc.eligible([_row("d", "X", .7, .5, True), _row("d", "X", .9, .5, False)], FAM)
        self.assertEqual(len(dup), 1)
        self.assertTrue(dup[0]["_win"])                                                    # first observed kept

    def test_record_follow_and_fade(self):
        rows = tc.eligible([_row("d", "A", .8, .5, False), _row("d", "B", .8, .5, False, stat="AST")], FAM)
        r = tc.record(rows)
        self.assertEqual((r["n"], r["wins"]), (2, 0))
        self.assertEqual(r["follow_units"], -2.0)
        self.assertGreater(r["fade_units"], 1.5)          # opposite of -110 at a 50% pair is about -110

    def test_overconfident_model_gets_zero_weight_and_market_verdict(self):
        rows = []
        for d in range(1, 8):
            for i in range(10):
                rows.append(_row(f"2026-11-0{d}", f"P{d}{i}", 0.85, 0.5, i < 4))   # model says 85%, wins 40%
        with tempfile.TemporaryDirectory() as t:
            led = Path(t) / "l.jsonl"
            led.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
            rep = tc.build(led, FAM)
        self.assertTrue(rep["research_only"])
        self.assertEqual(rep["top10"]["n"], 70)
        self.assertEqual(rep["temporal"]["fitted_shrink"], 0.0)
        self.assertEqual(rep["temporal"]["status"], "MARKET_AS_GOOD_OR_BETTER")
        self.assertGreater(rep["temporal"]["holdout_raw_model"]["brier"], rep["temporal"]["holdout_market"]["brier"])

    def test_informative_model_is_recognised(self):
        rows = []
        for d in range(1, 8):
            for i in range(10):
                rows.append(_row(f"2026-11-0{d}", f"P{d}{i}", 0.75, 0.5, i < 8))   # model 75%, wins 80%
        with tempfile.TemporaryDirectory() as t:
            led = Path(t) / "l.jsonl"
            led.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
            rep = tc.build(led, FAM)
        self.assertGreater(rep["temporal"]["fitted_shrink"], 0.5)
        self.assertEqual(rep["temporal"]["status"], "MODEL_ADDS_INFORMATION")

    def test_too_few_dates(self):
        with tempfile.TemporaryDirectory() as t:
            led = Path(t) / "l.jsonl"
            led.write_text(json.dumps(_row("2026-11-01", "A", .8, .5, True)) + "\n")
            rep = tc.build(led, FAM)
        self.assertEqual(rep["temporal"]["status"], "INSUFFICIENT_DATED_EVIDENCE")


if __name__ == "__main__":
    unittest.main()
