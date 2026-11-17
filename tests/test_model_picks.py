import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

import model_picks as mp

NOW = datetime(2026, 11, 18, 18, 0, tzinfo=timezone.utc)


def _prop(raw, side="OVER", book="draftkings", block=False):
    return {"pred": 20.0, "line": 18.5, "offer_side": side, "availability_block": block, "book_key": book,
            "odds": -110, "opposite_odds": -110, "raw_model_probability": raw, "market_probability": 0.5}


def _data(games, tip="2026-11-19T00:00Z"):
    out = []
    for gi in range(games):
        players = [{"player": f"G{gi}P{i}", "props": {"pts": _prop(0.51 + i / 100 + gi / 1000)}} for i in range(6)]
        out.append({"home": {"name": f"H{gi}"}, "away": {"name": f"A{gi}"}, "tip": tip, "props": players})
    return {"date": "2026-11-18", "generated": "x", "games": out}


class ModelPickTests(unittest.TestCase):
    def test_early_takes_ten_strongest_one_per_player(self):
        rows = mp.select(_data(3), "EARLY", [], 3, now=NOW)
        self.assertEqual(len(rows), 10)
        self.assertEqual(len({r["player"] for r in rows}), 10)
        self.assertEqual(rows[0]["player"], "G2P5")
        self.assertTrue(all(r["units"] == 1.0 and r["phase"] == "EARLY" for r in rows))

    def test_no_pick_after_tip_and_no_exchange_or_blocked(self):
        d = _data(1, tip="2026-11-18T17:00Z")           # already started at NOW
        self.assertEqual(mp.select(d, "EARLY", [], 1, now=NOW), [])
        d = _data(1)
        for p in d["games"][0]["props"]:
            p["props"]["pts"]["book_key"] = "novig"
        self.assertEqual(mp.select(d, "EARLY", [], 1, now=NOW), [])

    def test_adjusted_is_proportional_and_does_not_repeat_a_window(self):
        window = _data(2)                                # 2 of a 5-game slate -> ceil(10*2/5) = 4 picks
        rows = mp.select(window, "ADJUSTED", [], 5, now=NOW)
        self.assertEqual(len(rows), 4)
        again = mp.select(window, "ADJUSTED", rows, 5, now=NOW)
        self.assertEqual(again, [])                     # those games already have adjusted picks

    def test_record_appends_immutably(self):
        with tempfile.TemporaryDirectory() as t:
            pred = Path(t, "p.json")
            pred.write_text(json.dumps(_data(2)))
            path = Path(t, "picks.jsonl")
            mp.select  # noqa
            first = mp.select(json.loads(pred.read_text()), "EARLY", [], 2, now=NOW)
            path.write_text("".join(json.dumps(r) + "\n" for r in first))
            second = mp.select(json.loads(pred.read_text()), "EARLY", mp.read_picks(path), 2, now=NOW)
            self.assertEqual(second, [])                # same date/phase/player never re-picked


if __name__ == "__main__":
    unittest.main()
