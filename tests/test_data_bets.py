import unittest

import pandas as pd

import daily_runner as dr
import data_bets_report as rep


def _prop(raw, mkt, side="OVER", book="draftkings", block=False, odds=-110):
    return {"pred": 20.0, "line": 18.5, "edge": 1.5, "signal": None, "offer_side": side, "availability_block": block,
            "book": book, "book_key": book, "odds": odds, "raw_model_probability": raw, "model_probability": mkt,
            "market_probability": mkt, "probability_edge": 0.0, "ev": -0.04, "opposite_odds": -110}


def _games():
    players = []
    for i in range(14):
        players.append({"player": f"P{i}", "props": {"pts": _prop(0.50 + i / 100, 0.50), "reb": _prop(0.52 + i / 100, 0.50)}})
    players.append({"player": "Hurt", "props": {"pts": _prop(0.99, 0.50, block=True)}})
    players.append({"player": "Exchange", "props": {"pts": _prop(0.99, 0.50, book="novig")}})
    return [{"home": {"name": "H"}, "away": {"name": "A"}, "tip": "2026-11-18T00:00Z", "props": players}]


class DataCollectionBetTests(unittest.TestCase):
    def test_top_ten_strongest_one_per_player_skips_blocked_and_exchange(self):
        games = _games()
        picks = dr.data_collection_bets(games, existing=[])
        self.assertEqual(len(picks), 10)
        players = [p["player"] for p in picks]
        self.assertEqual(len(set(players)), 10)
        self.assertNotIn("Hurt", players)
        self.assertNotIn("Exchange", players)
        self.assertEqual(players[0], "P13")                      # largest raw - market gap first
        self.assertEqual(picks[0]["stat"], "REB")
        self.assertTrue(all(p["verdict"] == "BET" and p["units"] == 1.0 and p["bet_tier"] == "DATA_COLLECTION"
                            for p in picks))
        self.assertEqual(games[0]["props"][13]["props"]["reb"]["signal"], "OVER")

    def test_players_already_on_the_card_are_not_doubled(self):
        picks = dr.data_collection_bets(_games(), existing=[{"type": "PROP", "player": "P13"}])
        self.assertNotIn("P13", [p["player"] for p in picks])

    def test_opposite_odds_same_book_and_line(self):
        offers = [{"book_key": "fanduel", "side": "UNDER", "line": 18.5, "price": -125},
                  {"book_key": "draftkings", "side": "UNDER", "line": 18.5, "price": -105}]
        self.assertEqual(dr.opposite_odds(offers, "fanduel", 18.5, "OVER"), -125)
        self.assertIsNone(dr.opposite_odds(offers, "fanduel", 19.5, "OVER"))


class FadeReportTests(unittest.TestCase):
    def test_follow_and_fade_units_at_real_prices(self):
        rows = pd.DataFrame({"outcome": ["WIN", "LOSS", "LOSS"], "odds": [100, -110, 120], "opp_odds": [-120, -110, -140]})
        r = rep.record(rows)
        self.assertEqual((r["model_wins"], r["model_losses"]), (1, 2))
        self.assertAlmostEqual(r["follow_units"], 1.0 - 2.0)
        self.assertAlmostEqual(r["fade_units"], round(-1 + 100 / 110 + 100 / 140, 2))

    def test_opposite_price_from_pair(self):
        # -110 / -110 pair: no-vig 0.5 -> opposite -110
        self.assertEqual(rep.opposite_from_pair(-110, 0.5), -110)


if __name__ == "__main__":
    unittest.main()
