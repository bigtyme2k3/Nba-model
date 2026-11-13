import unittest

import pandas as pd

import quarter_records as qr


def _games(rows):
    df = pd.DataFrame(rows, columns=["game_id", "game_time", "home_team", "away_team", "home_q1", "away_q1"])
    df["game_time"] = pd.to_datetime(df["game_time"], utc=True)
    return df


class FirstQuarterParseTests(unittest.TestCase):
    def test_hoopr_linescores_old_and_new_formats(self):
        self.assertEqual(qr.first_quarter("[{'value': 22.0} {'value': 17.0}]"), 22.0)
        new = "[{'displayValue': '23', 'period': 1, 'value': 23.0}\n {'displayValue': '19', 'period': 2, 'value': 19.0}]"
        self.assertEqual(qr.first_quarter(new), 23.0)
        self.assertIsNone(qr.first_quarter(None))


class FirstQuarterRecordTests(unittest.TestCase):
    def setUp(self):
        self.games = _games([
            ("1", "2026-11-01", "Boston Celtics", "New York Knicks", 30, 25),   # BOS wins Q1
            ("2", "2026-11-03", "New York Knicks", "Boston Celtics", 28, 28),   # tie
            ("3", "2026-11-05", "Boston Celtics", "Miami Heat", 20, 26),        # BOS loses Q1
        ])

    def test_historical_record_uses_only_earlier_games(self):
        rec = qr.historical_q1_records(self.games).set_index(["game_id", "team"])
        first = rec.loc[("1", "Boston Celtics")]
        self.assertEqual((first.q1_wins, first.q1_losses, first.q1_ties), (0, 0, 0))
        self.assertEqual(first.q1_win_pct, 0.5)
        third = rec.loc[("3", "Boston Celtics")]
        self.assertEqual((third.q1_wins, third.q1_losses, third.q1_ties), (1, 0, 1))
        self.assertAlmostEqual(third.q1_win_pct, 0.75)
        self.assertAlmostEqual(third.q1_margin_l10, 2.5)  # (+5, 0) / 2

    def test_live_record_counts_every_completed_game(self):
        live = qr.live_q1_records(self.games, ["Boston Celtics", "Chicago Bulls"], 2026)
        bos = live["Boston Celtics"]
        self.assertEqual((bos["q1_wins"], bos["q1_losses"], bos["q1_ties"]), (1, 1, 1))
        self.assertEqual(live["Chicago Bulls"]["q1_win_pct"], 0.5)

    def test_record_resets_each_season(self):
        games = pd.concat([self.games, _games([
            ("4", "2027-10-25", "Boston Celtics", "Miami Heat", 10, 30)])])
        rec = qr.historical_q1_records(games).set_index(["game_id", "team"])
        opener = rec.loc[("4", "Boston Celtics")]
        self.assertEqual((opener.q1_wins, opener.q1_losses, opener.q1_ties), (0, 0, 0))

    def test_attach_gives_neutral_defaults_for_unknown_games(self):
        games = pd.DataFrame([{"game_id": "99", "home_team": "Boston Celtics", "away_team": "Miami Heat"}])
        out = qr.attach_to_games(games, qr.historical_q1_records(self.games))
        self.assertEqual(out.loc[0, "home_q1_win_pct"], 0.5)
        self.assertEqual(out.loc[0, "q1_win_pct_diff"], 0.0)


if __name__ == "__main__":
    unittest.main()
