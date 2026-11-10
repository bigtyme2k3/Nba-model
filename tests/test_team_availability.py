import unittest

import pandas as pd

import team_availability as ta


def _logs(rows):
    return ta.prepare_logs(pd.DataFrame(rows))


def _game(gid, date, team, player, minutes=30, pts=20, season=2026):
    return {"game_id": gid, "game_date": date, "season": season, "team": team, "player": player,
            "minutes": minutes, "pts": pts, "reb": 4, "ast": 2, "stl": 1, "blk": 0, "tov": 2}


class TeamAvailabilityTests(unittest.TestCase):
    def setUp(self):
        rows = []
        for i in range(1, 6):
            date = f"2026-11-0{i}"
            rows.append(_game(i, date, "Boston Celtics", "Star"))
            rows.append(_game(i, date, "Boston Celtics", "Bench", minutes=8, pts=3))
        # Game 6: the star sits; only the bench player appears.
        rows.append(_game(6, "2026-11-06", "Boston Celtics", "Bench", minutes=8, pts=3))
        self.logs = _logs(rows)

    def test_absent_regular_is_counted_with_prior_values_only(self):
        avail = ta.historical_availability(self.logs).set_index("game_id")
        self.assertEqual(list(avail.index), ["6"])
        row = avail.loc["6"]
        self.assertEqual(row["vac_players"], 1)
        self.assertAlmostEqual(row["vac_minutes"], 30.0)
        # 20 pts + 0.5*4 reb + 2 ast + 1 stl - 2 tov
        self.assertAlmostEqual(row["vac_value"], 23.0)

    def test_low_minute_player_is_not_a_regular(self):
        rows = [_game(i, f"2026-11-0{i}", "Boston Celtics", "Bench", minutes=8) for i in range(1, 5)]
        rows.append(_game(5, "2026-11-05", "Boston Celtics", "Other"))
        self.assertTrue(ta.historical_availability(_logs(rows)).empty)

    def test_live_counts_only_out_regulars_on_slate_teams(self):
        injuries = pd.DataFrame([
            {"player": "Star", "severity": "OUT", "is_out": True},
            {"player": "Bench", "severity": "OUT", "is_out": True},
        ])
        live = ta.live_availability(self.logs, injuries, ["Boston Celtics", "New York Knicks"])
        self.assertEqual(live["Boston Celtics"]["players"], ["Star"])
        self.assertAlmostEqual(live["Boston Celtics"]["vac_value"], 23.0)
        self.assertEqual(live["New York Knicks"]["vac_players"], 0)

    def test_questionable_is_not_counted_live(self):
        injuries = pd.DataFrame([{"player": "Star", "severity": "QUESTIONABLE", "is_out": False}])
        live = ta.live_availability(self.logs, injuries, ["Boston Celtics"])
        self.assertEqual(live["Boston Celtics"]["vac_players"], 0)

    def test_attach_fills_zero_when_nobody_missing(self):
        games = pd.DataFrame([{"game_id": "6", "home_team": "Boston Celtics", "away_team": "New York Knicks"}])
        out = ta.attach_to_games(games, ta.historical_availability(self.logs))
        self.assertAlmostEqual(out.loc[0, "home_vac_value"], 23.0)
        self.assertEqual(out.loc[0, "away_vac_value"], 0.0)
        self.assertAlmostEqual(out.loc[0, "vac_value_diff"], 23.0)


if __name__ == "__main__":
    unittest.main()
