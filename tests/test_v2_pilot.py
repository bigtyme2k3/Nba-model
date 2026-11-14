import unittest

import pandas as pd

from v2 import features, lineups


def _box():
    rows = []
    for side, prefix in (("home", "h"), ("away", "a")):
        for i in range(1, 7):
            rows.append({"game_id": "g1", "athlete_id": f"{prefix}{i}", "home_away": side,
                         "starter": i <= 5, "minutes": 0.0, "game_date": "2026-01-05",
                         "team": "Home" if side == "home" else "Away"})
    box = pd.DataFrame(rows)
    return box


def _event(play_no, period, clock, type_text, a1=None, a2=None, team="H", shooting=False,
           scoring=False, value=0, text=""):
    return {"game_id": "g1", "play_no": play_no, "period": period, "clock_remaining": clock,
            "type_text": type_text, "text": text, "team_id": team, "a1": a1, "a2": a2, "a3": None,
            "shooting_play": shooting, "scoring_play": scoring, "score_value": value,
            "home_team_id": "H", "away_team_id": "A"}


class LineupReconstructionTests(unittest.TestCase):
    def test_single_period_with_one_substitution(self):
        ev = pd.DataFrame([
            _event(1, 1, 700, "Jump Shot", "h1", "h2", shooting=True, scoring=True, value=2),  # h2 assists
            _event(2, 1, 690, "Jump Shot", "a1", team="A", shooting=True),                     # miss
            _event(3, 1, 689, "Defensive Rebound", "h3"),
            _event(4, 1, 360, "Substitution", "h6", "h5", text="h6 enters the game for h5"),
            _event(5, 1, 0, "End Period"),
        ])
        box = _box()
        # Box minutes consistent with the reconstruction (h5 out at 6:00, h6 in).
        box.loc[box["home_away"] == "away", "minutes"] = [12, 12, 12, 12, 12, 0]
        box.loc[box["home_away"] == "home", "minutes"] = [12, 12, 12, 12, 6, 6]
        stints, status = lineups.reconstruct_game(ev, box)
        self.assertTrue(status["valid"], status)
        per = stints.groupby("player_id").sum(numeric_only=True)
        self.assertAlmostEqual(per.loc["h5", "seconds"], 360)
        self.assertAlmostEqual(per.loc["h6", "seconds"], 360)
        self.assertEqual(per.loc["h2", "ast"], 1)
        self.assertEqual(per.loc["h1", "ast_chances"], 0)      # own make is not an assist chance
        self.assertEqual(per.loc["h2", "ast_chances"], 1)
        self.assertEqual(per.loc["h3", "dreb_chances"], 1)
        self.assertEqual(per.loc["a2", "oreb_chances"], 1)
        self.assertEqual(per.loc["h3", "reb"], 1)
        self.assertEqual(per.loc["h6", "dreb_chances"], 0)      # entered after the miss

    def test_substitution_removing_someone_off_court_invalidates_game(self):
        ev = pd.DataFrame([_event(1, 1, 500, "Substitution", "h6", "a1", text="bad")])
        _, status = lineups.reconstruct_game(ev, _box())
        self.assertFalse(status["valid"])


class PriorOnlyFeatureTests(unittest.TestCase):
    def test_a_games_own_stats_never_enter_its_features(self):
        pg = pd.DataFrame({
            "game_id": ["g1", "g2", "g3"], "player_id": ["p"] * 3,
            "game_date": pd.to_datetime(["2026-01-01", "2026-01-03", "2026-01-05"]),
            "season": [2025] * 3, "team": ["T"] * 3,
            "seconds": [1800.0] * 3, "oreb_chances": [10.0] * 3, "dreb_chances": [20.0] * 3,
            "ast_chances": [30.0] * 3, "team_fga": [40.0] * 3,
            "reb": [5.0, 5.0, 5.0], "ast": [3.0] * 3, "fga": [10.0] * 3, "pts": [12.0] * 3})
        before = features.opportunity_features(pg)
        changed = pg.copy()
        changed.loc[2, ["reb", "oreb_chances", "ast"]] = [30.0, 90.0, 20.0]   # game 3 blows up
        after = features.opportunity_features(changed)
        pd.testing.assert_frame_equal(before, after)                         # game 3's row unaffected
        self.assertTrue(pd.isna(before.loc[0, "reb_conv"]))                 # no prior games: NaN, not 0
        self.assertAlmostEqual(before.loc[1, "reb_conv"], 5 / 30)


if __name__ == "__main__":
    unittest.main()
