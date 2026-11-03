import tempfile
import unittest
from pathlib import Path

import pandas as pd

import build_team_features


class OpponentPointsTests(unittest.TestCase):
    def test_espn_current_box_without_opponent_score_gets_it_from_the_other_row(self):
        base = {"season": 2026, "game_date": "2026-11-02", "field_goals_attempted": 90,
                "free_throws_attempted": 20, "offensive_rebounds": 10, "turnovers": 14,
                "source": "espn_api"}
        rows = [
            {**base, "game_id": 1, "team_location": "Atlanta", "team_name": "Hawks",
             "team_abbreviation": "ATL", "home_away": "home", "points": 110},
            {**base, "game_id": 1, "team_location": "Boston", "team_name": "Celtics",
             "team_abbreviation": "BOS", "home_away": "away", "points": 100},
        ]
        with tempfile.TemporaryDirectory() as temp:
            pd.DataFrame(rows).to_csv(Path(temp) / "espn_current_team_box_2026.csv", index=False)
            boxes = build_team_features.load_team_boxes(temp)
        by_team = boxes.set_index("team")
        self.assertEqual(by_team.loc["Atlanta Hawks", "opp_points"], 100)
        self.assertEqual(by_team.loc["Boston Celtics", "margin"], -10)
        self.assertTrue(boxes["drtg"].notna().all())


if __name__ == "__main__":
    unittest.main()
