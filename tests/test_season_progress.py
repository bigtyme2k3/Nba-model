import unittest
import pandas as pd

from merge_data import add_rest_days
from spread_model import engineer_features


class TestSeasonProgressSemantics(unittest.TestCase):
    def test_team_game_number_not_league_game_number(self):
        df=pd.DataFrame([
            {"season":2026,"game_date":pd.Timestamp("2026-10-20"),"home_team":"A","away_team":"B"},
            {"season":2026,"game_date":pd.Timestamp("2026-10-20"),"home_team":"C","away_team":"D"},
            {"season":2026,"game_date":pd.Timestamp("2026-10-22"),"home_team":"A","away_team":"C"},
        ])
        out=add_rest_days(df).sort_values(["game_date","home_team"]).reset_index(drop=True)
        game=out[(out["home_team"]=="A") & (out["game_date"]==pd.Timestamp("2026-10-22"))].iloc[0]
        self.assertEqual(int(game["home_season_game_num"]),2)
        self.assertEqual(int(game["away_season_game_num"]),2)
        self.assertEqual(int(game["season_game_num"]),2)

    def test_spread_engineering_preserves_team_progress(self):
        df=pd.DataFrame([
            {
                "season":2026,"game_date":pd.Timestamp("2026-10-20"),
                "season_game_num":3,
                "home_rolling_5g":1.0,"away_rolling_5g":0.0,
                "home_rolling_10g":2.0,"away_rolling_10g":1.0,
            },
            {
                "season":2026,"game_date":pd.Timestamp("2026-10-21"),
                "season_game_num":4,
                "home_rolling_5g":1.0,"away_rolling_5g":0.0,
                "home_rolling_10g":2.0,"away_rolling_10g":1.0,
            },
        ])
        out=engineer_features(df)
        self.assertEqual(out["season_game_num"].tolist(),[3,4])


if __name__=="__main__":
    unittest.main()
