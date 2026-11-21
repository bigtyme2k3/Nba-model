import unittest
import pandas as pd
from unittest.mock import patch
import merge_data

from merge_data import attach_team_game_context


class TestTeamContextMerge(unittest.TestCase):
    def test_odds_provider_id_does_not_replace_canonical_game_id(self):
        games = pd.DataFrame([{"game_id": "espn-1", "season": 2026,
            "game_date": pd.Timestamp("2026-11-01"), "home_team": "Boston Celtics",
            "away_team": "Atlanta Hawks", "home_pts": 110, "away_pts": 100}])
        odds = pd.DataFrame([{"game_id": "odds-1", "game_date": pd.Timestamp("2026-11-01"),
            "home_team": "Boston Celtics", "away_team": "Atlanta Hawks", "avg_total_point": 220.5}])
        with patch.object(merge_data, "load_game_logs", return_value=games), \
             patch.object(merge_data, "load_odds", return_value=odds), \
             patch.object(merge_data, "load_team_game_context", return_value=pd.DataFrame()), \
             patch.object(merge_data, "load_team_stats", return_value=pd.DataFrame()), \
             patch.object(merge_data, "team_availability_table", return_value=pd.DataFrame(
                 columns=["game_id", "team", *merge_data.team_availability.VAC_COLS])), \
             patch.object(merge_data, "q1_records_table", return_value=pd.DataFrame(
                 columns=["game_id", "team", *merge_data.quarter_records.Q1_COLS])):
            out = merge_data.process_year(2026)
        self.assertEqual(out.iloc[0].game_id, "espn-1")
        self.assertEqual(out.iloc[0].game_id_odds, "odds-1")

    def test_numeric_and_string_game_ids_merge(self):
        games=pd.DataFrame([
            {"game_id":123,"home_team":"A","away_team":"B","game_date":"2026-10-20"}
        ])
        ctx=pd.DataFrame([
            {
                "game_id":"123","team":"A","games_prior":4,"pace":100.0,"ortg":115.0,
                "drtg":110.0,"net_rtg":5.0,"ts_pct":0.58,
                "roll5_net_rtg":4.0,"roll10_net_rtg":3.0,
                "roll5_points":114.0,"roll10_points":112.0,
                "roll5_allowed":109.0,"roll10_allowed":110.0,
            },
            {
                "game_id":"123","team":"B","games_prior":4,"pace":98.0,"ortg":111.0,
                "drtg":113.0,"net_rtg":-2.0,"ts_pct":0.55,
                "roll5_net_rtg":-1.0,"roll10_net_rtg":-2.0,
                "roll5_points":108.0,"roll10_points":109.0,
                "roll5_allowed":112.0,"roll10_allowed":111.0,
            },
        ])
        out=attach_team_game_context(games,ctx)
        self.assertEqual(len(out),1)
        self.assertAlmostEqual(out.iloc[0]["home_pace"],100.0)
        self.assertAlmostEqual(out.iloc[0]["away_pace"],98.0)
        self.assertEqual(int(out.iloc[0]["team_context_complete"]),1)

    def test_cardinality_never_changes(self):
        games=pd.DataFrame([
            {"game_id":"1","home_team":"A","away_team":"B"},
            {"game_id":"2","home_team":"C","away_team":"D"},
        ])
        ctx=pd.DataFrame([
            {"game_id":"1","team":"A","pace":100},
            {"game_id":"1","team":"B","pace":99},
            {"game_id":"2","team":"C","pace":101},
            {"game_id":"2","team":"D","pace":98},
        ])
        out=attach_team_game_context(games,ctx)
        self.assertEqual(len(out),len(games))


if __name__=="__main__":
    unittest.main()
