import unittest
import pandas as pd

from build_team_features import build_game_context


class TestTeamGameContext(unittest.TestCase):
    def test_context_uses_only_prior_games(self):
        rows=[]
        dates=["2026-10-20","2026-10-22","2026-10-24"]
        for i,(d,pace,ortg,drtg,net,ts,pts,opp) in enumerate([
            (dates[0],100,110,105,5,0.55,110,105),
            (dates[1],102,120,112,8,0.60,120,112),
            (dates[2],98,130,115,15,0.62,130,115),
        ],start=1):
            rows.append({
                "game_id":str(i),"season":2026,"game_date":pd.Timestamp(d),
                "team":"Test Team","poss":pace,"ortg":ortg,"drtg":drtg,
                "net_rtg":net,"ts_pct":ts,"points":pts,"opp_points":opp,
            })
        df=pd.DataFrame(rows)
        out=build_game_context(df).sort_values("game_date").reset_index(drop=True)

        self.assertTrue(pd.isna(out.loc[0,"pace"]))
        self.assertAlmostEqual(out.loc[1,"pace"],100.0)
        self.assertAlmostEqual(out.loc[2,"pace"],101.0)
        self.assertAlmostEqual(out.loc[2,"ortg"],115.0)
        self.assertAlmostEqual(out.loc[2,"net_rtg"],6.5)
        # Target game's own 130 ORTG / +15 net rating must not enter its context.
        self.assertNotAlmostEqual(out.loc[2,"ortg"],130.0)
        self.assertNotAlmostEqual(out.loc[2,"net_rtg"],15.0)

    def test_context_can_carry_previous_season_into_opener(self):
        df=pd.DataFrame([
            {
                "game_id":"old","season":2025,"game_date":pd.Timestamp("2026-04-10"),
                "team":"Test Team","poss":101,"ortg":116,"drtg":111,
                "net_rtg":5,"ts_pct":0.58,"points":116,"opp_points":111,
            },
            {
                "game_id":"new","season":2026,"game_date":pd.Timestamp("2026-10-20"),
                "team":"Test Team","poss":105,"ortg":125,"drtg":120,
                "net_rtg":5,"ts_pct":0.61,"points":125,"opp_points":120,
            },
        ])
        out=build_game_context(df).sort_values("game_date").reset_index(drop=True)
        opener=out[out["game_id"].eq("new")].iloc[0]
        self.assertAlmostEqual(opener["pace"],101.0)
        self.assertAlmostEqual(opener["ortg"],116.0)
        self.assertEqual(int(opener["games_prior"]),0)

    def test_context_unique_by_game_and_team(self):
        df=pd.DataFrame([
            {
                "game_id":"1","season":2026,"game_date":pd.Timestamp("2026-10-20"),
                "team":"A","poss":100,"ortg":110,"drtg":108,"net_rtg":2,
                "ts_pct":0.56,"points":110,"opp_points":108,
            },
            {
                "game_id":"1","season":2026,"game_date":pd.Timestamp("2026-10-20"),
                "team":"B","poss":100,"ortg":108,"drtg":110,"net_rtg":-2,
                "ts_pct":0.54,"points":108,"opp_points":110,
            },
        ])
        out=build_game_context(df)
        self.assertEqual(len(out),2)
        self.assertFalse(out.duplicated(["game_id","team"]).any())


if __name__=="__main__":
    unittest.main()
