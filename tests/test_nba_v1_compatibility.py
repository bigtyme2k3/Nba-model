import importlib
import unittest

import pandas as pd
import sklearn

from build_player_features import coalesce_team_aliases


class ExistingNBACompatibilityTests(unittest.TestCase):
    def test_existing_core_imports(self):
        for module in ("build_player_features","build_team_features","props_model","daily_runner","forward_ledger","forward_reconciler","results_tracker"):
            importlib.import_module(module)

    @unittest.skipUnless(sklearn.__version__ == "1.3.2", "Existing pickles require repository-pinned scikit-learn 1.3.2")
    def test_existing_model_artifacts_and_empty_slate_pipeline(self):
        import daily_runner
        from datetime import date
        result=daily_runner.run_pipeline(date(2026,10,4),[],posted_props={})
        self.assertEqual(result["games"],[])

    def test_current_collector_points_aliases_do_not_duplicate_columns(self):
        df=pd.DataFrame({"points":[114,None],"pts":[114,110],"turnovers":[10,12]})
        out=coalesce_team_aliases(df)
        self.assertEqual(out["team_points"].tolist(),[114,110])
        self.assertTrue(out.columns.is_unique)
        self.assertEqual(out["team_tov"].tolist(),[10,12])

    def test_historical_team_score_priority(self):
        df=pd.DataFrame({"team_score":[112,None],"points":[999,109],"pts":[888,109]})
        out=coalesce_team_aliases(df)
        self.assertEqual(out["team_points"].tolist(),[112,109])
        self.assertTrue(out.columns.is_unique)

    def test_existing_canonical_schema_unchanged(self):
        df=pd.DataFrame({"team_points":[111],"team_fga":[90],"home_away":["home"]})
        self.assertTrue(coalesce_team_aliases(df).equals(df))

    def test_team_context_uses_prior_not_current_boxscore(self):
        import tempfile
        from pathlib import Path
        from build_player_features import load_team_context
        with tempfile.TemporaryDirectory() as temp:
            rows=[]
            for i in range(4):
                rows.append({"game_id":str(i),"game_date":f"2026-10-{i+1:02d}","team_display_name":"Boston Celtics",
                             "home_away":"home","points":100+i*10,"pts":100+i*10,
                             "field_goals_attempted":80+i,"free_throws_attempted":20,"offensive_rebounds":10,"turnovers":10,"opp_points":100})
            pd.DataFrame(rows).to_csv(Path(temp)/"hoopr_team_box_2026.csv",index=False)
            games=pd.DataFrame(columns=["game_id"])
            context=load_team_context(temp,games)
            expected=sum(80+i+8.8 for i in range(3))/3
            self.assertAlmostEqual(context.iloc[3]["team_pace"],expected)


if __name__ == "__main__":
    unittest.main()
