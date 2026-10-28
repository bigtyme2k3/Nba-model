import unittest

from game_model_promotion import evaluate_model


def spread_bundle(mae=11.2,directional=0.64,semantics=True):
    return {
        "status":"trained",
        "model_name":"Ridge",
        "trained_through":"2026-10-26",
        "trained_rows":6600,
        "validation":"global_chronological_timeseries_split",
        "team_context_policy":"BOTH_TEAMS_PRIOR_ONLY_REQUIRED",
        "feature_semantics_version":(
            "NBA_GAME_FEATURES_V2_PRIOR_CONTEXT" if semantics else "OLD"
        ),
        "feature_names":["a"],
        "cv_results":{"Ridge":{"mae_mean":mae,"ats_mean":directional}},
    }


def totals_bundle(mae=15.0,semantics=True):
    return {
        "status":"trained",
        "model_name":"Ridge",
        "trained_through":"2026-10-26",
        "trained_rows":6600,
        "validation":"global_chronological_timeseries_split",
        "team_context_policy":"BOTH_TEAMS_PRIOR_ONLY_REQUIRED",
        "feature_semantics_version":(
            "NBA_GAME_FEATURES_V2_PRIOR_CONTEXT" if semantics else "OLD"
        ),
        "feature_names":["a"],
        "cv_results":{"Ridge":{"mae":mae}},
    }


class TestGameModelPromotion(unittest.TestCase):
    def test_good_spread_candidate_promotes(self):
        out=evaluate_model("spread",spread_bundle(11.35,0.634),spread_bundle(11.19,0.64))
        self.assertTrue(out["promotable"])

    def test_bad_semantics_rejected(self):
        out=evaluate_model("spread",spread_bundle(),spread_bundle(11.1,0.65,semantics=False))
        self.assertFalse(out["promotable"])
        self.assertFalse(out["checks"]["feature_semantics_version"])

    def test_material_mae_regression_rejected(self):
        out=evaluate_model("totals",totals_bundle(15.27),totals_bundle(15.40))
        self.assertFalse(out["promotable"])
        self.assertFalse(out["checks"]["mae_nonregression"])

    def test_directional_regression_rejected(self):
        out=evaluate_model("spread",spread_bundle(11.35,0.64),spread_bundle(11.2,0.62))
        self.assertFalse(out["promotable"])
        self.assertFalse(out["checks"]["directional_nonregression"])


if __name__=="__main__":
    unittest.main()
