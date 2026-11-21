import unittest

from game_model_promotion import evaluate_model
from team_box_matchups import VERSION, VALIDATION


def spread_bundle(mae=11.2,directional=0.64,semantics=True):
    return {
        "status":"trained",
        "model_name":"Ridge",
        "trained_through":"2026-10-26",
        "trained_rows":6600,
        "validation":VALIDATION,
        "team_context_policy":"BOTH_TEAMS_PRIOR_ONLY_REQUIRED",
        "feature_semantics_version":(
            VERSION if semantics else "OLD"
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
        "validation":VALIDATION,
        "team_context_policy":"BOTH_TEAMS_PRIOR_ONLY_REQUIRED",
        "feature_semantics_version":(
            VERSION if semantics else "OLD"
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

    def test_validation_migration_skips_incomparable_mae_and_direction(self):
        champion = spread_bundle(mae=9., directional=.8)
        champion["validation"] = "global_chronological_timeseries_split"
        candidate = spread_bundle(mae=11.2, directional=.64)
        out = evaluate_model("spread", champion, candidate)
        self.assertTrue(out["promotable"])
        self.assertFalse(out["metric_comparison"]["performed"])
        self.assertEqual(out["metric_comparison"]["reason"], "VALIDATION_METHOD_CHANGED")
        self.assertIsNone(out["mae_delta_candidate_minus_champion"])

    def test_validation_migration_keeps_absolute_sanity_and_scope_gates(self):
        champion = totals_bundle()
        champion["validation"] = "global_chronological_timeseries_split"
        bad = totals_bundle(mae=18.)
        out = evaluate_model("totals", champion, bad)
        self.assertFalse(out["promotable"])
        self.assertFalse(out["checks"]["absolute_mae_sane"])
        research = totals_bundle()
        research["research_only"] = True
        self.assertFalse(evaluate_model("totals", champion, research)["promotable"])

    def test_matching_methods_still_compare_metrics(self):
        out = evaluate_model("totals", totals_bundle(15.0), totals_bundle(15.1))
        self.assertTrue(out["metric_comparison"]["performed"])
        self.assertFalse(out["promotable"])
        self.assertAlmostEqual(out["mae_delta_candidate_minus_champion"], .1)

    def test_unknown_candidate_validation_is_rejected(self):
        candidate = totals_bundle()
        candidate["validation"] = "unknown"
        out = evaluate_model("totals", totals_bundle(), candidate)
        self.assertFalse(out["promotable"])
        self.assertFalse(out["checks"]["validation_method"])

    def test_nonfinite_metrics_cannot_bypass_migration_checks(self):
        champion = spread_bundle()
        champion["validation"] = "global_chronological_timeseries_split"
        for value in (float("nan"), float("-inf"), float("inf"), -1.):
            with self.subTest(value=value):
                self.assertFalse(evaluate_model("spread", champion, spread_bundle(mae=value))["promotable"])
        self.assertFalse(evaluate_model("spread", champion, spread_bundle(directional=1.5))["promotable"])


if __name__=="__main__":
    unittest.main()
