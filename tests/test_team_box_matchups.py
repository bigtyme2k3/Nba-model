"""Regression coverage for prior-only counts and the training/live boundary."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from build_team_features import build_game_context, build_profiles
from merge_data import attach_team_game_context
from team_box_matchups import (BOX_METRICS, MATCHUP_FEATURES, VERSION, attach_opponent_counts,
    prior_box_rates, latest_box_rates, add_matchup_features, live_matchup_features,
    training_features, whole_date_splits, save_research_candidate)


def boxes(n=25):
    rows = []
    for i in range(n):
        for team, sign in (("Boston Celtics", 1), ("Atlanta Hawks", -1)):
            net = sign * ((i * i % 29) - 10)
            fga = 80 + i % 17
            fgm = 35 + i % 7
            three_pm = 6 + i % 9
            points = 105 + net
            rows.append({"game_id": f"g{i:03}", "season": 2026 if i >= 10 else 2025,
                         "game_date": pd.Timestamp("2026-10-01") + pd.Timedelta(days=i),
                         "team": team, "points": points, "opp_points": 105 - net,
                         "fgm": fgm, "fga": fga, "three_pm": three_pm, "three_pa": 30 + i % 5,
                         "ftm": 15, "fta": 20, "oreb": 9 + i % 3, "dreb": 32 + i % 4,
                         "tov": 10 + i % 5, "poss": 100., "ortg": points,
                         "drtg": 105 - net, "net_rtg": 2 * net, "ts_pct": .57})
    return attach_opponent_counts(pd.DataFrame(rows))


class BoxRateTests(unittest.TestCase):
    def test_rates_pool_counts_and_use_opponent_rebound_chances(self):
        frame = boxes(3)
        # Make denominators differ to catch mean-of-percentages implementations.
        indices = frame.index[frame.team.eq("Boston Celtics")]
        frame.loc[indices[0], ["fgm", "fga", "three_pm", "oreb", "opp_dreb"]] = [10, 20, 2, 2, 8]
        frame.loc[indices[1], ["fgm", "fga", "three_pm", "oreb", "opp_dreb"]] = [40, 100, 10, 10, 10]
        rates = prior_box_rates(frame)
        self.assertAlmostEqual(rates.loc[indices[2], "efg_pct"], (50 + .5 * 12) / 120)
        self.assertAlmostEqual(rates.loc[indices[2], "oreb_pct"], 12 / 30)

    def test_target_and_future_box_scores_cannot_change_prior_rates(self):
        frame = boxes()
        before = prior_box_rates(frame)
        changed = frame.copy()
        mask = changed.game_date.ge(pd.Timestamp("2026-10-20"))
        for col in [c for c in frame if c not in ("game_id", "game_date", "team", "season")]:
            changed.loc[mask, col] = 999
        after = prior_box_rates(changed)
        through_target = frame.game_date.le(pd.Timestamp("2026-10-20"))
        pd.testing.assert_frame_equal(before.loc[through_target], after.loc[through_target])

    def test_missing_pairs_do_not_become_zero_opponent_stats(self):
        frame = boxes(2)
        incomplete = attach_opponent_counts(frame.iloc[[0, 2, 3]])
        self.assertTrue(pd.isna(incomplete.iloc[0].opp_fga))
        self.assertTrue(incomplete.iloc[1:].opp_fga.notna().all())

    def test_missing_component_excludes_both_numerator_and_denominator(self):
        frame = boxes(3)
        indices = frame.index[frame.team.eq("Boston Celtics")]
        frame.loc[indices[0], "three_pm"] = np.nan
        actual = prior_box_rates(frame).loc[indices[2], "efg_pct"]
        prior = frame.loc[indices[1]]
        self.assertAlmostEqual(actual, (prior.fgm + .5 * prior.three_pm) / prior.fga)

    def test_three_point_pct_shrinks_to_prior_and_excludes_recent_window_from_anchor(self):
        frame = boxes(13)
        ids = frame.index[frame.team.eq("Boston Celtics")]
        frame.loc[ids, "three_pm"] = 10
        frame.loc[ids, "three_pa"] = 20
        actual = prior_box_rates(frame).loc[ids[12], "three_pct"]
        older_rate = (2 * 10 + 100 * .36) / (2 * 20 + 100)
        self.assertAlmostEqual(actual, (10 * 10 + 100 * older_rate) / (10 * 20 + 100))

    def test_zero_attempts_leave_unobserved_percentage_missing(self):
        frame = boxes(3)
        frame["fta"] = frame["ftm"] = 0
        rates = prior_box_rates(frame)
        self.assertTrue(rates.ft_pct.isna().all())
        self.assertTrue(rates.ft_rate.dropna().eq(0).all())

    def test_every_profile_matches_historical_context_at_same_cutoff(self):
        frame = boxes()
        historical = build_game_context(frame)
        metrics = ["pace", "ortg", "drtg", "net_rtg", "ts_pct", "roll5_net_rtg", "roll10_net_rtg", *BOX_METRICS]
        # Includes windows crossing a season boundary, with nonconstant net ratings.
        for game in ("g005", "g010", "g015", "g024"):
            date = frame.loc[frame.game_id.eq(game), "game_date"].iloc[0]
            live = build_profiles(frame[frame.game_date.lt(date)]).set_index("team")
            expected = historical[historical.game_id.eq(game)].set_index("team")
            for team in live.index:
                for metric in metrics:
                    # Existing efficiency profiles round to 3/5 decimals.
                    self.assertAlmostEqual(live.loc[team, metric], expected.loc[team, metric], delta=.00051,
                                           msg=f"{game} {team} {metric}")


class MatchupIntegrationTests(unittest.TestCase):
    def fixture(self):
        frame = boxes()
        profiles = latest_box_rates(frame).set_index("team")
        home, away = profiles.loc["Boston Celtics"].to_dict(), profiles.loc["Atlanta Hawks"].to_dict()
        return home, away

    def test_live_and_historical_matchup_vectors_match(self):
        home, away = self.fixture()
        raw = {f"{side}_{key}": value for side, data in (("home", home), ("away", away)) for key, value in data.items()}
        historical = add_matchup_features(pd.DataFrame([raw])).iloc[0]
        live = live_matchup_features(home, away, MATCHUP_FEATURES)
        for feature in MATCHUP_FEATURES:
            self.assertAlmostEqual(historical[feature], live[feature])

    def test_merge_carries_all_box_context_to_both_sides(self):
        context = build_game_context(boxes())
        games = pd.DataFrame([{"game_id": "g024", "home_team": "Boston Celtics", "away_team": "Atlanta Hawks"}])
        merged = attach_team_game_context(games, context)
        self.assertEqual(len(merged), 1)
        for side in ("home", "away"):
            self.assertTrue(merged[[f"{side}_{m}" for m in BOX_METRICS]].notna().all().all())

    def test_new_models_fail_closed_with_missing_context_legacy_models_do_not_require_it(self):
        self.assertEqual(live_matchup_features({}, {}, ["home_pace"]), {})
        with self.assertRaises(ValueError):
            live_matchup_features({}, {}, MATCHUP_FEATURES)

    def test_matchup_training_requires_explicit_opt_in_and_coverage(self):
        frame = pd.DataFrame({"baseline": [1., 2.]})
        self.assertEqual(training_features(frame, ["baseline"]), ["baseline"])
        with self.assertRaises(ValueError):
            training_features(frame, ["baseline"], True)
        for feature in MATCHUP_FEATURES:
            frame[feature] = .5
        self.assertEqual(len(training_features(frame, ["baseline"], True)), 1 + len(MATCHUP_FEATURES))
        frame.loc[0, MATCHUP_FEATURES[0]] = np.nan
        with self.assertRaises(ValueError):
            training_features(frame, ["baseline"], True)

    def test_daily_spread_and_total_paths_receive_real_matchup_features(self):
        import daily_runner
        home, away = self.fixture()
        class Capture:
            def predict(self, frame):
                self.frame = frame
                return np.array([225.])
        model = Capture()
        bundle = {"model": model, "feature_names": list(MATCHUP_FEATURES)}
        ctx = {"home_rest_days": 2, "away_rest_days": 1, "home_b2b": False, "away_b2b": False,
               "home_3in4": False, "away_3in4": False, "long_travel": 0,
               "east_to_west": 0, "west_to_east": 0, "season_game_num": 20, "month": 11}
        with patch.dict(daily_runner.CURRENT_TEAM_STATS, {"H": home, "A": away}):
            for scorer in (daily_runner.run_spread, daily_runner.run_totals):
                self.assertIsNotNone(scorer("H", "A", ctx, bundle))
                self.assertTrue(np.isfinite(model.frame.to_numpy()).all())
                self.assertGreater(model.frame.home_efg_matchup.iloc[0], 0)
            daily_runner.CURRENT_TEAM_STATS["H"] = {}
            self.assertIsNone(daily_runner.run_spread("H", "A", ctx, bundle))
            self.assertIsNone(daily_runner.run_totals("H", "A", ctx, bundle))

    def test_research_bundle_cannot_replace_active_model(self):
        from game_model_promotion import evaluate_model
        with self.assertRaises(ValueError):
            save_research_candidate({}, "models/spread_model.pkl")
        with tempfile.TemporaryDirectory() as temp:
            import pickle
            path = Path(temp) / "candidate.pkl"
            save_research_candidate({"feature_semantics_version": VERSION}, path)
            bundle = pickle.loads(path.read_bytes())
            self.assertTrue(bundle["research_only"])
            self.assertFalse(evaluate_model("spread", None, bundle)["checks"]["production_scope"])

    def test_whole_date_validation_has_no_shared_dates_or_future_training(self):
        dates = pd.Series(pd.date_range("2023-01-01", periods=30).repeat(7))
        for train, test in whole_date_splits(dates):
            self.assertLess(dates.iloc[train].max(), dates.iloc[test].min())
            self.assertFalse(set(dates.iloc[train]) & set(dates.iloc[test]))

    def test_loaded_profiles_preserve_the_new_rates(self):
        import daily_runner
        profiles = build_profiles(boxes())
        with tempfile.TemporaryDirectory() as temp, \
             patch.dict(daily_runner.CURRENT_TEAM_STATS, {
                 key: dict(value) for key, value in daily_runner.CURRENT_TEAM_STATS.items()}), \
             patch.dict(daily_runner.TEAM_ROLLING, {
                 key: dict(value) for key, value in daily_runner.TEAM_ROLLING.items()}):
            path = Path(temp) / "profiles.csv"
            profiles.to_csv(path, index=False)
            self.assertEqual(daily_runner.load_team_profiles_into_context(path), 2)
            for _, row in profiles.iterrows():
                for metric in BOX_METRICS:
                    self.assertAlmostEqual(daily_runner.CURRENT_TEAM_STATS[row.team][metric], row[metric])

    def test_market_diagnostics_use_margin_sign_and_exclude_pushes(self):
        from research.game_box_matchups import market_diagnostics
        frame = pd.DataFrame({"spread_home": [-3., 2., -3., np.nan],
                              "actual_spread": [5., -3., 3., 10.]})
        result = market_diagnostics(frame, np.array([4., -1., 4., 9.]), "spread")
        self.assertEqual(result["line_rows"], 3)
        self.assertEqual(result["pushes"], 1)
        self.assertEqual(result["decisions"], 2)
        self.assertEqual(result["wins"], 1)
        self.assertFalse(result["pregame_quote_timing_validated"])

    def test_holdout_outcomes_never_choose_the_feature_group(self):
        from research.game_box_matchups import evaluate
        home, away = self.fixture()
        raw = {f"{side}_{key}": value for side, data in (("home", home), ("away", away)) for key, value in data.items()}
        rows = []
        for i in range(620):
            season = 2023 if i < 520 else 2025
            rows.append({**raw, "game_id": str(i), "game_date": pd.Timestamp(f"{season}-01-01") + pd.Timedelta(days=i % 520),
                         "season": season, "actual_spread": float(i % 11 - 5),
                         "home_rolling_5g": float(i % 5), "away_rolling_5g": float(i % 3)})
        frame = pd.DataFrame(rows)
        original = evaluate(frame, "spread", 2025)
        frame.loc[frame.season.eq(2025), "actual_spread"] += 100
        changed = evaluate(frame, "spread", 2025)
        self.assertEqual(original["cv"], changed["cv"])
        self.assertEqual(original["selected_using_development_only"], changed["selected_using_development_only"])


if __name__ == "__main__":
    unittest.main()
