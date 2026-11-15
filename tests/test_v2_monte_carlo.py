import unittest

import numpy as np
import pandas as pd

from v2.simulation import engine, probabilities as P


def _players(n=4):
    return pd.DataFrame({"game_id": ["g1", "g1", "g2", "g2"][:n], "minutes": [34.0, 28.0, 30.0, 12.0][:n],
                         "minutes_sd_l10": [3.0, 5.0, np.nan, 6.0][:n]})


MEANS = {"pts": np.array([24.0, 12.0, 18.0, 4.0]), "reb": np.array([8.0, 4.0, 6.0, 2.0]),
         "ast": np.array([6.0, 3.0, 2.0, 1.0]), "threes": np.array([2.5, 1.0, 1.5, 0.3])}


class EngineTests(unittest.TestCase):
    def test_same_seed_same_simulation(self):
        a = engine.simulate(_players(), MEANS, engine.Params(), 500, seed=11)
        b = engine.simulate(_players(), MEANS, engine.Params(), 500, seed=11)
        np.testing.assert_array_equal(a["pts"], b["pts"])

    def test_pra_is_the_same_path(self):
        s = engine.simulate(_players(), MEANS, engine.Params(), 200, seed=1)
        np.testing.assert_array_equal(s["pra"], s["pts"] + s["reb"] + s["ast"])

    def test_simulated_mean_matches_model_expectation(self):
        # The simulation adds shape only; it must not re-apply minutes or injury effects.
        s = engine.simulate(_players(), MEANS, engine.Params(player_sd=0.2, game_sd=0.1), 20000, seed=3)
        np.testing.assert_allclose(s["pts"].mean(axis=1), MEANS["pts"], rtol=0.03)
        np.testing.assert_allclose(s["reb"].mean(axis=1), MEANS["reb"], rtol=0.03)

    def test_shared_game_factor_correlates_teammates(self):
        p = engine.Params(game_sd=0.25, player_sd=0.0, minutes_sd_scale=0.01, early_exit_prob=0.0,
                          dispersion={"pts": 1000.0})
        s = engine.simulate(_players(), {"pts": MEANS["pts"]}, p, 20000, seed=5)
        same_game = np.corrcoef(s["pts"][0], s["pts"][1])[0, 1]
        other_game = np.corrcoef(s["pts"][0], s["pts"][2])[0, 1]
        self.assertGreater(same_game, 0.2)
        self.assertLess(abs(other_game), 0.05)

    def test_event_built_points_share_the_threes_path(self):
        s = engine.simulate(_players(), MEANS, engine.Params(), 4000, seed=4)
        self.assertTrue(((s["pts"] - 3 * s["threes"]) >= 0).all())
        # Points move with threes; the old count model had no link between them.
        self.assertGreater(np.corrcoef(s["pts"][0], s["threes"][0])[0, 1], 0.3)

    def test_event_built_points_are_wider_than_a_single_count(self):
        players = pd.DataFrame({"game_id": ["g"], "minutes": [34.0], "minutes_sd_l10": [0.01]})
        quiet = dict(minutes_sd_scale=0.01, early_exit_prob=0.0, game_sd=0.0, player_sd=0.0,
                     dispersion={"pts": 1000.0, "threes": 1000.0})
        means = {"pts": np.array([20.0]), "threes": np.array([2.0])}
        events = engine.simulate(players, means, engine.Params(**quiet), 40000, seed=8)["pts"]
        count = engine.simulate(players, means, engine.Params(**quiet, points_model="count"), 40000, seed=8)["pts"]
        self.assertAlmostEqual(events.mean(), 20.0, delta=0.15)
        self.assertGreater(events.var(), 1.8 * count.var())

    def test_minutes_stay_in_bounds(self):
        s = engine.simulate(_players(), {}, engine.Params(minutes_sd_scale=3.0), 2000, seed=2)
        self.assertGreaterEqual(s["minutes"].min(), 0)
        self.assertLessEqual(s["minutes"].max(), 48)


class ProbabilityTests(unittest.TestCase):
    def test_over_under_push_sum_to_one_and_push_only_on_whole_lines(self):
        sims = np.array([[3, 4, 4, 5, 6]])
        half = P.over_under_push(sims, [4.5])
        whole = P.over_under_push(sims, [4.0])
        self.assertAlmostEqual(half["over"][0] + half["under"][0] + half["push"][0], 1.0)
        self.assertEqual(half["push"][0], 0.0)
        self.assertAlmostEqual(whole["push"][0], 0.4)
        self.assertAlmostEqual(whole["over"][0], 0.4)

    def test_sample_crps_matches_normal_formula(self):
        rng = np.random.default_rng(0)
        sims = rng.normal(10, 3, (1, 200000))
        self.assertAlmostEqual(P.crps_samples(sims, [12])[0], P.crps_normal(10, 3, 12), places=2)

    def test_sampling_error_shrinks_with_more_simulations(self):
        self.assertLess(P.sampling_se(np.array([0.5]), 50000)[0], P.sampling_se(np.array([0.5]), 10000)[0])


class PriceResearchTests(unittest.TestCase):
    def test_side_probabilities_and_ev_at_american_odds(self):
        from v2.simulation import price_research as R
        sims = np.array([[3, 4, 4, 5, 6], [3, 4, 4, 5, 6]])
        win, push, loss = R.side_probabilities(sims, [4.0, 4.5], ["OVER", "UNDER"])
        np.testing.assert_allclose(win, [0.4, 0.6]); np.testing.assert_allclose(push, [0.4, 0.0])
        np.testing.assert_allclose(loss, [0.2, 0.4])
        np.testing.assert_allclose(R.american_to_decimal([-110, 150]), [1 + 100 / 110, 2.5])
        # A push returns the stake: EV counts only win and loss.
        self.assertAlmostEqual(R.expected_value([0.4], [0.2], [100])[0], 0.2)

    def test_gate_is_the_live_rule_unchanged(self):
        import calibrate_edges
        from v2.simulation import price_research as R
        rng = np.random.default_rng(0)
        n = 600
        rows = pd.DataFrame({"date": [f"2026-11-{1 + i % 10:02d}" for i in range(n)],
                             "p_market": rng.uniform(0.4, 0.6, n), "p_arm": rng.uniform(0.3, 0.7, n)})
        rows["outcome"] = np.where(rng.random(n) < rows["p_market"], "WIN", "LOSS")
        live = calibrate_edges.temporal_calibration(
            [{"market": "PROP", "model_family_id": "x", "date": r.date, "outcome": r.outcome,
              "market_probability": r.p_market, "raw_model_probability": r.p_arm} for r in rows.itertuples()],
            active_ids={"PROP": "x"})["prop_price"]
        gate = R.calibration_gate(rows, "arm")
        self.assertEqual(gate["status"], live["status"])
        self.assertEqual(gate["candidate_shrink"], live.get("candidate_shrink"))


class StrictGateTests(unittest.TestCase):
    def _rows(self, informative, n_dates=40, per_date=150, seed=0):
        rng = np.random.default_rng(seed)
        rows = []
        for d in range(n_dates):
            for k in range(per_date):
                truth = rng.uniform(0.3, 0.7)
                market = np.clip(truth + rng.normal(0, 0.12), 0.05, 0.95)
                arm = truth if informative else rng.uniform(0.3, 0.7)
                rows.append({"date": f"d{d:03d}", "stat": ["PTS", "REB", "AST"][k % 3], "p_market": market,
                             "p_arm": arm, "outcome": "WIN" if rng.random() < truth else "LOSS"})
        return pd.DataFrame(rows)

    def test_noise_fails_and_a_genuinely_better_arm_passes(self):
        from v2.simulation import gate
        self.assertEqual(gate.strict_gate(self._rows(False), "arm")["status"], "FAIL")
        self.assertEqual(gate.strict_gate(self._rows(True), "arm")["status"], "PASS")

    def test_too_few_holdout_dates_fails_even_when_better(self):
        from v2.simulation import gate
        res = gate.strict_gate(self._rows(True, n_dates=17, per_date=150), "arm")
        self.assertFalse(res["checks"]["enough_data"])
        self.assertEqual(res["status"], "FAIL")


class CouplingTests(unittest.TestCase):
    def test_coupling_keeps_each_players_distribution_and_sets_correlation(self):
        df = pd.DataFrame({"game_id": ["g"] * 6, "team": list("AAABBB"),
                           "minutes": [34, 30, 25, 33, 28, 20.0], "minutes_sd_l10": [4.0] * 6})
        m = {"pts": np.array([25, 15, 8, 22, 12, 6.0]), "reb": np.array([5, 6, 3, 8, 4, 2.0]),
             "ast": np.array([6, 3, 2, 4, 2, 1.0]), "threes": np.array([3, 1.5, 1, 2, 1, 0.5])}
        a = engine.simulate(df, m, engine.Params(), 20000, seed=1)
        b = engine.simulate(df, m, engine.Params(team_corr=-0.3, opp_corr=0.2), 20000, seed=1)
        for s in ("pts", "reb", "ast", "threes", "minutes"):
            for i in range(6):
                np.testing.assert_array_equal(np.sort(a[s][i]), np.sort(b[s][i]))
        np.testing.assert_array_equal(b["pra"], b["pts"] + b["reb"] + b["ast"])
        c = np.corrcoef(b["pts"])
        self.assertLess(c[0, 1], -0.15)
        self.assertGreater(c[0, 3], 0.1)


class CacheAndDiagnosticsTests(unittest.TestCase):
    def test_cache_misses_when_the_key_changes(self):
        import tempfile, os
        from v2.simulation import cache
        with tempfile.TemporaryDirectory() as t:
            path = os.path.join(t, "c.pkl")
            cache.save(path, cache.cache_key(features=["a"]), {"x": 1})
            self.assertEqual(cache.load(path, cache.cache_key(features=["a"])), {"x": 1})
            self.assertIsNone(cache.load(path, cache.cache_key(features=["a", "b"])))

    def test_overconfident_arm_has_slope_below_one_and_no_weight_over_the_market(self):
        from v2.simulation import price_research as R
        rng = np.random.default_rng(3)
        n = 20000
        truth = rng.uniform(0.35, 0.65, n)
        won = rng.random(n) < truth
        rows = pd.DataFrame({"side": "OVER", "won": won, "p_market": truth,
                             "p_arm": np.clip(0.5 + 2.5 * (truth - 0.5), 0.01, 0.99)})
        d = R.calibration_diagnostics(rows, "arm")
        self.assertLess(d["slope"], 0.6)
        self.assertLess(d["weight_on_top_of_market"], 0.2)


class LeakageTests(unittest.TestCase):
    def test_out_of_sample_models_train_only_on_earlier_seasons(self):
        from unittest.mock import patch
        from v2.simulation import validation
        seen = []
        rows = []
        for season in (2021, 2022, 2023):
            for i in range(30):
                rows.append({"game_id": f"{season}-{i}", "player_id": "p", "player": "P",
                             "game_date": pd.Timestamp(f"{season + 1}-01-01"), "v2_season": season,
                             "minutes": 30.0, "minutes_sd_l10": 4.0, "actual_minutes": 30.0,
                             **{s: float(i % 7) for s in engine.STATS},
                             **{f"roll5_{s}": 3.0 for s in engine.STATS}})
        df = pd.DataFrame(rows)
        for col in validation.pilot.OPP + validation.pilot.ABS:
            df[col] = np.nan

        class Spy:
            def fit(self, X, y):
                seen.append(sorted(df.loc[X.index, "v2_season"].unique()))
                return self

            def predict(self, X):
                return np.zeros(len(X))

        with patch.object(validation.pilot, "build_dataset", return_value=(df, {})), \
             patch.object(validation.pilot, "model", lambda: Spy()):
            validation.build_oos([2021, 2022, 2023], [2023], "unused")
        self.assertTrue(seen)
        self.assertTrue(all(max(s) < 2023 for s in seen))


if __name__ == "__main__":
    unittest.main()
