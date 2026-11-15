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
