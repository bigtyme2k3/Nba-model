import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import calibrate_edges
import kelly_sizing


class EdgeProbabilityTests(unittest.TestCase):
    def test_edge_only_probability_is_shrunk_and_capped(self):
        self.assertLess(kelly_sizing.win_prob_from_shrink(3, 0.25, 12.0), 0.53)
        self.assertEqual(kelly_sizing.win_prob_from_shrink(100, 0.25, 12.0),
                         kelly_sizing.MAX_EDGE_WIN_PROB)

    def test_under_and_over_edges_size_the_same(self):
        # Exercise sizing symmetry independently of today's market-only gate.
        with patch.object(kelly_sizing, 'load_calibration', return_value={'edge': dict(kelly_sizing.EDGE_SHRINK_PRIOR)}):
            under, over = kelly_sizing.size_bet(-8, "totals"), kelly_sizing.size_bet(8, "totals")
        self.assertEqual(under["win_prob"], over["win_prob"])
        self.assertGreater(under["win_prob"], 0.5)

    def test_prop_probability_moves_from_market_toward_model(self):
        self.assertAlmostEqual(kelly_sizing.prop_model_probability(0.70, 0.50, shrink=0.5), 0.60)
        self.assertEqual(kelly_sizing.prop_model_probability(0.99, 0.50, shrink=1.0), 0.75)


class CalibrationFitTests(unittest.TestCase):
    def test_prop_price_fit_recovers_known_shrink(self):
        rng = np.random.default_rng(0)
        rows = []
        for _ in range(4000):
            m = rng.uniform(0.40, 0.60)
            raw = float(np.clip(m + rng.normal(0, 0.2), 0.02, 0.98))
            p = m + 0.3 * (raw - m)
            rows.append({"market": "PROP", "raw_model_probability": raw, "market_probability": m,
                         "outcome": "WIN" if rng.random() < p else "LOSS", "decision": "PASS"})
        fitted = calibrate_edges.calibrate(rows)
        self.assertEqual(fitted["prop_price"]["status"], "FITTED")
        self.assertAlmostEqual(fitted["prop_price"]["shrink"], 0.3, delta=0.1)
        self.assertEqual(fitted["spread"]["status"], "PRIOR")

    def test_edge_fit_recovers_known_shrink(self):
        rng = np.random.default_rng(1)
        rows = []
        for e in rng.uniform(2, 12, 3000):
            p = kelly_sizing.win_prob_from_shrink(e, 0.4, kelly_sizing.OUTCOME_STD["spread"], cap=0.99)
            rows.append({"market": "SPREAD", "edge": e, "outcome": "WIN" if rng.random() < p else "LOSS"})
        fitted = calibrate_edges.calibrate(rows)
        self.assertEqual(fitted["spread"]["status"], "FITTED")
        self.assertAlmostEqual(fitted["spread"]["shrink"], 0.4, delta=0.08)

    def test_fitted_values_override_priors_only_when_fitted(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "cal.json"
            path.write_text(json.dumps({"markets": {
                "prop_price": {"status": "FITTED", "shrink": 0.1},
                "spread": {"status": "PRIOR", "shrink": None},
            }}))
            cal = kelly_sizing.load_calibration(str(path))
        self.assertEqual(cal["prop_price"], 0.1)
        self.assertEqual(cal["sources"]["prop_price"], "FITTED")
        self.assertEqual(cal["edge"]["spread"], kelly_sizing.EDGE_SHRINK_PRIOR["spread"])


if __name__ == "__main__":
    unittest.main()
