"""
calibrate_edges.py
------------------
Fits how much of the model's view to credit over the market, from the graded
forward ledger. kelly_sizing.py reads the result (models/edge_calibration.json).

Two fits, by maximum likelihood on a grid of shrinks in [0, 1]:
  prop_price  P = no_vig + s * (raw_model_p - no_vig), clipped to 1-99%.
              Uses every graded prop row (BET and PASS) that recorded both
              probabilities: fitting on BET rows alone would be biased upward,
              since those rows were selected for a high model probability.
  spread / totals / props (edge-only)
              P = Phi(s * |edge| / outcome_std), on graded WIN/LOSS rows of
              that market without a priced probability.

A market needs MIN_GRADED rows to be FITTED; otherwise its prior stays in
effect. Fits are uncapped: the sizing caps would flatten the likelihood.
VOID, PASS-outcome and PUSH rows are ignored.

Usage:
    python calibrate_edges.py
"""
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

from kelly_sizing import (CALIBRATION_PATH, EDGE_SHRINK_PRIOR, PROP_PRICE_SHRINK_PRIOR,
                          outcome_std, win_prob_from_shrink)

LEDGER = Path("data/history/nba_forward_predictions.jsonl")
EDGE_MARKETS = {"SPREAD": "spread", "TOTAL": "totals", "PROP": "props"}
MIN_GRADED = 200
SHRINK_GRID = [i / 100 for i in range(101)]


def read_graded(path=LEDGER):
    rows = []
    if not Path(path).exists():
        return rows
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if row.get("outcome") in {"WIN", "LOSS"}:
            rows.append(row)
    return rows


def _num(value):
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def has_price_probs(row):
    return _num(row.get("raw_model_probability")) is not None and _num(row.get("market_probability")) is not None


def _log_lik(rows, prob):
    total = 0.0
    for r in rows:
        p = min(0.99, max(0.01, prob(r)))
        total += math.log(p if r["outcome"] == "WIN" else 1 - p)
    return total


def fit_prop_price_shrink(rows):
    def prob(s):
        return lambda r: _num(r["market_probability"]) + s * (
            _num(r["raw_model_probability"]) - _num(r["market_probability"]))
    return max(SHRINK_GRID, key=lambda s: _log_lik(rows, prob(s)))


def fit_edge_shrink(rows, model_type):
    def prob(s):
        return lambda r: win_prob_from_shrink(abs(_num(r["edge"])), s,
                                              outcome_std(model_type, r.get("stat")), cap=0.99)
    return max(SHRINK_GRID, key=lambda s: _log_lik(rows, prob(s)))


def _entry(rows, prior, fit, min_graded):
    info = {"graded": len(rows), "wins": sum(r["outcome"] == "WIN" for r in rows), "prior": prior}
    if len(rows) >= min_graded:
        info.update(status="FITTED", shrink=fit(rows))
    else:
        info.update(status="PRIOR", shrink=None, note=f"needs {min_graded} graded rows to fit")
    return info


def calibrate(rows, min_graded=MIN_GRADED):
    priced = [r for r in rows if r.get("market") == "PROP" and has_price_probs(r)]
    markets = {"prop_price": _entry(priced, PROP_PRICE_SHRINK_PRIOR, fit_prop_price_shrink, min_graded)}
    for market, model_type in EDGE_MARKETS.items():
        subset = [r for r in rows if r.get("market") == market and not has_price_probs(r)
                  and _num(r.get("edge")) is not None]
        markets[model_type] = _entry(subset, EDGE_SHRINK_PRIOR[model_type],
                                     lambda rs, t=model_type: fit_edge_shrink(rs, t), min_graded)
    return markets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger", default=str(LEDGER))
    parser.add_argument("--out", default=CALIBRATION_PATH)
    parser.add_argument("--min-graded", type=int, default=MIN_GRADED)
    args = parser.parse_args()

    result = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": "MLE shrink of model view over the market; see calibrate_edges.py",
        "min_graded": args.min_graded,
        "markets": calibrate(read_graded(args.ledger), args.min_graded),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
