"""
calibrate_edges.py
------------------
Fits how much of the model's edge over the market is real, per market, from the
graded forward ledger. kelly_sizing.py reads the result to turn edges into win
probabilities.

For each market (spread / totals / props) with at least MIN_GRADED graded
WIN/LOSS bets, finds the shrink s in [0, 1] that maximizes the likelihood of the
outcomes under P(win) = Phi(s * |edge| / outcome_std). Markets with fewer graded
bets keep kelly_sizing.EDGE_SHRINK_PRIOR. VOID, PASS and PUSH rows are ignored.

Input:
  data/history/nba_forward_predictions.jsonl

Output:
  models/edge_calibration.json

Usage:
    python calibrate_edges.py
"""
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

from kelly_sizing import (CALIBRATION_PATH, EDGE_SHRINK_PRIOR, MAX_WIN_PROB,
                          outcome_std, win_prob_from_shrink)

LEDGER = Path("data/history/nba_forward_predictions.jsonl")
MARKET_TYPES = {"SPREAD": "spread", "TOTAL": "totals", "PROP": "props"}
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
        if row.get("outcome") in {"WIN", "LOSS"} and row.get("edge") is not None:
            rows.append(row)
    return rows


def fit_shrink(rows, model_type):
    """Maximum-likelihood shrink on a grid. rows: graded ledger rows of one market."""
    def log_lik(s):
        total = 0.0
        for r in rows:
            # Uncapped: the sizing cap would flatten the likelihood at large edges.
            p = win_prob_from_shrink(abs(float(r["edge"])), s, outcome_std(model_type, r.get("stat")), cap=0.99)
            total += math.log(p if r["outcome"] == "WIN" else 1 - p)
        return total
    return max(SHRINK_GRID, key=log_lik)


def calibrate(rows, min_graded=MIN_GRADED):
    markets = {}
    for market, model_type in MARKET_TYPES.items():
        subset = [r for r in rows if r.get("market") == market]
        wins = sum(r["outcome"] == "WIN" for r in subset)
        info = {"graded": len(subset), "wins": wins, "prior": EDGE_SHRINK_PRIOR[model_type]}
        if len(subset) >= min_graded:
            info.update(status="FITTED", shrink=fit_shrink(subset, model_type))
        else:
            info.update(status="PRIOR", shrink=None,
                        note=f"needs {min_graded} graded WIN/LOSS bets to fit")
        markets[model_type] = info
    return markets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger", default=str(LEDGER))
    parser.add_argument("--out", default=CALIBRATION_PATH)
    parser.add_argument("--min-graded", type=int, default=MIN_GRADED)
    args = parser.parse_args()

    result = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": "MLE shrink: P(win) = Phi(shrink * |edge| / outcome_std)",
        "max_win_prob": MAX_WIN_PROB,
        "min_graded": args.min_graded,
        "markets": calibrate(read_graded(args.ledger), args.min_graded),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
