"""
v2/simulation/gate.py
---------------------
Strict promotion gate for Monte Carlo probabilities at recorded prices.

Fixed on 2026-11-15, before any ledger date after 2026-11-13 was examined. The live
gate (calibrate_edges.temporal_calibration) accepts a candidate that ties or beats
the market by any amount on one hold-out; Phase 3 showed that lets a 0.00004
log-loss difference through. This gate keeps the same model of credit (shrink s
toward the arm, P = no_vig + s * (arm - no_vig), fit by likelihood on earlier dates)
and asks for evidence that the improvement is real:

  1. Enough data:   >= 7 hold-out dates and >= 500 hold-out rows in the final split.
  2. Something to credit: fitted shrink > 0 on the training dates.
  3. Margin:        hold-out log loss at least 0.002 below the market's.
  4. Not noise:     95% date-clustered bootstrap upper bounds of (candidate - market)
                    below zero for BOTH log loss and Brier.
  5. Stable over time: two walk-forward splits (train on the first 60% of dates and
                    test the next 20%; train on the first 80% and test the rest) both
                    beat the market on log loss.
  6. Broad:         no stat with >= 50 hold-out rows worse than the market by more
                    than 0.005 log loss.

All must pass. It is a report: nothing here writes the live calibration or gives any
arm bet authority; promotion stays a separate, explicit decision.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from calibrate_edges import fit_prop_price_shrink

RULES = {"min_holdout_dates": 7, "min_holdout_rows": 500, "min_log_loss_margin": 0.002,
         "max_stat_log_loss_deficit": 0.005, "min_stat_rows": 50, "splits": [(0.6, 0.8), (0.8, 1.0)],
         "bootstrap": 2000, "fixed_on": "2026-11-15", "data_through_when_fixed": "2026-11-13"}
CAP = (0.01, 0.75)   # the live prop-price clip


def _probability(rows: pd.DataFrame, arm: str, shrink: float) -> np.ndarray:
    m = rows["p_market"].to_numpy(float)
    return np.clip(m + shrink * (rows[f"p_{arm}"].to_numpy(float) - m), *CAP)


def _ll(p, y):
    p = np.clip(p, 0.01, 0.99)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def _split(rows, dates, lo, hi):
    a, b = int(len(dates) * lo), int(len(dates) * hi)
    train = rows[rows["date"].isin(dates[:a])]
    test = rows[rows["date"].isin(dates[a:b])]
    return train, test


def strict_gate(rows: pd.DataFrame, arm: str, rules: dict = RULES, seed: int = 0) -> dict:
    rows = rows[rows["outcome"].isin(["WIN", "LOSS"])]
    dates = sorted(rows["date"].unique())
    checks, info = {}, {"arm": arm, "rules": rules}
    splits = []
    for lo, hi in rules["splits"]:
        train, test = _split(rows, dates, lo, hi)
        if train.empty or test.empty:
            splits.append(None)
            continue
        recs = [{"market_probability": r.p_market, "raw_model_probability": getattr(r, f"p_{arm}"),
                 "outcome": r.outcome} for r in train.itertuples()]
        s = fit_prop_price_shrink(recs)
        y = test["outcome"].eq("WIN").to_numpy(float)
        pc, pm = _probability(test, arm, s), _probability(test, arm, 0.0)
        splits.append({"train_dates": int(train["date"].nunique()), "test_dates": int(test["date"].nunique()),
                       "test_rows": int(len(test)), "shrink": s, "test": test, "y": y, "pc": pc, "pm": pm,
                       "log_loss_delta": float(_ll(pc, y).mean() - _ll(pm, y).mean()),
                       "brier_delta": float(((pc - y) ** 2).mean() - ((pm - y) ** 2).mean())})
    final = splits[-1]
    info["splits"] = [None if sp is None else {k: v for k, v in sp.items() if k not in ("test", "y", "pc", "pm")}
                      for sp in splits]
    if final is None:
        return {**info, "status": "INSUFFICIENT_DATA", "checks": {"enough_data": False}}

    checks["enough_data"] = (final["test_dates"] >= rules["min_holdout_dates"]
                             and final["test_rows"] >= rules["min_holdout_rows"])
    checks["positive_shrink"] = final["shrink"] > 0
    checks["margin"] = final["log_loss_delta"] <= -rules["min_log_loss_margin"]

    test, y, pc, pm = final["test"], final["y"], final["pc"], final["pm"]
    ll_d, br_d = _ll(pc, y) - _ll(pm, y), (pc - y) ** 2 - (pm - y) ** 2
    groups = [np.where(test["date"].to_numpy() == d)[0] for d in test["date"].unique()]
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(rules["bootstrap"]):
        idx = np.concatenate([groups[k] for k in rng.integers(0, len(groups), len(groups))])
        boots.append((ll_d[idx].mean(), br_d[idx].mean()))
    boots = np.array(boots)
    info["holdout_ci95_upper"] = {"log_loss_delta": float(np.percentile(boots[:, 0], 97.5)),
                                  "brier_delta": float(np.percentile(boots[:, 1], 97.5))}
    checks["ci_excludes_zero"] = bool(info["holdout_ci95_upper"]["log_loss_delta"] < 0
                                      and info["holdout_ci95_upper"]["brier_delta"] < 0)
    checks["walk_forward_stable"] = all(sp is not None and sp["log_loss_delta"] < 0 for sp in splits)

    per_stat = {}
    for stat, idx in test.groupby("stat").indices.items():
        if len(idx) >= rules["min_stat_rows"]:
            per_stat[stat] = float(ll_d[idx].mean())
    info["holdout_log_loss_delta_by_stat"] = per_stat
    checks["no_stat_much_worse"] = all(v <= rules["max_stat_log_loss_deficit"] for v in per_stat.values())

    info["checks"] = {k: bool(v) for k, v in checks.items()}
    info["status"] = "PASS" if all(checks.values()) else "FAIL"
    return info
