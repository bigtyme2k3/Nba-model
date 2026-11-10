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
from forward_ledger import EVIDENCE_EPOCH
from game_markets import priced_evidence
from model_publication import model_family_id

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
        if (row.get("outcome") in {"WIN", "LOSS"} and row.get("evidence_epoch") == EVIDENCE_EPOCH
                and priced_evidence(row)):
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


def temporal_calibration(rows, min_graded=MIN_GRADED, active_ids=None):
    """Fit earlier whole dates; validate later dates against the market alone.

    Active family IDs prevent pooling different algorithms or feature semantics.
    Market-only is the fallback until the held-out comparison passes.
    """
    active_ids = active_ids or {}
    out = {}
    for name, market in [("prop_price", "PROP"), ("spread", "SPREAD"), ("totals", "TOTAL")]:
        subset = [r for r in rows if r.get("market") == market and r.get("model_family_id")
                  and r.get("model_family_id") == active_ids.get(market) and r.get("date")
                  and has_price_probs(r)]
        info = {"status": "MARKET_ONLY", "shrink": 0.0, "graded": len(subset),
                "model_family_id": active_ids.get(market), "validation": "INSUFFICIENT_DATED_EVIDENCE"}
        dates = sorted({r["date"] for r in subset})
        if len(dates) >= 5:
            cut = max(1, int(len(dates) * .8))
            training = [r for r in subset if r["date"] < dates[cut]]
            holdout = [r for r in subset if r["date"] >= dates[cut]]
            info.update(training_rows=len(training), holdout_rows=len(holdout),
                        train_through=dates[cut-1], holdout_from=dates[cut])
            if len(training) >= min_graded and len(holdout) >= 50:
                fitted = fit_prop_price_shrink(training)
                def probability(row, shrink):
                    value = _num(row["market_probability"]) + shrink * (_num(row["raw_model_probability"]) - _num(row["market_probability"]))
                    return min(.75 if name == "prop_price" else .60, max(.01, value))
                def scores(shrink):
                    probs = [(max(.01, min(.99, probability(r, shrink))), r["outcome"] == "WIN") for r in holdout]
                    return {"brier": sum((p-y)**2 for p,y in probs)/len(probs),
                            "log_loss": -sum(math.log(p if y else 1-p) for p,y in probs)/len(probs)}
                candidate, baseline = scores(fitted), scores(0)
                passed = candidate["brier"] <= baseline["brier"] and candidate["log_loss"] <= baseline["log_loss"]
                info.update(validation="PASSED" if passed else "REJECTED_HOLDOUT", candidate_shrink=fitted,
                            candidate_scores=candidate, market_scores=baseline)
                if passed:
                    info.update(status="VALIDATED", shrink=fitted)
        out[name] = info
    out["props"] = {"status": "MARKET_ONLY", "shrink": 0.0, "graded": 0}
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger", default=str(LEDGER))
    parser.add_argument("--out", default=CALIBRATION_PATH)
    parser.add_argument("--min-graded", type=int, default=MIN_GRADED)
    args = parser.parse_args()

    result = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": "earlier-date MLE fit; later-date Brier and log-loss nonregression versus market",
        "evidence_epoch": EVIDENCE_EPOCH,
        "min_graded": args.min_graded,
        "markets": temporal_calibration(read_graded(args.ledger), args.min_graded, {
            "PROP": model_family_id("models/props_models.pkl"),
            "SPREAD": model_family_id("models/spread_model.pkl"),
            "TOTAL": model_family_id("models/totals_model.pkl"),
        }),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
