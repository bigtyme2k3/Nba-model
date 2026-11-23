"""
top10_calibration.py
--------------------
Calibration check on the model's strongest opinions only (user request 2026-11-23).

The main prop calibration (calibrate_edges.py) uses every graded prop. This asks the
narrower question behind the data-collection picks: on each day's 10 largest
raw-model-minus-market gaps (one per player, FanDuel/DraftKings/Fanatics, current
props model family), does the model's probability beat the market's?

  record       wins, win rate, mean raw model probability, mean no-vig market
               probability, flat 1u follow units at the recorded price
  buckets      the same by gap size (how overconfident the model is as the gap grows)
  temporal     fit the shrink s in  P = market + s * (raw - market)  on earlier dates,
               score later dates (Brier, log loss) against the market alone (s = 0) and
               the raw model (s = 1); same date split rule as calibrate_edges.py

Research only. Nothing here changes models/edge_calibration.json, a probability, a
pick or a BET decision.

Usage:
    python top10_calibration.py
Output:
    data/tracking/top10_calibration.json
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

from calibrate_edges import LEDGER, MIN_CALIBRATION_DATES, SHRINK_GRID, _num, read_graded
from data_bets_report import opposite_from_pair
from model_publication import model_family_id

OUT = Path("data/tracking/top10_calibration.json")
BOOKS = {"fanduel", "draftkings", "fanatics"}
PER_DAY = 10
MIN_TRAIN, MIN_HOLDOUT = 40, 20
BUCKETS = [(0.0, 0.05), (0.05, 0.10), (0.10, 0.15), (0.15, 1.0)]


def _decimal(odds):
    o = _num(odds)
    if o is None or o == 0:
        return None
    return 1 + o / 100 if o > 0 else 1 + 100 / abs(o)


def calibration_family(path="models/edge_calibration.json"):
    """The family the main calibration used, if the props bundle can't be loaded here."""
    try:
        return json.loads(Path(path).read_text())["markets"]["prop_price"].get("model_family_id")
    except (OSError, KeyError, ValueError, TypeError):
        return None


def eligible(rows, family):
    out, seen = [], set()
    for r in rows:
        raw, mkt = _num(r.get("raw_model_probability")), _num(r.get("market_probability"))
        if (r.get("market") != "PROP" or raw is None or mkt is None or not r.get("date")
                or r.get("sportsbook_key") not in BOOKS
                or not family or r.get("model_family_id") != family):
            continue
        key = (r["date"], " ".join(str(r.get("player")).lower().split()), str(r.get("stat")).upper())
        if key in seen:          # first-observed row only
            continue
        seen.add(key)
        out.append({**r, "_raw": raw, "_mkt": mkt, "_gap": raw - mkt, "_win": r["outcome"] == "WIN"})
    return out


def top_per_day(rows, n=PER_DAY):
    picks = []
    for date in sorted({r["date"] for r in rows}):
        day = sorted((r for r in rows if r["date"] == date), key=lambda r: -r["_gap"])
        players = set()
        for r in day:
            p = " ".join(str(r.get("player")).lower().split())
            if p in players:
                continue
            players.add(p)
            picks.append(r)
            if sum(1 for x in picks if x["date"] == date) >= n:
                break
    return picks


def record(rows):
    if not rows:
        return {"n": 0}
    n = len(rows)
    wins = sum(r["_win"] for r in rows)
    units = [(_decimal(r.get("odds")) - 1 if r["_win"] else -1.0) for r in rows if _decimal(r.get("odds"))]
    fades = []
    for r in rows:   # the opposite side at the same book and line (recorded, else implied by the pair)
        opp = _num(r.get("opposite_odds"))
        opp = opp if opp is not None else opposite_from_pair(r.get("odds"), r["_mkt"])
        if _decimal(opp):
            fades.append(_decimal(opp) - 1 if not r["_win"] else -1.0)
    return {"n": n, "wins": wins, "win_rate": round(wins / n, 4),
            "mean_model_probability": round(sum(r["_raw"] for r in rows) / n, 4),
            "mean_market_probability": round(sum(r["_mkt"] for r in rows) / n, 4),
            "mean_gap": round(sum(r["_gap"] for r in rows) / n, 4),
            "follow_units": round(sum(units), 2) if units else None,
            "follow_roi": round(sum(units) / len(units), 4) if units else None,
            "fade_units": round(sum(fades), 2) if fades else None,
            "fade_roi": round(sum(fades) / len(fades), 4) if fades else None}


def _prob(r, s):
    return min(0.99, max(0.01, r["_mkt"] + s * (r["_raw"] - r["_mkt"])))


def scores(rows, s):
    n = len(rows)
    return {"brier": round(sum((_prob(r, s) - r["_win"]) ** 2 for r in rows) / n, 5),
            "log_loss": round(-sum(math.log(_prob(r, s) if r["_win"] else 1 - _prob(r, s)) for r in rows) / n, 5)}


def fit_shrink(rows):
    def ll(s):
        return sum(math.log(_prob(r, s) if r["_win"] else 1 - _prob(r, s)) for r in rows)
    return max(SHRINK_GRID, key=ll)


def temporal(rows):
    dates = sorted({r["date"] for r in rows})
    info = {"dates": len(dates), "min_dates": MIN_CALIBRATION_DATES, "status": "INSUFFICIENT_DATED_EVIDENCE"}
    if len(dates) < MIN_CALIBRATION_DATES:
        return info
    cut = max(1, int(len(dates) * .8))
    train = [r for r in rows if r["date"] < dates[cut]]
    hold = [r for r in rows if r["date"] >= dates[cut]]
    info.update(train_through=dates[cut - 1], holdout_from=dates[cut], training_rows=len(train), holdout_rows=len(hold),
                full_sample_shrink=fit_shrink(rows))
    if len(train) < MIN_TRAIN or len(hold) < MIN_HOLDOUT:
        info["status"] = "INSUFFICIENT_ROWS"
        return info
    s = fit_shrink(train)
    cand, market, raw = scores(hold, s), scores(hold, 0.0), scores(hold, 1.0)
    beats = cand["brier"] < market["brier"] and cand["log_loss"] < market["log_loss"]
    info.update(status="MODEL_ADDS_INFORMATION" if beats and s > 0 else "MARKET_AS_GOOD_OR_BETTER",
                fitted_shrink=s, holdout_fitted=cand, holdout_market=market, holdout_raw_model=raw,
                holdout_record=record(hold))
    return info


def build(ledger=LEDGER, family=None):
    if family is None:
        family = model_family_id("models/props_models.pkl") or calibration_family()
    pool = eligible(read_graded(ledger), family)
    top = top_per_day(pool)
    return {"generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "research_only": True,
            "rule": f"each date's {PER_DAY} largest raw-model-minus-market gaps, one per player, "
                    "FanDuel/DraftKings/Fanatics, current props model family",
            "model_family_id": family,
            "top10": record(top),
            "all_props_same_family": record(pool),
            "by_gap": {f"{lo:.0%}-{hi:.0%}" if hi < 1 else f"{lo:.0%}+": record([r for r in top if lo <= r["_gap"] < hi])
                       for lo, hi in BUCKETS},
            "by_side": {s: record([r for r in top if r.get("side") == s]) for s in ("OVER", "UNDER")},
            "temporal": temporal(top),
            "note": "shrink 0 = market probability, 1 = raw model probability. A fitted shrink above 0 that "
                    "also beats the market on later dates would mean the strongest opinions carry information. "
                    "A win rate well under the market probability is the fade signal (follow/fade units at recorded prices). "
                    "Research only: does not change calibration, picks or BET decisions."}


def main():
    report = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("top10", "temporal")}, indent=2))


if __name__ == "__main__":
    main()
