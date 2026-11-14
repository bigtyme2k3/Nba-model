"""
backtest_vs_close.py
--------------------
Out-of-sample test of the spread and totals models against near-closing market
lines from backfill_historical_lines.py.

For each month of the test season the model is refit (same features and
algorithm as spread_model.py / totals_model.py, Ridge) on every game before that
month, then predicts that month's games. Each prediction is compared with the
near-close line:
  - MAE of model vs MAE of the line against the final result
  - against-the-spread (or over/under) record of the model's side by edge size,
    with flat-stake ROI at -110
  - the edge shrink s in P(win) = Phi(s * edge / outcome_std), fit on the first
    half of the season and scored on the second half against s = 0 (market only)

Usage:
    python backtest_vs_close.py --season 2025
Output:
    data/tracking/historical_backtest_<season>.json
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import spread_model as sm
import totals_model as tm
from kelly_sizing import OUTCOME_STD, win_prob_from_shrink
from teams import normalize_team

EDGE_BUCKETS = [0, 1, 2, 3, 4, 6]
SHRINK_GRID = [i / 100 for i in range(101)]
PAYOUT_110 = 100 / 110


def load_lines(path: Path) -> pd.DataFrame:
    lines = pd.read_csv(path)
    for c in ("home_team", "away_team"):
        lines[c] = lines[c].map(normalize_team)
    lines["game_date"] = pd.to_datetime(lines["game_date"]).dt.date
    return lines


def join_lines(df: pd.DataFrame, lines: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["_date"] = pd.to_datetime(df["game_date"]).dt.date
    keyed = lines.rename(columns={"game_date": "_date"})[
        ["_date", "home_team", "away_team", "close_spread_home", "close_total", "hours_before_tip"]]
    out = df.merge(keyed, on=["_date", "home_team", "away_team"], how="left")
    # hoopR dates can be the UTC date for late tips; retry unmatched games a day earlier.
    miss = out["close_spread_home"].isna() & out["close_total"].isna()
    if miss.any():
        alt = keyed.assign(_date=lambda k: k["_date"] + pd.Timedelta(days=1))
        retry = out.loc[miss, df.columns].merge(alt, on=["_date", "home_team", "away_team"], how="left")
        out.loc[miss, ["close_spread_home", "close_total", "hours_before_tip"]] = \
            retry[["close_spread_home", "close_total", "hours_before_tip"]].values
    return out


def monthly_predictions(df, features, target, build, season_start):
    """Refit at the start of each test month on all earlier games; predict that month."""
    preds = pd.Series(np.nan, index=df.index)
    test = df[df["game_date"] >= season_start]
    for month, rows in test.groupby(test["game_date"].dt.to_period("M")):
        train = df[(df["game_date"] < month.start_time) & df[target].notna()]
        model = build()
        model.fit(train[features].fillna(0), train[target])
        preds.loc[rows.index] = model.predict(rows[features].fillna(0))
    return preds


def side_record(edge: pd.Series, result_vs_line: pd.Series) -> list[dict]:
    """Model side = sign(edge). result_vs_line > 0 means the side the edge favours won."""
    rows = []
    for lo in EDGE_BUCKETS:
        m = edge.abs() >= lo
        won = (np.sign(edge[m]) * result_vs_line[m])
        wins, losses = int((won > 0).sum()), int((won < 0).sum())
        n = wins + losses
        rate = wins / n if n else None
        # One-sided normal approximation against a coin flip.
        z = (wins - n / 2) / math.sqrt(n / 4) if n else None
        rows.append({"min_edge": lo, "bets": n, "wins": wins, "losses": losses,
                     "pushes": int((won == 0).sum()), "win_rate": rate,
                     "roi_at_110": (wins * PAYOUT_110 - losses) / n if n else None,
                     "z_vs_coin": z})
    return rows


def shrink_check(edge: pd.Series, won: pd.Series, std: float) -> dict:
    """Fit s on the first half of the season, score it on the second half vs s = 0."""
    keep = won.notna() & edge.notna()
    edge, won = edge[keep].abs(), won[keep].astype(bool)
    half = len(edge) // 2
    if half < 50:
        return {"status": "INSUFFICIENT_GAMES", "games": int(len(edge))}

    def log_loss(s, e, w):
        p = np.array([win_prob_from_shrink(x, s, std, cap=0.99) for x in e])
        return float(-np.mean(np.where(w, np.log(p), np.log(1 - p))))
    fit_e, fit_w, test_e, test_w = edge.iloc[:half], won.iloc[:half], edge.iloc[half:], won.iloc[half:]
    s = min(SHRINK_GRID, key=lambda x: log_loss(x, fit_e, fit_w))
    candidate, market = log_loss(s, test_e, test_w), log_loss(0.0, test_e, test_w)
    return {"status": "BEATS_MARKET" if candidate < market else "MARKET_BETTER",
            "fitted_shrink": s, "holdout_log_loss": candidate, "market_only_log_loss": market,
            "fit_games": int(half), "holdout_games": int(len(edge) - half)}


def evaluate(df, pred, actual, line, std, kind):
    have = df[line].notna() & pred.notna() & df[actual].notna()
    d, p = df[have], pred[have]
    if kind == "spread":
        market_margin = -d[line]                      # home spread -5.5 => home by 5.5
        edge = p - market_margin                      # >0: model likes home vs the number
        result = d[actual] - market_margin            # >0: home covered
    else:
        market_margin = d[line]
        edge = p - market_margin                      # >0: model likes the over
        result = d[actual] - market_margin            # >0: went over
    won = (np.sign(edge) * result).where(lambda x: x != 0) > 0
    return {
        "games": int(have.sum()),
        "model_mae": float(np.mean(np.abs(d[actual] - p))),
        "market_mae": float(np.mean(np.abs(d[actual] - market_margin))),
        "mean_abs_model_minus_market": float(np.mean(np.abs(edge))),
        "median_hours_before_tip": float(d["hours_before_tip"].median()),
        "record_by_edge": side_record(edge, result),
        "shrink": shrink_check(edge, won.where(result != 0), std),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=2025)
    ap.add_argument("--data", default="data/processed/master_all.csv")
    ap.add_argument("--lines", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    lines_path = Path(a.lines or f"data/raw/historical_lines_{a.season}.csv")
    out = Path(a.out or f"data/tracking/historical_backtest_{a.season}.json")
    season_start = pd.Timestamp(f"{a.season}-08-01")
    season_end = pd.Timestamp(f"{a.season + 1}-08-01")

    with contextlib.redirect_stdout(io.StringIO()):
        df = sm.load_and_prepare(a.data)
        df = tm.engineer_totals_features(df)
    df = df[df["game_date"] < season_end].sort_values("game_date").reset_index(drop=True)
    df = join_lines(df, load_lines(lines_path))
    in_season = df["game_date"] >= season_start

    sp_feats = [f for f in sm.SPREAD_FEATURES if f in df.columns]
    tot_feats = [f for f in tm.TOTALS_FEATURES if f in df.columns]
    sp_pred = monthly_predictions(df, sp_feats, "actual_spread", sm.build_ridge, season_start)
    tot_pred = monthly_predictions(df, tot_feats, "actual_total", sm.build_ridge, season_start)

    s = df[in_season]
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "season": a.season,
        "lines_file": str(lines_path),
        "season_games": int(in_season.sum()),
        "games_matched_to_lines": int(s["close_spread_home"].notna().sum()),
        "method": "monthly refit on all earlier games (Ridge); near-close median line across FanDuel/DraftKings/Fanatics",
        "spread": evaluate(s, sp_pred[in_season], "actual_spread", "close_spread_home", OUTCOME_STD["spread"], "spread"),
        "totals": evaluate(s, tot_pred[in_season], "actual_total", "close_total", OUTCOME_STD["totals"], "totals"),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
