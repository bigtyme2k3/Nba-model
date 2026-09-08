"""
results_tracker.py
------------------
Grades model picks when final scores are available and keeps model tracking files.

Inputs:
  predictions/predictions_YYYY-MM-DD.json
  data/raw/scores_YYYY-MM-DD.csv or scores_today.csv

Outputs:
  data/tracking/graded_bets.csv
  data/tracking/model_tracking.json
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import date, datetime, timezone

import pandas as pd

from betting_engine import GRADED_PATH, save_json, tracking_summary

RAW_DIR = "data/raw"
PREDICTIONS_DIR = "predictions"
TRACKING_DIR = "data/tracking"


def to_float(v, default=None):
    try:
        if pd.isna(v):
            return default
        return float(v)
    except Exception:
        return default


def load_predictions(target):
    path = os.path.join(PREDICTIONS_DIR, f"predictions_{target}.json")
    if not os.path.exists(path):
        return None, path
    with open(path) as f:
        return json.load(f), path


def load_scores(target):
    for path in [os.path.join(RAW_DIR, f"scores_{target}.csv"), os.path.join(RAW_DIR, "scores_today.csv")]:
        if os.path.exists(path):
            try:
                return pd.read_csv(path), path
            except Exception:
                pass
    return pd.DataFrame(), None


def score_lookup(scores_df):
    lookup = {}
    if scores_df.empty:
        return lookup
    for _, row in scores_df.iterrows():
        status = str(row.get("STATUS", row.get("status", ""))).upper()
        if "FINAL" not in status:
            continue
        home = str(row.get("HOME", row.get("home_team", "")))
        away = str(row.get("AWAY", row.get("away_team", "")))
        hs = to_float(row.get("HOME_SCORE", row.get("home_score")))
        aas = to_float(row.get("AWAY_SCORE", row.get("away_score")))
        if not home or not away or hs is None or aas is None:
            continue
        lookup[f"{away} @ {home}"] = {"home": home, "away": away, "home_score": hs, "away_score": aas, "total": hs + aas, "margin": hs - aas}
    return lookup


def grade_spread(bet, result):
    line = to_float(bet.get("market_line"))
    play = str(bet.get("play", ""))
    if line is None:
        return None
    if result["home"] in play:
        cover_margin = result["margin"] + line
    elif result["away"] in play:
        cover_margin = -result["margin"] - line
    else:
        cover_margin = abs(result["margin"]) - abs(line)
    if abs(cover_margin) < 0.01:
        return "PUSH", 0
    return ("WIN", 0.91) if cover_margin > 0 else ("LOSS", -1)


def grade_total(bet, result):
    line = to_float(bet.get("market_line"))
    play = str(bet.get("play", "")).upper()
    if line is None:
        return None
    diff = result["total"] - line
    if abs(diff) < 0.01:
        return "PUSH", 0
    if "OVER" in play:
        return ("WIN", 0.91) if diff > 0 else ("LOSS", -1)
    return ("WIN", 0.91) if diff < 0 else ("LOSS", -1)


def grade_predictions(data, score_map, target):
    rows = []
    for bet in data.get("best_bets", []):
        typ = str(bet.get("type", "")).upper()
        game = bet.get("game")
        result = score_map.get(game)
        if not result or typ not in {"SPREAD", "TOTAL"}:
            continue
        graded = grade_spread(bet, result) if typ == "SPREAD" else grade_total(bet, result)
        if not graded:
            continue
        outcome, profit = graded
        rows.append({
            "date": target, "type": typ, "game": game, "play": bet.get("play"),
            "conf": bet.get("conf"), "stars": bet.get("stars"), "edge": bet.get("edge"),
            "ev": bet.get("ev"), "grade": bet.get("grade"), "result": outcome,
            "profit_units": profit, "clv": None, "graded_at_utc": datetime.now(timezone.utc).isoformat(),
        })
    return rows


def upsert_rows(rows):
    os.makedirs(TRACKING_DIR, exist_ok=True)
    new_df = pd.DataFrame(rows)
    if new_df.empty:
        return 0
    if os.path.exists(GRADED_PATH):
        old = pd.read_csv(GRADED_PATH)
        key_cols = ["date", "type", "game", "play"]
        if all(c in old.columns for c in key_cols):
            old_key = old[key_cols].astype(str).agg("|".join, axis=1)
            new_key = new_df[key_cols].astype(str).agg("|".join, axis=1)
            old = old[~old_key.isin(set(new_key))]
        out = pd.concat([old, new_df], ignore_index=True)
    else:
        out = new_df
    out.to_csv(GRADED_PATH, index=False)
    return len(new_df)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=str(date.today()))
    args = parser.parse_args()

    print(f"\n═══ RESULTS TRACKER — {args.date} ═══\n")
    data, pred_path = load_predictions(args.date)
    if not data:
        print(f"  [WARN] No predictions file: {pred_path}")
        save_json(os.path.join(TRACKING_DIR, "model_tracking.json"), tracking_summary())
        return
    scores, score_path = load_scores(args.date)
    print(f"  Predictions: {pred_path}")
    print(f"  Scores: {score_path or 'missing'} ({len(scores)} rows)")
    rows = grade_predictions(data, score_lookup(scores), args.date)
    added = upsert_rows(rows)
    summary = tracking_summary()
    save_json(os.path.join(TRACKING_DIR, "model_tracking.json"), summary)
    print(f"  Graded bets: {added}")
    print(f"  Record: {summary.get('overall')} | ROI {summary.get('roi')}")
    print("✅ Results tracker complete.")


if __name__ == "__main__":
    main()
