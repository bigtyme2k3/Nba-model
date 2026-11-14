"""
v2/pilot.py
-----------
Phase 1 + 2 pilot: does lineup-based opportunity and teammate-absence data improve
player-prop projections over V1's features on the same games?

Steps
  1. Build the warehouse for one season and reconstruct lineups (stints table).
  2. Build pregame-only opportunity and teammate-absence features.
  3. Join V1's prop features and outcomes (data/processed/player_logs.csv) by
     game id and ESPN player id, games played only.
  4. Same chronological split for every arm (train before --cutoff, test after),
     same model as props_model.py (HistGradientBoosting), and compare MAE for
     PTS, REB, AST and PRA: V1 features alone, + opportunity, + absence, + both.

Absence uses box-score absence as the stand-in for a confirmed OUT; a live V2 would
need the injury report's timestamped status instead.

Usage:
    python -m v2.pilot --season 2025 --cutoff 2026-02-01
Output:
    data/tracking/v2_pilot_<season>.json
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

from prop_features import BASE_FEATURES
from v2 import features, lineups, warehouse

OPP = ["reb_chances36", "oreb_chances36", "dreb_chances36", "reb_conv",
       "ast_chances36", "ast_conv", "fga_share"]
ABS = ["n_absent_regulars", "absent_covered"] + [f"abs_{c}36_delta" for c in features.STATS]
TARGETS = ["pts", "reb", "ast", "pra"]


def model():
    # Same configuration as props_model.py's HistGBR candidate.
    return HistGradientBoostingRegressor(max_iter=180, learning_rate=0.05, max_leaf_nodes=15,
                                         min_samples_leaf=30, l2_regularization=1.0, random_state=42)


def build_dataset(season: int, db_path) -> tuple[pd.DataFrame, dict]:
    con = warehouse.build([season], db_path)
    checks = warehouse.warehouse_checks(con)
    stints, status = lineups.reconstruct_all(con, season)
    con.register("stints_df", stints.drop(columns=["teammates_on"]))
    con.execute("CREATE OR REPLACE TABLE stints AS SELECT * FROM stints_df")
    box = con.execute("SELECT * FROM player_box WHERE game_id IN (SELECT DISTINCT game_id FROM pbp)").df()
    pg = features.player_games(stints, box)
    opp = features.opportunity_features(pg)
    splits = features.without_splits(stints, pg)
    absn = features.absence_features(pg, splits, box)
    v2 = opp.merge(absn, on=["game_id", "player_id"], how="left")

    logs = pd.read_csv("data/processed/player_logs.csv", low_memory=False)
    logs["game_id"] = logs["game_id"].astype(str).str.replace(r"\.0$", "", regex=True)
    logs["player_id"] = logs["player_id"].astype(str).str.replace(r"\.0$", "", regex=True)
    logs["game_date"] = pd.to_datetime(logs["game_date"])
    logs = logs[logs["game_id"].isin(set(pg["game_id"])) & (pd.to_numeric(logs["actual_minutes"], errors="coerce") > 0)]
    df = logs.merge(v2, on=["game_id", "player_id"], how="inner")
    v = status[status["valid"]]
    report = {
        "warehouse_checks": checks,
        "games_with_pbp": int(len(status)), "games_valid_lineups": int(status["valid"].sum()),
        "lineup_minutes_mae": float(v["minutes_mae"].mean()),
        "lineup_minutes_max_err_p95": float(v["minutes_max_err"].quantile(.95)),
        "player_games_v1": int(len(logs)), "player_games_joined": int(len(df)),
        "opportunity_coverage": float(df["reb_chances36"].notna().mean()),
        "player_games_with_absent_regular": int((df["n_absent_regulars"] > 0).sum()),
        "absence_feature_coverage_when_absent": float(
            df.loc[df["n_absent_regulars"] > 0, "abs_reb36_delta"].notna().mean()),
    }
    return df, report


def evaluate(df: pd.DataFrame, cutoff: str) -> dict:
    base = [f for f in BASE_FEATURES + ["def_adj", "usage_pace"] if f in df.columns]
    arms = {"v1": base, "v1+opportunity": base + OPP, "v1+absence": base + ABS, "v1+both": base + OPP + ABS}
    train, test = df[df["game_date"] < cutoff], df[df["game_date"] >= cutoff]
    out = {"train_rows": int(len(train)), "test_rows": int(len(test)), "cutoff": cutoff, "targets": {}}
    absent_test = test["n_absent_regulars"] > 0
    for target in TARGETS:
        y_tr, y_te = train[target], test[target]
        res = {}
        for name, feats in arms.items():
            feats = [f for f in feats if f != target]
            m = model().fit(train[feats], y_tr)
            p = m.predict(test[feats])
            res[name] = {"mae": float(mean_absolute_error(y_te, p)),
                         "mae_when_teammate_absent": float(mean_absolute_error(y_te[absent_test], p[absent_test.values]))}
        out["targets"][target] = res
    out["test_rows_with_absent_regular"] = int(absent_test.sum())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=2025)
    ap.add_argument("--cutoff", default="2026-02-01")
    ap.add_argument("--db", default=str(warehouse.DB_PATH))
    a = ap.parse_args()
    df, report = build_dataset(a.season, a.db)
    report.update(generated_at_utc=datetime.now(timezone.utc).isoformat(), season=a.season,
                  evaluation=evaluate(df, a.cutoff))
    out = Path(f"data/tracking/v2_pilot_{a.season}.json")
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
