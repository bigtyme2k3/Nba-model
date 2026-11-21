"""Paired chronological ablation of box-score matchup groups; never publishes.

Choose a variant on older-date CV only, evaluate that frozen choice against a
corrected baseline on the last completed NBA season, then report newer games
separately as forward research. All variants use the same eligible game rows.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from spread_model import BASE_SPREAD_FEATURES, engineer_features
from totals_model import BASE_TOTALS_FEATURES, engineer_totals_features
from team_box_matchups import MATCHUP_GROUPS, MATCHUP_FEATURES, VERSION, whole_date_splits


def ridge():
    return Pipeline([("scale", StandardScaler()), ("ridge", Ridge(alpha=10.))])


def market_diagnostics(frame, predictions, kind):
    candidates = ("avg_spread_home_line", "spread_home") if kind == "spread" else ("avg_total_point", "total")
    column = next((col for col in candidates if col in frame and pd.to_numeric(frame[col], errors="coerce").notna().any()), None)
    if column is None:
        return {"status": "NO_RECORDED_MARKET_LINES", "priced_betting_edge_validated": False}
    line = pd.to_numeric(frame[column], errors="coerce").to_numpy()
    if kind == "spread":
        line = -line  # Raw sportsbook home handicap -> home-score margin.
    actual = frame["actual_spread" if kind == "spread" else "actual_total"].to_numpy()
    valid = np.isfinite(line)
    decisions = valid & (predictions != line) & (actual != line)
    wins = int(((predictions - line) * (actual - line) > 0)[decisions].sum())
    return {"status": "UNVERIFIED_QUOTE_LINE_DIAGNOSTIC", "line_column": column, "line_rows": int(valid.sum()),
            "decisions": int(decisions.sum()), "wins": wins,
            "pushes": int((valid & (actual == line)).sum()),
            "hit_rate": wins / int(decisions.sum()) if decisions.any() else None,
            "pregame_quote_timing_validated": False, "priced_betting_edge_validated": False}


def compare_period(train, test, baseline, extra, target, kind):
    if test.empty:
        return {"rows": 0}
    base = ridge().fit(train[baseline].fillna(0), train[target])
    full = ridge().fit(train[baseline + extra].fillna(0), train[target])
    pb = base.predict(test[baseline].fillna(0))
    pf = full.predict(test[baseline + extra].fillna(0))
    eb = np.abs(test[target].to_numpy() - pb)
    ef = np.abs(test[target].to_numpy() - pf)
    differences = pd.DataFrame({"date": test["game_date"].to_numpy(), "delta": ef - eb})
    daily = differences.groupby("date")["delta"].agg(["sum", "count"])
    rng = np.random.default_rng(42)
    samples = rng.integers(0, len(daily), size=(1000, len(daily)))
    changes = daily["sum"].to_numpy()[samples].sum(axis=1) / daily["count"].to_numpy()[samples].sum(axis=1)
    result = {"rows": len(test), "from": test["game_date"].min().date().isoformat(),
              "through": test["game_date"].max().date().isoformat(),
              "baseline_mae": float(eb.mean()), "candidate_mae": float(ef.mean()),
              "mae_delta_candidate_minus_baseline": float((ef - eb).mean()),
              "paired_date_bootstrap_95_ci": [float(x) for x in np.quantile(changes, [.025, .975])],
              "baseline_market": market_diagnostics(test, pb, kind),
              "candidate_market": market_diagnostics(test, pf, kind)}
    if kind == "spread":
        actual = test[target].to_numpy()
        result.update(baseline_winner_accuracy=float(((pb > 0) == (actual > 0)).mean()),
                      candidate_winner_accuracy=float(((pf > 0) == (actual > 0)).mean()))
    return result


def evaluate(frame, kind, holdout_season):
    target = "actual_spread" if kind == "spread" else "actual_total"
    original = len(frame)
    df = (engineer_features(frame.copy()) if kind == "spread" else engineer_totals_features(frame.copy()))
    missing = set(MATCHUP_FEATURES) - set(df.columns)
    if missing:
        raise ValueError(f"Rebuild team features and master first: {sorted(missing)}")
    valid = df[list(MATCHUP_FEATURES)].notna().all(axis=1) & df[target].notna()
    if "team_context_complete" in df:
        valid &= pd.to_numeric(df["team_context_complete"], errors="coerce").eq(1)
    df = df.loc[valid].sort_values(["game_date", "game_id"]).reset_index(drop=True)
    baseline = [f for f in (BASE_SPREAD_FEATURES if kind == "spread" else BASE_TOTALS_FEATURES) if f in df]
    development = df[df["season"] < holdout_season].reset_index(drop=True)
    holdout = df[df["season"] == holdout_season].reset_index(drop=True)
    forward = df[df["season"] > holdout_season].reset_index(drop=True)
    if len(development) < 500 or holdout.empty:
        raise ValueError("Need at least 500 development games and a separate holdout season")
    splits = list(whole_date_splits(development["game_date"]))
    variants = {"baseline": [], **{k: list(v) for k, v in MATCHUP_GROUPS.items()}, "all_box_matchups": list(MATCHUP_FEATURES)}
    cv = {}
    for name, extra in variants.items():
        features = baseline + extra
        folds = []
        for train, test in splits:
            model = ridge().fit(development.iloc[train][features].fillna(0), development.iloc[train][target])
            pred = model.predict(development.iloc[test][features].fillna(0))
            folds.append({"train_through": development.iloc[train]["game_date"].max().date().isoformat(),
                          "test_from": development.iloc[test]["game_date"].min().date().isoformat(),
                          "test_through": development.iloc[test]["game_date"].max().date().isoformat(),
                          "test_rows": len(test), "mae": float(np.abs(development.iloc[test][target].to_numpy() - pred).mean())})
        cv[name] = {"features_added": extra, "folds": folds,
                    "mae": sum(f["mae"] * f["test_rows"] for f in folds) / sum(f["test_rows"] for f in folds)}
    selected = min(cv, key=lambda name: cv[name]["mae"])
    extra = variants[selected]
    held = compare_period(development, holdout, baseline, extra, target, kind)
    # The choice is frozen before reading the holdout or forward result. No
    # search on either period; forward fits use the completed holdout as history.
    later = compare_period(pd.concat([development, holdout]), forward, baseline, extra, target, kind)
    return {"source_rows": original, "eligible_rows": len(df), "excluded_rows": original - len(df),
            "development_rows": len(development), "baseline_feature_count": len(baseline),
            "baseline_features": baseline,
            "availability_team_rows_nonzero": int(sum(df.get(f"{side}_vac_minutes", pd.Series(0, index=df.index)).gt(0).sum() for side in ("home", "away"))),
            "cv": cv, "selected_using_development_only": selected,
            "holdout": held, "forward": later,
            "supports_shadow_candidate": bool(extra and held["paired_date_bootstrap_95_ci"][1] < 0),
            "production_promotion_approved": False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/processed/master_all.csv")
    parser.add_argument("--holdout-season", type=int, required=True, help="NBA start year; e.g. 2025 means 2025-26")
    parser.add_argument("--out", default="data/tracking/game_box_matchups_ablation.json")
    args = parser.parse_args()
    frame = pd.read_csv(args.data, parse_dates=["game_date"], low_memory=False)
    if frame["game_id"].duplicated().any():
        raise ValueError("Ablation master has duplicate game IDs")
    report = {"generated_at_utc": datetime.now(timezone.utc).isoformat(), "feature_semantics_version": VERSION,
              "input_sha256": hashlib.sha256(Path(args.data).read_bytes()).hexdigest(),
              "policy": "DEVELOPMENT_DATE_GROUP_CV_THEN_FROZEN_SEASON_HOLDOUT; RESEARCH_ONLY_NO_PUBLICATION",
              "data_through": frame["game_date"].max().date().isoformat(),
              "holdout_season_start_year": args.holdout_season,
              "runtime": {"python": platform.python_version(), "sklearn": sklearn.__version__, "pandas": pd.__version__, "numpy": np.__version__},
              "spread": evaluate(frame, "spread", args.holdout_season),
              "totals": evaluate(frame, "totals", args.holdout_season)}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({kind: {key: report[kind][key] for key in
         ("eligible_rows", "selected_using_development_only", "holdout", "forward", "supports_shadow_candidate")}
         for kind in ("spread", "totals")}, indent=2))


if __name__ == "__main__":
    main()
