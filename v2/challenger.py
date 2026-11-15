"""
v2/challenger.py
----------------
Train the V2 player-prop challenger: V1's prop features plus V2 opportunity and
teammate-absence features, same HistGBR settings as props_model.py, fit on every
season with play-by-play including the current one up to yesterday.

The bundle is used only for shadow predictions (daily_runner records v2_pred next
to V1's prediction); V1 makes every signal and bet.

Usage:
    python -m v2.challenger --seasons 2021 2022 2023 2024 2025 2026
Output:
    models/v2_props_models.pkl
    data/tracking/v2_challenger_training.json
"""
from __future__ import annotations

import argparse
import json
import pickle
from datetime import date, datetime, timezone
from pathlib import Path

from prop_features import BASE_FEATURES
from season_filters import nba_season_start_year
from v2 import pilot, warehouse

MODEL_PATH = Path("models/v2_props_models.pkl")
VERSION = "V2_PROPS_SHADOW_1"


def feature_list(target: str, columns) -> list[str]:
    base = [f for f in BASE_FEATURES + ["def_adj", "usage_pace"] if f in columns]
    return [f for f in base + pilot.OPP + pilot.ABS if f != target]


def main():
    ap = argparse.ArgumentParser()
    current = nba_season_start_year(str(date.today()))
    ap.add_argument("--seasons", type=int, nargs="+", default=list(range(2021, current + 1)))
    ap.add_argument("--db", default=str(warehouse.DB_PATH))
    a = ap.parse_args()
    df, report = pilot.build_dataset(a.seasons, a.db, refresh=[current])
    models, feats = {}, {}
    for target in pilot.TARGETS:
        feats[target] = feature_list(target, df.columns)
        models[target] = pilot.model().fit(df[feats[target]], df[target])
    bundle = {"version": VERSION, "models": models, "features": feats, "seasons": a.seasons,
              "trained_through": str(df["game_date"].max().date()), "rows": int(len(df)),
              "trained_at_utc": datetime.now(timezone.utc).isoformat()}
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with MODEL_PATH.open("wb") as fh:
        pickle.dump(bundle, fh)
    report.update({k: v for k, v in bundle.items() if k not in ("models",)})
    Path("data/tracking/v2_challenger_training.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(json.dumps({k: report[k] for k in ("version", "trained_through", "rows", "seasons")}, indent=2))


if __name__ == "__main__":
    main()
