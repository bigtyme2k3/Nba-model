"""
build_team_features.py
----------------------
Builds latest NBA team context from real team box scores.

The output replaces seeded daily-runner ratings with prior completed-game evidence.
It intentionally uses only completed historical box scores and is safe to load
before a future game's prediction.

Outputs:
  data/processed/team_profiles_latest.csv
  data/processed/team_features_status.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd

from teams import TEAM_ABBR, normalize_team

RAW_DIR = "data/raw"
OUT_DIR = "data/processed"
ABBR_TO_TEAM = {abbr: team for team, abbr in TEAM_ABBR.items()}


def season_from_path(path):
    m = re.search(r"_(20\d{2})\.csv$", str(path))
    return int(m.group(1)) if m else None


def canonical_team(row):
    if "team_display_name" in row and pd.notna(row.get("team_display_name")):
        name = normalize_team(str(row.get("team_display_name")))
        if name in TEAM_ABBR:
            return name
    loc = str(row.get("team_location", "") or "").strip()
    name = str(row.get("team_name", "") or "").strip()
    combined = normalize_team(f"{loc} {name}".strip())
    if combined in TEAM_ABBR:
        return combined
    abbr = str(row.get("team_abbreviation", row.get("team_abbr", "")) or "").upper()
    return ABBR_TO_TEAM.get(abbr, normalize_team(name))


def num(df, candidates):
    for c in candidates:
        if c in df.columns:
            return pd.to_numeric(df[c], errors="coerce")
    return pd.Series(np.nan, index=df.index, dtype=float)


def load_team_boxes(raw_dir):
    frames = []
    for path in sorted(glob(os.path.join(raw_dir, "hoopr_team_box_20*.csv"))):
        try:
            df = pd.read_csv(path, low_memory=False)
        except Exception:
            continue
        if df.empty or "game_id" not in df.columns:
            continue
        season = season_from_path(path)
        if "season" not in df.columns and season is not None:
            df["season"] = season
        df["game_id"] = df["game_id"].astype(str)
        df["game_date"] = pd.to_datetime(df.get("game_date"), errors="coerce")
        df["team"] = df.apply(canonical_team, axis=1)
        df["points"] = num(df, ["team_score", "points", "pts"])
        df["opp_points"] = num(df, ["opponent_team_score", "opp_points"])
        df["fga"] = num(df, ["field_goals_attempted", "fga", "fg_att"])
        df["fta"] = num(df, ["free_throws_attempted", "fta", "ft_att"])
        df["oreb"] = num(df, ["offensive_rebounds", "oreb"])
        df["tov"] = num(df, ["turnovers", "tov"])
        df["poss"] = df["fga"] - df["oreb"] + df["tov"] + 0.44 * df["fta"]
        df.loc[df["poss"] <= 0, "poss"] = np.nan
        df["ortg"] = 100 * df["points"] / df["poss"]
        df["drtg"] = 100 * df["opp_points"] / df["poss"]
        df["net_rtg"] = df["ortg"] - df["drtg"]
        denom = 2 * (df["fga"] + 0.44 * df["fta"])
        df["ts_pct"] = np.where(denom > 0, df["points"] / denom, np.nan)
        df["margin"] = df["points"] - df["opp_points"]
        frames.append(df[[
            "game_id", "season", "game_date", "team", "points", "opp_points",
            "poss", "ortg", "drtg", "net_rtg", "ts_pct", "margin"
        ]])
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out = out[out["game_date"].notna() & out["team"].isin(TEAM_ABBR)].copy()
    return out.drop_duplicates(["game_id", "team"], keep="last")


def avg(frame, col):
    s = pd.to_numeric(frame[col], errors="coerce").dropna()
    return float(s.mean()) if not s.empty else np.nan


def build_profiles(df):
    rows = []
    for team, g in df.sort_values("game_date").groupby("team", sort=False):
        g = g.sort_values(["game_date", "game_id"])
        l5, l10 = g.tail(5), g.tail(10)
        latest = g.iloc[-1]
        rows.append({
            "team": team,
            "abbr": TEAM_ABBR.get(team, ""),
            "last_game_date": latest["game_date"].date().isoformat(),
            "games_history": int(len(g)),
            "pace": round(avg(l10, "poss"), 3),
            "ortg": round(avg(l10, "ortg"), 3),
            "drtg": round(avg(l10, "drtg"), 3),
            "net_rtg": round(avg(l10, "net_rtg"), 3),
            "ts_pct": round(avg(l10, "ts_pct"), 5),
            "roll5_net_rtg": round(avg(l5, "net_rtg"), 3),
            "roll10_net_rtg": round(avg(l10, "net_rtg"), 3),
            "roll5_points": round(avg(l5, "points"), 3),
            "roll10_points": round(avg(l10, "points"), 3),
            "roll5_allowed": round(avg(l5, "opp_points"), 3),
            "roll10_allowed": round(avg(l10, "opp_points"), 3),
            "source": "COMPLETED_TEAM_BOX_SCORES",
        })
    return pd.DataFrame(rows).sort_values("team").reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", default=RAW_DIR)
    parser.add_argument("--out", default=OUT_DIR)
    args = parser.parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)

    boxes = load_team_boxes(args.raw)
    if boxes.empty:
        status = {"status": "WAITING_FOR_TEAM_BOX_SCORES", "rows": 0, "teams": 0}
        Path(args.out, "team_features_status.json").write_text(json.dumps(status, indent=2) + "\n")
        print(json.dumps(status, indent=2))
        return

    profiles = build_profiles(boxes)
    out = Path(args.out, "team_profiles_latest.csv")
    profiles.to_csv(out, index=False)
    seasons = sorted(int(x) for x in pd.to_numeric(boxes["season"], errors="coerce").dropna().unique())
    status = {
        "status": "READY",
        "source_rows": int(len(boxes)),
        "teams": int(profiles["team"].nunique()),
        "seasons": seasons,
        "min_game_date": boxes["game_date"].min().date().isoformat(),
        "max_game_date": boxes["game_date"].max().date().isoformat(),
        "output": str(out),
        "policy": "COMPLETED_GAMES_ONLY",
    }
    Path(args.out, "team_features_status.json").write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
