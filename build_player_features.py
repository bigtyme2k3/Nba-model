""" 
build_player_features.py
------------------------
Builds the leakage-safe NBA player feature store used by the props and rotation layers.

Inputs
------
data/raw/hoopr_player_box_*.csv
data/raw/hoopr_games_*.csv
data/raw/hoopr_team_box_*.csv (optional, improves pace/defense context)

Outputs
-------
data/processed/player_logs.csv
data/processed/player_profiles_latest.csv
data/processed/rotation_profiles_latest.csv
data/processed/player_features_status.json

Policy
------
Training rows use only information available strictly before each target game.
Actual target-game minutes/usage/efficiency are kept in actual_* columns and are
never copied into prediction features.
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

PLAYER_RENAME = {
    "athlete_display_name": "player",
    "player_name": "player",
    "athlete_id": "player_id",
    "athlete_position_abbreviation": "position",
    "team_abbreviation": "team_abbr",
    "points": "pts",
    "rebounds": "reb",
    "assists": "ast",
    "three_point_field_goals_made": "threes",
    "threes_made": "threes",
    "field_goals_attempted": "fga",
    "fg_att": "fga",
    "free_throws_attempted": "fta",
    "ft_att": "fta",
    "turnovers": "tov",
    "tov": "tov",
}

TEAM_RENAME = {
    "team_home_away": "home_away",
    "team_score": "team_points",
    "points": "team_points",
    "pts": "team_points",
    "opponent_team_score": "opp_points",
    "field_goals_attempted": "team_fga",
    "free_throws_attempted": "team_fta",
    "offensive_rebounds": "team_oreb",
    "turnovers": "team_tov",
}

ABBR_TO_TEAM = {abbr: team for team, abbr in TEAM_ABBR.items()}


def coalesce_team_aliases(df: pd.DataFrame) -> pd.DataFrame:
    """Avoid duplicate canonical columns when old/new collectors emit multiple aliases."""
    out = df.copy()
    targets = dict.fromkeys(TEAM_RENAME.values())
    for target in targets:
        sources = ([target] if target in df.columns else []) + [
            source for source, dest in TEAM_RENAME.items()
            if dest == target and source in df.columns and source != target
        ]
        if not sources:
            continue
        merged = df[sources[0]].copy()
        for source in sources[1:]:
            merged = merged.combine_first(df[source])
        out = out.drop(columns=[source for source in sources if source != target])
        out[target] = merged
    return out


def season_from_path(path: str) -> int | None:
    m = re.search(r"_(20\d{2})\.csv$", str(path))
    return int(m.group(1)) if m else None


def team_name_from_row(row) -> str:
    raw_name = str(row.get("team_name", "") or "").strip()
    location = str(row.get("team_location", "") or "").strip()
    abbr = str(row.get("team_abbr", row.get("team_abbreviation", "")) or "").strip().upper()

    candidates = []
    if raw_name:
        candidates.append(raw_name)
    if location and raw_name:
        candidates.insert(0, f"{location} {raw_name}".strip())
    if abbr in ABBR_TO_TEAM:
        candidates.append(ABBR_TO_TEAM[abbr])

    for value in candidates:
        mapped = normalize_team(value)
        if mapped in TEAM_ABBR:
            return mapped
    return candidates[-1] if candidates else ""


def load_games(raw_dir: str) -> pd.DataFrame:
    frames = []
    for path in sorted(glob(os.path.join(raw_dir, "hoopr_games_20*.csv"))):
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if "game_id" not in df.columns or "game_date" not in df.columns:
            continue
        df["game_id"] = df["game_id"].astype(str)
        df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
        for col in ("home_team", "away_team"):
            if col in df.columns:
                df[col] = df[col].map(normalize_team)
        season = season_from_path(path)
        if "season" not in df.columns and season is not None:
            df["season"] = season
        frames.append(df[[c for c in [
            "game_id", "season", "game_date", "home_team", "away_team",
            "home_pts", "away_pts"
        ] if c in df.columns]])
    if not frames:
        return pd.DataFrame(columns=["game_id", "season", "game_date", "home_team", "away_team"])
    out = pd.concat(frames, ignore_index=True)
    return out.drop_duplicates(subset=["game_id"], keep="last")


def build_team_schedule(games: pd.DataFrame) -> pd.DataFrame:
    if games.empty:
        return pd.DataFrame(columns=["game_id", "team", "opp_team", "is_home", "rest_days"])
    home = games[["game_id", "game_date", "home_team", "away_team"]].copy()
    home.columns = ["game_id", "game_date", "team", "opp_team"]
    home["is_home"] = 1
    away = games[["game_id", "game_date", "away_team", "home_team"]].copy()
    away.columns = ["game_id", "game_date", "team", "opp_team"]
    away["is_home"] = 0
    sched = pd.concat([home, away], ignore_index=True)
    sched = sched.sort_values(["team", "game_date", "game_id"]).reset_index(drop=True)
    prev = sched.groupby("team")["game_date"].shift(1)
    sched["rest_days"] = ((sched["game_date"] - prev).dt.days - 1).clip(lower=0, upper=7)
    return sched


def load_team_context(raw_dir: str, games: pd.DataFrame) -> pd.DataFrame:
    frames = []
    game_map = games.set_index("game_id") if not games.empty else pd.DataFrame()
    for path in sorted(glob(os.path.join(raw_dir, "hoopr_team_box_20*.csv"))):
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if "game_id" not in df.columns:
            continue
        df = coalesce_team_aliases(df)
        df["game_id"] = df["game_id"].astype(str)
        if "game_date" in df.columns:
            df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
        else:
            date_map = game_map["game_date"].to_dict() if not game_map.empty and "game_date" in game_map else {}
            df["game_date"] = df["game_id"].map(date_map)

        if "team_display_name" in df.columns:
            df["team"] = df["team_display_name"].astype(str).map(normalize_team)
        else:
            df["team"] = df.apply(team_name_from_row, axis=1)

        if "opp_points" not in df.columns:
            df["opp_points"] = np.nan
        if not games.empty and {"home_pts", "away_pts"}.issubset(games.columns):
            gm = games.set_index("game_id")
            ha = df.get("home_away", pd.Series("", index=df.index)).astype(str).str.lower()
            home_pts = df["game_id"].map(gm["home_pts"].to_dict())
            away_pts = df["game_id"].map(gm["away_pts"].to_dict())
            inferred_opp = pd.Series(np.where(ha.eq("home"), away_pts, home_pts), index=df.index)
            df["opp_points"] = pd.to_numeric(df["opp_points"], errors="coerce").fillna(inferred_opp)

        for c in ["team_fga", "team_fta", "team_oreb", "team_tov", "team_points", "opp_points"]:
            if c not in df.columns:
                df[c] = np.nan
            df[c] = pd.to_numeric(df[c], errors="coerce")

        df["poss_actual"] = df["team_fga"] - df["team_oreb"] + df["team_tov"] + 0.44 * df["team_fta"]
        df.loc[df["poss_actual"] <= 0, "poss_actual"] = np.nan
        df["drtg_actual"] = 100.0 * df["opp_points"] / df["poss_actual"]
        df["ortg_actual"] = 100.0 * df["team_points"] / df["poss_actual"]
        frames.append(df[["game_id", "game_date", "team", "poss_actual", "drtg_actual", "ortg_actual"]])

    if not frames:
        return pd.DataFrame(columns=[
            "game_id", "team", "team_pace", "team_ortg_prior", "team_drtg_prior"
        ])

    ctx = pd.concat(frames, ignore_index=True)
    ctx = ctx.drop_duplicates(subset=["game_id", "team"], keep="last")
    ctx = ctx.sort_values(["team", "game_date", "game_id"]).reset_index(drop=True)

    for actual, prior in [
        ("poss_actual", "team_pace"),
        ("ortg_actual", "team_ortg_prior"),
        ("drtg_actual", "team_drtg_prior"),
    ]:
        ctx[prior] = ctx.groupby("team")[actual].transform(
            lambda s: s.shift(1).rolling(10, min_periods=3).mean()
        )
    return ctx


def load_players(raw_dir: str) -> pd.DataFrame:
    frames = []
    for path in sorted(glob(os.path.join(raw_dir, "hoopr_player_box_20*.csv"))):
        try:
            df = pd.read_csv(path, low_memory=False)
        except Exception:
            continue
        if df.empty or "game_id" not in df.columns:
            continue
        df = df.rename(columns={k: v for k, v in PLAYER_RENAME.items() if k in df.columns})
        season = season_from_path(path)
        if "season" not in df.columns and season is not None:
            df["season"] = season
        df["game_id"] = df["game_id"].astype(str)
        df["game_date"] = pd.to_datetime(df.get("game_date"), errors="coerce")
        if "player" not in df.columns:
            continue
        df["player"] = df["player"].astype(str).str.strip()
        df["player_id"] = df.get("player_id", pd.Series("", index=df.index)).astype(str)
        df["position"] = df.get("position", pd.Series("", index=df.index)).astype(str).str.upper().str.strip()
        df["team"] = df.apply(team_name_from_row, axis=1)
        df["starter"] = df.get("starter", pd.Series(False, index=df.index)).astype(str).str.lower().isin({"true", "1", "yes"})
        df["did_not_play"] = df.get("did_not_play", pd.Series(False, index=df.index)).astype(str).str.lower().isin({"true", "1", "yes"})

        for c in ["minutes", "pts", "reb", "ast", "threes", "fga", "fta", "tov"]:
            if c not in df.columns:
                df[c] = np.nan
            df[c] = pd.to_numeric(df[c], errors="coerce")

        df = df[df["player"].ne("") & df["game_date"].notna()].copy()
        df = df[~df["did_not_play"] & df["minutes"].fillna(0).gt(0)].copy()
        df["pra"] = df["pts"].fillna(0) + df["reb"].fillna(0) + df["ast"].fillna(0)
        df["usage_events"] = df["fga"].fillna(0) + 0.44 * df["fta"].fillna(0) + df["tov"].fillna(0)
        denom = 2.0 * (df["fga"].fillna(0) + 0.44 * df["fta"].fillna(0))
        df["ts_pct_actual"] = np.where(denom > 0, df["pts"].fillna(0) / denom, np.nan)
        frames.append(df)

    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    return out.drop_duplicates(subset=["game_id", "player_id", "player"], keep="last")


def add_usage(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    grp = df.groupby(["game_id", "team"], dropna=False)
    team_events = grp["usage_events"].transform("sum")
    team_minutes = grp["minutes"].transform("sum")
    numerator = df["usage_events"] * (team_minutes / 5.0)
    denominator = df["minutes"] * team_events
    df["usage_actual"] = np.where(denominator > 0, numerator / denominator, np.nan)
    df.loc[(df["usage_actual"] < 0) | (df["usage_actual"] > 0.8), "usage_actual"] = np.nan
    return df


def add_prior_player_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["player", "game_date", "game_id"]).reset_index(drop=True)
    metrics = ["minutes", "pts", "reb", "ast", "threes", "pra", "usage_actual", "ts_pct_actual"]

    for metric in metrics:
        for window in (5, 10):
            name = metric.replace("_actual", "")
            df[f"roll{window}_{name}"] = df.groupby("player")[metric].transform(
                lambda s, w=window: s.shift(1).rolling(w, min_periods=2).mean()
            )

    df["minutes_sd_l10"] = df.groupby("player")["minutes"].transform(
        lambda s: s.shift(1).rolling(10, min_periods=3).std(ddof=0)
    )
    df["_starter_num"] = df["starter"].astype(float)
    for window in (5, 10):
        df[f"starter_rate_l{window}"] = df.groupby("player")["_starter_num"].transform(
            lambda s, w=window: s.shift(1).rolling(w, min_periods=2).mean()
        )
    df = df.drop(columns=["_starter_num"])

    # Compatibility columns consumed by props_model.py. These are prior-only.
    df["actual_minutes"] = df["minutes"]
    df["minutes"] = df["roll5_minutes"]
    df["usage"] = df["roll10_usage"].fillna(df["roll5_usage"])
    df["ts_pct"] = df["roll10_ts_pct"].fillna(df["roll5_ts_pct"])
    return df


def enrich_context(players: pd.DataFrame, games: pd.DataFrame, raw_dir: str) -> pd.DataFrame:
    if players.empty:
        return players
    sched = build_team_schedule(games)
    if not sched.empty:
        players = players.merge(
            sched[["game_id", "team", "opp_team", "is_home", "rest_days"]],
            on=["game_id", "team"], how="left"
        )
    else:
        players["opp_team"] = ""
        players["is_home"] = np.nan
        players["rest_days"] = np.nan

    team_ctx = load_team_context(raw_dir, games)
    if not team_ctx.empty:
        players = players.merge(
            team_ctx[["game_id", "team", "team_pace", "team_ortg_prior", "team_drtg_prior"]],
            on=["game_id", "team"], how="left"
        )
        opp_ctx = team_ctx[["game_id", "team", "team_drtg_prior"]].rename(
            columns={"team": "opp_team", "team_drtg_prior": "opp_drtg_pos"}
        )
        players = players.merge(opp_ctx, on=["game_id", "opp_team"], how="left")
    else:
        players["team_pace"] = np.nan
        players["opp_drtg_pos"] = np.nan

    players["team_pace"] = pd.to_numeric(players.get("team_pace"), errors="coerce").fillna(99.0)
    players["opp_drtg_pos"] = pd.to_numeric(players.get("opp_drtg_pos"), errors="coerce").fillna(114.0)
    players["avg_pace"] = players["team_pace"]
    players["rest_days"] = pd.to_numeric(players.get("rest_days"), errors="coerce").fillna(2).clip(0, 7)
    players["is_home"] = pd.to_numeric(players.get("is_home"), errors="coerce").fillna(0).astype(int)
    players["season_game_num"] = players.groupby(["player", "season"]).cumcount() + 1
    players["month"] = players["game_date"].dt.month
    return players


def percentile(values, pct):
    vals = [float(x) for x in values if pd.notna(x)]
    return float(np.quantile(vals, pct)) if vals else np.nan


def build_profiles(actual: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if actual.empty:
        return pd.DataFrame(), pd.DataFrame()

    actual = actual.sort_values(["player", "game_date", "game_id"]).copy()
    max_date = actual["game_date"].max()
    cutoff = max_date - pd.Timedelta(days=240)
    rows = []
    rot_rows = []

    for player, g in actual.groupby("player", sort=False):
        g = g.sort_values(["game_date", "game_id"])
        latest = g.iloc[-1]
        if latest["game_date"] < cutoff:
            continue
        l10 = g.tail(10)
        l5 = g.tail(5)
        mins10 = l10["actual_minutes"].dropna()
        mins5 = l5["actual_minutes"].dropna()
        avg10 = float(mins10.mean()) if not mins10.empty else np.nan
        avg5 = float(mins5.mean()) if not mins5.empty else avg10
        sd10 = float(mins10.std(ddof=0)) if len(mins10) >= 2 else 0.0
        cv = sd10 / avg10 if pd.notna(avg10) and avg10 > 0 else np.nan
        start_rate = float(l10["starter"].astype(float).mean()) if len(l10) else np.nan
        consistency = max(start_rate, 1.0 - start_rate) if pd.notna(start_rate) else 0.5
        sample = min(1.0, len(g) / 10.0)
        volatility = max(0.0, min(1.0, 1.0 - (cv if pd.notna(cv) else 1.0)))
        stability = round(100 * (0.45 * sample + 0.35 * volatility + 0.20 * consistency), 2)
        risk = "STABLE" if stability >= 80 else ("MODERATE" if stability >= 60 else "VOLATILE")
        role = "STARTER" if start_rate >= 0.60 else ("BENCH" if start_rate <= 0.35 else "MIXED")
        projected_minutes = np.nan
        if pd.notna(avg5) and pd.notna(avg10):
            projected_minutes = max(0.0, min(42.0, 0.65 * avg5 + 0.35 * avg10))

        def mean_col(frame, col, default=np.nan):
            s = pd.to_numeric(frame[col], errors="coerce").dropna()
            return float(s.mean()) if not s.empty else default

        row = {
            "player": player,
            "player_id": latest.get("player_id", ""),
            "team": latest.get("team", ""),
            "pos": latest.get("position", ""),
            "last_game_date": latest["game_date"].date().isoformat(),
            "games_history": int(len(g)),
            "projected_minutes": round(projected_minutes, 3) if pd.notna(projected_minutes) else np.nan,
            "mpg": round(avg10, 3) if pd.notna(avg10) else np.nan,
            "usage": round(mean_col(l10, "usage_actual"), 5),
            "ts": round(mean_col(l10, "ts_pct_actual"), 5),
            "roll5_pts": round(mean_col(l5, "pts"), 3),
            "roll5_reb": round(mean_col(l5, "reb"), 3),
            "roll5_ast": round(mean_col(l5, "ast"), 3),
            "roll5_threes": round(mean_col(l5, "threes"), 3),
            "roll5_pra": round(mean_col(l5, "pra"), 3),
            "roll10_pts": round(mean_col(l10, "pts"), 3),
            "roll10_reb": round(mean_col(l10, "reb"), 3),
            "roll10_ast": round(mean_col(l10, "ast"), 3),
            "roll10_threes": round(mean_col(l10, "threes"), 3),
            "starter_rate_l10": round(start_rate, 4) if pd.notna(start_rate) else np.nan,
            "minutes_sd_l10": round(sd10, 3),
            "minutes_floor_p20": round(percentile(mins10, 0.20), 3) if len(mins10) else np.nan,
            "minutes_ceiling_p80": round(percentile(mins10, 0.80), 3) if len(mins10) else np.nan,
            "rotation_stability_score": stability,
            "rotation_risk_band": risk,
            "projected_role": role,
            "profile_source": "PRIOR_BOX_SCORES",
        }
        rows.append(row)
        rot_rows.append({
            k: row[k] for k in [
                "player", "player_id", "team", "pos", "last_game_date",
                "projected_minutes", "starter_rate_l10", "minutes_sd_l10",
                "minutes_floor_p20", "minutes_ceiling_p80",
                "rotation_stability_score", "rotation_risk_band",
                "projected_role", "profile_source"
            ]
        })

    return pd.DataFrame(rows), pd.DataFrame(rot_rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", default=RAW_DIR)
    parser.add_argument("--out", default=OUT_DIR)
    args = parser.parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)

    games = load_games(args.raw)
    players = load_players(args.raw)
    if players.empty:
        status = {
            "status": "WAITING_FOR_PLAYER_BOX_SCORES",
            "player_box_files": len(glob(os.path.join(args.raw, "hoopr_player_box_20*.csv"))),
            "rows": 0,
        }
        Path(args.out, "player_features_status.json").write_text(json.dumps(status, indent=2) + "\n")
        print(json.dumps(status, indent=2))
        return

    players = add_usage(players)
    players = enrich_context(players, games, args.raw)

    actual = players.copy().rename(columns={"minutes": "actual_minutes"})
    actual["minutes"] = actual["actual_minutes"]
    featured = add_prior_player_features(players.copy())

    featured = featured.sort_values(["game_date", "game_id", "player"]).reset_index(drop=True)
    featured["game_date"] = featured["game_date"].dt.date.astype(str)
    logs_path = Path(args.out, "player_logs.csv")
    featured.to_csv(logs_path, index=False)

    actual["game_date"] = pd.to_datetime(actual["game_date"], errors="coerce")
    profiles, rotations = build_profiles(actual)
    profiles_path = Path(args.out, "player_profiles_latest.csv")
    rotation_path = Path(args.out, "rotation_profiles_latest.csv")
    profiles.to_csv(profiles_path, index=False)
    rotations.to_csv(rotation_path, index=False)

    seasons = sorted(int(x) for x in pd.to_numeric(featured["season"], errors="coerce").dropna().unique())
    status = {
        "status": "READY",
        "rows": int(len(featured)),
        "players": int(featured["player"].nunique()),
        "seasons": seasons,
        "min_game_date": str(featured["game_date"].min()),
        "max_game_date": str(featured["game_date"].max()),
        "profile_rows": int(len(profiles)),
        "rotation_rows": int(len(rotations)),
        "leakage_policy": "ROLLING_FEATURES_SHIFTED_ONE_GAME",
        "actual_minutes_separate": True,
        "outputs": [str(logs_path), str(profiles_path), str(rotation_path)],
    }
    Path(args.out, "player_features_status.json").write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
