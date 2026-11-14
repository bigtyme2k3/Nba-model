"""
team_availability.py
--------------------
Who is missing from a team's rotation, as game-model features.

A player is a rotation regular for a team game when, as of his latest prior
appearance for that team, he averaged at least ROTATION_MIN_MINUTES over his last
10 games, and that appearance was within the team's last ABSENCE_LOOKBACK games
of the same season. A regular with no box-score row in the game was absent.

Per team-game:
  vac_minutes  sum of absent regulars' rolling minutes
  vac_value    sum of absent regulars' rolling box value
               (pts + 0.5*reb + ast + stl + blk - tov per game)
  vac_players  how many regulars were absent

Training uses actual absences from box scores. Live predictions use the same
rolling values for players the injury report lists as OUT, so the two sides
measure the same thing. Questionable/doubtful players are not counted live,
matching opportunity_redistribution.py.

All rolling values use only the player's games up to his latest prior
appearance, so nothing from the game being predicted leaks in.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from teams import normalize_team

PLAYER_LOGS = Path("data/processed/player_logs.csv")
ROTATION_MIN_MINUTES = 15.0
ABSENCE_LOOKBACK = 3
ROLL_GAMES = 10
STAT_COLS = ["minutes", "pts", "reb", "ast", "stl", "blk", "tov"]
VAC_COLS = ["vac_minutes", "vac_value", "vac_players"]


def norm_name(value) -> str:
    return " ".join(str(value or "").lower().replace(".", "").replace("'", "").split())


def load_player_logs(path=PLAYER_LOGS) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    cols = ["game_id", "game_date", "season", "team", "player"] + STAT_COLS
    df = pd.read_csv(path, low_memory=False, usecols=lambda c: c in cols)
    if df.empty or not {"game_id", "game_date", "team", "player"} <= set(df.columns):
        return pd.DataFrame()
    return prepare_logs(df)


def prepare_logs(df: pd.DataFrame) -> pd.DataFrame:
    """Add team game index and each player's rolling minutes/value as of each game."""
    df = df.copy()
    df["game_id"] = df["game_id"].astype(str).str.replace(r"\.0$", "", regex=True)
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    df = df[df["game_date"].notna() & df["team"].notna() & df["player"].notna()]
    for c in STAT_COLS:
        df[c] = pd.to_numeric(df.get(c), errors="coerce").fillna(0.0)
    if "season" not in df.columns:
        df["season"] = df["game_date"].dt.year + (df["game_date"].dt.month >= 8).astype(int)
    df["value"] = df["pts"] + 0.5 * df["reb"] + df["ast"] + df["stl"] + df["blk"] - df["tov"]

    games = (df[["game_id", "team", "game_date", "season"]]
             .drop_duplicates(["game_id", "team"])
             .sort_values(["team", "game_date", "game_id"]))
    games["gidx"] = games.groupby("team").cumcount()
    df = df.merge(games[["game_id", "team", "gidx"]], on=["game_id", "team"])

    df = df.sort_values(["player", "team", "game_date", "game_id"])
    grp = df.groupby(["player", "team"], sort=False)
    df["roll_minutes"] = grp["minutes"].transform(lambda s: s.rolling(ROLL_GAMES, min_periods=3).mean())
    df["roll_value"] = grp["value"].transform(lambda s: s.rolling(ROLL_GAMES, min_periods=3).mean())
    df.attrs["games"] = games
    return df


def historical_availability(logs: pd.DataFrame) -> pd.DataFrame:
    """One row per (game_id, team) that had at least one absent rotation regular."""
    if logs.empty:
        return pd.DataFrame(columns=["game_id", "team"] + VAC_COLS)
    games = logs.attrs["games"]
    regulars = logs[logs["roll_minutes"] >= ROTATION_MIN_MINUTES][
        ["player", "team", "season", "gidx", "roll_minutes", "roll_value"]]

    cands = []
    for lag in range(1, ABSENCE_LOOKBACK + 1):
        c = regulars.copy()
        c["target_gidx"] = c["gidx"] + lag
        c["lag"] = lag
        cands.append(c)
    # Keep each player's most recent appearance before the target game.
    cand = (pd.concat(cands).sort_values("lag")
            .drop_duplicates(["player", "team", "target_gidx"]))
    cand = cand.merge(
        games.rename(columns={"gidx": "target_gidx", "season": "target_season"}),
        on=["team", "target_gidx"])
    cand = cand[cand["season"] == cand["target_season"]]

    played = logs[["player", "team", "gidx"]].rename(columns={"gidx": "target_gidx"})
    played["played"] = True
    cand = cand.merge(played, on=["player", "team", "target_gidx"], how="left")
    absent = cand[cand["played"].isna()]

    return (absent.groupby(["game_id", "team"])
            .agg(vac_minutes=("roll_minutes", "sum"),
                 vac_value=("roll_value", "sum"),
                 vac_players=("player", "count"))
            .reset_index())


def live_availability(logs: pd.DataFrame, injuries: pd.DataFrame, teams) -> dict:
    """{team: {vac_minutes, vac_value, vac_players, players}} for players listed OUT.

    Each OUT player is credited to the team he last played for, only if he is a
    current rotation regular by the same rule as training.
    """
    out = {normalize_team(t): {"vac_minutes": 0.0, "vac_value": 0.0, "vac_players": 0, "players": []}
           for t in teams}
    if logs.empty or injuries is None or injuries.empty:
        return out
    sev = injuries.get("severity", injuries.get("status", pd.Series("", index=injuries.index)))
    is_out = sev.astype(str).str.upper().eq("OUT")
    if "is_out" in injuries.columns:
        is_out = is_out | injuries["is_out"].astype(str).str.lower().eq("true")
    out_names = {norm_name(n) for n in injuries.loc[is_out, "player"]}
    if not out_names:
        return out

    games = logs.attrs["games"]
    latest_team_game = games.groupby("team")["gidx"].max()
    latest_season = games.groupby("team")["season"].max()
    last = (logs.sort_values(["game_date", "game_id"])
            .groupby("player", sort=False).tail(1))
    for _, r in last.iterrows():
        if norm_name(r["player"]) not in out_names or r["team"] not in out:
            continue
        team = r["team"]
        recent = latest_team_game.get(team, -1) - r["gidx"] < ABSENCE_LOOKBACK
        same_season = r["season"] == latest_season.get(team)
        if recent and same_season and r["roll_minutes"] >= ROTATION_MIN_MINUTES:
            slot = out[team]
            slot["vac_minutes"] += float(r["roll_minutes"])
            slot["vac_value"] += float(r["roll_value"])
            slot["vac_players"] += 1
            slot["players"].append(r["player"])
    return out


def attach_to_games(game_df: pd.DataFrame, avail: pd.DataFrame) -> pd.DataFrame:
    """Add home_/away_ vac columns and diffs to a one-row-per-game frame (0 = nobody missing)."""
    df = game_df.copy()
    if "game_id" not in df.columns:
        return df
    df["game_id"] = df["game_id"].astype(str).str.replace(r"\.0$", "", regex=True)
    for side in ("home", "away"):
        a = avail.rename(columns={"team": f"{side}_team", **{c: f"{side}_{c}" for c in VAC_COLS}})
        df = df.merge(a, on=["game_id", f"{side}_team"], how="left")
        for c in VAC_COLS:
            df[f"{side}_{c}"] = df[f"{side}_{c}"].fillna(0.0)
    df["vac_value_diff"] = df["home_vac_value"] - df["away_vac_value"]
    return df
