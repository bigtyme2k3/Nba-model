"""
quarter_records.py
------------------
Each team's first-quarter record (wins, losses, ties) as game-model context.

Sources, by ESPN game id:
  data/raw/hoopr_schedule_*.csv   home_/away_linescores (completed past seasons)
  data/raw/scores_YYYY-MM-DD.csv  home_q1/away_q1 of STATUS_FINAL games (current season)

Per team-game, using only that team's earlier games of the same season:
  q1_wins, q1_losses, q1_ties   first-quarter record to date
  q1_win_pct                    (wins + 0.5 * ties) / games, 0.5 before any game
  q1_margin_l10                 average first-quarter margin over the last 10 games

live_q1_records() gives the same values for a team's next game (all completed
games counted), so training and live predictions see the same thing.

These are attached to master_all.csv and shown on each game, but they are not
SPREAD_FEATURES: in walk-forward tests (Ridge, 4 folds, 2021-22 to 2026-27) they
moved spread MAE from 11.115 to 11.114 and directional accuracy from 65.4% to
65.3%. A team's first-quarter record mostly restates team strength, which the
full-game rolling form already carries.
"""
from __future__ import annotations

import re
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd

from season_filters import nba_season_start_year
from teams import TEAM_ABBR, normalize_team

RAW_DIR = Path("data/raw")
Q1_COLS = ["q1_wins", "q1_losses", "q1_ties", "q1_win_pct", "q1_margin_l10"]
_VALUE = re.compile(r"'value':\s*(-?[\d.]+)")
_PERIOD = re.compile(r"'period':\s*(\d+)")


def first_quarter(linescores) -> float | None:
    """Q1 points from a hoopR linescores cell (a printed list of period dicts)."""
    if not isinstance(linescores, str):
        return None
    entries = re.findall(r"\{[^}]*\}", linescores)
    for i, entry in enumerate(entries):
        period = _PERIOD.search(entry)
        if (int(period.group(1)) if period else i + 1) == 1:
            value = _VALUE.search(entry)
            return float(value.group(1)) if value else None
    return None


def load_q1_games(raw_dir=RAW_DIR) -> pd.DataFrame:
    """One row per completed game: game_id, game_time, home/away team and Q1 points."""
    frames = []
    for path in sorted(glob(str(Path(raw_dir) / "hoopr_schedule_20*.csv"))):
        cols = ["game_id", "date", "game_date", "home_display_name", "away_display_name",
                "home_linescores", "away_linescores", "season_type"]
        s = pd.read_csv(path, low_memory=False, usecols=lambda c: c in cols)
        if not {"game_id", "home_linescores", "away_linescores"} <= set(s.columns):
            continue
        frames.append(pd.DataFrame({
            "game_id": s["game_id"],
            "game_time": pd.to_datetime(s.get("date", s.get("game_date")), errors="coerce", utc=True),
            "home_team": s["home_display_name"], "away_team": s["away_display_name"],
            "home_q1": s["home_linescores"].map(first_quarter),
            "away_q1": s["away_linescores"].map(first_quarter),
        }))
    for path in sorted(glob(str(Path(raw_dir) / "scores_20[0-9][0-9]-[0-9][0-9]-[0-9][0-9].csv"))):
        s = pd.read_csv(path)
        if s.empty or not {"home_q1", "away_q1", "status"} <= set(s.columns):
            continue
        s = s[s["status"].astype(str).str.contains("FINAL")]
        frames.append(pd.DataFrame({
            "game_id": s["game_id"],
            "game_time": pd.to_datetime(s["game_date"], errors="coerce", utc=True),
            "home_team": s["home_team"], "away_team": s["away_team"],
            "home_q1": pd.to_numeric(s["home_q1"], errors="coerce"),
            "away_q1": pd.to_numeric(s["away_q1"], errors="coerce"),
        }))
    if not frames:
        return pd.DataFrame(columns=["game_id", "game_time", "home_team", "away_team", "home_q1", "away_q1"])
    games = pd.concat(frames, ignore_index=True)
    games["game_id"] = games["game_id"].astype(str).str.replace(r"\.0$", "", regex=True)
    for side in ("home_team", "away_team"):
        games[side] = games[side].map(lambda t: normalize_team(str(t)))
    games = games.dropna(subset=["game_time", "home_q1", "away_q1"])
    games = games[games["home_team"].isin(TEAM_ABBR) & games["away_team"].isin(TEAM_ABBR)]
    # Later files win: a final score file supersedes an earlier copy of the same game.
    return games.drop_duplicates("game_id", keep="last").reset_index(drop=True)


def team_games(games: pd.DataFrame) -> pd.DataFrame:
    """One row per team per game with the team's Q1 margin, in time order."""
    home = games.assign(team=games["home_team"], q1_margin=games["home_q1"] - games["away_q1"])
    away = games.assign(team=games["away_team"], q1_margin=games["away_q1"] - games["home_q1"])
    tg = pd.concat([home, away])[["game_id", "game_time", "team", "q1_margin"]]
    tg["season"] = tg["game_time"].map(nba_season_start_year)
    return tg.sort_values(["team", "game_time", "game_id"]).reset_index(drop=True)


def _records(tg: pd.DataFrame, include_current: bool) -> pd.DataFrame:
    """Season-to-date Q1 record per team-game; prior games only unless include_current."""
    tg = tg.copy()
    grp = tg.groupby(["team", "season"], sort=False)
    shift = 0 if include_current else 1
    for name, mask in (("q1_wins", tg["q1_margin"] > 0), ("q1_losses", tg["q1_margin"] < 0),
                       ("q1_ties", tg["q1_margin"] == 0)):
        tg[name] = mask.astype(int).groupby([tg["team"], tg["season"]]).cumsum()
        if shift:
            tg[name] = tg[name] - mask.astype(int)
    played = tg["q1_wins"] + tg["q1_losses"] + tg["q1_ties"]
    tg["q1_win_pct"] = np.where(played > 0, (tg["q1_wins"] + 0.5 * tg["q1_ties"]) / played.clip(lower=1), 0.5)
    margin = grp["q1_margin"].shift(shift) if shift else tg["q1_margin"]
    tg["q1_margin_l10"] = (margin.groupby([tg["team"], tg["season"]])
                           .transform(lambda s: s.rolling(10, min_periods=1).mean())
                           .fillna(0.0))
    return tg


def historical_q1_records(games: pd.DataFrame) -> pd.DataFrame:
    """(game_id, team) -> each team's Q1 record before that game."""
    if games.empty:
        return pd.DataFrame(columns=["game_id", "team"] + Q1_COLS)
    return _records(team_games(games), include_current=False)[["game_id", "team"] + Q1_COLS]


def live_q1_records(games: pd.DataFrame, teams, season: int) -> dict:
    """{team: {q1 record entering its next game}} for this season; zeros before any game."""
    blank = {"q1_wins": 0, "q1_losses": 0, "q1_ties": 0, "q1_win_pct": 0.5, "q1_margin_l10": 0.0}
    out = {normalize_team(t): dict(blank) for t in teams}
    if games.empty:
        return out
    rec = _records(team_games(games), include_current=True)
    latest = rec[rec["season"] == season].groupby("team").tail(1).set_index("team")
    for team in out:
        if team in latest.index:
            r = latest.loc[team]
            out[team] = {"q1_wins": int(r["q1_wins"]), "q1_losses": int(r["q1_losses"]),
                         "q1_ties": int(r["q1_ties"]), "q1_win_pct": float(r["q1_win_pct"]),
                         "q1_margin_l10": float(r["q1_margin_l10"])}
    return out


def attach_to_games(game_df: pd.DataFrame, records: pd.DataFrame) -> pd.DataFrame:
    """Add home_/away_ Q1 columns and diffs to a one-row-per-game frame."""
    df = game_df.copy()
    if "game_id" not in df.columns:
        return df
    df["game_id"] = df["game_id"].astype(str).str.replace(r"\.0$", "", regex=True)
    defaults = {"q1_wins": 0, "q1_losses": 0, "q1_ties": 0, "q1_win_pct": 0.5, "q1_margin_l10": 0.0}
    for side in ("home", "away"):
        r = records.rename(columns={"team": f"{side}_team", **{c: f"{side}_{c}" for c in Q1_COLS}})
        df = df.merge(r, on=["game_id", f"{side}_team"], how="left")
        for c in Q1_COLS:
            df[f"{side}_{c}"] = df[f"{side}_{c}"].fillna(defaults[c])
    df["q1_win_pct_diff"] = df["home_q1_win_pct"] - df["away_q1_win_pct"]
    df["q1_margin_l10_diff"] = df["home_q1_margin_l10"] - df["away_q1_margin_l10"]
    return df
