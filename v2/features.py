"""
v2/features.py
--------------
Pregame-only opportunity and teammate-absence features from reconstructed stints.

Every feature for a player's game uses only that player's earlier games of the
same season (window frames end one game before, cumulative sums are shifted).
Missing coverage stays NaN with an explicit flag; it never becomes a zero.

Opportunity (last 10 games played):
  reb_chances36, oreb_chances36, dreb_chances36   missed shots while on court, per 36
  reb_conv                                         rebounds / rebound chances
  ast_chances36, ast_conv                          teammates' made FG while on court; assists / chances
  fga_share                                        own FGA / team FGA while on court

Teammate absence (season to date, from stints):
  For each rotation regular absent from today's game (10-game avg >= 15 min, seen in
  the team's last 3 games), the player's per-36 rebounds, assists, FGA and points in
  minutes that teammate was off the court, minus his overall per-36 rate. Summed over
  absent regulars that have at least MIN_WITHOUT_SECONDS of prior split data.
  Absence here is box-score absence, the research stand-in for a confirmed OUT.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

MIN_WITHOUT_SECONDS = 3600      # one hour of prior minutes without the teammate
STATS = ["reb", "ast", "fga", "pts"]
COUNTERS = ["seconds", "oreb_chances", "dreb_chances", "ast_chances", "team_fga"] + STATS


def player_games(stints: pd.DataFrame, box: pd.DataFrame) -> pd.DataFrame:
    """Per player-game totals from stints, with date, team and season."""
    for c in COUNTERS:
        if c not in stints:
            stints[c] = 0.0
    stints[COUNTERS] = stints[COUNTERS].fillna(0.0)
    pg = stints.groupby(["game_id", "player_id"], as_index=False)[COUNTERS].sum()
    meta = box[["game_id", "athlete_id", "game_date", "team"]].rename(columns={"athlete_id": "player_id"})
    pg = pg.merge(meta, on=["game_id", "player_id"], how="left")
    pg["game_date"] = pd.to_datetime(pg["game_date"])
    pg["season"] = pg["game_date"].dt.year - (pg["game_date"].dt.month < 7).astype(int)
    return pg.sort_values(["player_id", "game_date", "game_id"]).reset_index(drop=True)


def opportunity_features(pg: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    g = pg.groupby(["player_id", "season"], sort=False)
    def prior(col):
        return g[col].transform(lambda s: s.shift(1).rolling(window, min_periods=1).sum())
    sec, orc, drc = prior("seconds"), prior("oreb_chances"), prior("dreb_chances")
    out = pg[["game_id", "player_id"]].copy()
    per36 = lambda x: np.where(sec > 0, x / sec * 2160, np.nan)
    out["reb_chances36"] = per36(orc + drc)
    out["oreb_chances36"] = per36(orc)
    out["dreb_chances36"] = per36(drc)
    out["reb_conv"] = np.where(orc + drc > 0, prior("reb") / (orc + drc), np.nan)
    ac = prior("ast_chances")
    out["ast_chances36"] = per36(ac)
    out["ast_conv"] = np.where(ac > 0, prior("ast") / ac, np.nan)
    tf = prior("team_fga")
    out["fga_share"] = np.where(tf > 0, prior("fga") / tf, np.nan)
    out["opp_games_prior"] = g["seconds"].transform(lambda s: s.shift(1).rolling(window, min_periods=1).count())
    return out


def without_splits(stints: pd.DataFrame, pg: pd.DataFrame) -> pd.DataFrame:
    """(game, player, teammate) counters for the player's stints with that teammate off court.

    Teammates are everyone who has played for the team this season up to this game,
    so games a teammate missed entirely count as 'without' him.
    """
    team_of = pg.set_index(["game_id", "player_id"])["team"].to_dict()
    date_of = pg.groupby("game_id")["game_date"].first().to_dict()
    season_of = pg.groupby("game_id")["season"].first().to_dict()
    roster_to_date = defaultdict(set)   # (team, season) -> players seen so far
    rows = defaultdict(lambda: np.zeros(len(COUNTERS)))
    for game_id in sorted(stints["game_id"].unique(), key=lambda g: date_of.get(g)):
        gs = stints[stints["game_id"] == game_id]
        players_by_team = defaultdict(set)
        for p in gs["player_id"].unique():
            players_by_team[team_of.get((game_id, p))].add(p)
        for team, players in players_by_team.items():
            roster_to_date[(team, season_of[game_id])] |= players
        for r in gs.itertuples(index=False):
            team = team_of.get((game_id, r.player_id))
            roster = roster_to_date[(team, season_of[game_id])]
            vals = np.array([getattr(r, c, 0.0) or 0.0 for c in COUNTERS], dtype=float)
            for t in roster - set(r.teammates_on) - {r.player_id}:
                rows[(game_id, r.player_id, t)] += vals
    out = pd.DataFrame([(g, p, t, *v) for (g, p, t), v in rows.items()],
                       columns=["game_id", "player_id", "teammate_id"] + [f"wo_{c}" for c in COUNTERS])
    return out


def absence_features(pg: pd.DataFrame, splits: pd.DataFrame, box: pd.DataFrame) -> pd.DataFrame:
    """Per player-game: summed per-36 deltas for today's absent rotation regulars."""
    # Prior (season-to-date, before this game) totals per player and per (player, teammate).
    pg = pg.copy()
    g = pg.groupby(["player_id", "season"], sort=False)
    for c in ["seconds"] + STATS:
        pg[f"cum_{c}"] = g[c].cumsum() - pg[c]
    sp = splits.merge(pg[["game_id", "player_id", "game_date", "season"]], on=["game_id", "player_id"])
    sp = sp.sort_values(["player_id", "teammate_id", "game_date"])
    gg = sp.groupby(["player_id", "teammate_id", "season"], sort=False)
    for c in ["seconds"] + STATS:
        sp[f"cum_wo_{c}"] = gg[f"wo_{c}"].cumsum() - sp[f"wo_{c}"]

    # Rotation regulars and who is absent from each game (box score, minutes > 0 = played).
    b = box[box["athlete_id"].notna()].copy()
    b["game_date"] = pd.to_datetime(b["game_date"])
    b["season"] = b["game_date"].dt.year - (b["game_date"].dt.month < 7).astype(int)
    played = b[b["minutes"] > 0].sort_values(["athlete_id", "game_date"])
    played["roll_min"] = played.groupby(["athlete_id", "season"])["minutes"].transform(
        lambda s: s.shift(1).rolling(10, min_periods=3).mean())
    team_games = b[["game_id", "team", "game_date", "season"]].drop_duplicates(["game_id", "team"])
    team_games = team_games.sort_values(["team", "game_date"])
    team_games["gidx"] = team_games.groupby(["team", "season"]).cumcount()
    played = played.merge(team_games[["game_id", "team", "gidx"]], on=["game_id", "team"])
    # A regular is absent from team game j if his latest appearance (k < j, within 3 games)
    # left him with a 10-game average of at least 15 minutes and he has no row at j.
    last = played[["athlete_id", "team", "season", "gidx", "game_date", "minutes"]].copy()
    absences = []
    for (team, season), tg in team_games.groupby(["team", "season"]):
        appearances = last[(last["team"] == team) & (last["season"] == season)]
        by_game = {j: set(x["athlete_id"]) for j, x in appearances.groupby("gidx")}
        roll = played[(played["team"] == team) & (played["season"] == season)]
        for row in tg.itertuples():
            j = row.gidx
            here = by_game.get(j, set())
            recent = roll[(roll["gidx"] < j) & (roll["gidx"] >= j - 3)]
            if recent.empty:
                continue
            latest = recent.sort_values("gidx").groupby("athlete_id").tail(1)
            # Regular status uses minutes up to and including his latest appearance.
            hist = roll[roll["gidx"] < j].sort_values("gidx").groupby("athlete_id")["minutes"].apply(
                lambda s: s.tail(10).mean() if len(s) >= 3 else np.nan)
            for t in latest["athlete_id"]:
                if t not in here and hist.get(t, 0) >= 15:
                    absences.append((row.game_id, team, t))
    absent = pd.DataFrame(absences, columns=["game_id", "team", "teammate_id"])

    x = pg[["game_id", "player_id", "team", "cum_seconds"] + [f"cum_{c}" for c in STATS]].merge(
        absent, on=["game_id", "team"])
    x = x.merge(sp[["game_id", "player_id", "teammate_id", "cum_wo_seconds"] + [f"cum_wo_{c}" for c in STATS]],
                on=["game_id", "player_id", "teammate_id"], how="left")
    ok = (x["cum_wo_seconds"] >= MIN_WITHOUT_SECONDS) & (x["cum_seconds"] > 0)
    for c in STATS:
        x[f"abs_{c}36_delta"] = np.where(
            ok, x[f"cum_wo_{c}"] / x["cum_wo_seconds"] * 2160 - x[f"cum_{c}"] / x["cum_seconds"] * 2160, np.nan)
    x["covered"] = ok
    agg = x.groupby(["game_id", "player_id"]).agg(
        n_absent_regulars=("teammate_id", "count"), absent_covered=("covered", "sum"),
        **{f"abs_{c}36_delta": (f"abs_{c}36_delta", lambda s: s.sum(min_count=1)) for c in STATS}).reset_index()
    out = pg[["game_id", "player_id"]].merge(agg, on=["game_id", "player_id"], how="left")
    out["n_absent_regulars"] = out["n_absent_regulars"].fillna(0)
    out["absent_covered"] = out["absent_covered"].fillna(0)
    # No absent regular is a real zero adjustment; an uncovered absence stays NaN.
    none_absent = out["n_absent_regulars"] == 0
    for c in STATS:
        out.loc[none_absent, f"abs_{c}36_delta"] = 0.0
    return out
