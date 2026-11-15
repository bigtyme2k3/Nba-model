"""
v2/live.py
----------
V2 features for every player's next game, from this season's play-by-play and box
scores (republished daily) and today's injury report.

The features are the training features evaluated one game later: opportunity
rates over the player's last 10 completed games, and teammate-absence values for
rotation regulars listed OUT. Absence comes only from the injury report, and only
when daily_runner's injury-feed check passes; an unverified feed counts nobody out.

Usage:
    python -m v2.live --date 2026-11-14
Output:
    data/processed/v2_live_features.csv
"""
from __future__ import annotations

import argparse
import json
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from season_filters import nba_season_start_year
from teams import normalize_team
from v2 import features, lineups, warehouse

OUT_PATH = Path("data/processed/v2_live_features.csv")

# Training counts actual box-score absences; the morning report only catches some of
# them (OUT caught 62% of regulars who sat in 2026-27 through Nov 13). Live absence is
# therefore an expected count: each regular weighted by how often players with his
# status sat. Measured on 2,387 regular-games, Oct 20 - Nov 13, 2026: OUT 139/139,
# QUESTIONABLE 39/89, not listed 47/2,159. DOUBTFUL and PROBABLE did not occur; their
# weights are assumptions.
ABSENT_PROBABILITY = {"OUT": 1.0, "DOUBTFUL": 0.8, "QUESTIONABLE": 0.44, "PROBABLE": 0.05}
NOT_LISTED_PROBABILITY = 0.022
INJURY_PATH = Path("data/raw/injuries_today.csv")


def norm_name(value) -> str:
    return " ".join(str(value or "").lower().replace(".", "").replace("'", "").split())


def injury_statuses(injuries: pd.DataFrame) -> dict[str, str]:
    if injuries is None or injuries.empty:
        return {}
    sev = injuries.get("severity", injuries.get("status", pd.Series("", index=injuries.index)))
    return {norm_name(n): str(s).upper() for n, s in zip(injuries["player"], sev)}


def next_game_features(stints: pd.DataFrame, box: pd.DataFrame, target_date: str,
                       statuses: dict[str, str]) -> pd.DataFrame:
    pg = features.player_games(stints, box)
    pg["game_date"] = pd.to_datetime(pg["game_date"]).astype("datetime64[ns]")
    latest = pg.sort_values(["game_date", "game_id"]).groupby("player_id").tail(1)
    nxt = latest[["player_id", "team", "season"]].copy()
    nxt["game_id"] = "NEXT_" + nxt["team"].astype(str)
    nxt["game_date"] = pd.Series(pd.Timestamp(target_date), index=nxt.index).astype("datetime64[ns]")
    for c in features.COUNTERS:
        nxt[c] = 0.0
    opp = features.opportunity_features(pd.concat([pg, nxt], ignore_index=True)
                                        .sort_values(["player_id", "game_date", "game_id"]))
    opp = opp[opp["game_id"].str.startswith("NEXT_")].drop(columns="game_id")

    # Season-to-date totals and without-teammate totals over every completed game.
    totals = pg.groupby("player_id")[["seconds"] + features.STATS].sum()
    splits = features.without_splits(stints, pg)
    wo = splits.groupby(["player_id", "teammate_id"])[[f"wo_{c}" for c in ["seconds"] + features.STATS]].sum()

    # Rotation regulars for each team's next game: same rule as training.
    b = box[box["athlete_id"].notna() & (box["minutes"] > 0)].copy()
    b["game_date"] = pd.to_datetime(b["game_date"])
    team_games = b[["game_id", "team", "game_date"]].drop_duplicates(["game_id", "team"]).sort_values("game_date")
    team_games["gidx"] = team_games.groupby("team").cumcount()
    b = b.merge(team_games[["game_id", "team", "gidx"]], on=["game_id", "team"])
    names = box.drop_duplicates("athlete_id").set_index("athlete_id")["player"].to_dict()
    regulars = {}
    for team, tb in b.groupby("team"):
        nxt_idx = tb["gidx"].max() + 1
        recent = tb[tb["gidx"] >= nxt_idx - 3].groupby("athlete_id").size().index
        avg = tb.sort_values("gidx").groupby("athlete_id")["minutes"].apply(
            lambda s: s.tail(10).mean() if len(s) >= 3 else np.nan)
        regulars[team] = [a for a in recent if avg.get(a, 0) >= 15]

    rows = []
    for r in latest.itertuples():
        mates = [t for t in regulars.get(r.team, []) if t != r.player_id]
        prob = {t: ABSENT_PROBABILITY.get(statuses.get(norm_name(names.get(t))), NOT_LISTED_PROBABILITY)
                for t in mates}
        rec = {"player_id": r.player_id, "player": names.get(r.player_id), "team": normalize_team(str(r.team)),
               "n_absent_regulars": float(sum(prob.values())), "absent_covered": 0.0,
               "absent_players": "; ".join(f"{names.get(t)} ({statuses.get(norm_name(names.get(t)))})"
                                           for t in mates if prob[t] >= 0.4)}
        deltas = {c: 0.0 for c in features.STATS}
        uncovered = 0.0
        tot = totals.loc[r.player_id] if r.player_id in totals.index else None
        for t in mates:
            w = wo.loc[(r.player_id, t)] if (r.player_id, t) in wo.index else None
            if (tot is None or tot["seconds"] <= 0 or w is None
                    or w["wo_seconds"] < features.MIN_WITHOUT_SECONDS):
                uncovered += prob[t]
                continue
            rec["absent_covered"] += prob[t]
            for c in features.STATS:
                deltas[c] += prob[t] * (w[f"wo_{c}"] / w["wo_seconds"] * 2160 - tot[c] / tot["seconds"] * 2160)
        for c in features.STATS:
            # Expected adjustment over likely absences; NaN when the likely absences
            # (expected >= 0.5 of a player) have no split data, as in training.
            rec[f"abs_{c}36_delta"] = deltas[c] if uncovered < 0.5 else np.nan
        rows.append(rec)
    return pd.DataFrame(rows).merge(opp, on="player_id", how="left")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=str(date.today()))
    ap.add_argument("--out", default=str(OUT_PATH))
    ap.add_argument("--injury-healthy", default="auto", choices=["auto", "true", "false"])
    a = ap.parse_args()
    season = nba_season_start_year(a.date)
    con = warehouse.build([season], refresh=[season])
    stints, status = lineups.reconstruct_all(con, season)
    box = con.execute("SELECT * FROM player_box WHERE game_id IN (SELECT DISTINCT game_id FROM pbp)").df()

    healthy = a.injury_healthy == "true"
    if a.injury_healthy == "auto":
        from slate_injuries import assess_injury_feed
        healthy = bool(assess_injury_feed(date.fromisoformat(a.date)).get("healthy"))
    injuries = pd.read_csv(INJURY_PATH) if healthy and INJURY_PATH.exists() else pd.DataFrame()
    feats = next_game_features(stints, box, a.date, injury_statuses(injuries))
    feats["target_date"] = a.date
    feats["injury_feed_healthy"] = healthy
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    feats.to_csv(a.out, index=False)
    print(json.dumps({"date": a.date, "season": season, "games_with_pbp": int(len(status)),
                      "games_valid_lineups": int(status["valid"].sum()), "players": int(len(feats)),
                      "injury_feed_healthy": healthy,
                      "players_with_likely_absent_regular": int((feats["n_absent_regulars"] >= 0.5).sum()),
                      "generated_at_utc": datetime.now(timezone.utc).isoformat()}, indent=2))


if __name__ == "__main__":
    main()
