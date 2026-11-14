"""
v2/lineups.py
-------------
Five-player lineups (stints) from ESPN play-by-play, and what each player saw
on the court in every stint.

Period starters
  First quarter: the box-score starters. Later periods: players whose first
  appearance in the period is anything other than entering as a substitute;
  if that gives fewer than five, players on court at the end of the previous
  period who did not leave are added. A period that still lacks exactly five
  per team, or a substitution that removes someone not on court, marks the game
  invalid; invalid games are excluded, never repaired with guesses.

Per player per stint (both teams):
  seconds, rebound chances (own team / opponent missed shots: field goals and the
  last free throw of a trip), assist chances (teammates' made field goals),
  team field-goal attempts, and the player's own rebounds, assists, FGA, points.

Validation: reconstructed minutes per player-game against box-score minutes.
"""
from __future__ import annotations

import re
from collections import defaultdict

import pandas as pd

LAST_FT = re.compile(r"(\d) of \1")


def period_bounds(period: int) -> tuple[float, float]:
    """(start, length) in elapsed game seconds; 12-minute quarters, 5-minute overtimes."""
    if period <= 4:
        return (period - 1) * 720.0, 720.0
    return 2880.0 + (period - 5) * 300.0, 300.0


def is_missed_rebound_chance(row) -> bool:
    text = str(row.type_text)
    if row.shooting_play and not row.scoring_play:
        if text.startswith("Free Throw"):
            return bool(LAST_FT.search(text)) and "Technical" not in text and "Flagrant" not in text
        return True
    return False


def reconstruct_game(events: pd.DataFrame, box: pd.DataFrame):
    """Returns (stint_player_rows, status) for one game."""
    box = box[box["athlete_id"].notna()]
    side_of = dict(zip(box["athlete_id"], box["home_away"]))
    starters_q1 = {s: set(box[(box["starter"]) & (box["home_away"] == s)]["athlete_id"]) for s in ("home", "away")}
    team_side = {events["home_team_id"].iloc[0]: "home", events["away_team_id"].iloc[0]: "away"}
    rows, problems = [], []
    on = {"home": set(), "away": set()}
    events = events.sort_values(["period", "play_no"])

    for period, ev in events.groupby("period", sort=True):
        start, length = period_bounds(int(period))
        if period == 1:
            starters = {s: set(starters_q1[s]) for s in on}
        else:
            first = {}
            for r in ev.itertuples():
                if r.type_text == "Substitution":
                    for a, kind in ((r.a1, "in"), (r.a2, "start")):
                        if a in side_of and a not in first:
                            first[a] = kind
                else:
                    for a in (r.a1, r.a2, r.a3):
                        if a in side_of and a not in first:
                            first[a] = "start"
            starters = {s: {a for a, k in first.items() if k == "start" and side_of[a] == s} for s in on}
            for s in on:
                if len(starters[s]) < 5:
                    entered = {a for a, k in first.items() if k == "in"}
                    for a in sorted(on[s] - starters[s] - entered):
                        if len(starters[s]) < 5:
                            starters[s].add(a)
        if any(len(starters[s]) != 5 for s in on):
            problems.append(f"period {period}: starters {[len(starters[s]) for s in on]}")
            break
        on = {s: set(starters[s]) for s in on}

        stint_start = start
        acc = defaultdict(lambda: defaultdict(float))  # (player) -> counters for the open stint

        def close(t):
            secs = max(0.0, t - close.started)
            for s in on:
                for p in on[s]:
                    c = acc[p]
                    rows.append({"player_id": p, "side": s, "period": int(period),
                                 "start": close.started, "end": t, "seconds": secs,
                                 "teammates_on": tuple(sorted(on[s] - {p})),
                                 **c})
            acc.clear()
            close.started = t
        close.started = stint_start

        for r in ev.itertuples():
            t = start + length - float(r.clock_remaining) if pd.notna(r.clock_remaining) else None
            if r.type_text == "Substitution":
                s = side_of.get(r.a1)
                if s is None or side_of.get(r.a2) != s or t is None:
                    problems.append(f"unmapped substitution: {r.text}")
                    continue
                if r.a2 not in on[s] or r.a1 in on[s]:
                    problems.append(f"inconsistent substitution: {r.text}")
                    continue
                if t > close.started:
                    close(t)
                on[s].discard(r.a2)
                on[s].add(r.a1)
                continue
            shooter_side = side_of.get(r.a1) if r.shooting_play else None
            if shooter_side is None and r.shooting_play:
                shooter_side = team_side.get(r.team_id)
            if r.shooting_play and shooter_side:
                is_ft = str(r.type_text).startswith("Free Throw")
                if not is_ft:
                    for p in on[shooter_side]:
                        acc[p]["team_fga"] += 1
                    if r.a1 in side_of:
                        acc[r.a1]["fga"] += 1
                if r.scoring_play:
                    if r.a1 in side_of:
                        acc[r.a1]["pts"] += float(r.score_value or 0)
                    if not is_ft:
                        for p in on[shooter_side] - {r.a1}:
                            acc[p]["ast_chances"] += 1
                        if r.a2 in on[shooter_side]:
                            acc[r.a2]["ast"] += 1
                if is_missed_rebound_chance(r):
                    other = "away" if shooter_side == "home" else "home"
                    for p in on[shooter_side]:
                        acc[p]["oreb_chances"] += 1
                    for p in on[other]:
                        acc[p]["dreb_chances"] += 1
            if "Rebound" in str(r.type_text) and r.a1 in side_of:
                acc[r.a1]["reb"] += 1
        close(start + length)

    stints = pd.DataFrame(rows)
    # Counters are event counts: a stint with no rebound has 0 rebounds, not a missing value.
    counters = [c for c in stints.columns if c not in
                {"player_id", "side", "period", "start", "end", "seconds", "teammates_on"}]
    stints[counters] = stints[counters].fillna(0.0)
    status = {"valid": not problems, "problems": problems[:5]}
    if stints.empty or problems:
        return stints, status
    mins = stints.groupby("player_id")["seconds"].sum() / 60
    played = box[(box["minutes"] > 0)].set_index("athlete_id")["minutes"]
    diff = (mins.reindex(played.index).fillna(0) - played).abs()
    status.update(minutes_mae=float(diff.mean()), minutes_max_err=float(diff.max()),
                  players=int(len(played)))
    # A lineup that drifted (e.g. a missed substitution) shows up as minutes far off.
    if diff.max() > 6:
        status["valid"] = False
        status["problems"] = [f"minutes off by {diff.max():.1f} for {diff.idxmax()}"]
    return stints, status


def reconstruct_all(con, season: int):
    """Stint-player rows and per-game status for every play-by-play game of the season."""
    pbp = con.execute("""SELECT p.* FROM pbp p JOIN games g USING (game_id) WHERE g.season = ?
                         ORDER BY game_id, period, play_no""", [season]).df()
    box = con.execute("""SELECT b.* FROM player_box b WHERE b.game_id IN
                         (SELECT DISTINCT game_id FROM pbp)""").df()
    box_by_game = dict(tuple(box.groupby("game_id")))
    all_rows, statuses = [], []
    for game_id, ev in pbp.groupby("game_id", sort=False):
        b = box_by_game.get(game_id)
        if b is None:
            statuses.append({"game_id": game_id, "valid": False, "problems": ["no box score"]})
            continue
        stints, status = reconstruct_game(ev, b)
        statuses.append({"game_id": game_id, **status})
        if status["valid"]:
            all_rows.append(stints.assign(game_id=game_id))
    return (pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()), pd.DataFrame(statuses)
