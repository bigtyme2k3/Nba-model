"""
collect_result_box.py
---------------------
Fetch only FINAL NBA box scores for one slate date from ESPN.

This is a lightweight grading source. It does not rebuild the season warehouse.
It writes a dated player actuals file that forward_reconciler.py can consume
immediately after games finish.

Output
------
data/raw/result_player_box_YYYY-MM-DD.csv
data/raw/result_box_status_YYYY-MM-DD.json
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from collect_stats import (
    fetch_espn_scoreboard,
    fetch_espn_boxscore,
    parse_espn_player_box,
)
from season_filters import nba_season_start_year

COLUMNS = [
    "game_id","season","game_date","team_name","team_location","team_abbr",
    "home_away","player_id","player_name","position","starter","did_not_play",
    "minutes","pts","reb","ast","stl","blk","tov","oreb","dreb",
    "fg_made","fg_att","threes_made","threes_att","ft_made","ft_att",
    "plus_minus","source",
]


def collect(target, out="data/raw", attempts=2):
    """Publish available finals even if a different game or scoreboard fails."""
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    season = nba_season_start_year(target)
    status = {"target_date": target, "season": season, "final_games": 0,
              "player_rows": 0, "shooting_attempt_coverage": None, "errors": [],
              "generated_at_utc": datetime.now(timezone.utc).isoformat(),
              "source": "espn_final_box_only"}
    rows, seen, successful = [], set(), set()
    for season_type in (2, 3):
        events = None
        for attempt in range(attempts):
            try:
                events = fetch_espn_scoreboard(target, season_type=season_type)
                break
            except Exception as exc:
                error = str(exc)
        if events is None:
            status["errors"].append({"surface": season_type, "error": error})
            continue
        for event in events:
            game_id = str(event.get("id", ""))
            if not game_id or game_id in seen:
                continue
            state = str(event.get("status", {}).get("type", {}).get("name", "")).upper()
            if "FINAL" not in state or (event.get("season") or {}).get("type") == 1:
                continue
            seen.add(game_id)
            for attempt in range(attempts):
                try:
                    parsed = parse_espn_player_box(fetch_espn_boxscore(game_id), target, season)
                    if not parsed:
                        raise ValueError("Final box has no player rows")
                    rows.extend(parsed)
                    successful.add(game_id)
                    break
                except Exception as exc:
                    error = str(exc)
            else:
                status["errors"].append({"game_id": game_id, "error": error})
    path = out_dir / f"result_player_box_{target}.csv"
    frames = []
    if path.exists():
        try:
            old = pd.read_csv(path, dtype={"game_id": str})
            frames.append(old[~old["game_id"].isin(successful)])
        except (ValueError, KeyError, pd.errors.EmptyDataError):
            pass
    frames.append(pd.DataFrame(rows, columns=COLUMNS))
    df = pd.concat(frames, ignore_index=True).reindex(columns=COLUMNS)
    if not df.empty:
        df = df.drop_duplicates(["game_id", "player_id", "player_name"], keep="last")
        active = df[pd.to_numeric(df["minutes"], errors="coerce").fillna(0).gt(0)]
        if not active.empty:
            fga = pd.to_numeric(active["fg_att"], errors="coerce")
            # A real zero-attempt appearance is valid. Missing fields stay missing.
            status["shooting_attempt_coverage"] = round(float((fga.notna() & fga.ge(0)).mean()), 4)
    temp = path.with_suffix(".csv.tmp")
    df.to_csv(temp, index=False)
    temp.replace(path)
    status.update(status="PARTIAL" if status["errors"] else ("READY" if seen else "WAITING_FOR_FINALS"),
                  final_games=len(seen), fetched_games=len(successful), player_rows=len(df),
                  failed_game_ids=sorted(seen - successful), output=str(path))
    (out_dir / f"result_box_status_{target}.json").write_text(json.dumps(status, indent=2) + "\n")
    return status


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True)
    parser.add_argument("--out", default="data/raw")
    args = parser.parse_args()
    result = collect(args.date, args.out)
    print(json.dumps(result, indent=2))
    if result["status"] == "PARTIAL":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
