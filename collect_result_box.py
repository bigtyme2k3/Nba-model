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


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--date",required=True)
    p.add_argument("--out",default="data/raw")
    args=p.parse_args()

    out_dir=Path(args.out)
    out_dir.mkdir(parents=True,exist_ok=True)
    target=args.date
    season=nba_season_start_year(target)

    status={
        "status":"unknown",
        "target_date":target,
        "season":season,
        "final_games":0,
        "player_rows":0,
        "shooting_attempt_coverage":None,
        "generated_at_utc":datetime.now(timezone.utc).isoformat(),
        "source":"espn_final_box_only",
    }

    rows=[]
    seen=set()
    # Query both regular and postseason surfaces; dedupe by event id.
    for season_type in (2,3):
        try:
            events=fetch_espn_scoreboard(target,season_type=season_type)
        except Exception:
            continue
        for event in events:
            game_id=str(event.get("id",""))
            if not game_id or game_id in seen:
                continue
            state=str(event.get("status",{}).get("type",{}).get("name","")).upper()
            if "FINAL" not in state:
                continue
            raw_season=event.get("season",{}) or {}
            if raw_season.get("type")==1 or "preseason" in str(raw_season.get("slug","")).lower():
                continue
            seen.add(game_id)
            try:
                summary=fetch_espn_boxscore(game_id)
                rows.extend(parse_espn_player_box(summary,target,season))
            except Exception as exc:
                print(f"[WARN] Could not fetch final box {game_id}: {exc}")

    df=pd.DataFrame(rows,columns=COLUMNS) if rows else pd.DataFrame(columns=COLUMNS)
    if not df.empty:
        active=df[pd.to_numeric(df["minutes"],errors="coerce").fillna(0).gt(0)].copy()
        if not active.empty:
            fga=pd.to_numeric(active["fg_att"],errors="coerce")
            coverage=float((fga.notna() & fga.gt(0)).mean())
            status["shooting_attempt_coverage"]=round(coverage,4)
            if coverage < 0.85:
                raise RuntimeError(
                    f"Final-box ESPN shooting parse coverage too low: {coverage:.1%}"
                )

    path=out_dir/f"result_player_box_{target}.csv"
    df.to_csv(path,index=False)
    status.update({
        "status":"READY" if seen else "WAITING_FOR_FINALS",
        "final_games":len(seen),
        "player_rows":len(df),
        "output":str(path),
    })
    status_path=out_dir/f"result_box_status_{target}.json"
    status_path.write_text(json.dumps(status,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(status,indent=2))


if __name__=="__main__":
    main()
