"""
scrape_injuries.py
------------------
Creates daily NBA injury status files and minute adjustments.

Sources:
  1. ESPN team injury endpoints
  2. Optional data/raw/injury_overrides.json with optional expires date

Outputs:
  data/raw/injuries_today.csv
  data/raw/injuries_YYYY-MM-DD.csv
  data/raw/injuries_historical.csv
  data/raw/minute_projections.csv
  data/raw/injuries_status.json
"""
from __future__ import annotations

import argparse
import json
import os
import time
from datetime import date, datetime, timezone

import pandas as pd
import requests

from teams import TEAM_ESPN_ID

OUT_DIR = "data/raw"
HEADERS = {"User-Agent": "Mozilla/5.0 (Linux; Android 13; Tablet) AppleWebKit/537.36 Chrome/120 Safari/537.36"}
ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"
RAW_COLUMNS = ["game_date", "team", "player", "player_id", "position", "status", "severity", "injury_type", "detail", "return_date", "is_out", "source", "scraped_at"]
STATUS_SEVERITY = {"out":"OUT","doubtful":"DOUBTFUL","questionable":"QUESTIONABLE","probable":"PROBABLE","day-to-day":"QUESTIONABLE","day to day":"QUESTIONABLE","ir":"OUT","suspended":"OUT"}
BENCH_FILL_PCT = {"G":0.75,"F":0.70,"C":0.65,"":0.70}
# Rough placeholder starter minutes — overwritten in practice by roll5_minutes from real box scores.
DEFAULT_MINUTES = {}


def normalize_status(value):
    s=str(value or "").strip().lower()
    for key,label in STATUS_SEVERITY.items():
        if key in s:return label
    return "UNKNOWN" if s else "ACTIVE"


def empty_df():return pd.DataFrame(columns=RAW_COLUMNS)


def fetch_espn_injuries(team_name:str,team_id:int)->list:
    url=f"{ESPN_BASE}/teams/{team_id}/injuries"
    try:
        resp=requests.get(url,headers=HEADERS,timeout=12)
        if resp.status_code==404:return []
        resp.raise_for_status();data=resp.json()
    except Exception:return []
    injuries=[];now=datetime.now(timezone.utc).isoformat()
    for item in data.get("injuries",[]) or []:
        athlete=item.get("athlete",{}) or {};raw_status=item.get("status") or item.get("type") or item.get("shortComment") or "";detail=item.get("details",{}) if isinstance(item.get("details"),dict) else {};severity=normalize_status(raw_status)
        if severity in {"ACTIVE","UNKNOWN"}:continue
        injuries.append({"game_date":"","team":team_name,"player":athlete.get("displayName","") or athlete.get("fullName",""),"player_id":athlete.get("id",""),"position":(athlete.get("position",{}) or {}).get("abbreviation",""),"status":raw_status or severity,"severity":severity,"injury_type":item.get("type",""),"detail":detail.get("detail","") or item.get("longComment","") or item.get("shortComment",""),"return_date":detail.get("returnDate",""),"is_out":severity in {"OUT","DOUBTFUL"},"source":"espn","scraped_at":now})
    return injuries


def load_overrides(out_dir:str,target_date:str)->list:
    path=os.path.join(out_dir,"injury_overrides.json")
    if not os.path.exists(path):return []
    try:data=json.load(open(path))
    except Exception as exc:print(f"  [WARN] Could not read injury overrides: {exc}");return []
    rows=[];now=datetime.now(timezone.utc).isoformat()
    for player,info in (data or {}).items():
        if isinstance(info,str):info={"status":info}
        expires=str(info.get("expires","")).strip()
        if expires and expires<target_date:continue
        severity=normalize_status(info.get("status",""))
        rows.append({"game_date":target_date,"team":info.get("team",""),"player":player,"player_id":"","position":info.get("position",""),"status":info.get("status",severity),"severity":severity,"injury_type":info.get("injury_type",""),"detail":info.get("note",""),"return_date":expires,"is_out":severity in {"OUT","DOUBTFUL"},"source":info.get("source","manual"),"scraped_at":now})
    return rows


def fetch_all_injuries(out_dir:str,target_date:str)->pd.DataFrame:
    print(f"Fetching injury reports for {target_date}...");rows=[]
    for team,tid in TEAM_ESPN_ID.items():
        injuries=fetch_espn_injuries(team,tid);rows.extend(injuries)
        if injuries:print(f"  {team}: {', '.join([i['player']+' '+i['severity'] for i in injuries if i.get('player')])}")
        time.sleep(0.25)
    rows.extend(load_overrides(out_dir,target_date));df=pd.DataFrame(rows,columns=RAW_COLUMNS) if rows else empty_df()
    if not df.empty:df["game_date"]=target_date;df=df.drop_duplicates(subset=["player"],keep="last")
    os.makedirs(out_dir,exist_ok=True);today_path=os.path.join(out_dir,"injuries_today.csv");dated_path=os.path.join(out_dir,f"injuries_{target_date}.csv");df.to_csv(today_path,index=False);df.to_csv(dated_path,index=False)
    hist_path=os.path.join(out_dir,"injuries_historical.csv")
    if not df.empty:
        if os.path.exists(hist_path):
            hist=pd.read_csv(hist_path)
            if "game_date" in hist.columns:hist=hist[hist["game_date"]!=target_date]
            pd.concat([hist,df],ignore_index=True).to_csv(hist_path,index=False)
        else:df.to_csv(hist_path,index=False)
    print(f"  Saved → {today_path} ({len(df)} rows)");print(f"  Saved → {dated_path}");return df


def reallocate_minutes(injuries_df:pd.DataFrame,games:list,default_minutes:dict|None=None,bench_depth:dict|None=None)->dict:
    """Rough OUT/DOUBTFUL/QUESTIONABLE minute reallocation. Pass real roster data via
    default_minutes={player: mpg} and bench_depth={team: [(player, base_min), ...]}
    once data/processed has real rolling minutes — this ships with empty placeholders."""
    if injuries_df.empty:return {}
    default_minutes = default_minutes or DEFAULT_MINUTES
    bench_depth = bench_depth or {}
    if "is_out" not in injuries_df.columns:injuries_df["is_out"]=injuries_df.get("severity","").astype(str).isin(["OUT","DOUBTFUL"])
    adjustments={};teams_playing=set()
    for g in games or []:teams_playing.add(g.get("home",""));teams_playing.add(g.get("away",""))
    if not teams_playing:teams_playing=set(injuries_df.get("team",pd.Series(dtype=str)).dropna().astype(str))
    for team in teams_playing:
        bench=bench_depth.get(team,[]);team_df=injuries_df[injuries_df.get("team","")==team] if "team" in injuries_df.columns else injuries_df;total_missing=0
        for _,row in team_df.iterrows():
            player=row.get("player","");sev=str(row.get("severity","")).upper()
            if sev in {"OUT","DOUBTFUL"} or bool(row.get("is_out",False)):missing=default_minutes.get(player,25);total_missing+=missing;adjustments[player]=0
            elif sev=="QUESTIONABLE":adjustments[player]=default_minutes.get(player,25)*0.6
            elif sev=="PROBABLE":adjustments[player]=default_minutes.get(player,25)*0.9
        if total_missing>0 and bench:
            per_bench=total_missing/len(bench)
            for bench_player,base_min in bench:adjustments[bench_player]=min(38,base_min+per_bench*BENCH_FILL_PCT.get("",0.70))
    return adjustments


def save_projections(adjustments:dict,games:list,out_dir:str,target_date:str):
    rows=[{"game_date":target_date,"player":p,"proj_min":round(m,1),"adjusted":True} for p,m in adjustments.items()];df=pd.DataFrame(rows)
    if not df.empty:path=os.path.join(out_dir,"minute_projections.csv");df.to_csv(path,index=False);print(f"  Minute projections → {path} ({len(df)} players adjusted)")
    return df


def apply_injury_adjustments(player_games:list,adjustments:dict)->list:
    adjusted=[]
    for pg in player_games:
        player=pg.get("player","")
        if player in adjustments:
            new_min=adjustments[player];orig_min=pg.get("proj_minutes",pg.get("mpg",28));pg["injury_adjusted"]=True
            if new_min==0:pg["proj_minutes"]=0;pg["is_out"]=True
            else:
                scale=new_min/max(orig_min,1);pg["proj_minutes"]=new_min
                for k in ["roll5_pts","roll5_reb","roll5_ast","roll5_threes","roll5_pra"]:
                    if k in pg:pg[k]=pg[k]*scale
        adjusted.append(pg)
    return adjusted


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--date",default=str(date.today()));parser.add_argument("--out",default=OUT_DIR);args=parser.parse_args();os.makedirs(args.out,exist_ok=True)
    print(f"\n═══ Injury Report + Minute Projections — {args.date} ═══\n")
    started=datetime.now(timezone.utc);status={"status":"unknown","target_date":args.date,"rows":0,"out":0,"questionable":0,"error":None,"generated_at_utc":started.isoformat(),"sources":["espn","manual_override"],"teams_checked":len(TEAM_ESPN_ID)}
    try:
        injuries_df=fetch_all_injuries(args.out,args.date);adjustments=reallocate_minutes(injuries_df,[]);save_projections(adjustments,[],args.out,args.date)
        status.update({"status":"ok","rows":int(len(injuries_df)),"out":int(injuries_df["is_out"].sum()) if not injuries_df.empty else 0,"questionable":int((injuries_df.get("severity",pd.Series(dtype=str))=="QUESTIONABLE").sum()) if not injuries_df.empty else 0,"completed_at_utc":datetime.now(timezone.utc).isoformat()})
    except Exception as exc:
        status.update({"status":"error","error":str(exc),"completed_at_utc":datetime.now(timezone.utc).isoformat()});print(f"  [WARN] Injury step failed: {exc}");empty_df().to_csv(os.path.join(args.out,"injuries_today.csv"),index=False)
    with open(os.path.join(args.out,"injuries_status.json"),"w") as f:json.dump(status,f,indent=2)
    print(f"  Summary: {status['out']} OUT/DOUBTFUL, {status['questionable']} QUESTIONABLE");print("✅ Injury scrape complete.")
    if status["status"]=="error":raise SystemExit(1)


if __name__=="__main__":main()
