"""
execution_reconciler.py
-----------------------
Grades the closest valid NBA near-tip execution snapshot and compares it with the
immutable first-observed forecast.

This is diagnostic evidence, not a replacement for the forward ledger.
Promotion still uses first-observed forward performance + exact-book CLV.

Outputs
-------
data/tracking/nba_execution_performance.json
data/tracking/nba_execution_certified.csv
data/tracking/nba_forecast_vs_execution.csv
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from execution_ledger import OUT as EXECUTION_LEDGER
from forward_ledger import LEDGER as FORWARD_LEDGER, EVIDENCE_EPOCH
from forward_reconciler import load_scores, load_player_actuals, resolve_row, performance

TRACKING=Path("data/tracking")


def read_jsonl(path):
    rows=[]
    if not Path(path).exists():
        return rows
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row=json.loads(line)
        except Exception:
            continue
        if row.get("evidence_epoch")==EVIDENCE_EPOCH:
            rows.append(row)
    return rows


def f(value,default=None):
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def closest_execution(rows,target_date):
    selected={}
    for row in rows:
        if str(row.get("date"))!=str(target_date):
            continue
        mins=f(row.get("minutes_to_tip"))
        if mins is None or mins<=0:
            continue
        key=row.get("ranking_key")
        if not key:
            continue
        prior=selected.get(key)
        if prior is None or mins < f(prior.get("minutes_to_tip"),9999):
            selected[key]=row
    return list(selected.values())


def compare(forward_rows,execution_rows):
    fwd={r.get("ranking_key"):r for r in forward_rows}
    rows=[]
    for ex in execution_rows:
        fw=fwd.get(ex.get("ranking_key"))
        if not fw:
            continue
        rows.append({
            "ranking_key":ex.get("ranking_key"),
            "date":ex.get("date"),
            "game":ex.get("game"),
            "market":ex.get("market"),
            "player":ex.get("player"),
            "stat":ex.get("stat"),
            "execution_minutes_to_tip":ex.get("minutes_to_tip"),
            "forward_decision":fw.get("decision"),
            "execution_decision":ex.get("decision"),
            "decision_changed":fw.get("decision")!=ex.get("decision"),
            "forward_side":fw.get("side"),
            "execution_side":ex.get("side"),
            "side_changed":fw.get("side")!=ex.get("side"),
            "forward_line":fw.get("line"),
            "execution_line":ex.get("line"),
            "line_delta":(
                round(f(ex.get("line"))-f(fw.get("line")),4)
                if f(ex.get("line")) is not None and f(fw.get("line")) is not None
                else None
            ),
            "forward_odds":fw.get("odds"),
            "execution_odds":ex.get("odds"),
            "forward_projection":fw.get("projection"),
            "execution_projection":ex.get("projection"),
            "projection_delta":(
                round(f(ex.get("projection"))-f(fw.get("projection")),4)
                if f(ex.get("projection")) is not None and f(fw.get("projection")) is not None
                else None
            ),
            "forward_model_probability":fw.get("model_probability"),
            "execution_model_probability":ex.get("model_probability"),
            "forward_ev":fw.get("ev"),
            "execution_ev":ex.get("ev"),
            "forward_outcome":fw.get("outcome"),
            "execution_outcome":ex.get("outcome"),
            "actual":ex.get("actual"),
        })
    return rows


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--date",required=True)
    args=parser.parse_args()

    all_exec=read_jsonl(EXECUTION_LEDGER)
    selected=closest_execution(all_exec,args.date)
    forward=[
        r for r in read_jsonl(FORWARD_LEDGER)
        if str(r.get("date"))==str(args.date)
    ]

    scores=load_scores(args.date)
    players=load_player_actuals(args.date)
    now=datetime.now(timezone.utc).isoformat()

    graded=[]
    for row in selected:
        item=dict(row)
        resolve_row(item,scores,players,args.date,now)
        graded.append(item)

    perf=performance(graded)
    perf.update({
        "status":"READY" if graded else "NO_EXECUTION_SNAPSHOTS",
        "evidence_epoch":EVIDENCE_EPOCH,
        "target_date":args.date,
        "selected_execution_rows":len(graded),
        "selection_policy":"CLOSEST_POSITIVE_MINUTES_TO_TIP_PER_RANKING_KEY",
        "promotion_evidence":False,
    })

    TRACKING.mkdir(parents=True,exist_ok=True)
    (TRACKING/"nba_execution_performance.json").write_text(
        json.dumps(perf,indent=2)+"\n",encoding="utf-8"
    )
    pd.DataFrame(graded).to_csv(
        TRACKING/"nba_execution_certified.csv",index=False
    )

    comparison=compare(forward,graded)
    pd.DataFrame(comparison).to_csv(
        TRACKING/"nba_forecast_vs_execution.csv",index=False
    )

    changed=sum(bool(r.get("decision_changed")) for r in comparison)
    sides=sum(bool(r.get("side_changed")) for r in comparison)
    print(json.dumps({
        "status":perf["status"],
        "target_date":args.date,
        "execution_rows":len(graded),
        "execution_bets":perf.get("bet_rows"),
        "forecast_execution_matches":len(comparison),
        "decision_changes":changed,
        "side_changes":sides,
        "promotion_evidence":False,
    },indent=2))


if __name__=="__main__":
    main()
