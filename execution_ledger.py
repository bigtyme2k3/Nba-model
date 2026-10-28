"""
execution_ledger.py
-------------------
Append-only near-tip execution snapshots for NBA V1.

This ledger is intentionally separate from nba_forward_predictions.jsonl:
- forward ledger = first observed forecast, immutable by ranking_key
- execution ledger = every near-tip decision snapshot, append-only

A later evaluator can choose the closest valid pre-tip snapshot per ranking_key
without altering the original morning forecast.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from forward_ledger import flatten

OUT = Path("data/history/nba_pretip_execution_snapshots.jsonl")
STATUS = Path("data/tracking/execution_ledger_status.json")


def snapshot_id(row, generated, minutes_to_tip):
    raw = "|".join(map(str, [
        row.get("ranking_key"),
        generated,
        minutes_to_tip,
        row.get("sportsbook_key"),
        row.get("line"),
        row.get("odds"),
    ]))
    return hashlib.sha256(raw.encode()).hexdigest()[:24]


def read_existing_ids():
    if not OUT.exists():
        return set()
    ids=set()
    for line in OUT.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            ids.add(json.loads(line).get("execution_snapshot_id"))
        except Exception:
            pass
    return ids


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--date", required=True)
    parser.add_argument("--predictions", required=True)
    args=parser.parse_args()

    path=Path(args.predictions)
    if not path.exists():
        raise SystemExit(f"Prediction file missing: {path}")
    data=json.loads(path.read_text(encoding="utf-8"))
    generated=data.get("generated") or datetime.now(timezone.utc).isoformat()

    game_minutes={}
    for game in data.get("games",[]):
        away=game.get("away",{}).get("name","")
        home=game.get("home",{}).get("name","")
        game_minutes[f"{away} @ {home}"]=game.get("minutes_to_tip")

    rows=[]
    for row in flatten(data):
        mins=game_minutes.get(row.get("game"))
        snap={
            **row,
            "execution_snapshot_id": snapshot_id(row, generated, mins),
            "capture_class":"PRETIP_EXECUTION",
            "minutes_to_tip": mins,
            "execution_generated_at_utc": generated,
            "canonical_forward_ledger_mutated": False,
        }
        rows.append(snap)

    seen=read_existing_ids()
    new=[r for r in rows if r["execution_snapshot_id"] not in seen]
    OUT.parent.mkdir(parents=True,exist_ok=True)
    if new:
        with OUT.open("a",encoding="utf-8") as fh:
            for row in new:
                fh.write(json.dumps(row,allow_nan=False)+"\n")

    status={
        "status":"READY",
        "generated_at_utc":datetime.now(timezone.utc).isoformat(),
        "date":args.date,
        "candidate_snapshots":len(rows),
        "new_snapshots":len(new),
        "ledger_path":str(OUT),
        "policy":"APPEND_ALL_VALID_PRETIP_SNAPSHOTS; DO_NOT_REWRITE_FORWARD_LEDGER",
    }
    STATUS.parent.mkdir(parents=True,exist_ok=True)
    STATUS.write_text(json.dumps(status,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(status,indent=2))


if __name__=="__main__":
    main()
