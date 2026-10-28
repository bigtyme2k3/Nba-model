"""
closing_market.py
-----------------
Append-only sportsbook-specific NBA prop closing snapshots.

This script does not fetch markets itself. It consumes an isolated near-tip
sportsbook prop CSV produced by scrape_sportsbook_props.py with
--output-prefix sportsbook_props_closing.

The close ledger is immutable evidence. Each capture retains:
- exact sportsbook
- player/stat/side
- line and American price
- no-vig probability
- market update time
- capture time
- minutes to scheduled tip

Outputs
-------
data/history/nba_prop_closing_snapshots.jsonl
data/tracking/closing_market_status.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

OUT = Path("data/history/nba_prop_closing_snapshots.jsonl")
STATUS = Path("data/tracking/closing_market_status.json")


def clean(value):
    if pd.isna(value):
        return None
    return value


def snapshot_id(row: dict) -> str:
    raw = "|".join(map(str, [
        row.get("game_date"),
        row.get("event_id"),
        row.get("player"),
        row.get("stat"),
        row.get("book_key"),
        row.get("side"),
        row.get("line"),
        row.get("price"),
        row.get("captured_at_utc"),
    ]))
    return hashlib.sha256(raw.encode()).hexdigest()[:24]


def existing_ids():
    if not OUT.exists():
        return set()
    ids=set()
    for line in OUT.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            ids.add(json.loads(line).get("closing_snapshot_id"))
        except Exception:
            continue
    return ids


def minutes_to_tip(commence_time, captured):
    try:
        tip=datetime.fromisoformat(str(commence_time).replace("Z","+00:00")).astimezone(timezone.utc)
        cap=datetime.fromisoformat(str(captured).replace("Z","+00:00")).astimezone(timezone.utc)
        return round((tip-cap).total_seconds()/60.0, 3)
    except Exception:
        return None


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--date", required=True)
    parser.add_argument(
        "--input",
        default=None,
        help="Near-tip prop CSV; defaults to data/raw/sportsbook_props_closing_today.csv",
    )
    args=parser.parse_args()

    path=Path(args.input) if args.input else Path("data/raw/sportsbook_props_closing_today.csv")
    status={
        "status":"unknown",
        "date":args.date,
        "input":str(path),
        "generated_at_utc":datetime.now(timezone.utc).isoformat(),
        "candidate_rows":0,
        "new_rows":0,
        "books":[],
        "players":0,
        "min_minutes_to_tip":None,
        "max_minutes_to_tip":None,
        "policy":"APPEND_ONLY_EXACT_BOOK_PROP_CLOSES",
    }

    if not path.exists():
        status["status"]="missing_input"
        STATUS.parent.mkdir(parents=True,exist_ok=True)
        STATUS.write_text(json.dumps(status,indent=2)+"\n",encoding="utf-8")
        print(json.dumps(status,indent=2))
        return

    try:
        df=pd.read_csv(path)
    except Exception as exc:
        status["status"]="read_error"
        status["error"]=str(exc)
        STATUS.parent.mkdir(parents=True,exist_ok=True)
        STATUS.write_text(json.dumps(status,indent=2)+"\n",encoding="utf-8")
        print(json.dumps(status,indent=2))
        return

    if "game_date" in df.columns:
        df=df[df["game_date"].astype(str).eq(str(args.date))].copy()

    if df.empty:
        status["status"]="empty"
        STATUS.parent.mkdir(parents=True,exist_ok=True)
        STATUS.write_text(json.dumps(status,indent=2)+"\n",encoding="utf-8")
        print(json.dumps(status,indent=2))
        return

    rows=[]
    for _,r in df.iterrows():
        captured=str(clean(r.get("scraped_at")) or datetime.now(timezone.utc).isoformat())
        mins=minutes_to_tip(r.get("commence_time"),captured)
        if mins is None or mins <= 0:
            continue
        row={
            "capture_class":"SPORTSBOOK_PROP_CLOSE",
            "game_date":str(r.get("game_date","")),
            "event_id":str(r.get("event_id","")),
            "commence_time":str(r.get("commence_time","")),
            "game":f"{str(r.get('away_team','')).strip()} @ {str(r.get('home_team','')).strip()}",
            "home_team":str(r.get("home_team","")),
            "away_team":str(r.get("away_team","")),
            "player":str(r.get("player","")),
            "stat":str(r.get("stat","")).upper(),
            "market_key":str(r.get("market_key","")),
            "side":str(r.get("side","")).upper(),
            "line":float(r.get("line")),
            "price":int(float(r.get("price"))),
            "book":str(r.get("book","")),
            "book_key":str(r.get("book_key","")),
            "implied_probability":clean(r.get("implied_probability")),
            "no_vig_probability":clean(r.get("no_vig_probability")),
            "pair_overround":clean(r.get("pair_overround")),
            "book_last_update":str(clean(r.get("last_update")) or ""),
            "source":str(r.get("source","the-odds-api")),
            "captured_at_utc":captured,
            "minutes_to_tip":mins,
        }
        row["closing_snapshot_id"]=snapshot_id(row)
        rows.append(row)

    seen=existing_ids()
    new=[r for r in rows if r["closing_snapshot_id"] not in seen]
    OUT.parent.mkdir(parents=True,exist_ok=True)
    if new:
        with OUT.open("a",encoding="utf-8") as fh:
            for row in new:
                fh.write(json.dumps(row,allow_nan=False)+"\n")

    mins=[r["minutes_to_tip"] for r in rows if r.get("minutes_to_tip") is not None]
    status.update({
        "status":"READY",
        "candidate_rows":len(rows),
        "new_rows":len(new),
        "books":sorted({r["book_key"] for r in rows if r.get("book_key")}),
        "players":len({r["player"] for r in rows if r.get("player")}),
        "min_minutes_to_tip":round(min(mins),3) if mins else None,
        "max_minutes_to_tip":round(max(mins),3) if mins else None,
        "ledger":str(OUT),
    })
    STATUS.parent.mkdir(parents=True,exist_ok=True)
    STATUS.write_text(json.dumps(status,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(status,indent=2))


if __name__=="__main__":
    main()
