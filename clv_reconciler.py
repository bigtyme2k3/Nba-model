"""
clv_reconciler.py
-----------------
Sportsbook-specific NBA prop closing-line value (CLV) reconciliation.

Entry evidence comes from the immutable forward ledger. Closing evidence comes
from nba_prop_closing_snapshots.jsonl and is matched to the SAME sportsbook,
player, stat, side and game/date.

Positive line CLV is always defined as a better entry number:
  OVER 17.5 closing 18.5  -> +1.0
  UNDER 18.5 closing 17.5 -> +1.0

When the closing line is unchanged, price/no-vig probability CLV is also
reported:
  closing no-vig probability - entry no-vig probability
Positive means the market moved toward the side we entered.

Outputs
-------
data/tracking/nba_prop_clv.csv
data/tracking/nba_prop_clv_report.json
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from forward_ledger import EVIDENCE_EPOCH

FORWARD = Path("data/history/nba_forward_predictions.jsonl")
CLOSES = Path("data/history/nba_prop_closing_snapshots.jsonl")
OUT_CSV = Path("data/tracking/nba_prop_clv.csv")
OUT_JSON = Path("data/tracking/nba_prop_clv_report.json")


def norm(value):
    return " ".join(str(value or "").strip().lower().replace("’","'").split())


def f(value, default=None):
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def read_jsonl(path):
    rows=[]
    if not Path(path).exists():
        return rows
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            continue
    return rows


def match_key(row):
    return (
        str(row.get("date") or row.get("game_date") or ""),
        norm(row.get("game")),
        norm(row.get("player")),
        str(row.get("stat") or "").upper(),
        norm(row.get("sportsbook_key") or row.get("book_key")),
        str(row.get("side") or "").upper(),
    )


def select_close(candidates):
    """Closest valid snapshot before tip (minimum positive minutes_to_tip)."""
    valid=[]
    for row in candidates:
        mins=f(row.get("minutes_to_tip"))
        if mins is None or mins <= 0:
            continue
        valid.append((mins,row))
    if not valid:
        return None
    return min(valid,key=lambda x:x[0])[1]


def line_clv(entry_line, close_line, side):
    e=f(entry_line); c=f(close_line)
    if e is None or c is None:
        return None
    side=str(side or "").upper()
    if side=="OVER":
        return c-e
    if side=="UNDER":
        return e-c
    return None


def summarize(rows, field):
    groups=defaultdict(list)
    for r in rows:
        key=str(r.get(field) or "UNKNOWN")
        groups[key].append(r)
    out={}
    for key,vals in sorted(groups.items()):
        matched=[r for r in vals if r.get("close_matched")]
        line_vals=[f(r.get("line_clv")) for r in matched]
        line_vals=[x for x in line_vals if x is not None]
        price_vals=[
            f(r.get("same_line_price_clv_probability"))
            for r in matched
            if r.get("same_line")
        ]
        price_vals=[x for x in price_vals if x is not None]
        resolved=[r for r in matched if r.get("beat_close") in {True,False}]
        out[key]={
            "entries":len(vals),
            "matched_closes":len(matched),
            "match_rate":round(len(matched)/len(vals),4) if vals else None,
            "avg_line_clv":round(sum(line_vals)/len(line_vals),4) if line_vals else None,
            "positive_line_clv_rate":(
                round(sum(x>0 for x in line_vals)/len(line_vals),4) if line_vals else None
            ),
            "same_line_price_rows":len(price_vals),
            "avg_same_line_price_clv_probability":(
                round(sum(price_vals)/len(price_vals),5) if price_vals else None
            ),
            "beat_close_rows":len(resolved),
            "beat_close_rate":(
                round(sum(bool(r.get("beat_close")) for r in resolved)/len(resolved),4)
                if resolved else None
            ),
        }
    return out


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--date", default=None)
    args=parser.parse_args()

    entries=[
        r for r in read_jsonl(FORWARD)
        if r.get("evidence_epoch")==EVIDENCE_EPOCH
        and str(r.get("market","")).upper()=="PROP"
        and r.get("decision")=="BET"
        and norm(r.get("sportsbook_key"))
    ]
    if args.date:
        entries=[r for r in entries if str(r.get("date"))==str(args.date)]

    closes=read_jsonl(CLOSES)
    if args.date:
        closes=[r for r in closes if str(r.get("game_date"))==str(args.date)]

    close_map=defaultdict(list)
    for row in closes:
        close_map[match_key(row)].append(row)

    out=[]
    for entry in entries:
        key=match_key(entry)
        close=select_close(close_map.get(key,[]))
        base={
            "prediction_id":entry.get("prediction_id"),
            "ranking_key":entry.get("ranking_key"),
            "date":entry.get("date"),
            "game":entry.get("game"),
            "player":entry.get("player"),
            "stat":entry.get("stat"),
            "side":entry.get("side"),
            "sportsbook":entry.get("sportsbook"),
            "sportsbook_key":entry.get("sportsbook_key"),
            "entry_line":entry.get("line"),
            "entry_odds":entry.get("odds"),
            "entry_market_probability":entry.get("market_probability"),
            "entry_model_probability":entry.get("model_probability"),
            "entry_ev":entry.get("ev"),
            "entry_generated_at_utc":entry.get("prediction_generated_at_utc"),
            "close_matched":bool(close),
            "close_line":None,
            "close_odds":None,
            "close_market_probability":None,
            "close_minutes_to_tip":None,
            "close_captured_at_utc":None,
            "line_clv":None,
            "same_line":None,
            "same_line_price_clv_probability":None,
            "beat_close":None,
            "clv_basis":"EXACT_BOOK_PLAYER_STAT_SIDE",
        }
        if close:
            lc=line_clv(entry.get("line"),close.get("line"),entry.get("side"))
            same=(
                f(entry.get("line")) is not None
                and f(close.get("line")) is not None
                and abs(f(entry.get("line"))-f(close.get("line"))) < 1e-9
            )
            entry_mp=f(entry.get("market_probability"))
            close_mp=f(close.get("no_vig_probability"))
            price_clv=(close_mp-entry_mp) if same and entry_mp is not None and close_mp is not None else None

            # A line move dominates. If the line is unchanged, use no-vig price move.
            beat=None
            if lc is not None and abs(lc) > 1e-9:
                beat=lc>0
            elif price_clv is not None and abs(price_clv) > 0.001:
                beat=price_clv>0
            elif lc is not None or price_clv is not None:
                beat=False

            base.update({
                "close_line":close.get("line"),
                "close_odds":close.get("price"),
                "close_market_probability":close.get("no_vig_probability"),
                "close_minutes_to_tip":close.get("minutes_to_tip"),
                "close_captured_at_utc":close.get("captured_at_utc"),
                "line_clv":round(lc,4) if lc is not None else None,
                "same_line":same,
                "same_line_price_clv_probability":(
                    round(price_clv,5) if price_clv is not None else None
                ),
                "beat_close":beat,
            })
        out.append(base)

    OUT_CSV.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(out).to_csv(OUT_CSV,index=False)

    matched=[r for r in out if r.get("close_matched")]
    line_vals=[f(r.get("line_clv")) for r in matched]
    line_vals=[x for x in line_vals if x is not None]
    price_vals=[
        f(r.get("same_line_price_clv_probability"))
        for r in matched if r.get("same_line")
    ]
    price_vals=[x for x in price_vals if x is not None]
    resolved=[r for r in matched if r.get("beat_close") in {True,False}]

    report={
        "status":"READY" if entries else "NO_FORWARD_PROP_BETS",
        "generated_at_utc":datetime.now(timezone.utc).isoformat(),
        "date_filter":args.date,
        "evidence_epoch":EVIDENCE_EPOCH,
        "forward_prop_bets":len(entries),
        "matched_closes":len(matched),
        "match_rate":round(len(matched)/len(entries),4) if entries else None,
        "avg_line_clv":round(sum(line_vals)/len(line_vals),4) if line_vals else None,
        "positive_line_clv_rate":(
            round(sum(x>0 for x in line_vals)/len(line_vals),4) if line_vals else None
        ),
        "same_line_price_rows":len(price_vals),
        "avg_same_line_price_clv_probability":(
            round(sum(price_vals)/len(price_vals),5) if price_vals else None
        ),
        "beat_close_rows":len(resolved),
        "beat_close_rate":(
            round(sum(bool(r.get("beat_close")) for r in resolved)/len(resolved),4)
            if resolved else None
        ),
        "by_sportsbook":summarize(out,"sportsbook_key"),
        "by_stat":summarize(out,"stat"),
        "output_csv":str(OUT_CSV),
        "policy":(
            "EXACT_SPORTSBOOK_MATCH_ONLY; CLOSEST_PRETIP_SNAPSHOT; "
            "POSITIVE_LINE_CLV_MEANS_ENTRY_BEAT_CLOSE; "
            "SAME_LINE_PRICE_CLV_USES_NO_VIG_PROBABILITY_MOVE"
        ),
    }
    OUT_JSON.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    main()
