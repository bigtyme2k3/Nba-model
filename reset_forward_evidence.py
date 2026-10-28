"""
reset_forward_evidence.py
-------------------------
One-time clean evidence epoch migration for NBA V1.

Why:
The existing forward ledger contained legacy pre-clean predictions from before
the data/model repairs (including corrupted totals). Those rows remain preserved
for audit, but they must never contribute to V1 ROI, Brier, or CLV.

This script:
1) archives the current canonical ledger to a legacy audit file
2) rebuilds the canonical ledger from a specified corrected prediction artifact
3) resets forward performance/certified outputs for the clean evidence epoch
"""
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from forward_ledger import LEDGER, EVIDENCE_EPOCH, flatten

ARCHIVE = Path("data/history/nba_forward_predictions_legacy_pre_clean_v1.jsonl")
PERF = Path("data/tracking/nba_forward_performance.json")
CERT = Path("data/tracking/nba_forward_certified.csv")
STATUS = Path("data/tracking/evidence_epoch_reset_status.json")


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    args=parser.parse_args()

    pred_path=Path(args.predictions)
    if not pred_path.exists():
        raise SystemExit(f"Prediction file missing: {pred_path}")

    old_rows=0
    if LEDGER.exists():
        old_rows=sum(1 for line in LEDGER.read_text(encoding="utf-8").splitlines() if line.strip())
        ARCHIVE.parent.mkdir(parents=True,exist_ok=True)
        if not ARCHIVE.exists():
            shutil.copy2(LEDGER,ARCHIVE)

    data=json.loads(pred_path.read_text(encoding="utf-8"))
    rows=flatten(data)
    # flatten() stamps the current evidence epoch.
    for row in rows:
        row["evidence_epoch"]=EVIDENCE_EPOCH

    LEDGER.parent.mkdir(parents=True,exist_ok=True)
    with LEDGER.open("w",encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row,allow_nan=False)+"\n")

    PERF.parent.mkdir(parents=True,exist_ok=True)
    PERF.write_text(json.dumps({
        "generated_at_utc":datetime.now(timezone.utc).isoformat(),
        "evidence_epoch":EVIDENCE_EPOCH,
        "ledger_rows":len(rows),
        "certified_rows":0,
        "bet_rows":0,
        "wins":0,
        "losses":0,
        "pushes":0,
        "hit_rate":None,
        "profit_units":0.0,
        "roi":None,
        "probability_calibration":{"rows":0,"brier":None,"bins":[]},
        "by_market":{},
        "by_sportsbook":{},
        "by_prop_stat":{},
        "policy":"CLEAN_EVIDENCE_EPOCH_RESET; NO_LEGACY_ROWS_COUNT",
    },indent=2)+"\n",encoding="utf-8")
    pd.DataFrame().to_csv(CERT,index=False)

    bets=sum(r.get("decision")=="BET" for r in rows)
    status={
        "status":"READY",
        "generated_at_utc":datetime.now(timezone.utc).isoformat(),
        "evidence_epoch":EVIDENCE_EPOCH,
        "source_prediction":str(pred_path),
        "source_prediction_generated_at_utc":data.get("generated"),
        "legacy_rows_archived":old_rows,
        "legacy_archive":str(ARCHIVE),
        "clean_rows_written":len(rows),
        "clean_bet_rows":bets,
        "canonical_ledger":str(LEDGER),
        "legacy_rows_excluded_from_metrics":True,
    }
    STATUS.write_text(json.dumps(status,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(status,indent=2))


if __name__=="__main__":
    main()
