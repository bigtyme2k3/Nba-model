"""
v2/compare.py
-------------
Forward comparison of the V2 shadow against V1 on graded props: same rows, same
actual outcomes, MAE per stat. Rows come from the forward ledger, where each prop
carries V1's projection and, when available, the V2 shadow's (v2_projection).

Usage:
    python -m v2.compare
Output:
    data/tracking/v2_forward_comparison.json
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

LEDGER = Path("data/history/nba_forward_predictions.jsonl")
OUT = Path("data/tracking/v2_forward_comparison.json")


def compare(rows: pd.DataFrame) -> dict:
    for c in ("projection", "v2_projection", "actual"):
        rows[c] = pd.to_numeric(rows.get(c), errors="coerce")
    graded = rows[(rows["market"] == "PROP") & rows["projection"].notna()
                  & rows["v2_projection"].notna() & rows["actual"].notna()]
    out = {}
    for stat, g in graded.groupby("stat"):
        out[stat] = {"rows": int(len(g)), "dates": int(g["date"].nunique()),
                     "v1_mae": float((g["actual"] - g["projection"]).abs().mean()),
                     "v2_mae": float((g["actual"] - g["v2_projection"]).abs().mean()),
                     "v2_better_share": float(((g["actual"] - g["v2_projection"]).abs()
                                               < (g["actual"] - g["projection"]).abs()).mean())}
    return out


def main():
    rows = pd.read_json(LEDGER, lines=True) if LEDGER.exists() else pd.DataFrame()
    report = {"generated_at_utc": datetime.now(timezone.utc).isoformat(),
              "by_stat": compare(rows) if not rows.empty and "v2_projection" in rows else {}}
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
