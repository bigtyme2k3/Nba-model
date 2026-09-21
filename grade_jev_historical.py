#!/usr/bin/env python3
"""Grade a blind Jev shadow experiment against resolved WNBA outcomes.

The Jev input must not contain outcome/actual fields. This script runs only
after Jev has frozen its shadow decisions.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


def norm(v: Any) -> str:
    return str(v or "").strip().upper()


def fnum(v: Any) -> float | None:
    try:
        x = float(v)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def key(date: str, player: Any, stat: Any, side: Any, line: Any) -> tuple[str, str, str, str, float | None]:
    return (date[:10], norm(player), norm(stat), norm(side), fnum(line))


def profit_units(outcome: str, odds: float | None) -> float | None:
    result = norm(outcome)
    if result == "LOSS":
        return -1.0
    if result in {"PUSH", "VOID"}:
        return 0.0
    if result != "WIN" or odds is None or odds == 0:
        return None
    return (odds / 100.0) if odds > 0 else (100.0 / abs(odds))


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    resolved = [r for r in rows if norm(r.get("outcome")) in {"WIN", "LOSS", "PUSH", "VOID"}]
    wins = sum(norm(r.get("outcome")) == "WIN" for r in resolved)
    losses = sum(norm(r.get("outcome")) == "LOSS" for r in resolved)
    pushes = sum(norm(r.get("outcome")) in {"PUSH", "VOID"} for r in resolved)
    denom = wins + losses
    pnl = [r.get("profit_units") for r in resolved if r.get("profit_units") is not None]
    return {
        "rows": len(rows),
        "resolved": len(resolved),
        "wins": wins,
        "losses": losses,
        "pushes_or_voids": pushes,
        "hit_rate": round(wins / denom, 4) if denom else None,
        "profit_units": round(sum(pnl), 4) if pnl else None,
        "roi_per_wager": round(sum(pnl) / len(pnl), 4) if pnl else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shadow", required=True)
    ap.add_argument("--truth", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    shadow = json.loads(Path(args.shadow).read_text(encoding="utf-8"))
    target = str(shadow.get("target_date") or "")[:10]
    decisions = shadow.get("decisions") or []

    truth: dict[tuple[str, str, str, str, float | None], dict[str, Any]] = {}
    with Path(args.truth).open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if str(row.get("date") or "")[:10] != target:
                continue
            k = key(target, row.get("player"), row.get("stat"), row.get("signal"), row.get("line"))
            # The graded ledger may contain duplicates from repeated snapshots.
            # Preserve the first resolved record; outcome is the only field used
            # for correctness and market price is used for a rough unit P&L.
            if k not in truth and norm(row.get("outcome")) in {"WIN", "LOSS", "PUSH", "VOID"}:
                truth[k] = row

    graded: list[dict[str, Any]] = []
    unmatched: list[dict[str, Any]] = []
    for d in decisions:
        c = d.get("candidate") or {}
        side = c.get("side") or c.get("signal")
        stat = c.get("stat") or c.get("market")
        k = key(target, c.get("player"), stat, side, c.get("line"))
        t = truth.get(k)
        if not t:
            unmatched.append({
                "shadow_id": d.get("shadow_id"),
                "player": c.get("player"),
                "stat": stat,
                "side": side,
                "line": c.get("line"),
            })
            continue
        odds = fnum(c.get("odds") or c.get("american_odds") or t.get("american_odds"))
        outcome = norm(t.get("outcome"))
        graded.append({
            "shadow_id": d.get("shadow_id"),
            "player": c.get("player"),
            "game": c.get("game"),
            "stat": stat,
            "side": side,
            "line": c.get("line"),
            "odds": odds,
            "actual": fnum(t.get("actual")),
            "outcome": outcome,
            "profit_units": profit_units(outcome, odds),
            "jev_action": (d.get("jev") or {}).get("action"),
            "jev_action_probabilities": (d.get("jev") or {}).get("action_probabilities"),
            "jev_trust": (d.get("jev") or {}).get("trust"),
            "jev_trust_probabilities": (d.get("jev") or {}).get("trust_probabilities"),
            "jev_anomaly_probability": (d.get("jev") or {}).get("anomaly_probability"),
            "jev_review_probability": (d.get("jev") or {}).get("review_probability"),
        })

    by_action: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_trust: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in graded:
        by_action[str(r.get("jev_action") or "UNKNOWN")].append(r)
        by_trust[str(r.get("jev_trust") or "UNKNOWN")].append(r)

    bet = [r for r in graded if r.get("jev_action") == "BET"]
    bet_lean = [r for r in graded if r.get("jev_action") in {"BET", "LEAN"}]
    watch_pass = [r for r in graded if r.get("jev_action") in {"WATCH", "PASS"}]
    anomaly_hi = [r for r in graded if (fnum(r.get("jev_anomaly_probability")) or 0) >= 0.5]
    anomaly_lo = [r for r in graded if (fnum(r.get("jev_anomaly_probability")) or 0) < 0.5]
    review_hi = [r for r in graded if (fnum(r.get("jev_review_probability")) or 0) >= 0.5]
    review_lo = [r for r in graded if (fnum(r.get("jev_review_probability")) or 0) < 0.5]

    report = {
        "version": shadow.get("version"),
        "experiment": "JEV_BLIND_HISTORICAL_SHADOW",
        "target_date": target,
        "blindness_contract": (
            "Jev input contained only pre-result model/market fields. "
            "Outcome and actual were joined only after Jev decisions were frozen."
        ),
        "source_shadow": args.shadow,
        "truth_source": args.truth,
        "model": shadow.get("model"),
        "usage": shadow.get("usage"),
        "matched": len(graded),
        "unmatched": len(unmatched),
        "baseline_all_candidates": summarize(graded),
        "jev_bet_only": summarize(bet),
        "jev_bet_plus_lean": summarize(bet_lean),
        "jev_watch_plus_pass": summarize(watch_pass),
        "by_action": {k: summarize(v) for k, v in sorted(by_action.items())},
        "by_trust": {k: summarize(v) for k, v in sorted(by_trust.items())},
        "anomaly_ge_050": summarize(anomaly_hi),
        "anomaly_lt_050": summarize(anomaly_lo),
        "review_ge_050": summarize(review_hi),
        "review_lt_050": summarize(review_lo),
        "rows": graded,
        "unmatched_rows": unmatched,
        "research_only": True,
        "production_ready": False,
        "promotion_policy": (
            "One historical blind slate is diagnostic only. Require prospective "
            "shadow evidence across materially more resolved decisions before any "
            "production mutation."
        ),
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in (
        "matched", "unmatched", "baseline_all_candidates", "jev_bet_only",
        "jev_bet_plus_lean", "jev_watch_plus_pass"
    )}, indent=2))


if __name__ == "__main__":
    main()
