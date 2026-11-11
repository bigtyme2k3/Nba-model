"""Reconcile a slate plus a bounded, rotating backlog; retain partial results."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

STATUS = Path("data/tracking/reconciliation_status.json")


def pending_dates(target, max_backlog=3):
    from forward_ledger import EVIDENCE_EPOCH
    dates = set()
    resolved_execution = set()
    for archived in Path("data/tracking/execution_daily").glob("*.json"):
        snapshot = json.loads(archived.read_text())
        if snapshot.get("evidence_epoch") == EVIDENCE_EPOCH:
            resolved_execution.update(r.get("ranking_key") for r in snapshot.get("rows", [])
                                      if r.get("outcome") not in {None, "", "PENDING"})
    for path in (Path("data/history/nba_forward_predictions.jsonl"),
                 Path("data/history/nba_pretip_execution_snapshots.jsonl")):
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if "execution" in path.name and row.get("ranking_key") in resolved_execution:
                continue
            day = str(row.get("date", ""))
            if (row.get("evidence_epoch") == EVIDENCE_EPOCH and day and day < target
                    and row.get("outcome") in {None, "", "PENDING"}):
                dates.add(day)
    try:
        attempted = json.loads(STATUS.read_text()).get("last_attempt_by_date", {})
    except (OSError, ValueError):
        attempted = {}
    backlog = sorted(dates, key=lambda day: (attempted.get(day, ""), day))[:max_backlog]
    return [target, *backlog], attempted


def run_step(script, *args):
    try:
        result = subprocess.run([sys.executable, script, *args], capture_output=True, text=True, timeout=180)
        if result.returncode:
            print(f"[WARN] {script}: {result.stderr[-1500:]}")
        return {"step": script, "ok": result.returncode == 0, "returncode": result.returncode}
    except (subprocess.TimeoutExpired, OSError) as exc:
        return {"step": script, "ok": False, "error": str(exc)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True)
    parser.add_argument("--max-backlog", type=int, default=3)
    args = parser.parse_args()
    dates, attempted = pending_dates(args.date, args.max_backlog)
    steps = []
    for day in dates:
        outcomes = [run_step(script, "--date", day) for script in (
            "scrape_scores.py", "collect_result_box.py", "forward_reconciler.py", "execution_reconciler.py")]
        attempted[day] = datetime.now(timezone.utc).isoformat()
        steps.append({"date": day, "steps": outcomes})
    for script in ("clv_reconciler.py", "calibrate_edges.py", "nba_v1_readiness.py"):
        steps.append(run_step(script))
    report = {"status": "PARTIAL" if any(
        not s["ok"] for group in steps for s in group.get("steps", [group])) else "READY",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "target_date": args.date, "steps": steps, "last_attempt_by_date": attempted}
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    STATUS.write_text(json.dumps(report, indent=2) + "\n")
    # Regrading a historical slate must not change today's dashboard date.
    from active_slate_date import resolve_target_date
    os.environ["TARGET"] = resolve_target_date()
    dashboard = run_step("build_dashboard.py")
    report["steps"].append(dashboard)
    if not dashboard["ok"]:
        report["status"] = "PARTIAL"
    STATUS.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
