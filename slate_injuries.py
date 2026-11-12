"""Fail-closed, slate-dated injury feed validation.

The pipeline may publish research projections without a healthy injury feed, but
never qualify player props from an unverified or stale report.
"""
from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path


def assess_injury_feed(target_date, raw_dir="data/raw", now=None, max_age_hours=18):
    target_date = str(target_date)
    base = Path(raw_dir)
    status_path = base / "injuries_status.json"
    report_path = base / "injuries_today.csv"
    result = {
        "target_date": target_date,
        "healthy": False,
        "reason": "MISSING_STATUS",
        "age_hours": None,
        "rows": 0,
    }
    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return result
    result["feed_status"] = status.get("status")
    if status.get("target_date") != target_date:
        result["reason"] = "WRONG_SLATE_DATE"
        return result
    if status.get("status") != "ok" or status.get("empty_feed_warning"):
        result["reason"] = "FEED_UNVERIFIED"
        return result
    stamped = status.get("completed_at_utc") or status.get("generated_at_utc")
    try:
        checked = datetime.fromisoformat(str(stamped).replace("Z", "+00:00"))
        checked = checked.astimezone(timezone.utc)
        current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
        age = (current - checked).total_seconds() / 3600
    except (ValueError, TypeError, AttributeError):
        result["reason"] = "INVALID_TIMESTAMP"
        return result
    result["age_hours"] = round(age, 2)
    if not -0.1 <= age <= max_age_hours:
        result["reason"] = "STALE_REPORT"
        return result
    try:
        with report_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            dates = set()
            row_count = 0
            for row in reader:
                row_count += 1
                dates.add(row.get("game_date"))
    except (OSError, csv.Error):
        result["reason"] = "MISSING_REPORT"
        return result
    result["rows"] = row_count
    if not row_count or dates != {target_date}:
        result["reason"] = "REPORT_DATE_MISMATCH"
        return result
    result["healthy"] = True
    result["reason"] = "OK"
    return result
