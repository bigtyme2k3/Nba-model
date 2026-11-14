"""
odds_quota.py
-------------
Credit guard for optional Odds API pulls.

Core daily pulls (game odds, player props) always run. Optional pulls (first-quarter
lines, the historical backfill) first check the most recent `x-requests-remaining`
seen by any collector and skip when spending would drop credits below the reserve,
so the core pulls never run out before the monthly reset.
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

RAW_DIR = Path("data/raw")
STATUS_FILES = ("odds_api_status.json", "sportsbook_props_status.json", "period_odds_status.json")
# Credits kept back for the core daily pulls; override with ODDS_API_RESERVE.
DEFAULT_RESERVE = 3000


def reserve() -> int:
    try:
        return int(os.getenv("ODDS_API_RESERVE", DEFAULT_RESERVE))
    except ValueError:
        return DEFAULT_RESERVE


def _when(status: dict) -> datetime:
    for key in ("generated_at_utc", "checked_at_utc"):
        try:
            return datetime.fromisoformat(str(status[key]).replace("Z", "+00:00"))
        except (KeyError, ValueError):
            continue
    return datetime.min


def last_remaining(raw_dir=RAW_DIR) -> int | None:
    """Most recently observed remaining credits across collectors, or None if unknown."""
    seen = []
    for name in STATUS_FILES:
        try:
            status = json.loads((Path(raw_dir) / name).read_text())
            seen.append((_when(status).replace(tzinfo=None), int(float(status["api_requests_remaining"]))))
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return max(seen)[1] if seen else None


def allow(cost: int, remaining: int | None, label: str) -> tuple[bool, str]:
    """May an optional pull spend `cost` credits? Unknown remaining is allowed (first run)."""
    if remaining is None:
        return True, f"{label}: remaining credits unknown; proceeding"
    if remaining - cost < reserve():
        return False, (f"{label}: skipped, {remaining} credits left and {cost} needed "
                       f"would go below the {reserve()} reserve")
    return True, f"{label}: {remaining} credits left, spending up to {cost}"


def remaining_from(resp) -> int | None:
    try:
        return int(float(resp.headers.get("x-requests-remaining")))
    except (TypeError, ValueError):
        return None
