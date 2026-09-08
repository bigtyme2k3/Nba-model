"""
active_slate_date.py
--------------------
Resolves the NBA betting slate date using Eastern Time instead of UTC.

Why:
GitHub Actions runs on UTC. At 8:50 PM ET, UTC is already tomorrow, which would
flip the dashboard to the next day's slate while current games are still live.

Default behavior:
- Use America/New_York date.
- Do not flip to tomorrow until after local midnight.
- If a manual --date is supplied, return it unchanged.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from zoneinfo import ZoneInfo

DEFAULT_TZ = "America/New_York"

# NBA regular season tips off mid-October and the Finals wrap up mid-June.
SEASON_START_MONTH_DAY = (10, 1)   # preseason/early slate data can start flowing
SEASON_END_MONTH_DAY = (6, 30)


def resolve_target_date(manual_date: str = "", timezone_name: str = DEFAULT_TZ) -> str:
    manual_date = str(manual_date or "").strip()
    if manual_date:
        return manual_date
    return datetime.now(ZoneInfo(timezone_name)).strftime("%Y-%m-%d")


def resolve_year(timezone_name: str = DEFAULT_TZ) -> str:
    return datetime.now(ZoneInfo(timezone_name)).strftime("%Y")


def in_season(manual_date: str = "", timezone_name: str = DEFAULT_TZ) -> bool:
    """True roughly Oct 1 - Jun 30. Used by workflows to skip no-op runs in the off-season."""
    target = resolve_target_date(manual_date, timezone_name)
    month = int(target[5:7])
    return month >= SEASON_START_MONTH_DAY[0] or month <= SEASON_END_MONTH_DAY[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default="")
    ap.add_argument("--tz", default=DEFAULT_TZ)
    ap.add_argument("--year", action="store_true")
    ap.add_argument("--in-season", action="store_true", help="Print 'true'/'false' and exit")
    args = ap.parse_args()
    if args.in_season:
        print("true" if in_season(args.date, args.tz) else "false")
    elif args.year:
        print(resolve_year(args.tz))
    else:
        print(resolve_target_date(args.date, args.tz))


if __name__ == "__main__":
    main()
