"""
backfill_historical_lines.py
----------------------------
Near-closing spread and total lines for a past season from The Odds API
historical endpoint (paid plans), for backtesting the game models against the
market.

Snapshot plan, per game day (America/New_York):
  pass 1  five minutes before the day's first tip (covers every game that day)
  pass 2  18:55 ET on days whose first tip is before 18:30, so evening games are
          not priced hours early (only with --second-pass)
Each game keeps the latest pregame snapshot before its tip.

Cost: 10 x markets (2) x regions (3 books = 1) = 20 credits per snapshot.
Every snapshot response is cached in data/raw/historical_lines/, so reruns
never pay twice. Spending stops at --max-credits and before odds_quota's
reserve.

Outputs:
  data/raw/historical_lines/snap_<timestamp>.json   raw responses (cache)
  data/raw/historical_lines_<season>.csv            one row per game: median
                                                    home spread / total across books
  data/raw/historical_lines_status.json
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import median
from zoneinfo import ZoneInfo

import pandas as pd
import requests

import odds_quota
from game_markets import offers_from_event
from season_filters import nba_season_start_year
from teams import normalize_team

API_KEY = os.environ.get("ODDS_API_KEY") or os.environ.get("THE_ODDS_API_KEY") or os.environ.get("THEODDS_API_KEY")
URL = "https://api.the-odds-api.com/v4/historical/sports/basketball_nba/odds"
BOOKMAKERS = "fanduel,draftkings,fanatics"
MARKETS = "spreads,totals"
CREDITS_PER_SNAPSHOT = 20
ET = ZoneInfo("America/New_York")
RAW = Path("data/raw")


def schedule_tips(season: int, raw_dir=RAW, season_types=(2,)) -> pd.Series:
    """Tip times (ET) of the season's games from the hoopR schedule (labelled by end year)."""
    path = Path(raw_dir) / f"hoopr_schedule_{season + 1}.csv"
    s = pd.read_csv(path, low_memory=False)
    s = s[s["season_type"].isin(season_types)]
    return pd.to_datetime(s["date"], utc=True).dt.tz_convert(ET).sort_values()


def snapshot_plan(tips: pd.Series, second_pass: bool) -> list[datetime]:
    first = tips.groupby(tips.dt.date).min()
    plan = [t - timedelta(minutes=5) for t in first]
    if second_pass:
        for t in first:
            if t.hour + t.minute / 60 < 18.5:
                plan.append(t.replace(hour=18, minute=55, second=0, microsecond=0))
    return plan  # pass 1 first, so a budget cap never starves whole days


def iso_z(t: datetime) -> str:
    return t.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def fetch(plan, cache: Path, max_credits: int, status: dict):
    cache.mkdir(parents=True, exist_ok=True)
    remaining = odds_quota.last_remaining()
    for t in plan:
        ts = iso_z(t)
        path = cache / f"snap_{ts.replace(':', '')}.json"
        if path.exists():
            status["cached"] += 1
            continue
        if status["credits_spent"] + CREDITS_PER_SNAPSHOT > max_credits:
            status["stopped"] = f"credit cap {max_credits} reached"
            break
        allowed, reason = odds_quota.allow(CREDITS_PER_SNAPSHOT, remaining, "historical backfill")
        if not allowed:
            status["stopped"] = reason
            break
        resp = requests.get(URL, timeout=30, params={
            "apiKey": API_KEY, "bookmakers": BOOKMAKERS, "markets": MARKETS,
            "oddsFormat": "american", "dateFormat": "iso", "date": ts})
        remaining = odds_quota.remaining_from(resp) or remaining
        status["api_requests_remaining"] = remaining
        if resp.status_code in (401, 402, 403, 429):
            status["stopped"] = f"HTTP {resp.status_code}: {resp.text[:200]}"
            break
        resp.raise_for_status()
        status["credits_spent"] += int(float(resp.headers.get("x-requests-last") or CREDITS_PER_SNAPSHOT))
        status["fetched"] += 1
        path.write_text(json.dumps({"requested": ts, **resp.json()}))


def latest_offers(cache: Path, season: int) -> pd.DataFrame:
    """Latest pregame quote per game/book/market/side across the cached snapshots."""
    offers = []
    for path in sorted(cache.glob("snap_*.json")):
        snap = json.loads(path.read_text())
        taken = snap.get("timestamp") or snap.get("requested")
        taken_dt = datetime.fromisoformat(taken.replace("Z", "+00:00"))
        for event in snap.get("data", []) or []:
            tip = datetime.fromisoformat(event["commence_time"].replace("Z", "+00:00"))
            if tip <= taken_dt or nba_season_start_year(tip.astimezone(ET).date().isoformat()) != season:
                continue  # in-play or another season
            for o in offers_from_event(event, taken):
                offers.append({**o, "home_team": event["home_team"], "away_team": event["away_team"],
                               "snapshot": taken_dt, "tip": tip})
    if not offers:
        return pd.DataFrame()
    df = pd.DataFrame(offers).sort_values("snapshot")
    return df.drop_duplicates(["event_id", "book_key", "market", "side"], keep="last")


def consolidate(cache: Path, season: int) -> pd.DataFrame:
    """One row per game: median home spread and total across books."""
    df = latest_offers(cache, season)
    if df.empty:
        return df
    games = []
    for event_id, g in df.groupby("event_id"):
        spread = g[(g["market"] == "SPREAD") & (g["side"] == "HOME")]["line"].tolist()
        total = g[(g["market"] == "TOTAL") & (g["side"] == "OVER")]["line"].tolist()
        tip = g["tip"].iloc[0]
        games.append({
            "event_id": event_id,
            "game_date": tip.astimezone(ET).date().isoformat(),
            "commence_time": tip.isoformat(),
            "home_team": normalize_team(g["home_team"].iloc[0]),
            "away_team": normalize_team(g["away_team"].iloc[0]),
            "close_spread_home": median(spread) if spread else None,
            "close_total": median(total) if total else None,
            "n_books": int(g["book_key"].nunique()),
            "snapshot_utc": g["snapshot"].max().isoformat(),
            "hours_before_tip": round((tip - g["snapshot"].max()).total_seconds() / 3600, 2),
        })
    return pd.DataFrame(games).sort_values(["game_date", "commence_time"]).reset_index(drop=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--season", type=int, default=2025, help="Season start year (2025 = 2025-26)")
    p.add_argument("--max-credits", type=int, default=3400)
    p.add_argument("--second-pass", action="store_true")
    p.add_argument("--plan-only", action="store_true", help="Print the plan and cost; no API calls")
    p.add_argument("--raw", default=str(RAW))
    a = p.parse_args()
    raw = Path(a.raw)
    cache = raw / "historical_lines"

    plan = snapshot_plan(schedule_tips(a.season, raw), a.second_pass)
    cached = sum((cache / f"snap_{iso_z(t).replace(':', '')}.json").exists() for t in plan)
    status = {"season": a.season, "snapshots_planned": len(plan), "already_cached": cached,
              "estimated_new_credits": (len(plan) - cached) * CREDITS_PER_SNAPSHOT,
              "max_credits": a.max_credits, "reserve": odds_quota.reserve(),
              "fetched": 0, "cached": 0, "credits_spent": 0, "stopped": None,
              "generated_at_utc": datetime.now(timezone.utc).isoformat()}
    if a.plan_only:
        print(json.dumps(status, indent=2))
        return
    if not API_KEY:
        status["stopped"] = "missing_api_key"
    else:
        fetch(plan, cache, a.max_credits, status)

    lines = consolidate(cache, a.season)
    out = raw / f"historical_lines_{a.season}.csv"
    if not lines.empty:
        lines.to_csv(out, index=False)
    status.update(games_with_lines=int(len(lines)),
                  median_hours_before_tip=float(lines["hours_before_tip"].median()) if len(lines) else None,
                  output=str(out))
    (raw / "historical_lines_status.json").write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
