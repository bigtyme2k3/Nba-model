"""
scrape_period_odds.py
---------------------
First-quarter game lines (spread, total, moneyline) from FanDuel, DraftKings and
Fanatics via The Odds API event-odds endpoint. Collection only: these build the
dataset a first-quarter model would be tested on; nothing here is a bet.

Events come from the day's odds_{date}.csv (written by scrape_odds.py), so no
extra event lookup is spent. Each event costs one credit per market returned
(at most 3). The pull is optional and skips when odds_quota says credits would
drop below the reserve.

Outputs:
  data/raw/period_odds_{date}.csv
  data/raw/period_odds_status.json
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

import odds_quota
from scrape_sportsbook_props import BOOKMAKERS, target_events

API_KEY = os.environ.get("ODDS_API_KEY") or os.environ.get("THE_ODDS_API_KEY") or os.environ.get("THEODDS_API_KEY")
BASE = "https://api.the-odds-api.com/v4/sports/basketball_nba/events"
MARKETS = ["spreads_q1", "totals_q1", "h2h_q1"]
KIND = {"spreads_q1": "SPREAD_Q1", "totals_q1": "TOTAL_Q1", "h2h_q1": "ML_Q1"}
COLUMNS = ["game_date", "event_id", "commence_time", "home_team", "away_team", "book_key", "book",
           "market_key", "market", "side", "line", "price", "no_vig_probability",
           "last_update", "link", "scraped_at"]


def implied(price: int) -> float:
    return 100 / (price + 100) if price > 0 else -price / (-price + 100)


def parse_event(data: dict, target_date: str, scraped_at: str) -> list[dict]:
    """Two-way first-quarter markets only; ties in a first-quarter moneyline are a push."""
    rows = []
    home, away = data.get("home_team"), data.get("away_team")
    for book in data.get("bookmakers", []) or []:
        if book.get("key") not in BOOKMAKERS:
            continue
        for market in book.get("markets", []) or []:
            kind = KIND.get(market.get("key"))
            outcomes = market.get("outcomes", []) or []
            if not kind or len(outcomes) != 2:
                continue
            names = ("Over", "Under") if kind == "TOTAL_Q1" else (home, away)
            pair = {o.get("name"): o for o in outcomes}
            if set(pair) != set(names):
                continue
            try:
                prices = [int(float(pair[n]["price"])) for n in names]
            except (KeyError, TypeError, ValueError):
                continue
            probs = [implied(p) for p in prices]
            for i, name in enumerate(names):
                outcome = pair[name]
                side = (("OVER", "UNDER") if kind == "TOTAL_Q1" else ("HOME", "AWAY"))[i]
                rows.append({
                    "game_date": target_date, "event_id": data.get("id"),
                    "commence_time": data.get("commence_time"), "home_team": home, "away_team": away,
                    "book_key": book["key"], "book": book.get("title", book["key"]),
                    "market_key": market["key"], "market": kind, "side": side,
                    "line": outcome.get("point"), "price": prices[i],
                    "no_vig_probability": probs[i] / sum(probs),
                    "last_update": market.get("last_update") or book.get("last_update"),
                    "link": outcome.get("link") or market.get("link") or book.get("link"),
                    "scraped_at": scraped_at,
                })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True)
    parser.add_argument("--out", default="data/raw")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat()
    status = {"status": "ok", "target_date": args.date, "markets_requested": MARKETS,
              "bookmakers_requested": BOOKMAKERS, "events_found": 0, "events_queried": 0,
              "rows": 0, "http_calls": 0, "generated_at_utc": now, "error": None}

    events = target_events(args.date, args.out)
    status["events_found"] = len(events)
    rows = []
    allowed, reason = odds_quota.allow(len(events) * len(MARKETS), odds_quota.last_remaining(args.out), "first-quarter lines")
    print(f"  {reason}")
    if not API_KEY:
        status.update(status="missing_api_key", error="ODDS_API_KEY not set")
    elif not events:
        status["status"] = "no_events"
    elif not allowed:
        status.update(status="skipped_quota_reserve", error=reason)
    else:
        for event in events:
            try:
                resp = requests.get(f"{BASE}/{event['event_id']}/odds", timeout=25, params={
                    "apiKey": API_KEY, "bookmakers": ",".join(BOOKMAKERS), "markets": ",".join(MARKETS),
                    "oddsFormat": "american", "dateFormat": "iso", "includeLinks": "true"})
                status["http_calls"] += 1
                remaining = odds_quota.remaining_from(resp)
                if remaining is not None:
                    status["api_requests_remaining"] = remaining
                    status["api_requests_last"] = resp.headers.get("x-requests-last")
                if resp.status_code == 422:
                    continue
                resp.raise_for_status()
                status["events_queried"] += 1
                rows.extend(parse_event(resp.json(), args.date, now))
            except Exception as exc:
                status["error"] = f"{event['event_id']}: {exc}"
                print(f"  [WARN] {status['error']}")
        status["status"] = "ok" if rows else "empty_markets"

    df = pd.DataFrame(rows, columns=COLUMNS)
    df.to_csv(out / f"period_odds_{args.date}.csv", index=False)
    status["rows"] = len(df)
    status["markets_returned"] = sorted(df["market_key"].dropna().unique().tolist())
    (out / "period_odds_status.json").write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
