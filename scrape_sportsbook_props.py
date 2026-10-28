"""
scrape_sportsbook_props.py
--------------------------
Primary NBA player-prop market collector for FanDuel, DraftKings, and Fanatics
using The Odds API event-odds endpoint.

Why this exists:
- PrizePicks can block shared CI IPs with anti-bot protection.
- The Odds API already powers this repo's game odds.
- Event-specific player props include sportsbook price + line, which allows
  per-book EV, no-vig comparison, and CLV tracking.

Outputs
-------
data/raw/sportsbook_props_today.csv
data/raw/sportsbook_props_YYYY-MM-DD.csv
data/raw/sportsbook_props_status.json

Only events whose local America/New_York slate date matches --date are queried.
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests

API_KEY = os.getenv("ODDS_API_KEY")
SPORT = "basketball_nba"
BASE = "https://api.the-odds-api.com/v4/sports"
BOOKMAKERS = ["fanduel", "draftkings", "fanatics"]
MARKETS = [
    "player_points",
    "player_rebounds",
    "player_assists",
    "player_threes",
    "player_points_rebounds_assists",
]
STAT_MAP = {
    "player_points": "pts",
    "player_rebounds": "reb",
    "player_assists": "ast",
    "player_threes": "threes",
    "player_points_rebounds_assists": "pra",
}
COLUMNS = [
    "game_date","event_id","commence_time","home_team","away_team",
    "player","stat","market_key","side","line","price",
    "book","book_key","last_update","source","scraped_at"
]
ET = ZoneInfo("America/New_York")


def empty_df():
    return pd.DataFrame(columns=COLUMNS)


def american_implied(price):
    try:
        p = float(price)
    except Exception:
        return None
    if p > 0:
        return 100.0 / (p + 100.0)
    if p < 0:
        return (-p) / ((-p) + 100.0)
    return None


def target_events(target_date: str, out_dir: str):
    paths = [
        Path(out_dir) / f"odds_{target_date}.csv",
        Path(out_dir) / "odds_today.csv",
    ]
    path = next((p for p in paths if p.exists()), None)
    if path is None:
        return []
    try:
        df = pd.read_csv(path)
    except Exception:
        return []
    if df.empty:
        return []

    rows = []
    for _, r in df.iterrows():
        event_id = str(r.get("game_id", "") or "").strip()
        commence = str(r.get("commence_time", "") or "").strip()
        if not event_id or not commence:
            continue
        try:
            tip_utc = datetime.fromisoformat(commence.replace("Z", "+00:00")).astimezone(timezone.utc)
            dt = tip_utc.astimezone(ET)
        except Exception:
            continue
        if dt.date().isoformat() != target_date:
            continue
        # Pregame model only: The Odds API also serves in-play events/markets.
        if tip_utc <= datetime.now(timezone.utc):
            continue
        rows.append({
            "event_id": event_id,
            "commence_time": commence,
            "home_team": str(r.get("home_team", "") or "").strip(),
            "away_team": str(r.get("away_team", "") or "").strip(),
        })
    # One event id per actual game.
    seen = set()
    out = []
    for row in rows:
        if row["event_id"] not in seen:
            out.append(row)
            seen.add(row["event_id"])
    return out


def fetch_event_props(event: dict, status: dict):
    if not API_KEY:
        raise RuntimeError("ODDS_API_KEY not set")

    url = f"{BASE}/{SPORT}/events/{event['event_id']}/odds"
    params = {
        "apiKey": API_KEY,
        "bookmakers": ",".join(BOOKMAKERS),
        "markets": ",".join(MARKETS),
        "oddsFormat": "american",
        "dateFormat": "iso",
    }
    resp = requests.get(url, params=params, timeout=25)
    status["http_calls"] += 1
    if resp.headers.get("x-requests-used") is not None:
        status["api_requests_used"] = resp.headers.get("x-requests-used")
    if resp.headers.get("x-requests-remaining") is not None:
        status["api_requests_remaining"] = resp.headers.get("x-requests-remaining")
    if resp.headers.get("x-requests-last") is not None:
        status["api_requests_last"] = resp.headers.get("x-requests-last")

    if resp.status_code == 422:
        return None
    resp.raise_for_status()
    return resp.json()


def parse_event(data: dict, target_date: str):
    if not data:
        return []
    rows = []
    now = datetime.now(timezone.utc).isoformat()
    event_id = str(data.get("id", ""))
    commence = str(data.get("commence_time", ""))
    home = str(data.get("home_team", ""))
    away = str(data.get("away_team", ""))

    for bm in data.get("bookmakers", []) or []:
        book_key = str(bm.get("key", ""))
        if book_key not in BOOKMAKERS:
            continue
        book = str(bm.get("title", book_key))
        for market in bm.get("markets", []) or []:
            market_key = str(market.get("key", ""))
            stat = STAT_MAP.get(market_key)
            if not stat:
                continue
            last_update = market.get("last_update") or bm.get("last_update")
            for outcome in market.get("outcomes", []) or []:
                side = str(outcome.get("name", "")).upper()
                player = str(outcome.get("description", "") or "").strip()
                line = outcome.get("point")
                price = outcome.get("price")
                if side not in {"OVER", "UNDER"} or not player or line is None or price is None:
                    continue
                try:
                    line = float(line)
                    price = int(float(price))
                except Exception:
                    continue
                rows.append({
                    "game_date": target_date,
                    "event_id": event_id,
                    "commence_time": commence,
                    "home_team": home,
                    "away_team": away,
                    "player": player,
                    "stat": stat,
                    "market_key": market_key,
                    "side": side,
                    "line": line,
                    "price": price,
                    "book": book,
                    "book_key": book_key,
                    "last_update": last_update,
                    "source": "the-odds-api",
                    "scraped_at": now,
                })
    return rows


def add_no_vig(df: pd.DataFrame):
    if df.empty:
        for col in ["implied_probability","no_vig_probability","pair_overround"]:
            df[col] = pd.Series(dtype=float)
        return df

    df = df.copy()
    df["implied_probability"] = df["price"].map(american_implied)
    df["no_vig_probability"] = pd.NA
    df["pair_overround"] = pd.NA

    keys = ["event_id","player","stat","book_key","line"]
    for _, idx in df.groupby(keys).groups.items():
        group = df.loc[list(idx)]
        over = group[group["side"] == "OVER"]
        under = group[group["side"] == "UNDER"]
        if over.empty or under.empty:
            continue
        po = american_implied(over.iloc[0]["price"])
        pu = american_implied(under.iloc[0]["price"])
        if po is None or pu is None or (po + pu) <= 0:
            continue
        total = po + pu
        df.loc[over.index, "no_vig_probability"] = po / total
        df.loc[under.index, "no_vig_probability"] = pu / total
        df.loc[group.index, "pair_overround"] = total - 1.0

    df["no_vig_probability"] = pd.to_numeric(df["no_vig_probability"], errors="coerce")
    df["pair_overround"] = pd.to_numeric(df["pair_overround"], errors="coerce")
    return df


def save(df, target_date, out_dir):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    dated = Path(out_dir) / f"sportsbook_props_{target_date}.csv"
    today = Path(out_dir) / "sportsbook_props_today.csv"
    df.to_csv(dated, index=False)
    df.to_csv(today, index=False)
    return dated, today


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True)
    parser.add_argument("--out", default="data/raw")
    args = parser.parse_args()

    status = {
        "status": "unknown",
        "target_date": args.date,
        "bookmakers_requested": BOOKMAKERS,
        "markets_requested": MARKETS,
        "events_found": 0,
        "events_queried": 0,
        "rows": 0,
        "unique_players": 0,
        "books_returned": [],
        "markets_returned": [],
        "http_calls": 0,
        "api_requests_used": None,
        "api_requests_remaining": None,
        "api_requests_last": None,
        "error": None,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    events = target_events(args.date, args.out)
    status["events_found"] = len(events)
    rows = []

    if not API_KEY:
        status["status"] = "missing_api_key"
        status["error"] = "ODDS_API_KEY not set"
    else:
        try:
            for event in events:
                data = fetch_event_props(event, status)
                status["events_queried"] += 1
                rows.extend(parse_event(data, args.date))
            df = pd.DataFrame(rows, columns=COLUMNS) if rows else empty_df()
            df = add_no_vig(df)
            status["rows"] = int(len(df))
            status["unique_players"] = int(df["player"].nunique()) if not df.empty else 0
            status["books_returned"] = sorted(df["book_key"].dropna().unique().tolist()) if not df.empty else []
            status["markets_returned"] = sorted(df["market_key"].dropna().unique().tolist()) if not df.empty else []
            status["status"] = "ok" if not df.empty else ("no_events" if not events else "empty_markets")
        except Exception as exc:
            status["status"] = "error"
            status["error"] = str(exc)
            df = pd.DataFrame(rows, columns=COLUMNS) if rows else empty_df()
            df = add_no_vig(df)

    if "df" not in locals():
        df = add_no_vig(empty_df())

    save(df, args.date, args.out)
    Path(args.out, "sportsbook_props_status.json").write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps(status, indent=2))
    if not df.empty:
        sample = df.sort_values(["player","stat","book_key","line","side"]).head(30)
        print(sample[["player","stat","side","line","price","book_key","no_vig_probability"]].to_string(index=False))


if __name__ == "__main__":
    main()
