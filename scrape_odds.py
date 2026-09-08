"""
scrape_odds.py
--------------
Fetches NBA spreads, totals, and moneylines from The Odds API.

Quota-safe behavior:
If the API key is missing/invalid or monthly credits are exhausted, this script
writes empty-but-valid CSV outputs and a status JSON instead of failing the full
pipeline. Downstream model/dashboard steps can continue using historical and
committed data.
"""

import argparse
import json
import os
from datetime import date, datetime, timezone

import pandas as pd
import requests

API_KEY  = os.getenv("ODDS_API_KEY")
SPORT    = "basketball_nba"
REGIONS  = "us"
MARKETS  = "h2h,spreads,totals"
ODDS_FMT = "american"
BASE_URL = "https://api.the-odds-api.com/v4/sports"

COLUMNS = [
    "game_date", "game_id", "commence_time", "home_team", "away_team",
    "spread_home", "spread_home_juice", "total", "total_over_juice",
    "ml_home", "ml_away", "num_books", "source", "scraped_at"
]


def empty_df() -> pd.DataFrame:
    return pd.DataFrame(columns=COLUMNS)


def classify_api_error(resp: requests.Response) -> str:
    try:
        payload = resp.json()
    except Exception:
        payload = {}
    msg = str(payload.get("message") or payload.get("error") or resp.text or "").lower()
    if resp.status_code == 401:
        return "invalid_api_key_or_quota_exhausted"
    if resp.status_code == 429:
        return "rate_limited_or_quota_exhausted"
    if "quota" in msg or "requests" in msg or "credit" in msg:
        return "quota_exhausted"
    return f"http_{resp.status_code}"


def fetch_odds(status: dict):
    if not API_KEY:
        status["status"] = "missing_api_key"
        status["error"] = "ODDS_API_KEY not set."
        print("  [WARN] ODDS_API_KEY not set. Continuing with empty odds file.")
        return []

    url = f"{BASE_URL}/{SPORT}/odds"
    params = {
        "apiKey":     API_KEY,
        "regions":    REGIONS,
        "markets":    MARKETS,
        "oddsFormat": ODDS_FMT,
    }

    resp = requests.get(url, params=params, timeout=15)

    remaining = resp.headers.get("x-requests-remaining", "?")
    used      = resp.headers.get("x-requests-used", "?")
    status["api_requests_used"] = used
    status["api_requests_remaining"] = remaining
    status["http_status"] = resp.status_code
    print(f"  API quota — used: {used} | remaining: {remaining}")

    if resp.status_code in (401, 402, 403, 429):
        reason = classify_api_error(resp)
        status["status"] = reason
        status["error"] = f"The Odds API unavailable: {reason}"
        print(f"  [WARN] The Odds API unavailable ({reason}). Continuing with empty odds file.")
        return []

    if resp.status_code == 422:
        print("  [INFO] No NBA games currently available.")
        status["status"] = "empty"
        return []

    try:
        resp.raise_for_status()
    except Exception as exc:
        status["status"] = classify_api_error(resp)
        status["error"] = str(exc)
        print(f"  [WARN] Odds request failed: {exc}. Continuing with empty odds file.")
        return []

    status["status"] = "ok"
    return resp.json()


def parse_games(games: list, target_date: str) -> pd.DataFrame:
    rows = []
    for g in games:
        home = g.get("home_team","")
        away = g.get("away_team","")
        start= g.get("commence_time","")
        gid  = g.get("id","")

        spread_home_lines, spread_home_prices = [], []
        total_points, total_over_prices       = [], []
        ml_home_prices, ml_away_prices        = [], []

        for bm in g.get("bookmakers", []):
            for mkt in bm.get("markets", []):
                key      = mkt.get("key")
                outcomes = {o["name"]: o for o in mkt.get("outcomes", [])}

                if key == "spreads":
                    h = outcomes.get(home, {})
                    if h.get("point") is not None:
                        spread_home_lines.append(h["point"])
                    if h.get("price") is not None:
                        spread_home_prices.append(h["price"])

                elif key == "totals":
                    ov = outcomes.get("Over", {})
                    if ov.get("point") is not None:
                        total_points.append(ov["point"])
                    if ov.get("price") is not None:
                        total_over_prices.append(ov["price"])

                elif key == "h2h":
                    if outcomes.get(home, {}).get("price") is not None:
                        ml_home_prices.append(outcomes[home]["price"])
                    if outcomes.get(away, {}).get("price") is not None:
                        ml_away_prices.append(outcomes[away]["price"])

        def avg(lst): return round(sum(lst)/len(lst), 2) if lst else None

        rows.append({
            "game_date":        target_date,
            "game_id":          gid,
            "commence_time":    start,
            "home_team":        home,
            "away_team":        away,
            "spread_home":      avg(spread_home_lines),
            "spread_home_juice":avg(spread_home_prices),
            "total":            avg(total_points),
            "total_over_juice": avg(total_over_prices),
            "ml_home":          avg(ml_home_prices),
            "ml_away":          avg(ml_away_prices),
            "num_books":        len(g.get("bookmakers", [])),
            "source":           "the-odds-api",
            "scraped_at":       datetime.now(timezone.utc).isoformat(),
        })

    return pd.DataFrame(rows, columns=COLUMNS)


def save_outputs(df: pd.DataFrame, target: str, out_dir: str) -> None:
    if df.empty:
        df = empty_df()

    today_path = os.path.join(out_dir, "odds_today.csv")
    dated_path = os.path.join(out_dir, f"odds_{target}.csv")
    hist_path  = os.path.join(out_dir, "odds_historical.csv")

    df.to_csv(today_path, index=False)
    df.to_csv(dated_path, index=False)

    if os.path.exists(hist_path) and not df.empty:
        hist = pd.read_csv(hist_path)
        if "game_date" in hist.columns:
            hist = hist[hist["game_date"] != target]
        pd.concat([hist, df], ignore_index=True).to_csv(hist_path, index=False)
    elif not df.empty:
        df.to_csv(hist_path, index=False)

    print(f"\n  Saved → {today_path}")
    print(f"  Saved → {dated_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=str(date.today()))
    parser.add_argument("--out",  default="data/raw")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    target = args.date
    status = {
        "status": "unknown",
        "target_date": target,
        "source": "the-odds-api",
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "rows": 0,
        "http_status": None,
        "api_requests_used": None,
        "api_requests_remaining": None,
        "error": None,
    }

    print(f"Fetching NBA odds from The Odds API — {target}")
    games = fetch_odds(status)

    if not games:
        print("  No games/odds available. Writing empty odds file so pipeline can continue.")
        df = empty_df()
        if status["status"] == "unknown":
            status["status"] = "empty"
    else:
        df = parse_games(games, target)
        status["status"] = "ok"
        print(f"  Parsed {len(df)} games | Books per game: {df['num_books'].mean():.1f}")
        print(f"  Lines found: {df['spread_home'].notna().sum()} spreads, {df['total'].notna().sum()} totals")

    status["rows"] = int(len(df))
    save_outputs(df, target, args.out)

    status_path = os.path.join(args.out, "odds_api_status.json")
    with open(status_path, "w") as f:
        json.dump(status, f, indent=2)
    print(f"  Status → {status_path}")

    if not df.empty:
        print(f"\n  Today's lines:")
        cols = [c for c in ["away_team","home_team","spread_home","total","ml_home"] if c in df.columns]
        print(df[cols].to_string(index=False))

    print("✅ Odds scrape complete. Pipeline may continue even when API credits are exhausted.")


if __name__ == "__main__":
    main()
