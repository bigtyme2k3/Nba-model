"""Check all scheduled tips before installing dependencies or spending odds credits."""
from __future__ import annotations
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen


def window_games(payload, minimum, maximum, now):
    games = []
    for event in payload.get("events", []):
        try:
            tip = datetime.fromisoformat(event["date"].replace("Z", "+00:00"))
            minutes = (tip - now).total_seconds() / 60
            competitors = event["competitions"][0]["competitors"]
            teams = {c["homeAway"]: c["team"]["displayName"] for c in competitors}
        except (ValueError, KeyError, IndexError, TypeError):
            continue
        games.append({"event_id": str(event["id"]), "commence_time": event["date"],
                      "game": f"{teams['away']} @ {teams['home']}",
                      "minutes_to_tip": round(minutes, 2),
                      "eligible": minimum <= minutes <= maximum})
    return games


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True)
    parser.add_argument("--min", type=float, default=5)
    parser.add_argument("--max", type=float, default=30)
    args = parser.parse_args()
    url = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates=" + args.date.replace("-", "") + "&limit=30"
    with urlopen(url, timeout=20) as response:
        payload = json.load(response)
    now = datetime.now(timezone.utc)
    games = window_games(payload, args.min, args.max, now)
    path = Path("data/tracking/closing_schedule") / f"{args.date}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"date": args.date, "generated_at_utc": now.isoformat(), "games": games}, indent=2) + "\n")
    print("ELIGIBLE=" + ("true" if any(g["eligible"] for g in games) else "false"))


if __name__ == "__main__":
    main()
