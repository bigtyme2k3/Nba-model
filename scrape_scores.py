"""
scrape_scores.py — Live NBA scores and results from ESPN's public API.
No auth, no key. Clean JSON.

Endpoints:
  Today's scoreboard: https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard
  Specific date:      ...?dates=YYYYMMDD
  Game summary:       https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary?event={game_id}

Output:
  data/raw/scores_today.csv          — today's scores (updates live)
  data/raw/scores_YYYY-MM-DD.csv     — daily snapshot
  data/raw/scores_historical.csv     — master file, appended daily

Usage:
  python scrape_scores.py                         # today
  python scrape_scores.py --date 2026-11-05       # specific date
  python scrape_scores.py --historical 2023 2025  # bulk historical pull
  python scrape_scores.py --live                  # poll every 2 min during games
"""

import os, re, time, argparse
from datetime import date, datetime, timedelta
import requests, pandas as pd

OUT_DIR      = "data/raw"
ESPN_BASE    = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"
BOARD_URL    = f"{ESPN_BASE}/scoreboard"
SUMMARY_URL  = f"{ESPN_BASE}/summary"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
    "Accept":     "application/json",
}


def fetch_scoreboard(target_date: str = None) -> dict:
    """Fetch ESPN scoreboard for a date (YYYY-MM-DD) or today."""
    params = {"limit": 20}
    if target_date:
        params["dates"] = target_date.replace("-","")  # ESPN wants YYYYMMDD
    resp = requests.get(BOARD_URL, headers=HEADERS, params=params, timeout=15)
    resp.raise_for_status()
    return resp.json()


def fetch_game_summary(game_id: str) -> dict:
    """Fetch detailed game summary including box score and betting info."""
    resp = requests.get(SUMMARY_URL, headers=HEADERS, params={"event": game_id}, timeout=15)
    resp.raise_for_status()
    return resp.json()


def parse_scoreboard(data: dict, target_date: str) -> pd.DataFrame:
    """Parse ESPN scoreboard JSON into a clean game-level DataFrame."""
    events = data.get("events", [])
    rows   = []

    for event in events:
        game_id   = event.get("id")
        game_date = event.get("date","")[:10]   # "2026-11-05T00:00Z" → "2026-11-05"
        status    = event.get("status", {})
        state     = status.get("type", {}).get("name","")  # STATUS_FINAL, STATUS_IN_PROGRESS, etc.

        comps = event.get("competitions", [{}])[0]
        competitors = comps.get("competitors", [])

        if len(competitors) < 2:
            continue

        # ESPN: homeAway field identifies which is home
        home = next((c for c in competitors if c.get("homeAway") == "home"), competitors[0])
        away = next((c for c in competitors if c.get("homeAway") == "away"), competitors[1])

        home_team  = home.get("team", {}).get("displayName","")
        away_team  = away.get("team", {}).get("displayName","")
        home_score = home.get("score")
        away_score = away.get("score")

        # Betting line if available
        odds_obj   = comps.get("odds", [{}])
        spread     = odds_obj[0].get("details","")  if odds_obj else ""  # e.g. "BOS -8.5"
        over_under = odds_obj[0].get("overUnder")   if odds_obj else None

        # Quarter scores
        home_line_scores = [ls.get("value") for ls in home.get("linescores",[])]
        away_line_scores = [ls.get("value") for ls in away.get("linescores",[])]

        row = {
            "game_date":    game_date or target_date,
            "game_id":      game_id,
            "status":       state,
            "is_final":     "FINAL" in state.upper(),
            "in_progress":  "IN_PROGRESS" in state.upper() or "PROGRESS" in state.upper(),
            "home_team":    home_team,
            "away_team":    away_team,
            "home_score":   int(home_score) if home_score is not None else None,
            "away_score":   int(away_score) if away_score is not None else None,
            "actual_spread":int(home_score) - int(away_score) if (home_score and away_score) else None,
            "actual_total": int(home_score) + int(away_score) if (home_score and away_score) else None,
            "home_q1":      home_line_scores[0] if len(home_line_scores) > 0 else None,
            "home_q2":      home_line_scores[1] if len(home_line_scores) > 1 else None,
            "home_q3":      home_line_scores[2] if len(home_line_scores) > 2 else None,
            "home_q4":      home_line_scores[3] if len(home_line_scores) > 3 else None,
            "away_q1":      away_line_scores[0] if len(away_line_scores) > 0 else None,
            "away_q2":      away_line_scores[1] if len(away_line_scores) > 1 else None,
            "away_q3":      away_line_scores[2] if len(away_line_scores) > 2 else None,
            "away_q4":      away_line_scores[3] if len(away_line_scores) > 3 else None,
            "posted_spread":spread,
            "posted_total": over_under,
            "venue":        comps.get("venue",{}).get("fullName",""),
            "attendance":   comps.get("attendance"),
            "scraped_at":   datetime.now().isoformat(),
        }
        rows.append(row)

    return pd.DataFrame(rows)


def parse_box_score(summary: dict, game_id: str) -> pd.DataFrame:
    """
    Extract player-level box score from ESPN game summary.
    Returns a DataFrame with one row per player per game.
    """
    rows = []
    game_date = summary.get("header",{}).get("competitions",[{}])[0].get("date","")[:10]

    for team_data in summary.get("boxscore",{}).get("players",[]):
        team_name = team_data.get("team",{}).get("displayName","")
        for stat_group in team_data.get("statistics",[]):
            labels  = stat_group.get("labels",[])
            for athlete in stat_group.get("athletes",[]):
                stats   = athlete.get("stats",[])
                player  = athlete.get("athlete",{})
                stat_dict = dict(zip(labels, stats))

                def g(key, default=None):
                    val = stat_dict.get(key, default)
                    try: return float(val) if val not in (None,"--","") else default
                    except: return default

                rows.append({
                    "game_date": game_date,
                    "game_id":   game_id,
                    "player":    player.get("displayName",""),
                    "team":      team_name,
                    "position":  player.get("position",{}).get("abbreviation",""),
                    "starter":   athlete.get("starter", False),
                    "minutes":   g("MIN"),
                    "pts":       g("PTS"),
                    "reb":       g("REB"),
                    "ast":       g("AST"),
                    "stl":       g("STL"),
                    "blk":       g("BLK"),
                    "tov":       g("TO"),
                    "fgm":       g("FGM"),
                    "fga":       g("FGA"),
                    "threes":    g("3PM"),
                    "threes_att":g("3PA"),
                    "ftm":       g("FTM"),
                    "fta":       g("FTA"),
                    "plus_minus":g("+/-"),
                    "scraped_at":datetime.now().isoformat(),
                })

    return pd.DataFrame(rows)


def save(df, path, append_to_hist=True, hist_key="game_date"):
    """Save DataFrame to path and optionally append to historical master."""
    df.to_csv(path, index=False)
    if append_to_hist and not df.empty:
        base = os.path.dirname(path)
        name = os.path.basename(path)
        hist_name = re.sub(r"_\d{4}-\d{2}-\d{2}", "", name).replace("today","historical")
        hist_path = os.path.join(base, hist_name)
        if os.path.exists(hist_path):
            hist = pd.read_csv(hist_path)
            if hist_key in hist.columns and hist_key in df.columns:
                today_dates = df[hist_key].unique()
                hist = hist[~hist[hist_key].isin(today_dates)]
            pd.concat([hist, df], ignore_index=True).to_csv(hist_path, index=False)
        else:
            df.to_csv(hist_path, index=False)


def scrape_date(target_date: str, out_dir: str, include_boxscores: bool = False) -> pd.DataFrame:
    os.makedirs(out_dir, exist_ok=True)

    print(f"Fetching ESPN NBA scores — {target_date}")
    data = fetch_scoreboard(target_date)
    df   = parse_scoreboard(data, target_date)

    if df.empty:
        print("  No games found for this date.")
        return df

    finals    = df[df["is_final"]].shape[0]
    live      = df[df["in_progress"]].shape[0]
    scheduled = df.shape[0] - finals - live
    print(f"  {df.shape[0]} games: {finals} final  |  {live} live  |  {scheduled} upcoming")

    today_path = os.path.join(out_dir, "scores_today.csv")
    dated_path = os.path.join(out_dir, f"scores_{target_date}.csv")
    df.to_csv(today_path, index=False)
    save(df, dated_path, append_to_hist=True, hist_key="game_date")

    show_cols = [c for c in ["away_team","home_team","away_score","home_score","actual_spread","status"] if c in df.columns]
    print(f"\n  {'AWAY':<24} {'HOME':<24} {'AWAY':>5} {'HOME':>5} {'SPREAD':>7}  STATUS")
    print("  " + "─"*75)
    for _, r in df.iterrows():
        aw = str(r.get("away_score","—") or "—")
        hw = str(r.get("home_score","—") or "—")
        sp = f"{r['actual_spread']:+.0f}" if r.get("actual_spread") is not None else "—"
        st = r.get("status","").replace("STATUS_","")
        print(f"  {r['away_team']:<24} {r['home_team']:<24} {aw:>5} {hw:>5} {sp:>7}  {st}")

    if include_boxscores:
        print(f"\n  Fetching box scores for {finals} completed games...")
        box_rows = []
        for _, game in df[df["is_final"]].iterrows():
            try:
                summary = fetch_game_summary(str(game["game_id"]))
                box_df  = parse_box_score(summary, str(game["game_id"]))
                box_rows.append(box_df)
                time.sleep(1)
            except Exception as e:
                print(f"    Error for game {game['game_id']}: {e}")
        if box_rows:
            box_master = pd.concat(box_rows, ignore_index=True)
            box_path   = os.path.join(out_dir, f"boxscores_{target_date}.csv")
            box_master.to_csv(box_path, index=False)
            print(f"  Box scores → {box_path}  ({len(box_master)} player rows)")

    return df


def scrape_historical(start_year: int, end_year: int, out_dir: str):
    """Pull all results from start_year through end_year (season labeled by its start year)."""
    os.makedirs(out_dir, exist_ok=True)
    all_dfs = []

    for year in range(start_year, end_year + 1):
        # NBA season: roughly Oct 15 (year) – Jun 25 (year+1)
        start = date(year, 10, 15)
        end   = date(year + 1, 6, 25)
        current = start
        season_rows = []

        print(f"\nSeason {year}-{year+1}:")
        while current <= end:
            try:
                data = fetch_scoreboard(str(current))
                df   = parse_scoreboard(data, str(current))
                if not df.empty and df["is_final"].any():
                    season_rows.append(df[df["is_final"]])
                    print(f"  {current}: {df['is_final'].sum()} final games", end="\r")
            except Exception as e:
                print(f"  {current}: error — {e}")
            current += timedelta(days=1)
            time.sleep(1.0)  # Polite

        if season_rows:
            season_df = pd.concat(season_rows, ignore_index=True)
            season_df["season"] = year
            path = os.path.join(out_dir, f"scores_{year}.csv")
            season_df.to_csv(path, index=False)
            print(f"\n  Season {year}-{year+1}: {len(season_df)} games → {path}")
            all_dfs.append(season_df)

    if all_dfs:
        master = pd.concat(all_dfs, ignore_index=True)
        master.to_csv(os.path.join(out_dir, "scores_historical.csv"), index=False)
        print(f"\n✅ Historical scores: {len(master)} games saved.")


def live_poll(out_dir: str, interval: int = 120):
    """Poll every `interval` seconds during live games."""
    print(f"Live polling every {interval}s. Ctrl+C to stop.\n")
    while True:
        today = str(date.today())
        df = scrape_date(today, out_dir)
        live = df[df["in_progress"]].shape[0] if not df.empty else 0
        if live == 0:
            print("\nNo live games. Polling slowed to 10 min.")
            time.sleep(600)
        else:
            print(f"\n{live} games live. Next update in {interval}s...")
            time.sleep(interval)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date",       default=None)
    parser.add_argument("--out",        default="data/raw")
    parser.add_argument("--boxscores",  action="store_true", help="Also pull player box scores")
    parser.add_argument("--historical", nargs=2, type=int, metavar=("START","END"),
                        help="Pull historical results e.g. --historical 2023 2025")
    parser.add_argument("--live",       action="store_true", help="Poll live scores every 2 min")
    parser.add_argument("--interval",   default=120, type=int)
    args = parser.parse_args()

    if args.historical:
        scrape_historical(args.historical[0], args.historical[1], args.out)
    elif args.live:
        live_poll(args.out, args.interval)
    else:
        target = args.date or str(date.today())
        scrape_date(target, args.out, include_boxscores=args.boxscores)
        print("\n✅ Scores scrape complete.")

if __name__ == "__main__":
    main()
