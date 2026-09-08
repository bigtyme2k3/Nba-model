"""
scrape_refs.py
--------------
Builds an NBA referee tendency database and fetches today's assignments.

Two data sources:
  1. ESPN game summary API  — officials for every historical game
  2. RefMetrics.com         — today's assignments + foul/total stats

Outputs:
  data/raw/ref_stats.csv          — per-referee career stats (foul rate, total tendency)
  data/raw/ref_assignments_today.csv — today's game → referee crew mapping
  data/raw/ref_game_log.csv       — every game with ref crew + result (appended daily)

Key features added to model:
  crew_avg_total      — this crew's average total points allowed (vs league avg)
  crew_foul_rate      — fouls called per game
  crew_over_rate      — % of games going over
  crew_total_adj      — deviation from league average total (+/- pts)

Usage:
    python scrape_refs.py                        # today's assignments + update stats
    python scrape_refs.py --historical 2023 2025 # build historical dataset
    python scrape_refs.py --today-only           # only fetch today's assignments
"""

import os, time, argparse
import pandas as pd
import numpy as np
from datetime import date, timedelta
import requests

OUT_DIR    = "data/raw"
HEADERS    = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
ESPN_BASE  = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"
RM_BASE    = "https://www.refmetrics.com/nba"

# League average total for normalization (NBA totals run much higher than WNBA — recalibrate from real data).
LEAGUE_AVG_TOTAL = 224.0
LEAGUE_AVG_FOULS = 42.0   # combined fouls both teams


# ── ESPN: Extract officials from game summary ─────────────────────────────────

def fetch_game_summary(game_id: str) -> dict:
    url  = f"{ESPN_BASE}/summary"
    resp = requests.get(url, headers=HEADERS, params={"event": game_id}, timeout=15)
    resp.raise_for_status()
    return resp.json()


def parse_officials(summary: dict) -> list[str]:
    """Extract referee names from ESPN game summary."""
    officials = []

    for off in summary.get("officials", []):
        name = off.get("displayName") or off.get("fullName") or off.get("name","")
        if name:
            officials.append(name.strip())

    if not officials:
        comps = summary.get("header", {}).get("competitions", [{}])
        for off in comps[0].get("officials", []):
            name = off.get("displayName","")
            if name:
                officials.append(name.strip())

    return list(set(officials))


def parse_game_totals(summary: dict) -> dict:
    """Extract total points and fouls from game summary."""
    total_pts = 0
    total_fouls = 0
    home_pts  = 0
    away_pts  = 0

    for team_data in summary.get("boxscore", {}).get("teams", []):
        stats = {s["name"]: s.get("displayValue","0")
                 for s in team_data.get("statistics", [])}
        try:
            pts   = float(stats.get("points","0") or 0)
            fouls = float(stats.get("fouls","0") or 0)
            total_pts   += pts
            total_fouls += fouls
            if team_data.get("homeAway") == "home":
                home_pts = pts
            else:
                away_pts = pts
        except:
            pass

    return {
        "actual_total":  total_pts,
        "actual_spread": home_pts - away_pts,
        "total_fouls":   total_fouls,
        "home_pts":      home_pts,
        "away_pts":      away_pts,
    }


# ── ESPN: Build historical ref stats ─────────────────────────────────────────

def fetch_historical_refs(start_year: int, end_year: int, out_dir: str):
    """Loop through completed games in date range, fetch ESPN summaries, extract officials + totals."""
    print(f"\nBuilding historical ref database ({start_year}-{start_year+1} .. {end_year}-{end_year+1})...")
    rows = []

    for year in range(start_year, end_year + 1):
        print(f"\n  Season {year}-{year+1}:")
        season_start = date(year, 10, 15)
        season_end   = min(date(year + 1, 6, 25), date.today())
        current      = season_start

        while current <= season_end:
            date_str = str(current)
            try:
                url    = f"{ESPN_BASE}/scoreboard"
                params = {"dates": date_str.replace("-",""), "limit":20}
                resp   = requests.get(url, headers=HEADERS, params=params, timeout=15)
                resp.raise_for_status()
                events = resp.json().get("events", [])

                for event in events:
                    status = event.get("status",{}).get("type",{}).get("name","")
                    if "FINAL" not in status.upper():
                        continue

                    gid = event.get("id","")
                    try:
                        summary   = fetch_game_summary(gid)
                        officials = parse_officials(summary)
                        totals    = parse_game_totals(summary)

                        if not officials:
                            continue

                        comps = event.get("competitions",[{}])[0]
                        competitors = comps.get("competitors",[])
                        home = next((c for c in competitors if c.get("homeAway")=="home"),{})
                        away = next((c for c in competitors if c.get("homeAway")=="away"),{})

                        row = {
                            "game_id":      gid,
                            "game_date":    date_str,
                            "season":       year,
                            "home_team":    home.get("team",{}).get("displayName",""),
                            "away_team":    away.get("team",{}).get("displayName",""),
                            "refs":         "|".join(sorted(officials)),
                            "num_refs":     len(officials),
                            **totals,
                        }
                        rows.append(row)
                        time.sleep(0.4)

                    except Exception:
                        pass

            except Exception:
                pass

            current += timedelta(days=1)
            time.sleep(0.2)

        print(f"    {sum(1 for r in rows if r['season']==year)} games logged")

    if rows:
        df = pd.DataFrame(rows)
        path = os.path.join(out_dir, "ref_game_log.csv")

        if os.path.exists(path):
            existing = pd.read_csv(path)
            existing = existing[~existing["game_id"].isin(df["game_id"])]
            df = pd.concat([existing, df], ignore_index=True)

        df.to_csv(path, index=False)
        print(f"\n  Ref game log → {path} ({len(df)} games)")
        return df
    return pd.DataFrame()


# ── Compute per-referee stats ─────────────────────────────────────────────────

def compute_ref_stats(game_log: pd.DataFrame) -> pd.DataFrame:
    """From game log, compute per-referee tendency stats (one row per referee per game)."""
    if game_log.empty or "refs" not in game_log.columns:
        return pd.DataFrame()

    ref_rows = []
    for _, row in game_log.iterrows():
        refs = str(row.get("refs","")).split("|")
        for ref in refs:
            ref = ref.strip()
            if not ref:
                continue
            ref_rows.append({
                "ref_name":     ref,
                "game_id":      row["game_id"],
                "game_date":    row["game_date"],
                "season":       row["season"],
                "actual_total": row.get("actual_total", np.nan),
                "total_fouls":  row.get("total_fouls", np.nan),
                "actual_spread":row.get("actual_spread", np.nan),
            })

    if not ref_rows:
        return pd.DataFrame()

    ref_df = pd.DataFrame(ref_rows)

    stats = ref_df.groupby("ref_name").agg(
        games         = ("game_id",      "count"),
        avg_total     = ("actual_total", "mean"),
        std_total     = ("actual_total", "std"),
        avg_fouls     = ("total_fouls",  "mean"),
        over_rate     = ("actual_total", lambda x: (x > LEAGUE_AVG_TOTAL).mean()),
        last_season   = ("season",       "max"),
    ).reset_index()

    stats["total_tendency"] = stats["avg_total"] - LEAGUE_AVG_TOTAL
    stats["foul_tendency"]  = stats["avg_fouls"] - LEAGUE_AVG_FOULS
    stats["weight"] = np.minimum(stats["games"] / 30.0, 1.0)
    stats = stats[stats["games"] >= 5].sort_values("avg_total", ascending=False)

    return stats


# ── Today's assignments from RefMetrics ──────────────────────────────────────

def scrape_today_assignments() -> pd.DataFrame:
    """Scrape today's NBA referee assignments from RefMetrics.com. Falls back to ESPN game summaries."""
    rows = []

    try:
        from bs4 import BeautifulSoup
        url  = f"{RM_BASE}/referee-assignments-today"
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "lxml")

        tables = soup.find_all("table")
        for table in tables:
            df = pd.read_html(str(table))[0]
            df.columns = [c.lower().replace(" ","_") for c in df.columns]
            if any("ref" in c or "crew" in c or "official" in c for c in df.columns):
                df["scraped_from"] = "refmetrics"
                df["game_date"]    = str(date.today())
                rows.append(df)
                break

    except Exception as e:
        print(f"  [WARN] RefMetrics unavailable: {e}")

    if not rows:
        try:
            today_str = str(date.today())
            url    = f"{ESPN_BASE}/scoreboard"
            params = {"dates": today_str.replace("-",""), "limit":20}
            resp   = requests.get(url, headers=HEADERS, params=params, timeout=15)
            resp.raise_for_status()
            events = resp.json().get("events",[])

            espn_rows = []
            for event in events:
                gid   = event.get("id","")
                comps = event.get("competitions",[{}])[0]
                competitors = comps.get("competitors",[])
                home  = next((c for c in competitors if c.get("homeAway")=="home"),{})
                away  = next((c for c in competitors if c.get("homeAway")=="away"),{})

                try:
                    summary   = fetch_game_summary(gid)
                    officials = parse_officials(summary)
                    espn_rows.append({
                        "game_date": today_str,
                        "game_id":   gid,
                        "home_team": home.get("team",{}).get("displayName",""),
                        "away_team": away.get("team",{}).get("displayName",""),
                        "refs":      "|".join(sorted(officials)),
                        "scraped_from": "espn",
                    })
                    time.sleep(0.5)
                except:
                    pass

            if espn_rows:
                rows.append(pd.DataFrame(espn_rows))

        except Exception as e:
            print(f"  [WARN] ESPN ref fetch failed: {e}")

    if not rows:
        return pd.DataFrame()

    return pd.concat(rows, ignore_index=True)


# ── Look up crew stats for a game ─────────────────────────────────────────────

def get_crew_stats(refs_str: str, ref_stats: pd.DataFrame) -> dict:
    """Given a pipe-separated string of referee names, return aggregate crew stats."""
    default = {
        "crew_avg_total":   LEAGUE_AVG_TOTAL,
        "crew_total_adj":   0.0,
        "crew_foul_rate":   LEAGUE_AVG_FOULS,
        "crew_over_rate":   0.5,
        "crew_games":       0,
    }
    if not refs_str or ref_stats.empty:
        return default

    refs = [r.strip() for r in refs_str.split("|") if r.strip()]
    crew = ref_stats[ref_stats["ref_name"].isin(refs)]

    if crew.empty:
        return default

    w = crew["weight"].values
    w = w / w.sum() if w.sum() > 0 else np.ones(len(w)) / len(w)

    return {
        "crew_avg_total":   float(np.average(crew["avg_total"], weights=w)),
        "crew_total_adj":   float(np.average(crew["total_tendency"], weights=w)),
        "crew_foul_rate":   float(np.average(crew["avg_fouls"], weights=w)),
        "crew_over_rate":   float(np.average(crew["over_rate"], weights=w)),
        "crew_games":       int(crew["games"].mean()),
    }


# ── Daily update (run each morning) ──────────────────────────────────────────

def daily_update(out_dir: str) -> dict:
    os.makedirs(out_dir, exist_ok=True)

    stats_path = os.path.join(out_dir, "ref_stats.csv")
    if os.path.exists(stats_path):
        ref_stats = pd.read_csv(stats_path)
    else:
        log_path = os.path.join(out_dir, "ref_game_log.csv")
        if os.path.exists(log_path):
            game_log  = pd.read_csv(log_path)
            ref_stats = compute_ref_stats(game_log)
            ref_stats.to_csv(stats_path, index=False)
            print(f"  Computed ref stats for {len(ref_stats)} officials")
        else:
            ref_stats = pd.DataFrame()
            print("  [WARN] No ref stats available — skipping ref features")

    print("  Fetching today's referee assignments...")
    assignments = scrape_today_assignments()

    if assignments.empty:
        print("  [WARN] No assignments found")
        return {}

    path = os.path.join(out_dir, "ref_assignments_today.csv")
    assignments.to_csv(path, index=False)
    print(f"  {len(assignments)} games with refs → {path}")

    crew_lookup = {}
    for _, row in assignments.iterrows():
        refs_str = row.get("refs","")
        if not refs_str and "crew_chief" in row:
            refs_str = "|".join(filter(None, [
                str(row.get("crew_chief","")),
                str(row.get("referee","")),
                str(row.get("umpire",""))
            ]))

        home = str(row.get("home_team",""))
        away = str(row.get("away_team",""))
        key  = f"{away}@{home}"

        crew_lookup[key] = {
            "refs":         refs_str,
            "crew_stats":   get_crew_stats(refs_str, ref_stats),
        }

    print(f"\n  Today's referee crew stats:")
    for game_key, info in crew_lookup.items():
        cs = info["crew_stats"]
        adj = cs["crew_total_adj"]
        sign = "+" if adj >= 0 else ""
        print(f"    {game_key}: total_adj={sign}{adj:.1f}  over_rate={cs['crew_over_rate']:.0%}  games={cs['crew_games']}")

    return crew_lookup


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--historical", nargs=2, type=int, metavar=("START","END"))
    parser.add_argument("--today-only", action="store_true")
    parser.add_argument("--rebuild-stats", action="store_true")
    parser.add_argument("--out", default="data/raw")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    print("\n═══ NBA Referee Tracker ═══\n")

    if args.historical:
        start, end = args.historical
        game_log = fetch_historical_refs(start, end, args.out)
        if not game_log.empty:
            ref_stats = compute_ref_stats(game_log)
            path = os.path.join(args.out, "ref_stats.csv")
            ref_stats.to_csv(path, index=False)
            print(f"\n  Ref stats → {path} ({len(ref_stats)} officials)")

    elif args.rebuild_stats:
        log_path = os.path.join(args.out, "ref_game_log.csv")
        if os.path.exists(log_path):
            game_log  = pd.read_csv(log_path)
            ref_stats = compute_ref_stats(game_log)
            path      = os.path.join(args.out, "ref_stats.csv")
            ref_stats.to_csv(path, index=False)
            print(f"  Rebuilt ref stats → {path}")
        else:
            print("  No game log found. Run --historical first.")

    else:
        crew_lookup = daily_update(args.out)
        print(f"\n✅ Referee tracker complete. {len(crew_lookup)} games with crew stats.")


if __name__ == "__main__":
    main()
