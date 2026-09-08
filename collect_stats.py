"""Collect NBA team/player box scores from Sportsdataverse (hoopR) and ESPN."""
from __future__ import annotations

import argparse
import os
import tempfile
import time
from datetime import date, timedelta

import pandas as pd
import requests

OUT_DIR = "data/raw"
HEADERS = {"User-Agent": "Mozilla/5.0 (research project)"}
SDV_BASE = "https://github.com/sportsdataverse/sportsdataverse-data/releases/download"
ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"
SDV_URLS = {
    "team_box": f"{SDV_BASE}/espn_nba_team_boxscores/team_box_{{year}}.rds",
    "player_box": f"{SDV_BASE}/espn_nba_player_boxscores/player_box_{{year}}.rds",
    "schedule": f"{SDV_BASE}/espn_nba_schedules/nba_schedule_{{year}}.rds",
}
TEAM_BOX_COLS = [
    "game_id", "season", "game_date", "game_date_time", "team_id", "team_name",
    "team_location", "team_abbreviation", "home_away", "field_goals_made",
    "field_goals_attempted", "field_goal_pct", "three_point_field_goals_made",
    "three_point_field_goals_attempted", "three_point_field_goal_pct",
    "free_throws_made", "free_throws_attempted", "free_throw_pct",
    "offensive_rebounds", "defensive_rebounds", "rebounds", "assists", "steals",
    "blocks", "turnovers", "fouls", "points", "largest_lead", "team_turnovers",
    "total_technical_fouls",
]
PLAYER_BOX_COLS = [
    "game_id", "season", "game_date", "team_name", "team_location",
    "team_abbreviation", "athlete_id", "athlete_display_name",
    "athlete_position_abbreviation", "home_away", "starter", "did_not_play",
    "minutes", "field_goals_made", "field_goals_attempted",
    "three_point_field_goals_made", "three_point_field_goals_attempted",
    "free_throws_made", "free_throws_attempted", "offensive_rebounds",
    "defensive_rebounds", "rebounds", "assists", "steals", "blocks", "turnovers",
    "fouls", "points", "plus_minus",
]


def read_rds_url(url: str) -> pd.DataFrame:
    import pyreadr
    response = requests.get(url, headers=HEADERS, timeout=60, allow_redirects=True)
    if response.status_code == 404:
        return pd.DataFrame()
    response.raise_for_status()
    with tempfile.NamedTemporaryFile(suffix=".rds", delete=False) as tmp:
        tmp.write(response.content)
        path = tmp.name
    try:
        result = pyreadr.read_r(path)
        return list(result.values())[0]
    finally:
        os.unlink(path)


def fetch_historical_season(year: int, out_dir: str) -> None:
    print(f"\n── Season {year}-{year+1} (Sportsdataverse / hoopR) ──")
    for kind, filename in (
        ("team_box", f"hoopr_team_box_{year}.csv"),
        ("player_box", f"hoopr_player_box_{year}.csv"),
        ("schedule", f"hoopr_schedule_{year}.csv"),
    ):
        print(f"  Downloading {kind}...", end="", flush=True)
        df = read_rds_url(SDV_URLS[kind].format(year=year))
        if df.empty:
            print(" [404 — not available]")
            continue
        if kind == "team_box":
            df = df[[c for c in TEAM_BOX_COLS if c in df.columns]].copy()
            if "points" in df.columns:
                df["pts"] = pd.to_numeric(df["points"], errors="coerce")
        elif kind == "player_box":
            df = df[[c for c in PLAYER_BOX_COLS if c in df.columns]].copy()
        path = os.path.join(out_dir, filename)
        df.to_csv(path, index=False)
        print(f" {len(df)} rows → {path}")
        time.sleep(1)


def fetch_espn_scoreboard(target_date: str) -> list:
    response = requests.get(
        f"{ESPN_BASE}/scoreboard",
        headers=HEADERS,
        params={"dates": target_date.replace("-", ""), "limit": 20},
        timeout=20,
    )
    response.raise_for_status()
    return response.json().get("events", [])


def fetch_espn_boxscore(game_id: str) -> dict:
    response = requests.get(
        f"{ESPN_BASE}/summary", headers=HEADERS, params={"event": game_id}, timeout=20
    )
    response.raise_for_status()
    return response.json()


def competitor_index(summary: dict) -> dict[str, dict]:
    competition = summary.get("header", {}).get("competitions", [{}])[0]
    result = {}
    for competitor in competition.get("competitors", []):
        team_id = str(competitor.get("team", {}).get("id", ""))
        if team_id:
            result[team_id] = competitor
    return result


def number(value):
    try:
        return float(str(value).replace("%", ""))
    except (TypeError, ValueError):
        return None


def parse_espn_team_box(summary: dict, game_date: str, season: int) -> list:
    competition = summary.get("header", {}).get("competitions", [{}])[0]
    game_id = str(competition.get("id", ""))
    competitors = competitor_index(summary)
    rows = []
    for team_data in summary.get("boxscore", {}).get("teams", []):
        team = team_data.get("team", {})
        team_id = str(team.get("id", ""))
        competitor = competitors.get(team_id, {})
        stats = {s.get("name"): s.get("displayValue") for s in team_data.get("statistics", [])}
        score = number(competitor.get("score"))
        home_away = competitor.get("homeAway") or team_data.get("homeAway") or ""
        rows.append({
            "game_id": game_id,
            "season": season,
            "game_date": game_date,
            "team_id": team_id,
            "team_name": team.get("name", ""),
            "team_location": team.get("location", ""),
            "team_abbreviation": team.get("abbreviation", ""),
            "home_away": home_away,
            "points": score,
            "pts": score,
            "field_goal_pct": number(stats.get("fieldGoalPct")),
            "three_point_pct": number(stats.get("threePointPct")),
            "free_throw_pct": number(stats.get("freeThrowPct")),
            "rebounds": number(stats.get("totalRebounds")),
            "assists": number(stats.get("assists")),
            "steals": number(stats.get("steals")),
            "blocks": number(stats.get("blocks")),
            "turnovers": number(stats.get("turnovers")),
            "fouls": number(stats.get("fouls")),
            "offensive_rebounds": number(stats.get("offensiveRebounds")),
            "source": "espn_api",
        })
    return rows


def parse_espn_player_box(summary: dict, game_date: str, season: int) -> list:
    competition = summary.get("header", {}).get("competitions", [{}])[0]
    game_id = str(competition.get("id", ""))
    competitors = competitor_index(summary)
    rows = []
    for team_data in summary.get("boxscore", {}).get("players", []):
        team = team_data.get("team", {})
        home_away = competitors.get(str(team.get("id", "")), {}).get("homeAway", "")
        for group in team_data.get("statistics", []):
            labels = group.get("labels", [])
            for athlete in group.get("athletes", []):
                player = athlete.get("athlete", {})
                values = dict(zip(labels, athlete.get("stats", [])))
                rows.append({
                    "game_id": game_id, "season": season, "game_date": game_date,
                    "team_name": team.get("name", ""),
                    "team_location": team.get("location", ""),
                    "team_abbr": team.get("abbreviation", ""), "home_away": home_away,
                    "player_id": player.get("id", ""),
                    "player_name": player.get("displayName", ""),
                    "position": player.get("position", {}).get("abbreviation", ""),
                    "starter": athlete.get("starter", False),
                    "did_not_play": athlete.get("didNotPlay", False),
                    "minutes": number(values.get("MIN")), "pts": number(values.get("PTS")),
                    "reb": number(values.get("REB")), "ast": number(values.get("AST")),
                    "stl": number(values.get("STL")), "blk": number(values.get("BLK")),
                    "tov": number(values.get("TO")), "fg_made": number(values.get("FGM")),
                    "fg_att": number(values.get("FGA")), "threes_made": number(values.get("3PM")),
                    "threes_att": number(values.get("3PA")), "ft_made": number(values.get("FTM")),
                    "ft_att": number(values.get("FTA")), "plus_minus": number(values.get("+/-")),
                    "source": "espn_api",
                })
    return rows


def fetch_current_season(season_year: int, out_dir: str) -> None:
    """season_year is the year the season STARTS (e.g. 2025 for the 2025-26 season)."""
    print(f"\n── Season {season_year}-{season_year+1} (ESPN API — current) ──")
    today = date.today()
    current = date(season_year, 10, 1)
    end = min(date(season_year + 1, 6, 30), today)
    team_rows, player_rows, seen_games = [], [], set()
    while current <= end:
        date_str = current.isoformat()
        try:
            for event in fetch_espn_scoreboard(date_str):
                game_id = str(event.get("id", ""))
                status = str(event.get("status", {}).get("type", {}).get("name", ""))
                if "FINAL" not in status.upper() or game_id in seen_games:
                    continue
                seen_games.add(game_id)
                summary = fetch_espn_boxscore(game_id)
                team_rows.extend(parse_espn_team_box(summary, date_str, season_year))
                player_rows.extend(parse_espn_player_box(summary, date_str, season_year))
                time.sleep(0.35)
        except Exception as exc:
            print(f"  [WARN] {date_str}: {exc}")
        current += timedelta(days=1)
        time.sleep(0.15)
    print(f"  Total: {len(seen_games)} games")
    if team_rows:
        path = os.path.join(out_dir, f"hoopr_team_box_{season_year}.csv")
        pd.DataFrame(team_rows).to_csv(path, index=False)
        print(f"  Team box → {path} ({len(team_rows)} rows)")
    if player_rows:
        path = os.path.join(out_dir, f"hoopr_player_box_{season_year}.csv")
        pd.DataFrame(player_rows).to_csv(path, index=False)
        print(f"  Player box → {path} ({len(player_rows)} rows)")


def build_stats_master(out_dir: str) -> None:
    files = sorted(f for f in os.listdir(out_dir) if f.startswith("hoopr_team_box_") and f.endswith(".csv"))
    if not files:
        print("  No hoopR team box files found.")
        return
    master = pd.concat([pd.read_csv(os.path.join(out_dir, f)) for f in files], ignore_index=True)
    if "home_away" not in master.columns:
        master.to_csv(os.path.join(out_dir, "hoopr_master.csv"), index=False)
        return
    home = master[master["home_away"].astype(str).str.lower() == "home"].copy()
    away = master[master["home_away"].astype(str).str.lower() == "away"].copy()
    keys = [c for c in ("game_id", "season", "game_date") if c in master.columns]
    home = home.rename(columns={c: f"home_{c}" for c in home.columns if c not in keys})
    away = away.rename(columns={c: f"away_{c}" for c in away.columns if c not in keys})
    games = home.merge(away, on=keys, how="inner")
    if "home_pts" in games.columns and "away_pts" in games.columns:
        games["actual_spread"] = pd.to_numeric(games["home_pts"], errors="coerce") - pd.to_numeric(games["away_pts"], errors="coerce")
        games["actual_total"] = pd.to_numeric(games["home_pts"], errors="coerce") + pd.to_numeric(games["away_pts"], errors="coerce")
    path = os.path.join(out_dir, "hoopr_games_master.csv")
    games.to_csv(path, index=False)
    print(f"  Master game file → {path} ({len(games)} games)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect NBA data via hoopR/ESPN")
    parser.add_argument("--start", type=int, default=None, help="First season start-year, e.g. 2022")
    parser.add_argument("--end", type=int, default=2024, help="Last season start-year")
    parser.add_argument("--current", action="store_true")
    parser.add_argument("--out", default=OUT_DIR)
    parser.add_argument("--master", action="store_true")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    print("\n═══ NBA Data Collection — hoopR + ESPN ═══\n")
    if args.start:
        for year in range(args.start, args.end + 1):
            try:
                fetch_historical_season(year, args.out)
            except Exception as exc:
                print(f"  [ERROR] Season {year}: {exc}")
    if args.current:
        today = date.today()
        # NBA season labeled by its start year; before October it's still last season's playoffs tail.
        season_start_year = today.year if today.month >= 8 else today.year - 1
        fetch_current_season(season_start_year, args.out)
    if args.master or args.start or args.current:
        print("\nBuilding master game dataset...")
        build_stats_master(args.out)
    print("\n✅ hoopR/ESPN collection complete.")


if __name__ == "__main__":
    main()
