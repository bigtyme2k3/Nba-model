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
# Real espn_nba_team_boxscores schema (verified against the actual release, not assumed):
# one row per TEAM per game, home/away flag is `team_home_away`, this team's score
# is `team_score`, and the opponent's name/score are already inlined as
# `opponent_team_display_name` / `opponent_team_score` — no self-join needed.
TEAM_BOX_COLS = [
    "game_id", "season", "season_type", "game_date", "game_date_time", "team_id",
    "team_display_name", "team_location", "team_name", "team_abbreviation",
    "team_home_away", "team_score", "field_goals_made", "field_goals_attempted",
    "field_goal_pct", "three_point_field_goals_made",
    "three_point_field_goals_attempted", "three_point_field_goal_pct",
    "free_throws_made", "free_throws_attempted", "free_throw_pct",
    "offensive_rebounds", "defensive_rebounds", "total_rebounds", "assists",
    "steals", "blocks", "turnovers", "fouls", "largest_lead", "team_turnovers",
    "total_technical_fouls", "opponent_team_id", "opponent_team_display_name",
    "opponent_team_score",
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


def build_games_from_team_box(df: pd.DataFrame, year: int) -> pd.DataFrame:
    """Reshape one-row-per-team-per-game hoopR data into one-row-per-game,
    in the same shape merge_data.py already expects from scores_{year}.csv
    (game_date, home_team, away_team, home_pts, away_pts, actual_spread/total)."""
    if df.empty or "team_home_away" not in df.columns:
        return pd.DataFrame()
    home = df[df["team_home_away"] == "home"].copy()
    if home.empty:
        return pd.DataFrame()
    games = home.rename(columns={
        "team_display_name": "home_team",
        "opponent_team_display_name": "away_team",
        "team_score": "home_pts",
        "opponent_team_score": "away_pts",
    })[["game_id", "game_date", "home_team", "away_team", "home_pts", "away_pts"]
      + (["season_type"] if "season_type" in home.columns else [])].copy()
    games["season"] = year
    games["home_pts"] = pd.to_numeric(games["home_pts"], errors="coerce")
    games["away_pts"] = pd.to_numeric(games["away_pts"], errors="coerce")
    games["actual_spread"] = games["home_pts"] - games["away_pts"]
    games["actual_total"] = games["home_pts"] + games["away_pts"]
    return games


def fetch_historical_season(year: int, out_dir: str) -> None:
    """year is hoopR's season label: the year the season ENDS (2026 = 2025-26)."""
    print(f"\n── Season {year-1}-{str(year)[-2:]} (Sportsdataverse / hoopR, label {year}) ──")
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
            games = build_games_from_team_box(df, year)
            if not games.empty:
                games_path = os.path.join(out_dir, f"hoopr_games_{year}.csv")
                games.to_csv(games_path, index=False)
                print(f" {len(df)} team-rows, {len(games)} games → {games_path}", end="")
            df = df[[c for c in TEAM_BOX_COLS if c in df.columns]].copy()
        elif kind == "player_box":
            df = df[[c for c in PLAYER_BOX_COLS if c in df.columns]].copy()
        path = os.path.join(out_dir, filename)
        df.to_csv(path, index=False)
        print(f" (raw {kind}: {len(df)} rows → {path})")
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


def parse_espn_team_box(summary: dict, game_date: str, season: int, season_type: int | None = None) -> list:
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
            "season_type": season_type,
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


# ESPN season types: 1 preseason, 2 regular season, 3 postseason, 5 play-in.
PRESEASON_TYPE = 1


def fetch_current_season(season_year: int, out_dir: str) -> None:
    """season_year is the year the season STARTS (e.g. 2026 for the 2026-27 season).

    Files and the `season` column use hoopR's label — the year the season ENDS — so
    the current season never overwrites last season's hoopR files. Preseason games
    are skipped.
    """
    label = season_year + 1
    print(f"\n── Season {season_year}-{str(label)[-2:]} (ESPN API — current, label {label}) ──")
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
                season_type = event.get("season", {}).get("type")
                if "FINAL" not in status.upper() or game_id in seen_games or season_type == PRESEASON_TYPE:
                    continue
                seen_games.add(game_id)
                summary = fetch_espn_boxscore(game_id)
                team_rows.extend(parse_espn_team_box(summary, date_str, label, season_type))
                player_rows.extend(parse_espn_player_box(summary, date_str, label))
                time.sleep(0.35)
        except Exception as exc:
            print(f"  [WARN] {date_str}: {exc}")
        current += timedelta(days=1)
        time.sleep(0.15)
    print(f"  Total: {len(seen_games)} games")
    if team_rows:
        team_df = pd.DataFrame(team_rows)
        path = os.path.join(out_dir, f"hoopr_team_box_{label}.csv")
        team_df.to_csv(path, index=False)
        print(f"  Team box → {path} ({len(team_rows)} rows)")

        home = team_df[team_df["home_away"] == "home"].copy()
        away = team_df[team_df["home_away"] == "away"].copy()
        keys = [c for c in ("game_id", "season", "season_type", "game_date") if c in team_df.columns]
        games = home[keys + ["team_name", "pts"]].rename(columns={"team_name": "home_team", "pts": "home_pts"}).merge(
            away[keys + ["team_name", "pts"]].rename(columns={"team_name": "away_team", "pts": "away_pts"}),
            on=keys, how="inner")
        if not games.empty:
            games["actual_spread"] = pd.to_numeric(games["home_pts"], errors="coerce") - pd.to_numeric(games["away_pts"], errors="coerce")
            games["actual_total"] = pd.to_numeric(games["home_pts"], errors="coerce") + pd.to_numeric(games["away_pts"], errors="coerce")
            games_path = os.path.join(out_dir, f"hoopr_games_{label}.csv")
            games.to_csv(games_path, index=False)
            print(f"  Games → {games_path} ({len(games)} games)")
    if player_rows:
        path = os.path.join(out_dir, f"hoopr_player_box_{label}.csv")
        pd.DataFrame(player_rows).to_csv(path, index=False)
        print(f"  Player box → {path} ({len(player_rows)} rows)")


def build_stats_master(out_dir: str) -> None:
    """Concatenate the per-season hoopr_games_{year}.csv files (already one-row-per-game,
    written by fetch_historical_season/fetch_current_season) into a single reference file.
    merge_data.py reads the per-season files directly and doesn't need this master file —
    it's just a convenience for eyeballing everything collected so far."""
    files = sorted(f for f in os.listdir(out_dir) if f.startswith("hoopr_games_") and f.endswith(".csv") and f != "hoopr_games_master.csv")
    if not files:
        print("  No hoopr_games_*.csv files found.")
        return
    games = pd.concat([pd.read_csv(os.path.join(out_dir, f)) for f in files], ignore_index=True)
    path = os.path.join(out_dir, "hoopr_games_master.csv")
    games.to_csv(path, index=False)
    print(f"  Master game file → {path} ({len(games)} games)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect NBA data via hoopR/ESPN")
    parser.add_argument("--start", type=int, default=None, help="First hoopR season label (end-year), e.g. 2022 = 2021-22")
    parser.add_argument("--end", type=int, default=2026, help="Last hoopR season label (end-year)")
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
        # Season start year; before August it's still last season's playoffs tail.
        season_start_year = today.year if today.month >= 8 else today.year - 1
        fetch_current_season(season_start_year, args.out)
    if args.master or args.start or args.current:
        print("\nBuilding master game dataset...")
        build_stats_master(args.out)
    print("\n✅ hoopR/ESPN collection complete.")


if __name__ == "__main__":
    main()
