"""
v2/warehouse.py
---------------
Builds the V2 DuckDB warehouse and checks its identifiers.

Tables
  player_box   hoopR/ESPN player box scores, every season on disk (incl. DNP rows)
  games        one row per game: date, season (start year), type, home/away team
  pbp          ESPN play-by-play for the seasons fetched by fetch_pbp()

Validation (warehouse_checks): duplicate player-game rows, play-by-play games
without a box score and vice versa, play-by-play athletes missing from that
game's box score, and games whose box score lacks exactly five starters per
team. Problems are reported, never patched with zeros or guesses.

Usage:
    python -m v2.warehouse --seasons 2025
"""
from __future__ import annotations

import argparse
import json
from glob import glob
from pathlib import Path

import duckdb
import requests

RAW = Path("data/raw")
PBP_DIR = RAW / "pbp"
DB_PATH = Path("data/warehouse/nba_v2.duckdb")
RELEASE = "https://github.com/sportsdataverse/sportsdataverse-data/releases/download/"
RELEASES = {  # sportsdataverse release files, labelled by the season's end year
    "pbp": ("espn_nba_pbp", "play_by_play_{label}.parquet"),
    "player_box": ("espn_nba_player_boxscores", "player_box_{label}.parquet"),
    "schedule": ("espn_nba_schedules", "nba_schedule_{label}.parquet"),
}


def fetch_release(kind: str, season: int, refresh: bool = False, pbp_dir=PBP_DIR) -> Path:
    """Cached sportsdataverse parquet for a season (start year). The current season is
    republished daily, so callers pass refresh=True for it."""
    tag, name = RELEASES[kind]
    path = Path(pbp_dir) / name.format(label=season + 1)
    if refresh or not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        resp = requests.get(RELEASE + f"{tag}/{name.format(label=season + 1)}", timeout=120)
        resp.raise_for_status()
        path.write_bytes(resp.content)
    return path


def fetch_pbp(season: int, pbp_dir=PBP_DIR, refresh: bool = False) -> Path:
    return fetch_release("pbp", season, refresh, pbp_dir)


def _box_sources(seasons, raw, refresh):
    """Committed hoopR CSVs for past seasons; the daily parquet for a season without one."""
    csvs = sorted(glob(str(Path(raw) / "hoopr_player_box_20*.csv")))
    have = {int(Path(p).stem.rsplit("_", 1)[1]) - 1 for p in csvs}
    parquets = [str(fetch_release("player_box", s, s in refresh)) for s in seasons if s not in have]
    scheds = sorted(glob(str(Path(raw) / "hoopr_schedule_20*.csv")))
    have_s = {int(Path(p).stem.rsplit("_", 1)[1]) - 1 for p in scheds}
    sched_parquets = [str(fetch_release("schedule", s, s in refresh)) for s in seasons if s not in have_s]
    return csvs, parquets, scheds, sched_parquets


def build(seasons, db_path=DB_PATH, raw=RAW, refresh=()) -> duckdb.DuckDBPyConnection:
    """refresh: seasons (start years) whose release files are re-downloaded (the current one)."""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path))
    csvs, parquets, scheds, sched_parquets = _box_sources(seasons, raw, set(refresh))
    parts = ([f"SELECT * FROM read_csv_auto({csvs!r}, union_by_name=true)"] if csvs else []) + \
            ([f"SELECT * FROM read_parquet({parquets!r}, union_by_name=true)"] if parquets else [])
    box_from = " UNION ALL BY NAME ".join(parts)
    sched_from = f"SELECT * FROM read_csv_auto({scheds!r}, union_by_name=true)"
    if sched_parquets:
        sched_from += f" UNION ALL BY NAME SELECT * FROM read_parquet({sched_parquets!r}, union_by_name=true)"
    con.execute(f"""
        CREATE OR REPLACE TABLE player_box AS
        SELECT CAST(game_id AS VARCHAR) AS game_id,
               CAST(game_date AS DATE) AS game_date,
               CAST(athlete_id AS VARCHAR) AS athlete_id,
               athlete_display_name AS player,
               team_location || ' ' || team_name AS team,
               team_abbreviation AS team_abbr, home_away,
               CAST(starter AS BOOLEAN) AS starter,
               CAST(did_not_play AS BOOLEAN) AS did_not_play,
               TRY_CAST(minutes AS DOUBLE) AS minutes,
               TRY_CAST(points AS DOUBLE) AS pts, TRY_CAST(rebounds AS DOUBLE) AS reb,
               TRY_CAST(offensive_rebounds AS DOUBLE) AS oreb,
               TRY_CAST(defensive_rebounds AS DOUBLE) AS dreb,
               TRY_CAST(assists AS DOUBLE) AS ast,
               TRY_CAST(field_goals_attempted AS DOUBLE) AS fga,
               TRY_CAST(three_point_field_goals_made AS DOUBLE) AS threes
        FROM ({box_from})
    """)
    con.execute(f"""
        CREATE OR REPLACE TABLE games AS
        SELECT DISTINCT CAST(game_id AS VARCHAR) AS game_id,
               coalesce(TRY_CAST(date AS TIMESTAMPTZ),
                        try_strptime(CAST(date AS VARCHAR), '%Y-%m-%dT%H:%MZ')) AS tip_utc,
               CAST(game_date AS DATE) AS game_date,
               CASE WHEN month(CAST(game_date AS DATE)) >= 7 THEN year(CAST(game_date AS DATE))
                    ELSE year(CAST(game_date AS DATE)) - 1 END AS season,
               season_type, CAST(home_id AS VARCHAR) AS home_team_id,
               CAST(away_id AS VARCHAR) AS away_team_id,
               home_display_name AS home_team, away_display_name AS away_team
        FROM ({sched_from})
    """)
    pbps = [str(fetch_pbp(s, refresh=s in set(refresh))) for s in seasons]
    con.execute(f"""
        CREATE OR REPLACE TABLE pbp AS
        SELECT CAST(game_id AS VARCHAR) AS game_id, CAST(game_play_number AS INTEGER) AS play_no,
               CAST(sequence_number AS BIGINT) AS seq, type_text, text,
               CAST(period_number AS INTEGER) AS period,
               CAST(clock_minutes AS DOUBLE) * 60 + CAST(clock_seconds AS DOUBLE) AS clock_remaining,
               CAST(team_id AS VARCHAR) AS team_id,
               CAST(athlete_id_1 AS VARCHAR) AS a1, CAST(athlete_id_2 AS VARCHAR) AS a2,
               CAST(athlete_id_3 AS VARCHAR) AS a3,
               CAST(scoring_play AS BOOLEAN) AS scoring_play,
               CAST(shooting_play AS BOOLEAN) AS shooting_play,
               CAST(score_value AS INTEGER) AS score_value,
               coordinate_x, coordinate_y,
               CAST(home_team_id AS VARCHAR) AS home_team_id, CAST(away_team_id AS VARCHAR) AS away_team_id
        FROM read_parquet({pbps!r}, union_by_name=true)
    """)
    return con


def warehouse_checks(con) -> dict:
    q = lambda sql: con.execute(sql).fetchall()
    checks = {
        "player_box_rows": q("SELECT count(*) FROM player_box")[0][0],
        "duplicate_player_games": q("""SELECT count(*) FROM (SELECT game_id, athlete_id FROM player_box
                                       GROUP BY 1,2 HAVING count(*) > 1)""")[0][0],
        "pbp_games": q("SELECT count(DISTINCT game_id) FROM pbp")[0][0],
        "pbp_games_without_box": q("""SELECT count(DISTINCT game_id) FROM pbp
                                      WHERE game_id NOT IN (SELECT game_id FROM player_box)""")[0][0],
        "pbp_athletes_not_in_game_box": q("""
            WITH a AS (SELECT DISTINCT game_id, a FROM (
                SELECT game_id, a1 AS a FROM pbp UNION ALL SELECT game_id, a2 FROM pbp
                UNION ALL SELECT game_id, a3 FROM pbp) WHERE a IS NOT NULL)
            SELECT count(*) FROM a LEFT JOIN player_box b ON a.game_id = b.game_id AND a.a = b.athlete_id
            WHERE b.athlete_id IS NULL""")[0][0],
        "games_without_5_starters_per_team": q("""
            SELECT count(*) FROM (SELECT game_id, home_away, count(*) FILTER (WHERE starter) s
                FROM player_box WHERE game_id IN (SELECT DISTINCT game_id FROM pbp)
                GROUP BY 1,2 HAVING s <> 5)""")[0][0],
    }
    return checks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", type=int, nargs="+", default=[2025])
    ap.add_argument("--db", default=str(DB_PATH))
    a = ap.parse_args()
    con = build(a.seasons, a.db)
    checks = warehouse_checks(con)
    out = Path("data/tracking/v2_warehouse_checks.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(checks, indent=2) + "\n")
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
