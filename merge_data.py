"""
merge_data.py
-------------
Merges box-score stats with odds/scores into a single model-ready master dataset.

Usage:
    python merge_data.py --year 2023
    python merge_data.py --year all
"""

import os
import argparse
import numpy as np
import pandas as pd
from glob import glob

from teams import EASTERN_CONFERENCE, WESTERN_CONFERENCE, TEAM_NAMES, normalize_team
from season_filters import filter_games_for_source, nba_season_start_year

RAW_DIR       = "data/raw"
PROCESSED_DIR = "data/processed"


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_game_logs(year: int) -> pd.DataFrame:
    """Load one canonical NBA season keyed by its START year.

    Historical Sportsdataverse releases use ending-year filenames:
      hoopR 2026 => 2025-26 season
    Live ESPN files use start-year filenames:
      espn_current 2026 => 2026-27 season
    """
    candidates = [
        os.path.join(RAW_DIR, f"game_logs_{year}.csv"),
        os.path.join(RAW_DIR, f"espn_current_games_{year}.csv"),
        os.path.join(RAW_DIR, f"hoopr_games_{year + 1}.csv"),
        os.path.join(RAW_DIR, f"scores_{year}.csv"),
    ]
    path = next((p for p in candidates if os.path.exists(p)), None)
    if path is None:
        print(f"  [WARN] Missing game log for canonical season {year}-{str(year+1)[-2:]}")
        return pd.DataFrame()

    df = pd.read_csv(path)
    if "game_id" in df.columns:
        df, _ = filter_games_for_source(df, RAW_DIR, path, label="game-model rows")

    rename = {
        "date":"game_date","visitor/neutral":"away_team",
        "home/neutral":"home_team","pts":"away_pts","pts.1":"home_pts",
    }
    df = df.rename(columns={k:v for k,v in rename.items() if k in df.columns})

    if "game_date" in df.columns:
        df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
        df = df[df["game_date"].notna()].copy()
        df["season"] = df["game_date"].map(nba_season_start_year)
        df = df[pd.to_numeric(df["season"], errors="coerce").eq(year)].copy()
    else:
        df["season"] = year

    if "game_id" in df.columns:
        ids = df["game_id"].astype(str).str.strip()
        nonblank = df["game_id"].notna() & ids.ne("") & ids.ne("nan")
        if ids[nonblank].duplicated().any():
            raise RuntimeError(f"Duplicate game_id rows in {os.path.basename(path)}")
        df = df.drop_duplicates(subset=["game_id"], keep="last").copy()

    if "home_pts" in df.columns and "away_pts" in df.columns:
        df["home_pts"] = pd.to_numeric(df["home_pts"], errors="coerce")
        df["away_pts"] = pd.to_numeric(df["away_pts"], errors="coerce")
        mask = df["home_pts"].notna() & df["away_pts"].notna()
        df.loc[mask, "actual_spread"] = df.loc[mask,"home_pts"] - df.loc[mask,"away_pts"]
        df.loc[mask, "actual_total"] = df.loc[mask,"home_pts"] + df.loc[mask,"away_pts"]

    print(f"  Game source: {os.path.basename(path)}")
    return df.reset_index(drop=True)


def load_odds(year: int) -> pd.DataFrame:
    """Load one canonical odds snapshot per game for the requested NBA season."""
    for fname in [f"odds_{year}.csv", "odds_consensus.csv", "odds_historical.csv"]:
        path = os.path.join(RAW_DIR, fname)
        if not os.path.exists(path):
            continue

        df = pd.read_csv(path)
        if df.empty:
            return df

        # Prefer the explicit Eastern slate date written by scrape_odds.py.
        if "game_date" in df.columns:
            df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce").dt.normalize()
        else:
            starts = pd.to_datetime(df.get("start_time", df.get("commence_time","")), errors="coerce", utc=True)
            try:
                df["game_date"] = starts.dt.tz_convert("America/New_York").dt.tz_localize(None).dt.normalize()
            except Exception:
                df["game_date"] = starts.dt.tz_localize(None).dt.normalize()

        if "season" in df.columns:
            df = df[pd.to_numeric(df["season"], errors="coerce").eq(year)].copy()
        else:
            season_start = pd.Timestamp(year=year, month=9, day=1)
            season_end = pd.Timestamp(year=year + 1, month=7, day=15)
            df = df[df["game_date"].between(season_start, season_end, inclusive="both")].copy()

        for col in ("home_team","away_team"):
            if col in df.columns:
                df[col] = df[col].map(normalize_team)

        keys = [k for k in ["game_date","home_team","away_team"] if k in df.columns]
        if keys:
            if "scraped_at" in df.columns:
                df["_scraped"] = pd.to_datetime(df["scraped_at"], errors="coerce", utc=True)
                df = df.sort_values("_scraped").drop_duplicates(keys, keep="last").drop(columns=["_scraped"])
            else:
                df = df.drop_duplicates(keys, keep="last")

        return df.reset_index(drop=True)
    return pd.DataFrame()


def load_team_stats(year: int) -> pd.DataFrame:
    for fname in [f"team_advanced_{year}.csv", f"team_stats_{year}.csv"]:
        path = os.path.join(RAW_DIR, fname)
        if os.path.exists(path):
            df = pd.read_csv(path)
            df["season"] = year
            df.columns = [c.strip().lower().replace("/","_").replace(" ","_") for c in df.columns]
            if "ortg" in df.columns and "drtg" in df.columns:
                df["net_rtg"] = pd.to_numeric(df["ortg"],errors="coerce") - pd.to_numeric(df["drtg"],errors="coerce")
            return df
    return pd.DataFrame()


# ── Feature Engineering ───────────────────────────────────────────────────────

def add_rest_days(df: pd.DataFrame) -> pd.DataFrame:
    """Compute true rest/fatigue from every prior team game, regardless of venue."""
    df = df.sort_values("game_date").reset_index(drop=True).copy()
    df["_row_id"] = np.arange(len(df))

    home = df[["_row_id","game_date","home_team"]].rename(columns={"home_team":"team"})
    home["side"] = "home"
    away = df[["_row_id","game_date","away_team"]].rename(columns={"away_team":"team"})
    away["side"] = "away"
    sched = pd.concat([home, away], ignore_index=True)
    sched = sched.sort_values(["team","game_date","_row_id"]).reset_index(drop=True)

    prev1 = sched.groupby("team")["game_date"].shift(1)
    prev2 = sched.groupby("team")["game_date"].shift(2)
    sched["rest_days"] = (sched["game_date"] - prev1).dt.days - 1
    sched["back_to_back"] = sched["rest_days"].eq(0)
    sched["three_in_four"] = (
        (sched["game_date"] - prev2).dt.days.le(3) & prev2.notna()
    )

    for side in ("home","away"):
        part = sched[sched["side"].eq(side)].set_index("_row_id")
        df[f"{side}_rest_days"] = df["_row_id"].map(part["rest_days"])
        df[f"{side}_back_to_back"] = df["_row_id"].map(part["back_to_back"]).fillna(False).astype(bool)
        df[f"{side}_three_in_four"] = df["_row_id"].map(part["three_in_four"]).fillna(False).astype(bool)

    return df.drop(columns=["_row_id"])


def add_travel_flag(df: pd.DataFrame) -> pd.DataFrame:
    WEST = WESTERN_CONFERENCE
    EAST = EASTERN_CONFERENCE

    def direction(away, home):
        if away in EAST and home in WEST: return "east_to_west"
        if away in WEST and home in EAST: return "west_to_east"
        return "same_region"

    df["travel_direction"] = df.apply(lambda r: direction(
        r.get("away_team",""), r.get("home_team","")), axis=1)
    df["long_travel"]  = df["travel_direction"].isin(["east_to_west","west_to_east"]).astype(int)
    df["east_to_west"] = (df["travel_direction"] == "east_to_west").astype(int)
    df["west_to_east"] = (df["travel_direction"] == "west_to_east").astype(int)
    return df


def add_rolling_efficiency(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """Rolling margin with shift(1), preserving exactly one row per game."""
    df = df.sort_values("game_date").reset_index(drop=True).copy()
    base_len = len(df)

    if "actual_spread" not in df.columns:
        return df

    home_rec = df[["game_date","home_team","actual_spread"]].copy()
    home_rec.columns = ["game_date","team","margin"]

    away_rec = df[["game_date","away_team","actual_spread"]].copy()
    away_rec["actual_spread"] = -away_rec["actual_spread"]
    away_rec.columns = ["game_date","team","margin"]

    all_rec = pd.concat([home_rec, away_rec], ignore_index=True)
    all_rec = all_rec[all_rec["margin"].notna()].copy()
    all_rec = all_rec.sort_values(["team","game_date"]).reset_index(drop=True)

    key_cols = ["game_date","team"]
    dupes = all_rec.duplicated(key_cols, keep=False)
    if dupes.any():
        sample = all_rec.loc[dupes, key_cols].head(10).to_dict("records")
        raise RuntimeError(
            f"Rolling efficiency history is not unique by team/date: "
            f"{int(dupes.sum())} duplicate rows; sample={sample}"
        )

    all_rec[f"rolling_{window}g"] = (
        all_rec.groupby("team")["margin"]
        .transform(lambda x: x.shift(1).rolling(window, min_periods=2).mean())
    )

    for side, prefix in [("home_team","home"), ("away_team","away")]:
        lookup = all_rec[["game_date","team",f"rolling_{window}g"]].copy()
        lookup.columns = ["game_date",side,f"{prefix}_rolling_{window}g"]
        lookup = lookup.drop_duplicates(["game_date",side], keep="last")
        df = df.merge(
            lookup,
            on=["game_date",side],
            how="left",
            validate="many_to_one",
        )
        if len(df) != base_len:
            raise RuntimeError(
                f"Rolling efficiency merge changed cardinality after {side}: "
                f"{base_len} -> {len(df)}"
            )

    df[f"rolling_diff_{window}g"] = (
        df.get(f"home_rolling_{window}g", 0).fillna(0) -
        df.get(f"away_rolling_{window}g", 0).fillna(0)
    )
    df[f"pred_spread_rolling_{window}g"] = df[f"rolling_diff_{window}g"] * 0.5
    return df


def add_model_labels(df: pd.DataFrame) -> pd.DataFrame:
    spread_col = "avg_spread_home_line"
    total_col  = "avg_total_point"

    if "actual_spread" in df.columns and spread_col in df.columns:
        valid = df[spread_col].notna() & df["actual_spread"].notna()
        df.loc[valid, "home_covered"] = df.loc[valid,"actual_spread"] > -df.loc[valid,spread_col]

    if "actual_total" in df.columns and total_col in df.columns:
        valid = df[total_col].notna() & df["actual_total"].notna()
        df.loc[valid, "went_over"] = df.loc[valid,"actual_total"] > df.loc[valid,total_col]

    return df


# ── Main Merge ────────────────────────────────────────────────────────────────

def process_year(year: int) -> pd.DataFrame:
    print(f"\n── Processing canonical season {year}-{str(year+1)[-2:]} ──")

    game_df  = load_game_logs(year)
    odds_df  = load_odds(year)
    stats_df = load_team_stats(year)

    if game_df.empty:
        print(f"  [SKIP] No game data for {year}")
        return pd.DataFrame()

    print(f"  Source game rows: {len(game_df)}")
    if "game_id" in game_df.columns:
        nonblank = game_df["game_id"].notna() & game_df["game_id"].astype(str).str.strip().ne("")
        if game_df.loc[nonblank, "game_id"].astype(str).duplicated().any():
            raise RuntimeError(f"Duplicate nonblank game_id rows detected for source {year}")

    for col in ("home_team", "away_team"):
        if col in game_df.columns:
            game_df[col] = game_df[col].map(normalize_team)

    before_franchise_filter = len(game_df)
    game_df = game_df[
        game_df["home_team"].isin(TEAM_NAMES) & game_df["away_team"].isin(TEAM_NAMES)
    ].copy()
    excluded_exhibitions = before_franchise_filter - len(game_df)
    if excluded_exhibitions:
        print(f"  Excluded non-franchise/exhibition games: {excluded_exhibitions}")

    game_df = add_rest_days(game_df)
    game_df = add_travel_flag(game_df)

    if "actual_spread" in game_df.columns:
        game_df = add_rolling_efficiency(game_df, window=5)
        game_df = add_rolling_efficiency(game_df, window=10)

    if not odds_df.empty:
        keys = [k for k in ["game_date","home_team","away_team"] if k in odds_df.columns and k in game_df.columns]
        if keys:
            before = len(game_df)
            merged = game_df.merge(odds_df, on=keys, how="left", validate="many_to_one")
            if len(merged) != before:
                raise RuntimeError(
                    f"Odds merge changed game cardinality: {before} -> {len(merged)}"
                )
            print(f"  Odds merged: {merged.shape}")
        else:
            merged = game_df.copy()
    else:
        merged = game_df.copy()
        print("  [INFO] No odds data for this year")

    if not stats_df.empty and "team" in stats_df.columns:
        for side, col in [("home_team","home"), ("away_team","away")]:
            side_stats = stats_df.copy()
            side_stats.columns = [
                f"{col}_{c}" if c not in ["team","season"] else c
                for c in side_stats.columns
            ]
            merged = merged.merge(
                side_stats.rename(columns={"team": side}),
                on=[side,"season"], how="left"
            )

    merged = add_model_labels(merged)

    if "game_date" in merged.columns and "season" in merged.columns:
        merged = merged.sort_values(["season","game_date"]).reset_index(drop=True)
        merged["season_game_num"] = merged.groupby("season").cumcount() + 1
        merged["month"] = pd.to_datetime(merged["game_date"]).dt.month
        # NBA playoffs run April - June; regular season is Oct - April.
        merged["is_playoff"] = merged["month"].isin([4, 5, 6]).astype(int)

    if len(merged) != len(game_df):
        raise RuntimeError(
            f"process_year changed game cardinality: source={len(game_df)} final={len(merged)}"
        )
    print(f"  Final shape: {merged.shape}")
    return merged


def main():
    global RAW_DIR, PROCESSED_DIR

    parser = argparse.ArgumentParser(description="Merge NBA stats + odds into model-ready dataset")
    parser.add_argument("--year", type=str, default="all")
    parser.add_argument("--raw", type=str, default=RAW_DIR)
    parser.add_argument("--out", type=str, default=PROCESSED_DIR)
    args = parser.parse_args()

    RAW_DIR       = args.raw
    PROCESSED_DIR = args.out
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    if args.year == "all":
        years = set()

        for f in glob(os.path.join(RAW_DIR, "hoopr_games_20*.csv")):
            token = os.path.basename(f).split("_")[-1].replace(".csv","")
            if token.isdigit():
                years.add(int(token) - 1)

        for pattern in ["espn_current_games_20*.csv", "game_logs_20*.csv", "scores_20*.csv"]:
            for f in glob(os.path.join(RAW_DIR, pattern)):
                token = os.path.basename(f).split("_")[-1].replace(".csv","")
                if token.isdigit():
                    years.add(int(token))

        years = sorted(y for y in years if y >= 2000)
        if not years:
            print("[WARN] No game log files found in data/raw/")
            print("       Run collect_stats.py first.")
            return
        print(f"Found canonical season-start years: {years}")
    else:
        years = [int(args.year)]

    all_dfs = []
    for year in years:
        df = process_year(year)
        if not df.empty:
            path = os.path.join(PROCESSED_DIR, f"master_{year}.csv")
            df.to_csv(path, index=False)
            print(f"  Saved → {path}")
            all_dfs.append(df)

    if len(all_dfs) > 1:
        master = pd.concat(all_dfs, ignore_index=True)
        master_path = os.path.join(PROCESSED_DIR, "master_all.csv")
        master.to_csv(master_path, index=False)
        print(f"\n✅ Master dataset → {master_path}")
        print(f"   Total rows: {len(master)} | Seasons: {master['season'].nunique()}")
    elif len(all_dfs) == 1:
        master = all_dfs[0]
        master.to_csv(os.path.join(PROCESSED_DIR, "master_all.csv"), index=False)
        print(f"\n✅ Master dataset saved ({len(master)} rows)")

    print("\n✅ Merge complete.")


if __name__ == "__main__":
    main()
