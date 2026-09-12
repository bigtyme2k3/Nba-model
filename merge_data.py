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

from teams import EASTERN_CONFERENCE, WESTERN_CONFERENCE, normalize_team

RAW_DIR       = "data/raw"
PROCESSED_DIR = "data/processed"


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_game_logs(year: int) -> pd.DataFrame:
    # Preference order: a hand-provided game log, ESPN scoreboard scrape results,
    # then collect_stats.py's hoopR-derived per-season games file (already in this
    # same home_team/away_team/home_pts/away_pts/game_date shape).
    path = os.path.join(RAW_DIR, f"game_logs_{year}.csv")
    if not os.path.exists(path):
        path = os.path.join(RAW_DIR, f"scores_{year}.csv")
        if not os.path.exists(path):
            path = os.path.join(RAW_DIR, f"hoopr_games_{year}.csv")
            if not os.path.exists(path):
                print(f"  [WARN] Missing game log: year {year}")
                return pd.DataFrame()

    df = pd.read_csv(path)
    df["season"] = year

    rename = {
        "date":"game_date","visitor/neutral":"away_team",
        "home/neutral":"home_team","pts":"away_pts","pts.1":"home_pts",
    }
    df = df.rename(columns={k:v for k,v in rename.items() if k in df.columns})

    if "game_date" in df.columns:
        df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
        df = df[df["game_date"].notna()]

    if "home_pts" in df.columns and "away_pts" in df.columns:
        df["home_pts"]      = pd.to_numeric(df["home_pts"], errors="coerce")
        df["away_pts"]      = pd.to_numeric(df["away_pts"], errors="coerce")
        mask = df["home_pts"].notna() & df["away_pts"].notna()
        df.loc[mask, "actual_spread"] = df.loc[mask,"home_pts"] - df.loc[mask,"away_pts"]
        df.loc[mask, "actual_total"]  = df.loc[mask,"home_pts"] + df.loc[mask,"away_pts"]

    return df


def load_odds(year: int) -> pd.DataFrame:
    for fname in [f"odds_{year}.csv", "odds_consensus.csv", "odds_historical.csv"]:
        path = os.path.join(RAW_DIR, fname)
        if os.path.exists(path):
            df = pd.read_csv(path)
            df["start_time"] = pd.to_datetime(df.get("start_time", df.get("game_date","")), errors="coerce")
            df["game_date"]  = df["start_time"].dt.normalize()
            if "season" in df.columns:
                df = df[df["season"] == year].copy()
            return df
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
    """Compute rest days with correct index alignment."""
    df = df.sort_values("game_date").reset_index(drop=True)

    for side in ["home_team", "away_team"]:
        prefix   = side.split("_")[0]
        rest_col = f"{prefix}_rest_days"
        b2b_col  = f"{prefix}_back_to_back"
        t4_col   = f"{prefix}_three_in_four"

        df[rest_col] = np.nan
        df[b2b_col]  = False
        df[t4_col]   = False

        for team in df[side].dropna().unique():
            idx = df.index[df[side] == team].tolist()
            dates = df.loc[idx, "game_date"]

            prev1 = dates.shift(1)
            rest  = (dates.values - prev1.values) / np.timedelta64(1,"D") - 1
            df.loc[idx, rest_col] = rest
            df.loc[idx, b2b_col]  = rest == 0

            prev2  = dates.shift(2)
            span2  = (dates.values - prev2.values) / np.timedelta64(1,"D")
            df.loc[idx, t4_col] = span2 <= 3

    return df


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
    """Rolling margin with shift(1) to prevent data leakage."""
    df = df.sort_values("game_date").reset_index(drop=True)

    if "actual_spread" not in df.columns:
        return df

    home_rec = df[["game_date","home_team","actual_spread"]].copy()
    home_rec.columns = ["game_date","team","margin"]

    away_rec = df[["game_date","away_team","actual_spread"]].copy()
    away_rec["actual_spread"] = -away_rec["actual_spread"]
    away_rec.columns = ["game_date","team","margin"]

    all_rec = pd.concat([home_rec, away_rec]).sort_values(["team","game_date"]).reset_index(drop=True)
    all_rec = all_rec[all_rec["margin"].notna()]

    all_rec[f"rolling_{window}g"] = (
        all_rec.groupby("team")["margin"]
        .transform(lambda x: x.shift(1).rolling(window, min_periods=2).mean())
    )

    home_m = all_rec[["game_date","team",f"rolling_{window}g"]].copy()
    home_m.columns = ["game_date","home_team",f"home_rolling_{window}g"]
    away_m = all_rec[["game_date","team",f"rolling_{window}g"]].copy()
    away_m.columns = ["game_date","away_team",f"away_rolling_{window}g"]

    df = df.merge(home_m, on=["game_date","home_team"], how="left")
    df = df.merge(away_m, on=["game_date","away_team"], how="left")

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
    print(f"\n── Processing season {year}-{year+1} ──")

    game_df  = load_game_logs(year)
    odds_df  = load_odds(year)
    stats_df = load_team_stats(year)

    if game_df.empty:
        print(f"  [SKIP] No game data for {year}")
        return pd.DataFrame()

    for col in ("home_team", "away_team"):
        if col in game_df.columns:
            game_df[col] = game_df[col].map(normalize_team)

    game_df = add_rest_days(game_df)
    game_df = add_travel_flag(game_df)

    if "actual_spread" in game_df.columns:
        game_df = add_rolling_efficiency(game_df, window=5)
        game_df = add_rolling_efficiency(game_df, window=10)

    if not odds_df.empty:
        keys = [k for k in ["game_date","home_team","away_team"] if k in odds_df.columns and k in game_df.columns]
        if keys:
            merged = game_df.merge(odds_df, on=keys, how="left")
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
        patterns = [
            glob(os.path.join(RAW_DIR, "game_logs_*.csv")),
            glob(os.path.join(RAW_DIR, "scores_20*.csv")),
            glob(os.path.join(RAW_DIR, "hoopr_games_20*.csv")),
        ]
        files = [f for group in patterns for f in group]
        years = sorted(set([
            int(f.split("_")[-1].replace(".csv",""))
            for f in files
            if f.split("_")[-1].replace(".csv","").isdigit()
        ]))
        if not years:
            print("[WARN] No game log files found in data/raw/")
            print("       Run collect_stats.py or scrape_scores.py first.")
            return
        print(f"Found seasons: {years}")
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
