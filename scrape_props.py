"""
scrape_props.py — Pulls NBA player props from PrizePicks public API.
No auth, no key. JSON returned directly.

Endpoint: https://api.prizepicks.com/projections?league_id=7&per_page=250&single_stat=true
NBA league_id = 7

Always writes:
  data/raw/props_today.csv
  data/raw/props_raw_YYYY-MM-DD.csv
  data/raw/props_wide_YYYY-MM-DD.csv
  data/raw/props_fetch_status.json

Usage:
  python scrape_props.py --date 2026-11-05 --out data/raw
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import date, datetime, timezone

import pandas as pd
import requests

OUT_DIR = "data/raw"
LEAGUE_ID = 7   # NBA on PrizePicks
BASE_URL = "https://api.prizepicks.com"
PROJ_URL = f"{BASE_URL}/projections"

RAW_COLUMNS = [
    "game_date", "player", "team", "position", "opp_team", "is_home",
    "stat_raw", "stat", "line", "odds_type", "game_time", "projection_id",
    "source", "scraped_at"
]

WIDE_COLUMNS = [
    "game_date", "player", "team", "opp_team", "is_home", "game_time",
    "posted_pts", "posted_reb", "posted_ast", "posted_threes", "posted_pra"
]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Linux; Android 13; Tablet) AppleWebKit/537.36 Chrome/120 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://app.prizepicks.com/",
    "Origin": "https://app.prizepicks.com",
    "Content-Type": "application/json",
}

# PrizePicks stat_type → our model prop name
STAT_MAP = {
    "Points": "pts",
    "Rebounds": "reb",
    "Assists": "ast",
    "3-Point Made": "threes",
    "3-Pointers Made": "threes",
    "Pts+Rebs+Asts": "pra",
    "Points+Rebounds+Assists": "pra",
    "Pts+Asts": "pts_ast",
    "Pts+Rebs": "pts_reb",
    "Rebs+Asts": "reb_ast",
    "Blocked Shots": "blk",
    "Steals": "stl",
    "Turnovers": "tov",
    "Fantasy Score": "fantasy",
    "Minutes Played": "min",
}


def empty_raw_df() -> pd.DataFrame:
    return pd.DataFrame(columns=RAW_COLUMNS)


def empty_wide_df() -> pd.DataFrame:
    return pd.DataFrame(columns=WIDE_COLUMNS)


def write_status(out_dir: str, status: dict):
    path = os.path.join(out_dir, "props_fetch_status.json")
    with open(path, "w") as f:
        json.dump(status, f, indent=2)
    print(f"  Status → {path}")


def fetch_projections(league_id=LEAGUE_ID):
    """Fetch all active props for a league from PrizePicks public API."""
    params = {
        "league_id": league_id,
        "per_page": 250,
        "single_stat": "true",
    }
    resp = requests.get(PROJ_URL, headers=HEADERS, params=params, timeout=20)
    print(f"  PrizePicks HTTP status: {resp.status_code}")
    print(f"  Final URL: {resp.url}")

    if resp.status_code != 200:
        preview = resp.text[:300].replace("\n", " ")
        raise RuntimeError(f"PrizePicks returned HTTP {resp.status_code}: {preview}")

    try:
        return resp.json()
    except Exception as exc:
        preview = resp.text[:300].replace("\n", " ")
        raise RuntimeError(f"PrizePicks response was not JSON: {exc}; preview={preview}")


def parse_projections(data: dict, target_date: str) -> pd.DataFrame:
    projections = data.get("data", []) or []
    included = data.get("included", []) or []

    lookup = {obj.get("id"): obj for obj in included if obj.get("id") is not None}

    rows = []
    for proj in projections:
        if proj.get("type") != "projection":
            continue

        attrs = proj.get("attributes", {}) or {}
        rels = proj.get("relationships", {}) or {}

        player_ref = rels.get("new_player", rels.get("player", {})).get("data", {}) or {}
        player_id = player_ref.get("id")
        player_obj = lookup.get(player_id, {}) or {}
        player_attr = player_obj.get("attributes", {}) or {}

        game_ref = rels.get("game", {}).get("data", {}) or {}
        game_obj = lookup.get(game_ref.get("id", ""), {}) or {}
        game_attr = game_obj.get("attributes", {}) or {}

        stat_raw = attrs.get("stat_type") or attrs.get("stat_display") or ""
        stat_norm = STAT_MAP.get(stat_raw, str(stat_raw).lower().replace(" ", "_"))

        line = attrs.get("line_score", attrs.get("projection", None))
        if line is None:
            continue

        try:
            line = float(line)
        except Exception:
            continue

        rows.append({
            "game_date": target_date,
            "player": player_attr.get("display_name") or player_attr.get("name") or attrs.get("name", ""),
            "team": player_attr.get("team") or player_attr.get("team_name") or "",
            "position": player_attr.get("position", ""),
            "opp_team": str(attrs.get("description", "")).replace("vs", "").replace("@", "").strip(),
            "is_home": "@" not in str(attrs.get("description", "")),
            "stat_raw": stat_raw,
            "stat": stat_norm,
            "line": line,
            "odds_type": attrs.get("odds_type", "standard"),
            "game_time": attrs.get("start_time", game_attr.get("start_time", "")),
            "projection_id": proj.get("id"),
            "source": "prizepicks",
            "scraped_at": datetime.now(timezone.utc).isoformat(),
        })

    if not rows:
        return empty_raw_df()

    return pd.DataFrame(rows, columns=RAW_COLUMNS)


def pivot_to_model_format(df: pd.DataFrame) -> pd.DataFrame:
    """Reshape from long to one row per player-game."""
    if df.empty:
        return empty_wide_df()

    model_stats = {"pts", "reb", "ast", "threes", "pra"}
    df_model = df[df["stat"].isin(model_stats)].copy()
    if df_model.empty:
        return empty_wide_df()

    pivot = df_model.pivot_table(
        index=["game_date", "player", "team", "opp_team", "is_home", "game_time"],
        columns="stat",
        values="line",
        aggfunc="first"
    ).reset_index()

    pivot.columns = [c if isinstance(c, str) else c for c in pivot.columns]

    for stat in model_stats:
        if stat in pivot.columns:
            pivot.rename(columns={stat: f"posted_{stat}"}, inplace=True)

    for col in WIDE_COLUMNS:
        if col not in pivot.columns:
            pivot[col] = None

    return pivot[WIDE_COLUMNS]


def save_outputs(df: pd.DataFrame, target: str, out_dir: str):
    raw_path = os.path.join(out_dir, f"props_raw_{target}.csv")
    today_path = os.path.join(out_dir, "props_today.csv")
    wide_path = os.path.join(out_dir, f"props_wide_{target}.csv")
    wide_today_path = os.path.join(out_dir, "props_wide_today.csv")

    if df.empty:
        df = empty_raw_df()

    df.to_csv(raw_path, index=False)
    df.to_csv(today_path, index=False)

    pivot_df = pivot_to_model_format(df)
    pivot_df.to_csv(wide_path, index=False)
    pivot_df.to_csv(wide_today_path, index=False)

    hist_path = os.path.join(out_dir, "props_historical.csv")
    if not df.empty:
        if os.path.exists(hist_path):
            hist = pd.read_csv(hist_path)
            if "game_date" in hist.columns:
                hist = hist[hist["game_date"] != target]
            pd.concat([hist, df], ignore_index=True).to_csv(hist_path, index=False)
        else:
            df.to_csv(hist_path, index=False)

    print(f"  Saved → {today_path}")
    print(f"  Saved → {raw_path}")
    print(f"  Saved → {wide_path}")
    print(f"  Saved → {wide_today_path}")
    return raw_path, today_path, wide_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=None)
    parser.add_argument("--out", default="data/raw")
    parser.add_argument("--league-id", default=LEAGUE_ID, type=int, help="PrizePicks league ID (7=NBA)")
    args = parser.parse_args()

    target = args.date or str(date.today())
    os.makedirs(args.out, exist_ok=True)

    status = {
        "status": "unknown",
        "target_date": target,
        "league_id": args.league_id,
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "raw_projection_count": 0,
        "included_count": 0,
        "parsed_rows": 0,
        "points_rows": 0,
        "unique_players": 0,
        "error": None,
    }

    print(f"Fetching PrizePicks NBA props — {target}")

    try:
        data = fetch_projections(args.league_id)
        status["raw_projection_count"] = len(data.get("data", []) or [])
        status["included_count"] = len(data.get("included", []) or [])
        print(f"  Raw response: {status['raw_projection_count']} projections, {status['included_count']} included objects")

        df = parse_projections(data, target)
        status["parsed_rows"] = int(len(df))
        if not df.empty:
            status["points_rows"] = int(df["stat"].astype(str).str.lower().eq("pts").sum())
            status["unique_players"] = int(df["player"].nunique())
            print(f"  Parsed {len(df)} props for {df['player'].nunique()} players")
            print(f"  Stat types: {sorted(df['stat'].dropna().unique())}")
        else:
            print("  [WARN] PrizePicks returned no parsed props. Writing empty files with headers.")

        save_outputs(df, target, args.out)
        status["status"] = "ok" if not df.empty else "empty"

        if not df.empty:
            print("\n  Today's props sample:")
            sample = df[df["stat"].isin({"pts", "reb", "ast", "pra"})].sort_values("player")
            if sample.empty:
                sample = df.sort_values("player")
            print(sample[["player", "team", "opp_team", "stat", "line", "odds_type"]].head(15).to_string(index=False))

    except Exception as exc:
        status["status"] = "error"
        status["error"] = str(exc)
        print(f"  [ERROR] Props scrape failed: {exc}")
        print("  Writing empty props files so downstream steps can diagnose the issue.")
        save_outputs(empty_raw_df(), target, args.out)

    write_status(args.out, status)
    print("\n✅ Props scrape step complete.")


if __name__ == "__main__":
    main()
