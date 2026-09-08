"""
line_movement.py
----------------
Tracks opening vs closing line movement to detect sharp money.
High CLV (closing line value) = model predicted correctly before market adjusted.

Logic:
  - Save odds when first scraped (opening line)
  - Re-scrape odds 2 hours before tip (closing line)
  - Large movement TOWARD our model's prediction = sharp money agrees with us
  - Large movement AWAY = fade signal

Output:
  data/raw/line_movement_today.csv  — today's movement summary
  data/raw/clv_log.csv             — historical CLV tracker

Usage:
    python line_movement.py                    # update today's lines
    python line_movement.py --closing          # capture closing lines
    python line_movement.py --report           # print CLV report
"""

import os, json, glob, argparse
import pandas as pd
import numpy as np
from datetime import date, datetime

OUT_DIR     = "data/raw"
CLV_LOG     = "data/raw/clv_log.csv"
PRED_DIR    = "predictions"


def load_odds_snapshot(target_date: str, snapshot: str = "opening") -> pd.DataFrame:
    """
    Load odds file for a date.
    snapshot = 'opening' | 'closing'
    """
    fname = f"odds_{snapshot}_{target_date}.csv"
    path  = os.path.join(OUT_DIR, fname)
    if not os.path.exists(path):
        path = os.path.join(OUT_DIR, f"odds_{target_date}.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)


def compute_movement(opening: pd.DataFrame, closing: pd.DataFrame) -> pd.DataFrame:
    """Compare opening vs closing lines to detect movement."""
    if opening.empty or closing.empty:
        return pd.DataFrame()

    merge_keys = ["home_team","away_team"]
    cols = merge_keys + ["spread_home","total"]

    op = opening[cols].copy() if all(c in opening.columns for c in cols) else pd.DataFrame()
    cl = closing[cols].copy() if all(c in closing.columns for c in cols) else pd.DataFrame()

    if op.empty or cl.empty:
        return pd.DataFrame()

    merged = op.merge(cl, on=merge_keys, suffixes=("_open","_close"))

    merged["spread_movement"] = merged["spread_home_close"] - merged["spread_home_open"]
    merged["total_movement"]  = merged["total_close"] - merged["total_open"]
    merged["spread_moved"]    = merged["spread_movement"].abs() >= 0.5
    merged["total_moved"]     = merged["total_movement"].abs() >= 0.5

    return merged


def compute_clv(predictions: dict, closing_odds: pd.DataFrame,
                actual_results: dict = None) -> list:
    """
    For each best bet, compute:
    - CLV = model edge at time of prediction vs closing line
    - Result = WIN/LOSS if actual results available
    """
    rows = []
    bets = predictions.get("best_bets", [])

    for bet in bets:
        game_key = bet["game"]
        parts    = game_key.split(" @ ")
        if len(parts) != 2:
            continue
        away, home = parts[0], parts[1]

        game_odds = closing_odds[
            (closing_odds.get("home_team","") == home) &
            (closing_odds.get("away_team","") == away)
        ] if not closing_odds.empty and "home_team" in closing_odds.columns else pd.DataFrame()

        closing_spread = game_odds["spread_home"].iloc[0] if not game_odds.empty else None
        closing_total  = game_odds["total"].iloc[0]  if not game_odds.empty else None

        edge       = bet.get("edge", 0) or 0
        bet_type   = bet.get("type","")

        clv = None
        if bet_type == "SPREAD" and closing_spread is not None:
            clv = edge  # Simplified: our edge vs closing line IS the CLV
        elif bet_type == "TOTAL" and closing_total is not None:
            clv = edge

        rows.append({
            "date":           predictions.get("date",""),
            "game":           game_key,
            "type":           bet_type,
            "play":           bet.get("play",""),
            "model_edge":     edge,
            "closing_spread": closing_spread,
            "closing_total":  closing_total,
            "clv":            clv,
            "conf":           bet.get("conf",""),
            "stars":          bet.get("stars",1),
        })

    return rows


def update_clv_log(new_rows: list):
    """Append new CLV entries to historical log."""
    if not new_rows:
        return
    new_df = pd.DataFrame(new_rows)
    if os.path.exists(CLV_LOG):
        existing = pd.read_csv(CLV_LOG)
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["date","game","type","play"])
    else:
        combined = new_df
    combined.to_csv(CLV_LOG, index=False)
    print(f"  CLV log updated → {CLV_LOG} ({len(combined)} entries)")


def clv_report() -> dict:
    """Print a CLV summary report."""
    if not os.path.exists(CLV_LOG):
        print("  No CLV log yet. Run after some games complete.")
        return {}

    df = pd.read_csv(CLV_LOG)
    df = df[df["clv"].notna()]

    if df.empty:
        return {}

    print("\n═══ Closing Line Value Report ═══\n")

    avg_clv = df["clv"].mean()
    pos_clv = (df["clv"] > 0).mean()
    print(f"  Average CLV:          {avg_clv:+.2f} pts")
    print(f"  % positive CLV:       {pos_clv:.1%}")
    print(f"  Total bets tracked:   {len(df)}")

    print(f"\n  By confidence:")
    for conf in ["HIGH","MED","LOW"]:
        sub = df[df["conf"]==conf]
        if not sub.empty:
            print(f"    {conf}: avg CLV={sub['clv'].mean():+.2f}  n={len(sub)}")

    print(f"\n  By type:")
    for t in ["SPREAD","TOTAL","PROP"]:
        sub = df[df["type"]==t]
        if not sub.empty:
            print(f"    {t}: avg CLV={sub['clv'].mean():+.2f}  n={len(sub)}")

    print(f"\n  Best CLV bets:")
    best = df.nlargest(5,"clv")[["date","play","clv","conf"]]
    print(best.to_string(index=False))

    return {
        "avg_clv": round(avg_clv, 2),
        "pos_clv_pct": round(pos_clv, 3),
        "total_bets": len(df),
    }


def save_opening_lines(target_date: str):
    """Copy today's odds file as the 'opening' snapshot. Call right after scrape_odds.py runs."""
    src = os.path.join(OUT_DIR, f"odds_{target_date}.csv")
    dst = os.path.join(OUT_DIR, f"odds_opening_{target_date}.csv")
    if os.path.exists(src) and not os.path.exists(dst):
        import shutil
        shutil.copy2(src, dst)
        print(f"  Opening lines saved → {dst}")


def save_closing_lines(target_date: str):
    """Copy current odds file as the 'closing' snapshot. Call ~2 hours before first tip-off."""
    src = os.path.join(OUT_DIR, f"odds_{target_date}.csv")
    dst = os.path.join(OUT_DIR, f"odds_closing_{target_date}.csv")
    if os.path.exists(src):
        import shutil
        shutil.copy2(src, dst)
        print(f"  Closing lines saved → {dst}")


def daily_movement_report(target_date: str):
    """Full daily workflow: compare opening vs closing, compute CLV."""
    print(f"\n═══ Line Movement — {target_date} ═══\n")

    opening = load_odds_snapshot(target_date, "opening")
    closing = load_odds_snapshot(target_date, "closing")

    if opening.empty:
        print("  No opening lines found.")
        return

    movement = compute_movement(opening, closing)
    if not movement.empty:
        moved = movement[movement["spread_moved"] | movement["total_moved"]]
        if not moved.empty:
            print(f"  Line movement detected in {len(moved)} games:")
            for _, row in moved.iterrows():
                if row["spread_moved"]:
                    print(f"    {row['away_team']} @ {row['home_team']}: "
                          f"spread {row['spread_home_open']:+.1f} → {row['spread_home_close']:+.1f} "
                          f"({row['spread_movement']:+.1f})")
                if row["total_moved"]:
                    print(f"    {row['away_team']} @ {row['home_team']}: "
                          f"total {row['total_open']:.1f} → {row['total_close']:.1f} "
                          f"({row['total_movement']:+.1f})")

            path = os.path.join(OUT_DIR, f"line_movement_{target_date}.csv")
            movement.to_csv(path, index=False)
            print(f"\n  Movement saved → {path}")
        else:
            print("  No significant line movement today.")

    pred_path = os.path.join(PRED_DIR, f"predictions_{target_date}.json")
    if os.path.exists(pred_path):
        with open(pred_path) as f:
            preds = json.load(f)
        clv_rows = compute_clv(preds, closing)
        update_clv_log(clv_rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date",    default=str(date.today()))
    parser.add_argument("--opening", action="store_true", help="Save opening line snapshot")
    parser.add_argument("--closing", action="store_true", help="Save closing line snapshot")
    parser.add_argument("--report",  action="store_true", help="Print CLV report")
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(PRED_DIR, exist_ok=True)

    if args.opening:
        save_opening_lines(args.date)
    elif args.closing:
        save_closing_lines(args.date)
    elif args.report:
        clv_report()
    else:
        daily_movement_report(args.date)
        print("\n✅ Line movement tracking complete.")


if __name__ == "__main__":
    main()
