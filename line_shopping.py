"""
line_shopping.py
----------------
What taking the best line and price across FanDuel, DraftKings and Fanatics is
worth, measured on graded near-close quotes from backfill_historical_lines.py.

Every side of every spread and total is "bet" for 1 unit at each book, at the
best-of-books quote, and at a randomly chosen book. Betting both sides of every
game cancels any selection effect, so differences in realized ROI come from price
and line alone. Best-of-books picks the most favourable line for the side, then
the best price at that line.

Usage:
    python line_shopping.py --season 2025
Output:
    data/tracking/line_shopping_<season>.json
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import backfill_historical_lines as bh
import spread_model as sm
from game_markets import decimal


def graded_offers(season: int, cache: Path, master: str) -> pd.DataFrame:
    """Latest pregame quote per game/book/market/side, graded against the final score."""
    offers = bh.latest_offers(cache, season)
    games = bh.consolidate(cache, season)[["event_id", "game_date", "home_team", "away_team"]]
    games["game_date"] = pd.to_datetime(games["game_date"]).dt.date
    with contextlib.redirect_stdout(io.StringIO()):
        results = sm.load_and_prepare(master)
    results = results[results["game_date"] >= pd.Timestamp(f"{season}-08-01")]
    results = results.assign(game_date=results["game_date"].dt.date)[
        ["game_date", "home_team", "away_team", "actual_spread", "actual_total"]]
    keyed = games.merge(results, on=["game_date", "home_team", "away_team"], how="left")
    # hoopR can date late tips by UTC; retry unmatched games one day later.
    miss = keyed["actual_spread"].isna()
    if miss.any():
        retry = games[miss.values].assign(game_date=lambda g: g["game_date"] + pd.Timedelta(days=1)) \
            .merge(results, on=["game_date", "home_team", "away_team"], how="left")
        keyed.loc[miss, ["actual_spread", "actual_total"]] = retry[["actual_spread", "actual_total"]].values
    df = offers.merge(keyed.dropna(subset=["actual_spread"])[["event_id", "actual_spread", "actual_total"]],
                      on="event_id")
    home_margin = df["actual_spread"]
    df["margin_vs_line"] = np.select(
        [df["side"] == "HOME", df["side"] == "AWAY", df["side"] == "OVER", df["side"] == "UNDER"],
        [home_margin + df["line"], -home_margin + df["line"],
         df["actual_total"] - df["line"], df["line"] - df["actual_total"]])
    df["profit"] = np.where(df["margin_vs_line"] > 0, df["price"].map(decimal) - 1,
                            np.where(df["margin_vs_line"] < 0, -1.0, 0.0))
    return df


def best_of_books(df: pd.DataFrame) -> pd.DataFrame:
    """Most favourable line for the side, then the best price at that line."""
    # For every side a higher line is better (HOME/AWAY get more points, OVER... is
    # the exception: a lower total is better for OVER).
    better = np.where(df["side"] == "OVER", -df["line"], df["line"])
    ranked = df.assign(_line_rank=better, _dec=df["price"].map(decimal))
    ranked = ranked.sort_values(["_line_rank", "_dec"], ascending=False)
    return ranked.drop_duplicates(["event_id", "market", "side"])


def summarize(df: pd.DataFrame, seed: int = 0) -> dict:
    out = {}
    rng = np.random.default_rng(seed)
    for market in ("SPREAD", "TOTAL"):
        m = df[df["market"] == market]
        multi = m.groupby(["event_id", "side"])["book_key"].transform("nunique") >= 2
        m = m[multi]
        best = best_of_books(m)
        per_book = {b: {"bets": int(len(g)), "roi": float(g["profit"].mean())}
                    for b, g in m.groupby("book_key")}
        random_pick = m.groupby(["event_id", "side"], group_keys=False).apply(
            lambda g: g.iloc[[rng.integers(len(g))]])
        med = m.groupby(["event_id", "side"], as_index=False)["line"].median().rename(columns={"line": "median_line"})
        cmp = best.merge(med, on=["event_id", "side"])
        line_gain = np.where(cmp["side"] == "OVER", cmp["median_line"] - cmp["line"], cmp["line"] - cmp["median_line"])
        # Combined hold of the best price on each side, at games where both best
        # sides share a line (a true two-way market).
        hold = []
        for _, g in best.groupby("event_id"):
            if len(g) == 2 and abs(abs(g["line"].iloc[0]) - abs(g["line"].iloc[1])) < 1e-9:
                hold.append(sum(1 / g["price"].map(decimal)) - 1)
        single_hold = []
        for _, g in m.groupby(["event_id", "book_key"]):
            if len(g) == 2:
                single_hold.append(sum(1 / g["price"].map(decimal)) - 1)
        out[market] = {
            "games": int(m["event_id"].nunique()),
            "side_bets": int(len(best)),
            "roi_by_book": per_book,
            "roi_random_book": float(random_pick["profit"].mean()),
            "roi_best_of_books": float(best["profit"].mean()),
            "gain_vs_random_book": float(best["profit"].mean() - random_pick["profit"].mean()),
            "avg_points_better_than_median_line": float(np.mean(line_gain)),
            "share_sides_with_better_line_than_median": float(np.mean(np.asarray(line_gain) > 0)),
            "avg_single_book_hold": float(np.mean(single_hold)) if single_hold else None,
            "avg_best_price_hold_same_line": float(np.mean(hold)) if hold else None,
            "games_with_negative_hold": int(sum(h < 0 for h in hold)),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=2025)
    ap.add_argument("--data", default="data/processed/master_all.csv")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    df = graded_offers(a.season, Path("data/raw/historical_lines"), a.data)
    report = {"generated_at_utc": datetime.now(timezone.utc).isoformat(), "season": a.season,
              "method": "1u on every side of every game at each book, best-of-books and a random book; "
                        "latest pregame near-close quotes; graded on final results",
              **summarize(df)}
    out = Path(a.out or f"data/tracking/line_shopping_{a.season}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
