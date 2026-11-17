"""
data_bets_report.py
-------------------
Follow-vs-fade record of V1's data-collection bets (daily_runner.DATA_BETS_PER_DAY
strongest prop opinions, flat 1 unit; user decision 2026-11-17).

  live           BET rows in the forward ledger with bet_tier DATA_COLLECTION, graded at
                 the recorded price (follow) and at the opposite side's recorded price
                 at the same book and line (fade).
  retrospective  the same rule applied to earlier dates' recorded props (each day's
                 10 largest raw-model-minus-market gaps, one per player, sportsbooks
                 only, no availability block): what the rule would have shown. Kept
                 separate because those rows were not issued as bets at the time.

Fading is profitable only if the model's side wins less often than the opposite
price's break-even (about 47.6% at -110 both ways).

  phases         EARLY (morning read) and ADJUSTED (after the injury report) picks from
                 data/history/nba_model_picks.jsonl (model_picks.py), graded on the
                 forward ledger's actual stat lines, with how often the adjusted pass
                 changed or dropped the early pick.

Usage:
    python data_bets_report.py
Output:
    data/tracking/data_bets_report.json
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

LEDGER = Path("data/history/nba_forward_predictions.jsonl")
OUT = Path("data/tracking/data_bets_report.json")
RAW = Path("data/raw")
BOOKS = {"fanduel", "draftkings", "fanatics"}
PER_DAY = 10


def decimal(odds):
    o = float(odds)
    return 1 + o / 100 if o > 0 else 1 + 100 / abs(o)


def opposite_from_pair(odds, no_vig):
    """Opposite price implied by the recorded side's price and the pair's no-vig
    probability (exact when both came from the same book's two-way market)."""
    try:
        imp, p = 1 / decimal(odds), float(no_vig)
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    if not 0 < p < 1:
        return None
    io = imp * (1 - p) / p
    if not 0 < io < 1:
        return None
    return round(-100 * io / (1 - io)) if io >= 0.5 else round(100 * (1 - io) / io)


def quoted_opposite(row, cache):
    path = RAW / f"sportsbook_props_{row['date']}.csv"
    if path not in cache:
        cache[path] = pd.read_csv(path, low_memory=False) if path.exists() else None
    q = cache[path]
    if q is None:
        return None
    other = "UNDER" if row["side"] == "OVER" else "OVER"
    m = q[(q["book_key"] == row["sportsbook_key"]) & (q["player"].str.lower() == str(row["player"]).lower())
          & (q["stat"].str.upper() == row["stat"]) & (q["line"] == row["line"]) & (q["side"] == other)]
    return float(m["price"].iloc[-1]) if len(m) else None


def record(rows: pd.DataFrame) -> dict:
    if rows.empty:
        return {"bets": 0}
    won = rows["outcome"].eq("WIN")
    follow = [decimal(o) - 1 if w else -1.0 for o, w in zip(rows["odds"], won)]
    fade = [(decimal(o) - 1 if not w else -1.0) for o, w in zip(rows["opp_odds"], won)]
    return {"bets": int(len(rows)), "model_wins": int(won.sum()), "model_losses": int((~won).sum()),
            "model_win_rate": round(float(won.mean()), 4),
            "follow_units": round(sum(follow), 2), "follow_roi": round(sum(follow) / len(rows), 4),
            "fade_units": round(sum(fade), 2), "fade_roi": round(sum(fade) / len(rows), 4)}


def breakdown(rows: pd.DataFrame) -> dict:
    return {"overall": record(rows),
            "by_stat": {k: record(g) for k, g in rows.groupby("stat")},
            "by_side": {k: record(g) for k, g in rows.groupby("side")},
            "by_date": {k: record(g) for k, g in rows.groupby("date")}}


def load(path=LEDGER) -> pd.DataFrame:
    rows = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    df = pd.DataFrame([r for r in rows if r.get("market") == "PROP"])
    for c in ("line", "odds", "market_probability", "raw_model_probability", "opposite_odds"):
        if c not in df:
            df[c] = None
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def with_opposite(rows: pd.DataFrame) -> pd.DataFrame:
    rows = rows.copy()
    cache, opp = {}, []
    for r in rows.to_dict("records"):
        o = r.get("opposite_odds")
        if o is None or (isinstance(o, float) and math.isnan(o)):
            o = quoted_opposite(r, cache) or opposite_from_pair(r["odds"], r["market_probability"])
        opp.append(o)
    rows["opp_odds"] = opp
    return rows.dropna(subset=["opp_odds"])


def retrospective(df: pd.DataFrame) -> pd.DataFrame:
    g = df[df["outcome"].isin(["WIN", "LOSS"]) & df["sportsbook_key"].isin(BOOKS)
           & df["raw_model_probability"].notna() & df["market_probability"].notna()].copy()
    if "bet_tier" in g:
        g = g[g["bet_tier"].ne("DATA_COLLECTION")]
    g["strength"] = g["raw_model_probability"] - g["market_probability"]
    picks = []
    for _, day in g.sort_values("strength", ascending=False).groupby("date"):
        picks.append(day.drop_duplicates("player").head(PER_DAY))
    return pd.concat(picks) if picks else g.iloc[:0]


PICKS = Path("data/history/nba_model_picks.jsonl")


def actuals(df: pd.DataFrame) -> dict:
    a = df.dropna(subset=["actual"]) if "actual" in df else df.iloc[:0]
    return {(r["date"], " ".join(str(r["player"]).lower().split()), str(r["stat"]).upper()): float(r["actual"])
            for r in a.to_dict("records")}


def graded_picks(ledger: pd.DataFrame, path=PICKS) -> pd.DataFrame:
    if not Path(path).exists():
        return pd.DataFrame()
    p = pd.DataFrame([json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()])
    if p.empty:
        return p
    act = actuals(ledger)
    p["actual"] = [act.get((d, " ".join(str(n).lower().split()), s)) for d, n, s in zip(p["date"], p["player"], p["stat"])]
    p = p.dropna(subset=["actual"]).copy()
    over = p["actual"] > p["line"]
    under = p["actual"] < p["line"]
    p["outcome"] = ["PUSH" if not (o or u) else ("WIN" if (o if s == "OVER" else u) else "LOSS")
                    for o, u, s in zip(over, under, p["side"])]
    p = p[p["outcome"].ne("PUSH")]
    cache, opp = {}, []
    for r in p.to_dict("records"):
        o = _num(r.get("opposite_odds"))
        opp.append(o if o is not None else (quoted_opposite({**r, "sportsbook_key": r["book_key"]}, cache)
                                            or opposite_from_pair(r["odds"], r["market_probability"])))
    p["opp_odds"] = opp
    return p.dropna(subset=["opp_odds"])


def _num(v):
    try:
        v = float(v)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def phase_changes(path=PICKS) -> dict:
    if not Path(path).exists():
        return {}
    p = pd.DataFrame([json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()])
    if p.empty or not {"EARLY", "ADJUSTED"} <= set(p["phase"]):
        return {}
    early = p[p["phase"].eq("EARLY")].set_index(["date", "player", "stat"])
    adj = p[p["phase"].eq("ADJUSTED")].set_index(["date", "player", "stat"])
    both = early.join(adj, how="inner", lsuffix="_early", rsuffix="_adj")
    days = sorted(set(early.index.get_level_values(0)) & set(adj.index.get_level_values(0)))
    e_days = early[early.index.get_level_values(0).isin(days)]
    return {"dates_with_both": len(days), "early_picks": int(len(e_days)),
            "kept_same_side": int((both["side_early"] == both["side_adj"]).sum()),
            "flipped_side": int((both["side_early"] != both["side_adj"]).sum()),
            "dropped_after_report": int(len(e_days) - len(both))}


def main():
    df = load()
    tier = df["bet_tier"] if "bet_tier" in df else pd.Series(None, index=df.index)
    live = df[tier.eq("DATA_COLLECTION") & df["decision"].eq("BET") & df["outcome"].isin(["WIN", "LOSS"])]
    live_dates = set(live["date"])
    retro = retrospective(df[~df["date"].isin(live_dates)])
    report = {"generated_at_utc": datetime.now(timezone.utc).isoformat(),
              "rule": f"top {PER_DAY} per day by raw model probability minus no-vig market probability; flat 1 unit",
              "fade_break_even_note": "fading needs the model side to win less than ~47.6% at -110 both ways",
              "live": breakdown(with_opposite(live)) if len(live) else {"overall": {"bets": 0}},
              "retrospective_same_rule": breakdown(with_opposite(retro))}
    picks = graded_picks(df)
    report["phases"] = {ph: breakdown(picks[picks["phase"].eq(ph)]) if len(picks) and (picks["phase"] == ph).any()
                        else {"overall": {"bets": 0}} for ph in ("EARLY", "ADJUSTED")}
    report["phase_changes"] = phase_changes()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(json.dumps({"live": report["live"]["overall"], "retrospective": report["retrospective_same_rule"]["overall"],
                      "early": report["phases"]["EARLY"]["overall"], "adjusted": report["phases"]["ADJUSTED"]["overall"]},
                     indent=2))


if __name__ == "__main__":
    main()
