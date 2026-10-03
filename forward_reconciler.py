"""
forward_reconciler.py
---------------------
Grades the canonical NBA forward ledger without changing any original prediction
field. Only resolution fields (actual/outcome/graded_at_utc) may be updated.

Sources:
  data/raw/scores_YYYY-MM-DD.csv / scores_today.csv
  data/raw/hoopr_player_box_*.csv

Outputs:
  data/history/nba_forward_predictions.jsonl
  data/tracking/nba_forward_performance.json
  data/tracking/nba_forward_certified.csv
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from glob import glob
from pathlib import Path

import pandas as pd

LEDGER = Path("data/history/nba_forward_predictions.jsonl")
RAW = Path("data/raw")
TRACKING = Path("data/tracking")


def f(value, default=None):
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def norm(value):
    return " ".join(str(value or "").strip().lower().replace("’", "'").split())


def read_ledger():
    rows = []
    if not LEDGER.exists():
        return rows
    for line in LEDGER.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            pass
    return rows


def load_scores(target):
    candidates = [RAW / f"scores_{target}.csv", RAW / "scores_today.csv"]
    for path in candidates:
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        out = {}
        for _, r in df.iterrows():
            status = str(r.get("STATUS", r.get("status", ""))).upper()
            if status and "FINAL" not in status:
                continue
            home = str(r.get("HOME", r.get("home_team", ""))).strip()
            away = str(r.get("AWAY", r.get("away_team", ""))).strip()
            hs = f(r.get("HOME_SCORE", r.get("home_score")))
            aas = f(r.get("AWAY_SCORE", r.get("away_score")))
            if not home or not away or hs is None or aas is None:
                continue
            out[norm(f"{away} @ {home}")] = {
                "home": home, "away": away,
                "home_score": hs, "away_score": aas,
                "margin": hs - aas, "total": hs + aas,
            }
        if out:
            return out
    return {}


def load_player_actuals(target):
    rows = {}
    for path in sorted(glob(str(RAW / "hoopr_player_box_20*.csv"))):
        try:
            df = pd.read_csv(path, low_memory=False)
        except Exception:
            continue
        if "game_date" not in df.columns:
            continue
        dates = pd.to_datetime(df["game_date"], errors="coerce").dt.date.astype(str)
        sub = df[dates.eq(target)].copy()
        if sub.empty:
            continue

        player_col = "athlete_display_name" if "athlete_display_name" in sub.columns else (
            "player_name" if "player_name" in sub.columns else None
        )
        if not player_col:
            continue

        def col(*names):
            for name in names:
                if name in sub.columns:
                    return pd.to_numeric(sub[name], errors="coerce")
            return pd.Series(0.0, index=sub.index)

        pts = col("points", "pts")
        reb = col("rebounds", "reb")
        ast = col("assists", "ast")
        threes = col("three_point_field_goals_made", "threes_made", "threes")

        for idx, r in sub.iterrows():
            player = str(r.get(player_col, "")).strip()
            if not player:
                continue
            p = f(pts.loc[idx], 0.0)
            rb = f(reb.loc[idx], 0.0)
            a = f(ast.loc[idx], 0.0)
            t = f(threes.loc[idx], 0.0)
            rows[norm(player)] = {
                "PTS": p, "REB": rb, "AST": a, "THREES": t,
                "PRA": p + rb + a,
            }
    return rows


def grade_side(actual, line, side):
    if actual is None or line is None:
        return None
    if abs(actual - line) < 1e-9:
        return "PUSH"
    side = str(side or "").upper()
    if side == "OVER":
        return "WIN" if actual > line else "LOSS"
    if side == "UNDER":
        return "WIN" if actual < line else "LOSS"
    return None


def resolve_row(row, scores, players, target, now):
    if row.get("date") != target or row.get("outcome") not in (None, "", "PENDING"):
        return False
    market = str(row.get("market", "")).upper()
    game = scores.get(norm(row.get("game")))
    actual = None
    outcome = None

    if market == "TOTAL":
        if not game:
            return False
        actual = game["total"]
        outcome = grade_side(actual, f(row.get("line")), row.get("side"))

    elif market == "SPREAD":
        if not game:
            return False
        line = f(row.get("line"))
        if line is None:
            return False
        side = str(row.get("side") or "").upper()
        if side == "HOME":
            actual = game["margin"]
            cover = actual + line
        elif side == "AWAY":
            actual = -game["margin"]
            cover = actual - line
        else:
            # PASS rows still receive the actual margin, but no betting outcome.
            row["actual"] = game["margin"]
            row["outcome"] = "PASS"
            row["graded_at_utc"] = now
            return True
        outcome = "PUSH" if abs(cover) < 1e-9 else ("WIN" if cover > 0 else "LOSS")

    elif market == "PROP":
        pdata = players.get(norm(row.get("player")))
        stat = str(row.get("stat", "")).upper()
        if not pdata or stat not in pdata:
            return False
        actual = pdata[stat]
        if str(row.get("side") or "").upper() == "PASS":
            outcome = "PASS"
        else:
            outcome = grade_side(actual, f(row.get("line")), row.get("side"))
    else:
        return False

    if outcome is None:
        return False
    row["actual"] = actual
    row["outcome"] = outcome
    row["graded_at_utc"] = now
    return True


def performance(rows):
    certified = [r for r in rows if r.get("outcome") in {"WIN", "LOSS", "PUSH"}]
    bets = [r for r in certified if r.get("decision") == "BET"]
    wins = sum(r.get("outcome") == "WIN" for r in bets)
    losses = sum(r.get("outcome") == "LOSS" for r in bets)
    pushes = sum(r.get("outcome") == "PUSH" for r in bets)

    by_market = {}
    for market in sorted({str(r.get("market")) for r in bets}):
        subset = [r for r in bets if str(r.get("market")) == market]
        w = sum(r.get("outcome") == "WIN" for r in subset)
        l = sum(r.get("outcome") == "LOSS" for r in subset)
        p = sum(r.get("outcome") == "PUSH" for r in subset)
        denom = w + l
        by_market[market] = {
            "bets": len(subset), "wins": w, "losses": l, "pushes": p,
            "hit_rate": round(w / denom, 4) if denom else None,
        }

    denom = wins + losses
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "ledger_rows": len(rows),
        "certified_rows": len(certified),
        "bet_rows": len(bets),
        "wins": wins, "losses": losses, "pushes": pushes,
        "hit_rate": round(wins / denom, 4) if denom else None,
        "by_market": by_market,
        "policy": "ONLY_FORWARD_ROWS_WITH_DECISION_BET_COUNT_TOWARD_HIT_RATE",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True)
    args = parser.parse_args()

    rows = read_ledger()
    if not rows:
        print(json.dumps({"status": "WAITING_FOR_FORWARD_PREDICTIONS", "date": args.date}, indent=2))
        return

    scores = load_scores(args.date)
    players = load_player_actuals(args.date)
    now = datetime.now(timezone.utc).isoformat()
    resolved = sum(resolve_row(r, scores, players, args.date, now) for r in rows)

    # Prediction fields are preserved byte-for-value in each object; only resolution
    # keys above are mutated. Rewrite is atomic at file level to resolve outcomes.
    tmp = LEDGER.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, allow_nan=False) + "\n")
    tmp.replace(LEDGER)

    TRACKING.mkdir(parents=True, exist_ok=True)
    perf = performance(rows)
    perf["target_date"] = args.date
    perf["newly_resolved"] = resolved
    perf["score_games_available"] = len(scores)
    perf["player_actuals_available"] = len(players)
    (TRACKING / "nba_forward_performance.json").write_text(
        json.dumps(perf, indent=2) + "\n", encoding="utf-8"
    )

    cert = pd.DataFrame([r for r in rows if r.get("outcome") in {"WIN", "LOSS", "PUSH"}])
    cert.to_csv(TRACKING / "nba_forward_certified.csv", index=False)
    print(json.dumps(perf, indent=2))


if __name__ == "__main__":
    main()
