"""
game_predictions_report.py
--------------------------
Game-level prediction record (regular season): winner, spread and total.

  winner   the model's predicted winner (sign of its projected margin, from
           predictions/predictions_<date>.json) against the final score, next to the
           market favorite's record (from the recorded spread quote).
  spread   the model's side of every graded forward-ledger spread row, at the recorded
           price (follow) and at the opposite side's price (fade).
  total    the same for totals.
  error    mean absolute error of the model's margin and total, and of the market's
           total line, on the same games.

Usage:
    python game_predictions_report.py
Output:
    data/tracking/game_predictions_report.json
"""
from __future__ import annotations

import glob
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from data_bets_report import decimal, opposite_from_pair
from teams import normalize_team

LEDGER = Path("data/history/nba_forward_predictions.jsonl")
SCORES = Path("data/raw/scores.csv")
OUT = Path("data/tracking/game_predictions_report.json")
SEASON_START = "2026-10-27"


def _key(date, away, home):
    return f"{date}|{normalize_team(str(away))}|{normalize_team(str(home))}"


def finals(path=SCORES) -> dict:
    s = pd.read_csv(path, low_memory=False)
    s = s[s["is_final"].astype(str).str.lower().eq("true")]
    return {_key(r.game_date, r.away_team, r.home_team): (float(r.home_score), float(r.away_score))
            for r in s.itertuples()}


def market_home_lines(ledger: list[dict]) -> dict:
    """Home spread from the first recorded spread quote of each game (negative = home favored)."""
    out = {}
    for r in ledger:
        if r.get("market") != "SPREAD" or r.get("line") is None:
            continue
        away, _, home = str(r.get("game", "")).partition(" @ ")
        k = _key(r["date"], away, home)
        if k not in out:
            line = float(r["line"])
            out[k] = line if r.get("side") == "HOME" else -line
    return out


def follow_fade(rows: list[dict]) -> dict:
    graded = [r for r in rows if r.get("outcome") in ("WIN", "LOSS") and r.get("odds") is not None]
    if not graded:
        return {"games": 0}
    follow, fade, wins = 0.0, 0.0, 0
    for r in graded:
        won = r["outcome"] == "WIN"
        wins += won
        follow += decimal(r["odds"]) - 1 if won else -1.0
        opp = opposite_from_pair(r["odds"], r.get("market_probability"))
        fade += (decimal(opp) - 1 if not won else -1.0) if opp is not None else 0.0
    n = len(graded)
    return {"games": n, "model_wins": wins, "model_losses": n - wins, "model_win_rate": round(wins / n, 4),
            "follow_units": round(follow, 2), "follow_roi": round(follow / n, 4),
            "fade_units": round(fade, 2), "fade_roi": round(fade / n, 4),
            "qualified_bets": f"{sum(r['outcome'] == 'WIN' for r in graded if r.get('decision') == 'BET')}-"
                              f"{sum(r['outcome'] == 'LOSS' for r in graded if r.get('decision') == 'BET')}"}


def main():
    ledger = [json.loads(l) for l in LEDGER.read_text().splitlines() if l.strip()]
    ledger = [r for r in ledger if str(r.get("date", "")) >= SEASON_START]
    fin = finals()
    mkt = market_home_lines(ledger)
    games = []
    for f in sorted(glob.glob("predictions/predictions_*.json")):
        d = json.loads(Path(f).read_text())
        if str(d.get("date", "")) < SEASON_START:
            continue
        for g in d.get("games", []):
            k = _key(d["date"], g["away"]["name"], g["home"]["name"])
            margin_pred = (g.get("spread") or {}).get("pred")
            if k not in fin or margin_pred is None:
                continue
            hs, as_ = fin[k]
            games.append({"date": d["date"], "pred_margin": float(margin_pred), "margin": hs - as_,
                          "pred_total": (g.get("totals") or {}).get("pred"), "total": hs + as_,
                          "market_total": (g.get("totals") or {}).get("line"), "market_home_line": mkt.get(k)})
    G = pd.DataFrame(games)
    report = {"generated_at_utc": datetime.now(timezone.utc).isoformat(), "season_from": SEASON_START}
    if G.empty:
        report["games"] = 0
    else:
        G["model_right"] = (G["pred_margin"] > 0) == (G["margin"] > 0)
        m = G.dropna(subset=["market_home_line"])
        m = m[m["market_home_line"] != 0]
        fav_right = (m["market_home_line"] < 0) == (m["margin"] > 0)
        t = G.dropna(subset=["pred_total", "market_total"])
        report.update({
            "games": int(len(G)), "first_date": G["date"].min(), "last_date": G["date"].max(),
            "winner": {"model_wins": int(G["model_right"].sum()), "model_losses": int((~G["model_right"]).sum()),
                       "model_rate": round(float(G["model_right"].mean()), 4),
                       "market_favorite_wins": int(fav_right.sum()), "market_favorite_losses": int((~fav_right).sum()),
                       "market_favorite_rate": round(float(fav_right.mean()), 4) if len(m) else None,
                       "games_with_market_line": int(len(m))},
            "spread": follow_fade([r for r in ledger if r.get("market") == "SPREAD"]),
            "total": follow_fade([r for r in ledger if r.get("market") == "TOTAL"]),
            "error": {"model_margin_mae": round(float((G["pred_margin"] - G["margin"]).abs().mean()), 2),
                      "model_total_mae": round(float((t["pred_total"] - t["total"]).abs().mean()), 2) if len(t) else None,
                      "market_total_mae": round(float((t["market_total"] - t["total"]).abs().mean()), 2) if len(t) else None,
                      "total_games": int(len(t))},
        })
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
