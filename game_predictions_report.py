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


GAME_PHASES = Path("data/history/nba_game_predictions.jsonl")


def phase_records(fin: dict, path=GAME_PHASES) -> dict:
    """Winner, ATS and O/U records and errors for EARLY and ADJUSTED game predictions,
    graded against the market line each phase recorded; plus how much the adjusted
    pass moved the early prediction."""
    if not Path(path).exists():
        return {}
    rows = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    out = {}
    graded = {}
    for r in rows:
        k = _key(r["date"], r["away"], r["home"])
        if k not in fin:
            continue
        hs, as_ = fin[k]
        margin, total = hs - as_, hs + as_
        g = {"winner": (r["pred_home_margin"] > 0) == (margin > 0),
             "margin_err": abs(r["pred_home_margin"] - margin)}
        line = r.get("market_home_line")
        if line is not None and margin + line != 0 and r["pred_home_margin"] + line != 0:
            g["ats"] = ((r["pred_home_margin"] + line) > 0) == ((margin + line) > 0)
        if r.get("pred_total") is not None:
            g["total_err"] = abs(r["pred_total"] - total)
            mt = r.get("market_total")
            if mt is not None and total != mt and r["pred_total"] != mt:
                g["ou"] = (r["pred_total"] > mt) == (total > mt)
                g["line_total_err"] = abs(mt - total)
        graded.setdefault(r["phase"], []).append(g)
        graded.setdefault(("pair", r["date"], r["game"]), {})[r["phase"]] = r
    for phase in ("EARLY", "ADJUSTED"):
        gs = graded.get(phase, [])
        if not gs:
            out[phase] = {"games": 0}
            continue
        def rec(field):
            v = [g[field] for g in gs if field in g]
            return {"wins": int(sum(v)), "losses": int(len(v) - sum(v)), "rate": round(sum(v) / len(v), 4) if v else None}
        mean = lambda f: round(sum(g[f] for g in gs if f in g) / max(1, sum(f in g for g in gs)), 2)
        out[phase] = {"games": len(gs), "winner": rec("winner"), "ats": rec("ats"), "ou": rec("ou"),
                      "margin_mae": mean("margin_err"), "total_mae": mean("total_err"),
                      "line_total_mae": mean("line_total_err")}
    pairs = [v for k, v in graded.items() if isinstance(k, tuple) and k[0] == "pair" and {"EARLY", "ADJUSTED"} <= set(v)]
    if pairs:
        dm = [abs(p["ADJUSTED"]["pred_home_margin"] - p["EARLY"]["pred_home_margin"]) for p in pairs]
        dt = [abs((p["ADJUSTED"]["pred_total"] or 0) - (p["EARLY"]["pred_total"] or 0)) for p in pairs
              if p["ADJUSTED"]["pred_total"] is not None and p["EARLY"]["pred_total"] is not None]
        out["changes"] = {"games_with_both": len(pairs),
                          "spread_moved": int(sum(d > 0.05 for d in dm)), "mean_spread_move": round(sum(dm) / len(dm), 2),
                          "total_moved": int(sum(d > 0.05 for d in dt)), "mean_total_move": round(sum(dt) / len(dt), 2) if dt else None,
                          "winner_flipped": int(sum((p["EARLY"]["pred_home_margin"] > 0) != (p["ADJUSTED"]["pred_home_margin"] > 0)
                                                    for p in pairs))}
    return out


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
    report["phases"] = phase_records(fin)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
