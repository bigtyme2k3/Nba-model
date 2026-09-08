"""
betting_engine.py
-----------------
Shared betting utilities for the NBA dashboard.

Adds:
  - Expected value calculations for American odds
  - Fair odds / implied probability
  - Conservative edge-to-probability conversion
  - Decision score 0-100
  - Automatic bet scoring/ranking
  - Lightweight model tracking summary
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import pandas as pd

TRACKING_DIR = "data/tracking"
GRADED_PATH = os.path.join(TRACKING_DIR, "graded_bets.csv")


def implied_prob_american(odds):
    try:
        odds = float(odds)
    except Exception:
        odds = -110.0
    if odds < 0:
        return abs(odds) / (abs(odds) + 100.0)
    return 100.0 / (odds + 100.0)


def american_from_prob(prob):
    """Convert no-vig/fair probability to American fair odds."""
    try:
        p = max(0.001, min(0.999, float(prob)))
    except Exception:
        p = 0.5
    if p >= 0.5:
        return int(round(-(p / (1 - p)) * 100))
    return int(round(((1 - p) / p) * 100))


def payout_profit_per_unit(odds):
    try:
        odds = float(odds)
    except Exception:
        odds = -110.0
    if odds < 0:
        return 100.0 / abs(odds)
    return odds / 100.0


def edge_to_prob(edge, market_type):
    """Convert model edge in stat/score points into a conservative win probability."""
    try:
        ae = abs(float(edge or 0.0))
    except Exception:
        ae = 0.0
    base = {"SPREAD": 0.515, "TOTAL": 0.515, "PROP": 0.505}.get(str(market_type).upper(), 0.51)
    scale = {"SPREAD": 0.018, "TOTAL": 0.014, "PROP": 0.026}.get(str(market_type).upper(), 0.018)
    cap = {"SPREAD": 0.63, "TOTAL": 0.62, "PROP": 0.66}.get(str(market_type).upper(), 0.62)
    return round(min(cap, base + ae * scale), 4)


def expected_value(prob, odds=-110):
    profit = payout_profit_per_unit(odds)
    ev = float(prob) * profit - (1.0 - float(prob))
    return round(ev, 4)


def kelly_fraction(prob, odds=-110, fraction=0.25, cap=0.05):
    b = payout_profit_per_unit(odds)
    p = float(prob)
    q = 1.0 - p
    raw = (b * p - q) / b if b else 0.0
    return round(max(0.0, min(cap, raw * fraction)), 4)


def grade_bucket(ev, stars):
    if ev >= 0.15 and stars >= 3:
        return "A+"
    if ev >= 0.10 and stars >= 3:
        return "A"
    if ev >= 0.05 and stars >= 2:
        return "B"
    if ev > 0:
        return "C"
    return "PASS"


def decision_score(prob, odds, edge=0, stars=1, injury_status="ACTIVE", market_books=1):
    """Professional-style 0-100 score based on EV, edge, confidence, injury, and market depth."""
    ev = expected_value(prob, odds)
    try:
        edge_abs = abs(float(edge or 0))
    except Exception:
        edge_abs = 0.0
    score = 50
    score += min(25, max(0, ev * 120))
    score += min(12, edge_abs * 2.5)
    score += int(stars or 1) * 4
    score += min(6, int(market_books or 1) * 1.5)
    status = str(injury_status or "ACTIVE").upper()
    if status == "QUESTIONABLE":
        score -= 10
    elif status == "PROBABLE":
        score -= 4
    elif status in {"OUT", "DOUBTFUL"}:
        score = 0
    return int(round(max(0, min(100, score))))


def decision_label(score):
    if score >= 92:
        return "ELITE"
    if score >= 85:
        return "STRONG"
    if score >= 75:
        return "GOOD"
    if score >= 65:
        return "LEAN"
    return "PASS"


def explain_edge(b):
    parts = []
    if b.get("ev_pct") is not None:
        parts.append(f"EV {b.get('ev_pct')}%")
    if b.get("edge") is not None:
        parts.append(f"edge {b.get('edge')}")
    if b.get("best_book_title") or b.get("best_book"):
        parts.append(f"best at {b.get('best_book_title') or b.get('best_book')}")
    if b.get("injury_status") and b.get("injury_status") != "ACTIVE":
        parts.append(f"injury {b.get('injury_status')}")
    if b.get("available_books"):
        parts.append(f"{b.get('available_books')} books")
    return " · ".join(parts) if parts else "Model edge and market price combined."


def enrich_bet(bet, market_type=None):
    b = dict(bet)
    typ = (market_type or b.get("type") or "BET").upper()
    odds = b.get("best_odds") or b.get("odds") or b.get("price") or b.get("over_price") or b.get("under_price") or -110
    prob = b.get("model_prob") or edge_to_prob(b.get("edge", b.get("edge_abs", 0)), typ)
    ev = expected_value(prob, odds)
    stars = int(b.get("stars", 1) or 1)
    score = decision_score(prob, odds, b.get("edge", b.get("edge_abs", 0)), stars, b.get("injury_status", "ACTIVE"), b.get("available_books", 1))
    b.update({
        "type": typ,
        "odds": odds,
        "model_prob": prob,
        "model_prob_pct": round(prob * 100, 1),
        "implied_prob": round(implied_prob_american(odds), 4),
        "implied_prob_pct": round(implied_prob_american(odds) * 100, 1),
        "fair_odds": american_from_prob(prob),
        "ev": ev,
        "ev_pct": round(ev * 100, 1),
        "kelly_frac": kelly_fraction(prob, odds),
        "units": round(kelly_fraction(prob, odds) / 0.05, 2),
        "grade": grade_bucket(ev, stars),
        "score": score,
        "score_label": decision_label(score),
    })
    b["ev_reason"] = b.get("ev_reason") or explain_edge(b)
    return b


def rank_bets(bets, limit=20):
    enriched = [enrich_bet(b) for b in bets]
    enriched.sort(key=lambda b: (-float(b.get("score", 0) or 0), -{"A+": 5, "A": 4, "B": 3, "C": 2, "PASS": 1}.get(b.get("grade"), 0), -float(b.get("ev", 0) or 0), -int(b.get("stars", 1) or 1), -abs(float(b.get("edge", b.get("edge_abs", 0)) or 0))))
    for i, b in enumerate(enriched[:limit], 1):
        b["rank"] = i
    return enriched[:limit]


def tracking_summary(path=GRADED_PATH):
    summary = {
        "overall": "0-0", "wins": 0, "losses": 0, "pushes": 0, "win_pct": 0,
        "roi": 0, "profit_units": 0, "clv_avg": 0, "by_type": {}, "by_conf": {},
        "recent_10": [], "last_updated_utc": datetime.now(timezone.utc).isoformat(),
    }
    if not os.path.exists(path):
        return summary
    try:
        df = pd.read_csv(path)
    except Exception:
        return summary
    if df.empty or "result" not in df.columns:
        return summary
    wins = int((df["result"] == "WIN").sum())
    losses = int((df["result"] == "LOSS").sum())
    pushes = int((df["result"] == "PUSH").sum())
    graded = wins + losses
    summary.update({
        "wins": wins, "losses": losses, "pushes": pushes,
        "overall": f"{wins}-{losses}-{pushes}",
        "win_pct": round(wins / graded, 3) if graded else 0,
        "profit_units": round(float(df.get("profit_units", pd.Series(dtype=float)).fillna(0).sum()), 2),
        "roi": round(float(df.get("profit_units", pd.Series(dtype=float)).fillna(0).sum()) / max(1, len(df)), 3),
        "clv_avg": round(float(df.get("clv", pd.Series(dtype=float)).fillna(0).mean()), 2) if "clv" in df.columns else 0,
    })
    for col, key in [("type", "by_type"), ("conf", "by_conf")]:
        if col in df.columns:
            out = {}
            for value, group in df.groupby(col):
                gw = int((group["result"] == "WIN").sum())
                gl = int((group["result"] == "LOSS").sum())
                gg = gw + gl
                out[str(value)] = {"record": f"{gw}-{gl}", "win_pct": round(gw / gg, 3) if gg else 0, "bets": int(len(group))}
            summary[key] = out
    summary["recent_10"] = df.tail(10).to_dict("records")
    return summary


def save_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
