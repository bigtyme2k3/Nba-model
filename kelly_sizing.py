"""
kelly_sizing.py
---------------
Converts model edge (predicted spread vs posted line) into
actual bet sizes using fractional Kelly criterion.

Kelly formula: f = (bp - q) / b
  b = decimal odds - 1  (at -110, b = 100/110 = 0.909)
  p = model win probability
  q = 1 - p

We use 25% Kelly (conservative) capped at 5% of bankroll.

Also converts spread margin predictions to win probabilities
using a logistic model calibrated to NBA game variance.

Usage:
    from kelly_sizing import size_bet, edge_to_prob, kelly_units
"""

import numpy as np
import pandas as pd

# Standard juice
JUICE_110  = -110   # Most spread/total bets
JUICE_EVEN = -100

# Kelly fraction (conservative — reduces variance significantly)
KELLY_FRACTION = 0.25

# Max bet as % of bankroll
MAX_BET_PCT = 0.05

# NBA game margin standard deviation (pts) — ~12 pts is the commonly cited figure;
# recalibrate against your own backtest once you have a season of real results.
NBA_GAME_STD = 12.0

# Minimum edge required to place a bet
MIN_EDGE_SPREAD = 2.0   # pts
MIN_EDGE_TOTAL  = 2.0   # pts (NBA totals run ~220-235, so the same abs-edge threshold as WNBA is looser relatively)
MIN_EDGE_PROP   = 1.0   # stat units


def juice_to_implied_prob(juice: int) -> float:
    """Convert American odds to implied probability."""
    if juice < 0:
        return abs(juice) / (abs(juice) + 100)
    else:
        return 100 / (juice + 100)


def implied_to_decimal(juice: int) -> float:
    """Convert American odds to decimal odds."""
    if juice < 0:
        return 1 + (100 / abs(juice))
    else:
        return 1 + (juice / 100)


def edge_to_prob(edge_pts: float, model_type: str = "spread") -> float:
    """
    Convert a model edge in points to a win probability.

    Uses logistic function calibrated to NBA variance.
    At 0 edge: 50% win probability
    At +3 pts edge: ~59% win probability
    At +6 pts edge: ~67% win probability
    At +10 pts edge: ~77% win probability
    """
    scales = {
        "spread": NBA_GAME_STD,
        "totals": NBA_GAME_STD * 1.2,  # Totals have higher variance
        "props":  NBA_GAME_STD * 0.6,  # Props have lower variance
    }
    scale = scales.get(model_type, NBA_GAME_STD)

    k = np.pi / (np.sqrt(3) * scale)
    prob = 1 / (1 + np.exp(-k * edge_pts))
    return float(np.clip(prob, 0.01, 0.99))


def kelly_fraction_bet(win_prob: float, juice: int = JUICE_110) -> float:
    """Full Kelly fraction (not yet scaled by KELLY_FRACTION). Returns 0-1."""
    b = implied_to_decimal(juice) - 1   # Net profit per unit
    p = win_prob
    q = 1 - p

    kelly = (b * p - q) / b

    return max(0.0, kelly)


def size_bet(edge_pts: float, model_type: str = "spread",
             juice: int = JUICE_110, bankroll: float = 1000.0,
             min_edge: float = None) -> dict:
    """
    Full bet sizing calculation.

    Returns dict with:
        win_prob:    model's estimated win probability
        kelly_full:  full Kelly fraction
        kelly_frac:  fractional Kelly (25%)
        units:       bet size in units (1 unit = 1% of bankroll)
        dollar_amt:  dollar amount to bet
        edge_pts:    input edge
        juice:       input juice
        verdict:     BET / PASS / STRONG
    """
    thresholds = {"spread": MIN_EDGE_SPREAD, "totals": MIN_EDGE_TOTAL, "props": MIN_EDGE_PROP}
    if min_edge is None:
        min_edge = thresholds.get(model_type, 2.0)

    if abs(edge_pts) < min_edge:
        return {
            "win_prob":   0.5,
            "kelly_full": 0.0,
            "kelly_frac": 0.0,
            "units":      0.0,
            "dollar_amt": 0.0,
            "edge_pts":   edge_pts,
            "juice":      juice,
            "verdict":    "PASS",
        }

    win_prob   = edge_to_prob(edge_pts, model_type)
    kelly_full = kelly_fraction_bet(win_prob, juice)
    kelly_frac = kelly_full * KELLY_FRACTION
    kelly_frac = min(kelly_frac, MAX_BET_PCT)

    dollar_amt = round(kelly_frac * bankroll, 2)
    units      = round(kelly_frac * 100, 2)  # 1 unit = 1% bankroll

    if kelly_frac >= 0.03:
        verdict = "STRONG"
    elif kelly_frac >= 0.01:
        verdict = "BET"
    else:
        verdict = "PASS"

    return {
        "win_prob":   round(win_prob, 3),
        "kelly_full": round(kelly_full, 4),
        "kelly_frac": round(kelly_frac, 4),
        "units":      units,
        "dollar_amt": dollar_amt,
        "edge_pts":   edge_pts,
        "juice":      juice,
        "verdict":    verdict,
    }


def kelly_units(edge_pts: float, model_type: str = "spread",
                juice: int = JUICE_110) -> float:
    """Shorthand — return just the unit size (0 if no bet)."""
    result = size_bet(edge_pts, model_type, juice)
    return result["units"]


def size_all_bets(best_bets: list, bankroll: float = 1000.0) -> list:
    """
    Apply Kelly sizing to a list of best bets from daily_runner.py.
    Adds 'units', 'dollar_amt', 'win_prob', 'verdict' to each bet.
    """
    TYPE_MAP = {"SPREAD":"spread", "TOTAL":"totals", "PROP":"props"}
    sized = []

    for bet in best_bets:
        edge     = bet.get("edge", 0) or 0
        bet_type = TYPE_MAP.get(bet.get("type","SPREAD"), "spread")
        result   = size_bet(edge, bet_type, bankroll=bankroll)

        sized.append({
            **bet,
            "win_prob":   result["win_prob"],
            "units":      result["units"],
            "dollar_amt": result["dollar_amt"],
            "verdict":    result["verdict"],
        })

    sized.sort(key=lambda b: (-b["units"], -abs(b.get("edge",0))))
    for i, b in enumerate(sized):
        b["rank"] = i + 1

    return sized


# ── Quick reference table ─────────────────────────────────────────────────────

def print_sizing_table():
    """Print a reference table of edge → units at -110."""
    print("\n═══ Kelly Sizing Reference (-110 juice, $1000 bankroll) ═══\n")
    print(f"{'Edge':>8} {'Win%':>8} {'Full K':>8} {'Frac K':>8} {'Units':>8} {'$Amt':>8}  Verdict")
    print("─" * 68)

    for edge in [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]:
        result = size_bet(edge, "spread", JUICE_110, 1000.0)
        print(f"{edge:>8.1f} {result['win_prob']:>8.1%} {result['kelly_full']:>8.3f} "
              f"{result['kelly_frac']:>8.3f} {result['units']:>8.2f} "
              f"${result['dollar_amt']:>7.2f}  {result['verdict']}")


if __name__ == "__main__":
    print_sizing_table()

    print("\n\n═══ Sample Bet Sizing ═══\n")
    sample_bets = [
        {"type":"SPREAD","play":"BOS -4.5","edge":2.5,"conf":"MED","stars":2,"rank":1},
        {"type":"TOTAL", "play":"OVER 228.5","edge":6.4,"conf":"HIGH","stars":3,"rank":2},
        {"type":"PROP",  "play":"Jokic REB U11.5","edge":-2.2,"conf":"MED","stars":2,"rank":3},
    ]
    sized = size_all_bets(sample_bets, bankroll=1000.0)
    print(f"{'Play':<35} {'Edge':>6} {'Win%':>7} {'Units':>7} {'$Amt':>8}  Verdict")
    print("─"*72)
    for b in sized:
        print(f"{b['play']:<35} {b.get('edge',0):>+6.1f} "
              f"{b['win_prob']:>7.1%} {b['units']:>7.2f} "
              f"${b['dollar_amt']:>7.2f}  {b['verdict']}")
