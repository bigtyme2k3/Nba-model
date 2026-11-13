"""Explain why research projections did not become qualifying NBA bets.

Audits published model outputs only. Does not modify any betting decision.
"""
from collections import Counter


def summarize_qualification(games, best_bets):
    counts = Counter()
    examined = 0
    for game in games:
        for kind in ("spread", "totals"):
            entry = game.get(kind) or {}
            examined += 1
            if entry.get("selected_quote"):
                counts["QUALIFIED_GAME_MARKET"] += 1
            elif not entry.get("reference_quote"):
                counts["NO_VALID_GAME_QUOTE"] += 1
            else:
                counts["GAME_PRICE_OR_CALIBRATION_GATE"] += 1
        for player in game.get("props") or []:
            for stat, prop in (player.get("props") or {}).items():
                examined += 1
                if prop.get("signal"):
                    counts["PRICED_PROP_SIGNAL"] += 1
                elif prop.get("availability_block"):
                    reason = prop.get("block_reason") or "PLAYER_AVAILABILITY"
                    counts[reason] += 1
                elif prop.get("odds") is None or prop.get("line") is None:
                    counts["MISSING_EXACT_PROP_PRICE"] += 1
                elif (prop.get("probability_calibration") or "").endswith("_market_only"):
                    counts["PROP_CALIBRATION_MARKET_ONLY"] += 1
                elif (prop.get("probability_edge") or 0) < 0.020:
                    counts["PROP_PROBABILITY_EDGE_BELOW_2_PCT"] += 1
                elif (prop.get("ev") or 0) < 0.025:
                    counts["PROP_EV_BELOW_2_5_PCT"] += 1
                else:
                    counts["PROP_LINE_OR_CONFIDENCE_GATE"] += 1
    return {
        "examined_market_projections": examined,
        "final_bets": len(best_bets or []),
        "rejection_counts": dict(sorted(counts.items())),
        "note": "Each row has one primary diagnostic. No rejected row is a BET.",
    }
