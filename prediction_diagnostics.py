"""Date-locked, research-only NBA forecast diagnostics versus the sportsbook market.

Uses forecasts saved before tip and FINAL results from the same game date.
Never turns a research lean into a bet or changes the immutable forward ledger.
"""
from __future__ import annotations

import csv
import json
import math
from datetime import date, datetime, timedelta, timezone
from pathlib import Path


MARKETS = ("pts", "reb", "ast", "threes", "pra")
# Shadow-only: monitor market anchoring; never used in pricing or bet selection.
SHADOW_MODEL_WEIGHT = 0.35


def number(value):
    try:
        x = float(value)
        return x if math.isfinite(x) else None
    except (ValueError, TypeError):
        return None


def read_csv(path):
    try:
        with Path(path).open(encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))
    except (OSError, csv.Error):
        return []


def normalize_name(value):
    import unicodedata
    value = unicodedata.normalize("NFKD", str(value or ""))
    return "".join(c.lower() for c in value if c.isalpha() and not unicodedata.combining(c))


def is_final(row):
    return str(row.get("is_final", "")).lower() in {"1", "true", "yes"} or row.get("status") == "STATUS_FINAL"


def before_tip(generated, tip):
    try:
        published = datetime.fromisoformat(str(generated).replace("Z", "+00:00"))
        starts = datetime.fromisoformat(str(tip).replace("Z", "+00:00"))
        return published.astimezone(timezone.utc) < starts.astimezone(timezone.utc)
    except (TypeError, ValueError):
        return False


def average_absolute(rows, left, right):
    pairs = [(number(r.get(left)), number(r.get(right))) for r in rows]
    errors = [abs(a - b) for a, b in pairs if a is not None and b is not None]
    return round(sum(errors) / len(errors), 2) if errors else None


def evaluate_day(target_date, predictions_dir="predictions", raw_dir="data/raw"):
    """Return benchmark evidence for a date, or an explicit unavailable status."""
    target_date = str(target_date)
    paths = (Path(predictions_dir) / f"predictions_{target_date}.json",
             Path(raw_dir) / f"scores_{target_date}.csv",
             Path(raw_dir) / f"result_player_box_{target_date}.csv")
    out = {"date": target_date, "status": "AWAITING_RESULTS", "games": [], "stats": {},
           "player_outliers": [], "qualified_bets": 0,
           "method": "PREGAME_FORECAST_VS_FINAL; RESEARCH_ONLY; MARKET_BLEND_SHADOW_35_PERCENT"}
    try:
        predictions = json.loads(paths[0].read_text(encoding="utf-8"))
    except (OSError, ValueError):
        out["status"] = "MISSING_PREDICTIONS"
        return out
    if predictions.get("date") != target_date:
        out["status"] = "PREDICTION_DATE_MISMATCH"
        return out
    out["qualified_bets"] = len(predictions.get("best_bets") or [])
    scores = {
        (r.get("home_team"), r.get("away_team")): r
        for r in read_csv(paths[1])
        if r.get("game_date") == target_date and is_final(r)
    }
    boxes = {}
    for row in read_csv(paths[2]):
        if row.get("game_date") != target_date:
            continue
        boxes[(str(row.get("game_id")), normalize_name(row.get("player_name")))] = row
    player_rows = []
    expected_games = len(predictions.get("games") or [])
    late_snapshots = 0
    for game in predictions.get("games") or []:
        home = (game.get("home") or {}).get("name")
        away = (game.get("away") or {}).get("name")
        final = scores.get((home, away))
        if not final:
            continue
        if not before_tip(predictions.get("generated"), game.get("tip")):
            late_snapshots += 1
            continue

        actual_margin, actual_total = number(final.get("actual_spread")), number(final.get("actual_total"))
        model_margin = number((game.get("spread") or {}).get("pred"))
        posted_spread = number((game.get("spread") or {}).get("posted_line"))
        market_margin = -posted_spread if posted_spread is not None else None
        model_total = number((game.get("totals") or {}).get("pred"))
        market_total = number((game.get("totals") or {}).get("line"))
        shadow_margin = round(SHADOW_MODEL_WEIGHT * model_margin + (1 - SHADOW_MODEL_WEIGHT) * market_margin, 2) if model_margin is not None and market_margin is not None else None
        line = {
            "game": f"{away} @ {home}", "game_id": str(final.get("game_id")),
            "actual_margin": actual_margin, "model_margin": model_margin,
            "market_margin": market_margin, "shadow_margin": shadow_margin,
            "actual_total": actual_total, "model_total": model_total,
            "market_total": market_total,
            "winner_correct": (model_margin > 0) == (actual_margin > 0) if model_margin is not None and actual_margin not in (None, 0) and model_margin != 0 else None,
            "spread_lean_correct": (model_margin - market_margin > 0) == (actual_margin - market_margin > 0) if None not in (model_margin, market_margin, actual_margin) and model_margin != market_margin and actual_margin != market_margin else None,
            "total_lean_correct": (model_total > market_total) == (actual_total > market_total) if None not in (model_total, market_total, actual_total) and model_total != market_total and actual_total != market_total else None,
        }
        out["games"].append(line)
        game_id = str(final.get("game_id"))
        for player in game.get("props") or []:
            observed = boxes.get((game_id, normalize_name(player.get("player"))))
            if not observed or str(observed.get("did_not_play", "")).lower() in {"true", "1", "yes"}:
                continue
            minutes = number(observed.get("minutes"))
            projected_minutes = number(player.get("proj_min"))
            if minutes is None:
                continue
            player_rows.append({"player": player.get("player"), "game": line["game"],
                                "stat": "MINUTES", "pred": projected_minutes,
                                "actual": minutes, "line": None, "correct": None})
            values = {stat: number(observed.get("threes_made" if stat == "threes" else stat)) for stat in ("pts", "reb", "ast", "threes")}
            values["pra"] = sum(values[s] for s in ("pts", "reb", "ast")) if all(values[s] is not None for s in ("pts", "reb", "ast")) else None
            for stat, projection in (player.get("props") or {}).items():
                if stat not in MARKETS:
                    continue
                pred = number(projection.get("pred"))
                actual = values[stat]
                if pred is None or actual is None:
                    continue
                market_line = number(projection.get("line")) if projection.get("odds") is not None else None
                correct = (pred > market_line) == (actual > market_line) if market_line is not None and pred != market_line and actual != market_line else None
                player_rows.append({"player": player.get("player"), "game": line["game"],
                                    "stat": stat.upper(), "pred": pred, "actual": actual,
                                    "line": market_line, "correct": correct})
    if not out["games"]:
        out["status"] = "PREGAME_SNAPSHOT_INVALID" if late_snapshots else "AWAITING_RESULTS"
        return out
    out["status"] = "COMPLETE" if len(out["games"]) == expected_games else "PARTIAL_RESULTS"
    out["summary"] = {
        "completed_games": len(out["games"]), "scheduled_games": expected_games,
        "winner_correct": sum(g["winner_correct"] is True for g in out["games"]),
        "winner_sample": sum(g["winner_correct"] is not None for g in out["games"]),
        "spread_lean_correct": sum(g["spread_lean_correct"] is True for g in out["games"]),
        "spread_lean_sample": sum(g["spread_lean_correct"] is not None for g in out["games"]),
        "total_lean_correct": sum(g["total_lean_correct"] is True for g in out["games"]),
        "total_lean_sample": sum(g["total_lean_correct"] is not None for g in out["games"]),
        "margin_mae_model": average_absolute(out["games"], "model_margin", "actual_margin"),
        "margin_mae_market": average_absolute(out["games"], "market_margin", "actual_margin"),
        "margin_mae_shadow": average_absolute(out["games"], "shadow_margin", "actual_margin"),
        "total_mae_model": average_absolute(out["games"], "model_total", "actual_total"),
        "total_mae_market": average_absolute(out["games"], "market_total", "actual_total"),
        "minutes_mae": average_absolute([r for r in player_rows if r["stat"] == "MINUTES"], "pred", "actual"),
    }
    for stat in (*[s.upper() for s in MARKETS], "MINUTES"):
        rows = [r for r in player_rows if r["stat"] == stat]
        counted = [r for r in rows if r["correct"] is not None]
        out["stats"][stat] = {
            "sample": len(rows), "mae": average_absolute(rows, "pred", "actual"),
            "priced_directions": len(counted), "directions_correct": sum(r["correct"] is True for r in counted),
        }
    out["player_outliers"] = sorted(
        [{"player": r["player"], "game": r["game"], "stat": r["stat"],
          "projected": r["pred"], "actual": r["actual"],
          "error": round(r["pred"] - r["actual"], 2)}
         for r in player_rows if r["stat"] in {"PTS", "MINUTES"}],
        key=lambda r: abs(r["error"]), reverse=True
    )[:12]
    return out


def evaluate_history(through_date, lookback_days=14, predictions_dir="predictions", raw_dir="data/raw"):
    """Rolling pregame-error benchmark from dated finalized results; no bet grading.

    Historical model snapshots are compared without fitting a weight on the same
    data. Shadow values remain a hypothesis until independently validated.
    """
    end = date.fromisoformat(str(through_date))
    games = []
    covered_dates = []
    for offset in range(max(0, int(lookback_days))):
        day = (end - timedelta(days=offset)).isoformat()
        report = evaluate_day(day, predictions_dir=predictions_dir, raw_dir=raw_dir)
        if report["status"] not in {"COMPLETE", "PARTIAL_RESULTS"}:
            continue
        covered_dates.append(day)
        games.extend(report["games"])
    return {
        "through_date": str(through_date), "lookback_days": lookback_days,
        "completed_dates": len(covered_dates), "completed_games": len(games),
        "margin_mae_model": average_absolute(games, "model_margin", "actual_margin"),
        "margin_mae_market": average_absolute(games, "market_margin", "actual_margin"),
        "margin_mae_shadow": average_absolute(games, "shadow_margin", "actual_margin"),
        "total_mae_model": average_absolute(games, "model_total", "actual_total"),
        "total_mae_market": average_absolute(games, "market_total", "actual_total"),
        "note": "Research-only paired MAE on completed pregame forecasts; not evidence of profitable bets.",
    }
