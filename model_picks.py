"""
model_picks.py
--------------
Two-phase daily model picks (user decision 2026-11-17).

  EARLY     the morning read, before the day's injury report settles: the daily run's
            data-collection bets (daily_runner.DATA_BETS_PER_DAY strongest prop
            opinions across the slate).
  ADJUSTED  re-picked after the injury report, shortly before each group of tips, by
            the pre-tip refresh: the strongest opinions among the games in that
            window, from predictions re-run with the latest injuries, odds and props.
            Each window adds picks in proportion to its share of the slate's games, so
            the day totals about the same number as the early read.

Both phases are recorded in data/history/nba_model_picks.jsonl: one immutable row per
(date, phase, player, stat), written only before that game's tip, flat 1 unit, with the
recorded price and the opposite side's price at the same book and line so a fade can be
graded. Grading (data_bets_report.py) uses the forward ledger's actual stat lines.

Usage:
    python model_picks.py record --date 2026-11-18 --phase EARLY
    python model_picks.py record --date 2026-11-18 --phase ADJUSTED --predictions predictions/pretip_2026-11-18.json
"""
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

PICKS = Path("data/history/nba_model_picks.jsonl")
PRED_DIR = Path("predictions")
PER_DAY = 10
BOOKS = {"fanduel", "draftkings", "fanatics"}
PHASES = {"EARLY", "ADJUSTED"}


def _num(v):
    try:
        v = float(v)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def _tip(value):
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).astimezone(timezone.utc)
    except (TypeError, ValueError):
        return None


def pick_key(date, phase, player, stat):
    return "|".join([str(date), phase, " ".join(str(player).lower().split()), str(stat).upper()])


def read_picks(path=PICKS) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def candidates(data: dict) -> list[dict]:
    """Every priced, available sportsbook prop in a prediction file, strongest first."""
    out = []
    for g in data.get("games", []):
        game = f"{g.get('away', {}).get('name')} @ {g.get('home', {}).get('name')}"
        for p in g.get("props", []):
            for stat, pr in (p.get("props") or {}).items():
                raw, mkt = _num(pr.get("raw_model_probability")), _num(pr.get("market_probability"))
                side = pr.get("offer_side")
                if (pr.get("availability_block") or side not in {"OVER", "UNDER"} or raw is None or mkt is None
                        or pr.get("book_key") not in BOOKS or _num(pr.get("odds")) is None):
                    continue
                out.append({"game": game, "tip": g.get("tip"), "player": p.get("player"), "team": p.get("team"),
                            "stat": stat.upper(), "side": side, "line": _num(pr.get("line")),
                            "odds": int(_num(pr.get("odds"))), "opposite_odds": pr.get("opposite_odds"),
                            "book_key": pr.get("book_key"), "projection": _num(pr.get("pred")),
                            "v2_projection": _num(pr.get("v2_pred")), "raw_model_probability": raw,
                            "market_probability": mkt, "strength": raw - mkt,
                            "injury_status": p.get("injury_status"), "projected_minutes": _num(p.get("proj_min"))})
    return sorted(out, key=lambda c: -c["strength"])


def select(data: dict, phase: str, existing: list[dict], slate_games: int | None, now=None) -> list[dict]:
    now = now or datetime.now(timezone.utc)
    date = data.get("date")
    have = {r["pick_key"] for r in existing}
    picked_players = {(r["date"], r["phase"], " ".join(str(r["player"]).lower().split())) for r in existing}
    games = {f"{g.get('away', {}).get('name')} @ {g.get('home', {}).get('name')}" for g in data.get("games", [])}
    if phase == "ADJUSTED":
        done = {r["game"] for r in existing if r["date"] == date and r["phase"] == "ADJUSTED"}
        games -= done
        n = math.ceil(PER_DAY * len(games) / max(slate_games or len(games) or 1, 1)) if games else 0
    else:
        # One morning read per day: a later daily run (e.g. GitHub's delayed backstop) adds nothing.
        n = 0 if any(r["date"] == date and r["phase"] == "EARLY" for r in existing) else PER_DAY
    rows = []
    if n <= 0:
        return rows
    for c in candidates(data):
        tip = _tip(c["tip"])
        player_key = (date, phase, " ".join(str(c["player"]).lower().split()))
        key = pick_key(date, phase, c["player"], c["stat"])
        if (c["game"] not in games or key in have or player_key in picked_players
                or tip is None or tip <= now):
            continue
        picked_players.add(player_key)
        rows.append({"pick_key": key, "date": date, "phase": phase, "recorded_at_utc": now.isoformat(),
                     "prediction_generated_at": data.get("generated"),
                     "injury_feed": (data.get("injury_feed") or {}).get("reason") or (data.get("injury_feed") or {}).get("healthy"),
                     "units": 1.0, **{k: v for k, v in c.items()}})
        if len(rows) >= n:
            break
    return rows


def record(date: str, phase: str, predictions: str | None = None, path=PICKS) -> dict:
    if phase not in PHASES:
        raise SystemExit(f"phase must be one of {sorted(PHASES)}")
    pred_path = Path(predictions) if predictions else PRED_DIR / f"predictions_{date}.json"
    data = json.loads(pred_path.read_text())
    existing = read_picks(path)
    slate = None
    daily = PRED_DIR / f"predictions_{date}.json"
    if daily.exists():
        slate = len(json.loads(daily.read_text()).get("games", []))
    rows = select(data, phase, existing, slate)
    if rows:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with Path(path).open("a", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, allow_nan=False, default=str) + "\n")
    status = {"date": date, "phase": phase, "predictions": str(pred_path), "games_in_file": len(data.get("games", [])),
              "slate_games": slate, "picks_recorded": len(rows),
              "picks": [f"{r['player']} {r['stat']} {r['side']} {r['line']} ({r['book_key']} {r['odds']})" for r in rows]}
    print(json.dumps(status, indent=2))
    return status


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("record")
    r.add_argument("--date", required=True)
    r.add_argument("--phase", required=True, choices=sorted(PHASES))
    r.add_argument("--predictions", default=None)
    a = ap.parse_args()
    record(a.date, a.phase, a.predictions)


if __name__ == "__main__":
    main()
