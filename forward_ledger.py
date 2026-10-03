"""
forward_ledger.py
-----------------
Append-only canonical forward prediction ledger for the NBA model.

The first market-bearing prediction for a ranking_key is immutable. Later daily
runs may update dashboard files, but they never replace the original forward
evidence used for calibration and champion/challenger evaluation.

Input:
  predictions/predictions_YYYY-MM-DD.json

Output:
  data/history/nba_forward_predictions.jsonl
  data/tracking/forward_ledger_status.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

PRED_DIR = Path("predictions")
LEDGER = Path("data/history/nba_forward_predictions.jsonl")
STATUS = Path("data/tracking/forward_ledger_status.json")


def norm(value):
    return " ".join(str(value or "").strip().lower().split())


def stable_key(parts):
    return "|".join(norm(x) for x in parts)


def prediction_id(ranking_key, generated):
    raw = f"{ranking_key}|{generated}"
    return hashlib.sha256(raw.encode()).hexdigest()[:24]


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


def spread_rows(data):
    date = data.get("date")
    generated = data.get("generated")
    version = data.get("model_version", "NBA_V1.0")
    for game in data.get("games", []):
        home = game.get("home", {}).get("name", "")
        away = game.get("away", {}).get("name", "")
        matchup = f"{away} @ {home}"
        sp = game.get("spread", {})
        line = sp.get("posted_line")
        projection = sp.get("pred")
        if line is None or projection is None:
            continue

        edge = sp.get("edge")
        play = sp.get("play")
        side = "PASS"
        if play:
            if norm(play).startswith(norm(home)):
                side = "HOME"
            elif norm(play).startswith(norm(away)):
                side = "AWAY"
        decision = "BET" if play and sp.get("stars", 0) >= 2 else "PASS"
        rk = stable_key([date, matchup, "SPREAD"])
        yield {
            "prediction_id": prediction_id(rk, generated),
            "ranking_key": rk,
            "prediction_generated_at_utc": generated,
            "date": date,
            "game": matchup,
            "market": "SPREAD",
            "player": None,
            "stat": None,
            "side": side,
            "line": line,
            "odds": sp.get("juice"),
            "sportsbook": None,
            "market_source": sp.get("source"),
            "projection": projection,
            "edge": edge,
            "decision": decision,
            "confidence": sp.get("conf"),
            "stars": sp.get("stars"),
            "model_version": version,
            "actual": None,
            "outcome": "PENDING",
            "graded_at_utc": None,
        }


def total_rows(data):
    date = data.get("date")
    generated = data.get("generated")
    version = data.get("model_version", "NBA_V1.0")
    for game in data.get("games", []):
        home = game.get("home", {}).get("name", "")
        away = game.get("away", {}).get("name", "")
        matchup = f"{away} @ {home}"
        total = game.get("totals", {})
        line = total.get("line")
        projection = total.get("pred")
        if line is None or projection is None:
            continue
        side = str(total.get("play") or "PASS").upper()
        decision = "BET" if side in {"OVER", "UNDER"} and total.get("stars", 0) >= 2 else "PASS"
        rk = stable_key([date, matchup, "TOTAL"])
        yield {
            "prediction_id": prediction_id(rk, generated),
            "ranking_key": rk,
            "prediction_generated_at_utc": generated,
            "date": date,
            "game": matchup,
            "market": "TOTAL",
            "player": None,
            "stat": None,
            "side": side,
            "line": line,
            "odds": total.get("juice"),
            "sportsbook": None,
            "market_source": total.get("source"),
            "projection": projection,
            "edge": total.get("edge"),
            "decision": decision,
            "confidence": total.get("conf"),
            "stars": total.get("stars"),
            "model_version": version,
            "actual": None,
            "outcome": "PENDING",
            "graded_at_utc": None,
        }


def prop_rows(data):
    date = data.get("date")
    generated = data.get("generated")
    version = data.get("model_version", "NBA_V1.0")
    for game in data.get("games", []):
        home = game.get("home", {}).get("name", "")
        away = game.get("away", {}).get("name", "")
        matchup = f"{away} @ {home}"
        for player in game.get("props", []):
            pname = player.get("player")
            for stat, prop in (player.get("props") or {}).items():
                line = prop.get("line")
                projection = prop.get("pred")
                if line is None or projection is None:
                    continue
                signal = str(prop.get("signal") or "PASS").upper()
                decision = "BET" if signal in {"OVER", "UNDER"} else "PASS"
                rk = stable_key([date, matchup, "PROP", pname, stat])
                yield {
                    "prediction_id": prediction_id(rk, generated),
                    "ranking_key": rk,
                    "prediction_generated_at_utc": generated,
                    "date": date,
                    "game": matchup,
                    "market": "PROP",
                    "player": pname,
                    "stat": stat.upper(),
                    "side": signal,
                    "line": line,
                    "odds": None,
                    "sportsbook": "PrizePicks" if prop.get("line") is not None else None,
                    "market_source": "prizepicks",
                    "projection": projection,
                    "projected_minutes": player.get("proj_min"),
                    "rotation_risk_band": player.get("rotation_risk_band"),
                    "rotation_stability_score": player.get("rotation_stability_score"),
                    "projected_role": player.get("projected_role"),
                    "edge": prop.get("edge"),
                    "decision": decision,
                    "confidence": None,
                    "stars": None,
                    "model_version": version,
                    "actual": None,
                    "outcome": "PENDING",
                    "graded_at_utc": None,
                }


def flatten(data):
    return list(spread_rows(data)) + list(total_rows(data)) + list(prop_rows(data))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True)
    parser.add_argument("--predictions", default=None)
    args = parser.parse_args()

    pred_path = Path(args.predictions) if args.predictions else PRED_DIR / f"predictions_{args.date}.json"
    if not pred_path.exists():
        raise SystemExit(f"Prediction file missing: {pred_path}")

    data = json.loads(pred_path.read_text(encoding="utf-8"))
    candidates = flatten(data)
    existing = read_ledger()
    seen = {r.get("ranking_key") for r in existing}
    new_rows = [r for r in candidates if r.get("ranking_key") not in seen]

    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    if new_rows:
        with LEDGER.open("a", encoding="utf-8") as fh:
            for row in new_rows:
                fh.write(json.dumps(row, allow_nan=False) + "\n")

    status = {
        "status": "READY",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "date": args.date,
        "prediction_file": str(pred_path),
        "candidate_rows": len(candidates),
        "new_rows_appended": len(new_rows),
        "existing_rows_preserved": len(existing),
        "total_rows": len(existing) + len(new_rows),
        "first_prediction_immutable": True,
        "policy": "FIRST_OBSERVED_RANKING_KEY_ONLY",
        "model_version": data.get("model_version", "NBA_V1.0"),
    }
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    STATUS.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
