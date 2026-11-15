"""
v2/simulation/shadow.py
-----------------------
Daily Monte Carlo shadow forecasts, published before tip and graded after.

    python -m v2.simulation.shadow publish --date 2026-11-15
    python -m v2.simulation.shadow grade

publish (after daily_runner and forward_ledger)
  Simulates every projected player in predictions/predictions_<date>.json with the
  frozen V2_MC parameters (models/v2_mc_calibration.json), twice:
    mc_v1  around V1's projections (the champion's mean, simulated shape)
    mc_v2  around V2's shadow projections (v2_pred); skipped for players without one
  and prices every recorded quote in data/raw/sportsbook_props_<date>.csv (FanDuel,
  DraftKings, Fanatics; Novig kept separate).
  - data/history/nba_mc_shadow_forecasts.jsonl: one immutable row per forward-ledger
    prop row (same ranking_key and prediction_id), first observation only, holding
    the probability of the ledger's recorded side at its recorded line and odds.
  - data/simulation/lab_today.json: the Research Lab's slate payload (all quotes,
    distribution summaries and histograms, scenario sensitivity). Rewritten daily.

grade
  Joins the shadow rows to the ledger's own outcomes (forward_reconciler grades the
  ledger) and scores only rows published before tip: Brier, log loss, calibration
  diagnostics, hypothetical ROI at the recorded odds, the unchanged live gate and the
  strict promotion gate (v2/simulation/gate.py). Sportsbooks and Novig are reported
  separately. Output: data/tracking/v2_mc_shadow_report.json.

Shadow only: nothing here changes a prediction, the ledger, calibration or any
verdict. BET eligibility belongs to V1.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from forward_ledger import read_ledger, stable_key
from teams import normalize_team
from v2.live import norm_name
from v2.simulation import engine, gate, price_research as R

PRED_DIR = Path("predictions")
RAW_DIR = Path("data/raw")
SHADOW = Path("data/history/nba_mc_shadow_forecasts.jsonl")
LAB = Path("data/simulation/lab_today.json")
REPORT = Path("data/tracking/v2_mc_shadow_report.json")
LOGS = Path("data/processed/player_logs.csv")
V2_FEATURES = Path("data/processed/v2_live_features.csv")
CALIBRATION = Path("models/v2_mc_calibration.json")
N_SIMS = 10000
STATS = {"pts": "PTS", "reb": "REB", "ast": "AST", "threes": "THREES", "pra": "PRA"}
SCENARIOS = {"reduced_minutes": {"minutes": 0.8}, "increased_minutes": {"minutes": 1.1},
             "increased_usage": {"usage": 1.1}}


def _num(v):
    try:
        v = float(v)
        return v if np.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def load_params() -> tuple[engine.Params, str, str]:
    cal = json.loads(CALIBRATION.read_text())
    params = engine.Params.from_dict(cal["params"])
    digest = hashlib.sha256(json.dumps(cal["params"], sort_keys=True).encode()).hexdigest()[:12]
    return params, cal.get("version", "unknown"), digest


def minutes_sd(date: str, logs_path=LOGS) -> dict:
    """10-game SD of actual minutes per player, from games before the date only."""
    try:
        logs = pd.read_csv(logs_path, usecols=["player", "game_date", "actual_minutes"], low_memory=False)
    except (OSError, ValueError):
        return {}
    logs = logs[pd.to_datetime(logs["game_date"]) < pd.Timestamp(date)].dropna(subset=["actual_minutes"])
    logs = logs.sort_values("game_date")
    sd = logs.groupby(logs["player"].map(norm_name))["actual_minutes"].apply(
        lambda s: s.tail(10).std() if len(s) >= 3 else np.nan)
    return sd.dropna().to_dict()


def player_frame(data: dict, sd: dict) -> pd.DataFrame:
    """One row per projected player: matchup, team, minutes, V1 and V2 stat means."""
    rows = []
    for game in data.get("games", []):
        home, away = game.get("home", {}).get("name", ""), game.get("away", {}).get("name", "")
        matchup = f"{away} @ {home}"   # forward_ledger's game key
        for p in game.get("props", []):
            props = p.get("props") or {}
            rec = {"game": matchup, "game_id": matchup, "tip": game.get("tip"), "player": p.get("player"),
                   "team": p.get("team"), "opp": p.get("opp"), "minutes": _num(p.get("proj_min")),
                   "injury_status": p.get("injury_status"), "minutes_sd_l10": sd.get(norm_name(p.get("player")))}
            for stat in STATS:
                rec[f"v1_{stat}"] = _num((props.get(stat) or {}).get("pred"))
                rec[f"v2_{stat}"] = _num((props.get(stat) or {}).get("v2_pred"))
            if rec["minutes"] is None or rec["player"] is None:
                continue
            rows.append(rec)
    return pd.DataFrame(rows)


def arm_means(df: pd.DataFrame, arm: str) -> tuple[dict, dict, np.ndarray]:
    """Means for single stats and for PRA (components rescaled to the PRA projection).
    Missing components fall back to the other arm's value, then V1, then 0; a player
    is 'available' for the arm only when the arm has its own projection."""
    get = lambda s: df[f"{arm}_{s}"].to_numpy(float)
    v1 = lambda s: df[f"v1_{s}"].to_numpy(float)
    single = {}
    for s in engine.STATS:
        x = get(s) if arm == "v1" or s != "threes" else v1(s)     # V2 has no threes model
        single[s] = np.clip(np.nan_to_num(np.where(np.isfinite(x), x, v1(s)), nan=0.0), 0, None)
    total = single["pts"] + single["reb"] + single["ast"]
    pra = get("pra")
    k = np.where(np.isfinite(pra) & (total > 0), pra / np.maximum(total, 1e-9), 1.0)
    pra_means = {s: single[s] * k for s in ("pts", "reb", "ast")}
    pra_means["threes"] = np.minimum(single["threes"], pra_means["pts"] / 3)
    own = np.isfinite(df[[f"{arm}_{s}" for s in ("pts", "reb", "ast", "pra")]].to_numpy(float)).any(axis=1)
    return single, pra_means, own


def simulate_arms(df: pd.DataFrame, params: engine.Params, n_sims: int, seed: int) -> dict:
    out = {}
    for arm in ("v1", "v2"):
        single, pra_means, own = arm_means(df, arm)
        s = engine.simulate(df, single, params, n_sims, seed)
        p = engine.simulate(df, pra_means, params, n_sims, seed)
        out[f"mc_{arm}"] = {"single": s, "pra": p["pra"], "available": own}
    return out


def sims_for(arms: dict, arm: str, stat: str) -> np.ndarray:
    a = arms[arm]
    return a["pra"] if stat == "pra" else a["single"][stat]


def quote_book(date: str, raw_dir=RAW_DIR) -> pd.DataFrame:
    path = Path(raw_dir) / f"sportsbook_props_{date}.csv"
    if not path.exists():
        return pd.DataFrame()
    q = pd.read_csv(path, low_memory=False)
    q = q[q["book_key"].isin(R.SPORTSBOOKS | R.EXCHANGES) & q["side"].isin(["OVER", "UNDER"])].copy()
    q["matchup"] = (q["away_team"].astype(str).map(normalize_team) + " @ "
                    + q["home_team"].astype(str).map(normalize_team))
    q["key"] = q["player"].map(norm_name)
    q["stat"] = q["stat"].astype(str).str.lower()
    return q


def normalize_team_matchup(game: str) -> str:
    away, _, home = str(game).partition(" @ ")
    return f"{normalize_team(away)} @ {normalize_team(home)}"


def v2_feature_status(date: str) -> dict:
    try:
        f = pd.read_csv(V2_FEATURES, usecols=["target_date", "injury_feed_healthy"])
    except (OSError, ValueError):
        return {"status": "MISSING"}
    fresh = bool(len(f) and str(f["target_date"].iloc[0]) == date)
    return {"status": "FRESH" if fresh else "STALE", "target_date": str(f["target_date"].iloc[0]) if len(f) else None,
            "injury_feed_healthy": bool(f["injury_feed_healthy"].iloc[0]) if len(f) else None}


def summary(x: np.ndarray) -> dict:
    q = np.percentile(x, [10, 25, 50, 75, 90], method="nearest")   # outcomes are whole numbers
    return {"mean": round(float(x.mean()), 2), "p10": float(q[0]), "p25": float(q[1]), "p50": float(q[2]),
            "p75": float(q[3]), "p90": float(q[4])}


def histogram(x: np.ndarray, max_bins: int = 60) -> dict:
    lo, hi = int(x.min()), int(x.max())
    width = max(1, int(np.ceil((hi - lo + 1) / max_bins)))
    edges = np.arange(lo, hi + width + 1, width)
    counts, _ = np.histogram(x, bins=edges)
    return {"start": lo, "width": width, "counts": counts.tolist()}


def side_probability(x: np.ndarray, line: float, side: str) -> tuple[float, float]:
    """(win given no push, push) of one side from one player's simulations."""
    over, under = float((x > line).mean()), float((x < line).mean())
    push = 1 - over - under
    win, loss = (over, under) if side == "OVER" else (under, over)
    return (win / (win + loss) if win + loss > 0 else None), push


def research_label(p_mc: float | None, p_market: float | None, available: bool) -> str:
    if not available or p_mc is None or p_market is None:
        return "INSUFFICIENT_EVIDENCE"
    gap = p_mc - p_market
    return "AGREES_WITH_MARKET" if abs(gap) < 0.03 else ("MC_FAVORS_SIDE" if gap > 0 else "MC_AGAINST_SIDE")


def publish(date: str, n_sims: int = N_SIMS, predictions: str | None = None, lab: bool = True) -> dict:
    pred_path = Path(predictions) if predictions else PRED_DIR / f"predictions_{date}.json"
    data = json.loads(pred_path.read_text())
    params, version, digest = load_params()
    published = datetime.now(timezone.utc)
    df = player_frame(data, minutes_sd(date))
    status = {"date": date, "version": version, "params_digest": digest, "published_at_utc": published.isoformat(),
              "players": int(len(df)), "n_sims": n_sims}
    if df.empty:
        status["status"] = "NO_PLAYERS"
        return status
    seed = int(hashlib.sha256(date.encode()).hexdigest()[:8], 16)
    arms = simulate_arms(df, params, n_sims, seed)
    pos = {(g, norm_name(p)): i for i, (g, p) in enumerate(zip(df["game"], df["player"]))}

    # 1) Immutable shadow rows, one per ledger prop row of this date.
    ledger = [r for r in read_ledger() if r.get("market") == "PROP" and r.get("date") == date]
    existing = set()
    if SHADOW.exists():
        existing = {json.loads(l).get("ranking_key") for l in SHADOW.read_text().splitlines() if l.strip()}
    new_rows = []
    for r in ledger:
        if r.get("ranking_key") in existing:
            continue
        i = pos.get((r.get("game"), norm_name(r.get("player"))))
        stat = str(r.get("stat", "")).lower()
        line, odds, side = _num(r.get("line")), _num(r.get("odds")), str(r.get("side", "")).upper()
        if i is None or stat not in STATS or line is None or odds is None or side not in ("OVER", "UNDER"):
            continue
        row = {"ranking_key": r["ranking_key"], "prediction_id": r.get("prediction_id"), "date": date,
               "game": r.get("game"), "commence_time": r.get("commence_time"), "player": r.get("player"),
               "stat": r.get("stat"), "side": side, "line": line, "odds": int(odds),
               "sportsbook_key": r.get("sportsbook_key"), "market_probability": _num(r.get("market_probability")),
               "raw_model_probability": _num(r.get("raw_model_probability")),
               "v1_projection": _num(r.get("projection")), "v2_projection": _num(r.get("v2_projection")),
               "mc_version": version, "mc_params_digest": digest, "n_sims": n_sims,
               "published_at_utc": published.isoformat(), "evidence": "SHADOW_ONLY"}
        for arm in ("mc_v1", "mc_v2"):
            x = sims_for(arms, arm, stat)[i]
            ok = bool(arms[arm]["available"][i])
            p, push = side_probability(x, line, side) if ok else (None, None)
            row[f"p_{arm}"] = None if p is None else round(p, 5)
            row[f"push_{arm}"] = None if push is None else round(push, 5)
            row[f"mean_{arm}"] = round(float(x.mean()), 3) if ok else None
            if p is not None:
                win = p * (1 - push)
                row[f"ev_{arm}"] = round(float(R.expected_value([win], [1 - push - win], [odds])[0]), 5)
        new_rows.append(row)
    if new_rows:
        SHADOW.parent.mkdir(parents=True, exist_ok=True)
        with SHADOW.open("a", encoding="utf-8") as fh:
            for row in new_rows:
                fh.write(json.dumps(row, allow_nan=False) + "\n")
    status.update(ledger_prop_rows=len(ledger), shadow_rows_appended=len(new_rows),
                  shadow_rows_already_present=sum(r.get("ranking_key") in existing for r in ledger))

    # 2) Research Lab payload: every quote, distributions and scenarios.
    if lab:
        status["lab"] = write_lab(date, data, df, arms, params, seed, n_sims, version, digest, published)
    status["status"] = "PUBLISHED"
    return status


def write_lab(date, data, df, arms, params, seed, n_sims, version, digest, published) -> dict:
    quotes = quote_book(date)
    ledger = {r["ranking_key"]: r for r in read_ledger() if r.get("market") == "PROP" and r.get("date") == date}
    # Scenario sensitivity: same seed, one input changed, V2 arm where available else V1.
    scen = {}
    for name, change in SCENARIOS.items():
        d = df.copy()
        if "minutes" in change:
            d["minutes"] = d["minutes"] * change["minutes"]
        single, pra_means, _ = arm_means(df, "v2")
        f = change.get("minutes", 1.0) * change.get("usage", 1.0)
        s = engine.simulate(d, {k: v * f for k, v in single.items()}, params, 2000, seed)
        p = engine.simulate(d, {k: v * f for k, v in pra_means.items()}, params, 2000, seed)
        s["pra"] = p["pra"]
        scen[name] = s
    cards = []
    for i, r in df.reset_index(drop=True).iterrows():
        for stat, label in STATS.items():
            if _num(r[f"v1_{stat}"]) is None:
                continue
            arm = "mc_v2" if arms["mc_v2"]["available"][i] else "mc_v1"
            x = sims_for(arms, arm, stat)[i]
            qs = quotes[(quotes["matchup"] == normalize_team_matchup(r["game"])) & (quotes["key"] == norm_name(r["player"]))
                        & (quotes["stat"] == stat)] if not quotes.empty else pd.DataFrame()
            offers = []
            for (book, line), g in (qs.groupby(["book_key", "line"]) if not qs.empty else []):
                prices = {s: int(v) for s, v in zip(g["side"], g["price"])}
                nv = g.set_index("side")["no_vig_probability"].to_dict()
                p_over, push = side_probability(x, float(line), "OVER")
                offers.append({"book_key": book, "exchange": book in R.EXCHANGES, "line": float(line),
                               "over": prices.get("OVER"), "under": prices.get("UNDER"),
                               "no_vig_over": _num(nv.get("OVER")), "mc_over": None if p_over is None else round(p_over, 4),
                               "push": round(push, 4)})
            rk = stable_key([date, r["game"], "PROP", r["player"], stat])
            led = ledger.get(rk)
            rec_line = _num(led.get("line")) if led else None
            p_side = side_probability(x, rec_line, str(led.get("side")).upper())[0] if led and rec_line is not None else None
            cards.append({
                "player": r["player"], "team": r["team"], "opp": r["opp"], "game": r["game"], "tip": r["tip"],
                "injury_status": r["injury_status"], "minutes": r["minutes"], "stat": label,
                "v1_projection": r[f"v1_{stat}"], "v2_projection": _num(r[f"v2_{stat}"]),
                "mc_arm": arm, "distribution": summary(x), "histogram": histogram(x),
                "scenarios": {n: summary(s[stat][i]) for n, s in scen.items()},
                "recorded": None if not led else {
                    "book_key": led.get("sportsbook_key"), "side": led.get("side"), "line": rec_line,
                    "odds": led.get("odds"), "market_probability": _num(led.get("market_probability")),
                    "v1_probability": _num(led.get("raw_model_probability")), "mc_probability": p_side,
                    "v1_decision": led.get("decision")},
                "research_label": research_label(p_side, _num(led.get("market_probability")) if led else None,
                                                 led is not None),
                "offers": sorted(offers, key=lambda o: (o["exchange"], o["book_key"], o["line"]))})
    payload = {"date": date, "generated_at_utc": published.isoformat(), "mc_version": version,
               "params_digest": digest, "n_sims": n_sims, "games": int(df["game"].nunique()),
               "players": int(len(df)), "v2_players": int(arms["mc_v2"]["available"].sum()),
               "v2_features": v2_feature_status(date),
               "injury_feed": data.get("injury_feed"), "cards": cards,
               "policy": "SHADOW_ONLY: research labels only; BET eligibility belongs to V1."}
    LAB.parent.mkdir(parents=True, exist_ok=True)
    LAB.write_text(json.dumps(payload, separators=(",", ":"), allow_nan=False, default=str) + "\n")
    return {"cards": len(cards), "path": str(LAB)}


def graded_rows() -> pd.DataFrame:
    if not SHADOW.exists():
        return pd.DataFrame()
    shadow = pd.read_json(SHADOW, lines=True)
    ledger = pd.DataFrame([r for r in read_ledger() if r.get("market") == "PROP"])
    if shadow.empty or ledger.empty:
        return pd.DataFrame()
    out = shadow.merge(ledger[["ranking_key", "outcome", "actual"]], on="ranking_key", how="inner")
    out = out[out["outcome"].isin(["WIN", "LOSS", "PUSH"])].copy()
    tip = pd.to_datetime(out["commence_time"], utc=True, errors="coerce")
    pub = pd.to_datetime(out["published_at_utc"], utc=True, errors="coerce")
    out["pregame"] = tip.isna() | (pub < tip)
    out["won"] = out["outcome"].eq("WIN")
    out["p_market"], out["p_v1"] = out["market_probability"], out["raw_model_probability"]
    for arm in ("market", "v1"):
        p = out[f"p_{arm}"].astype(float)
        out[f"ev_{arm}"] = R.expected_value(p, 1 - p, out["odds"])
    return out


def grade() -> dict:
    rows = graded_rows()
    report = {"generated_at_utc": datetime.now(timezone.utc).isoformat(), "policy": "SHADOW_ONLY",
              "strict_gate_rules": gate.RULES}
    if rows.empty:
        report.update(status="NO_GRADED_ROWS", graded_rows=0)
    else:
        late = int((~rows["pregame"]).sum())
        rows = rows[rows["pregame"] & rows["outcome"].isin(["WIN", "LOSS"])]
        report.update(status="GRADED", graded_rows=int(len(rows)), excluded_published_after_tip=late,
                      dates=int(rows["date"].nunique()), first_date=str(rows["date"].min()) if len(rows) else None,
                      last_date=str(rows["date"].max()) if len(rows) else None)
        for name, books in (("sportsbooks", R.SPORTSBOOKS), ("exchanges_novig", R.EXCHANGES)):
            g = rows[rows["sportsbook_key"].isin(books)]
            both = g.dropna(subset=["p_mc_v1", "p_mc_v2", "p_v1", "p_market"])
            v1only = g.dropna(subset=["p_mc_v1", "p_v1", "p_market"])
            report[name] = {
                "all_arms_common_rows": R.evaluate(both, R.ARMS) if len(both) >= 20 else {"rows": int(len(both))},
                "v1_anchored_all_rows": (R.evaluate(v1only, ["market", "v1", "mc_v1"]) if len(v1only) >= 20
                                         else {"rows": int(len(v1only))})}
        daily = []
        for d, g in rows[rows["sportsbook_key"].isin(R.SPORTSBOOKS)].groupby("date"):
            y = g["won"].to_numpy(float)
            item = {"date": d, "rows": int(len(g))}
            for arm in ("market", "v1", "mc_v1", "mc_v2"):
                p = g[f"p_{arm}"].dropna()
                if len(p):
                    item[f"log_loss_{arm}"] = R.log_loss(p.to_numpy(float), g.loc[p.index, "won"].to_numpy(float))
            daily.append(item)
        report["daily"] = daily
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2, default=str, allow_nan=False) + "\n")
    return report


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("publish")
    p.add_argument("--date", required=True)
    p.add_argument("--n-sims", type=int, default=N_SIMS)
    p.add_argument("--predictions", default=None, help="prediction file (default predictions_<date>.json)")
    p.add_argument("--no-lab", action="store_true", help="append shadow rows only (pre-tip refresh)")
    sub.add_parser("grade")
    a = ap.parse_args()
    out = publish(a.date, a.n_sims, a.predictions, not a.no_lab) if a.cmd == "publish" else grade()
    print(json.dumps({k: v for k, v in out.items() if k not in ("sportsbooks", "exchanges_novig", "daily",
                                                                  "strict_gate_rules")}, indent=2, default=str))


if __name__ == "__main__":
    main()
