"""
v2/simulation/price_research.py
-------------------------------
Phase 3: price-aware research. Monte Carlo probabilities scored at the exact
sportsbook quotes the forward ledger recorded before each game (FanDuel,
DraftKings, Fanatics; Novig reported separately), against the market and V1.

Arms, on the same graded rows:
  market   the recorded quote's no-vig probability (market_probability)
  v1       V1's recorded raw probability (Normal around V1's projection)
  mc_v1    Monte Carlo around V1's recorded projection and projected minutes:
           the same mean as v1, only the shape changes
  mc_v2    Monte Carlo around V2's mean: V2 models trained on seasons before this
           one, prior-only features, teammate absence from the injury report as it
           stood before the ledger's predictions (the git-history version of
           data/raw/injuries_<date>.csv scraped before them; no such version means
           no V2 row that date, as live V2 skips an unverified feed)

Everything is point-in-time: projections, minutes and quotes come from the ledger
row written before tip; V2's features use only games before the date.

Scores: Brier and log loss of the recorded side winning (pushes excluded; a
whole-number line's push probability is removed from the simulated side), paired
differences against the market with date-clustered bootstrap intervals, the
existing calibration gate (calibrate_edges.temporal_calibration, unchanged) and the
strict promotion gate (v2/simulation/gate.py) applied to each arm, and hypothetical ROI at the recorded American odds for "bet when
expected value >= threshold". Research only: nothing here writes the live
calibration, changes a verdict or gives any arm bet authority.

Usage:
    python -m v2.simulation.price_research
Output:
    data/tracking/v2_mc_price_research.json
"""
from __future__ import annotations

import argparse
import json
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd

import calibrate_edges
from prop_features import BASE_FEATURES
from v2 import lineups, live, pilot, warehouse
from v2.simulation import engine, gate

LEDGER = Path("data/history/nba_forward_predictions.jsonl")
OUT = Path("data/tracking/v2_mc_price_research.json")
CALIBRATION = Path("models/v2_mc_calibration.json")
STAT_KEY = {"PTS": "pts", "REB": "reb", "AST": "ast", "THREES": "threes", "PRA": "pra"}
SPORTSBOOKS = {"fanduel", "draftkings", "fanatics"}
EXCHANGES = {"novig"}
EV_THRESHOLDS = [0.0, 0.03, 0.05, 0.10]
ARMS = ["market", "v1", "mc_v1", "mc_v2"]


# ---------------------------------------------------------------- prices and scores
def american_to_decimal(odds) -> np.ndarray:
    o = np.asarray(odds, float)
    return np.where(o > 0, 1 + o / 100, 1 + 100 / np.abs(o))


def side_probabilities(sims: np.ndarray, line, side) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(win, push, loss) of the recorded side from simulated outcomes."""
    line = np.asarray(line, float)[:, None]
    over, under = (sims > line).mean(axis=1), (sims < line).mean(axis=1)
    push = 1 - over - under
    is_over = np.asarray([str(s).upper() == "OVER" for s in side])
    win = np.where(is_over, over, under)
    return win, push, np.where(is_over, under, over)


def expected_value(p_win, p_loss, odds) -> np.ndarray:
    return np.asarray(p_win) * (american_to_decimal(odds) - 1) - np.asarray(p_loss)


def brier(p, y):
    return float(np.mean((np.asarray(p) - y) ** 2))


def log_loss(p, y):
    p = np.clip(np.asarray(p, float), 0.01, 0.99)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def date_bootstrap(dates, stat_fn, n_boot=2000, seed=0):
    """Percentile interval of stat_fn(index array), resampling whole dates."""
    rng = np.random.default_rng(seed)
    groups = defaultdict(list)
    for i, d in enumerate(dates):
        groups[d].append(i)
    keys = list(groups)
    vals = []
    for _ in range(n_boot):
        idx = np.concatenate([groups[keys[j]] for j in rng.integers(0, len(keys), len(keys))])
        v = stat_fn(idx)
        if v is not None and np.isfinite(v):
            vals.append(v)
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))] if vals else [None, None]


# ---------------------------------------------------------------- point-in-time inputs
def ledger_props(path=LEDGER) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    props = pd.DataFrame([r for r in rows if r.get("market") == "PROP"])
    graded = props[props["outcome"].isin(["WIN", "LOSS", "PUSH"])].copy()
    for c in ("line", "odds", "actual", "projection", "market_probability", "raw_model_probability",
              "projected_minutes"):
        graded[c] = pd.to_numeric(graded[c], errors="coerce")
    graded = graded.dropna(subset=["line", "odds", "actual", "projection", "market_probability"])
    graded["stat_key"] = graded["stat"].map(STAT_KEY)
    graded["name_key"] = graded["player"].map(live.norm_name)
    graded["generated"] = pd.to_datetime(graded["prediction_generated_at_utc"], utc=True, errors="coerce")
    return graded.reset_index(drop=True), props


def injury_snapshot_before(day: str, cutoff: pd.Timestamp) -> tuple[pd.DataFrame | None, str | None]:
    """Latest version of data/raw/injuries_<day>.csv scraped before cutoff, from git history."""
    path = f"data/raw/injuries_{day}.csv"
    try:
        commits = subprocess.run(["git", "log", "--format=%H", "--", path], capture_output=True,
                                 text=True, check=True).stdout.split()
    except (subprocess.CalledProcessError, FileNotFoundError):
        commits = []
    best = (None, None, None)
    for h in commits:
        text = subprocess.run(["git", "show", f"{h}:{path}"], capture_output=True, text=True).stdout
        if not text.strip():
            continue
        df = pd.read_csv(StringIO(text))
        if "scraped_at" not in df or df.empty:
            continue
        scraped = pd.to_datetime(df["scraped_at"], utc=True, errors="coerce").max()
        if pd.notna(scraped) and scraped <= cutoff and (best[0] is None or scraped > best[0]):
            best = (scraped, df, h)
    return best[1], (str(best[0]) if best[0] is not None else None)


def v2_inputs(season: int, train_seasons, db_path, graded: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Per (date, player): V1-model and V2-model means, minutes inputs, point-in-time absences."""
    df, _ = pilot.build_dataset(list(train_seasons) + [season], db_path, refresh=[season])
    df["name_key"] = df["player"].map(live.norm_name)
    df["date"] = df["game_date"].dt.strftime("%Y-%m-%d")
    cur = df[(df["v2_season"] == season) & df["date"].isin(set(graded["date"]))].copy()
    train = df[df["v2_season"] < season]

    con = warehouse.build(list(train_seasons) + [season], db_path)
    stints, _ = lineups.reconstruct_all(con, season)
    box = con.execute("SELECT * FROM player_box WHERE game_id IN (SELECT DISTINCT game_id FROM pbp)").df()
    box["game_date"] = pd.to_datetime(box["game_date"])
    stints = stints.merge(box[["game_id", "game_date"]].drop_duplicates("game_id"), on="game_id", how="left")

    report = {"dates": {}}
    parts = []
    for day, rows in cur.groupby("date"):
        cutoff = graded.loc[graded["date"] == day, "generated"].min()
        snap, scraped = injury_snapshot_before(day, cutoff)
        info = {"predictions_from_utc": str(cutoff), "injury_snapshot_scraped_utc": scraped}
        rows = rows.copy()
        if snap is None:
            info["v2"] = "skipped: no injury report scraped before the predictions"
            rows[pilot.ABS] = np.nan
            rows["v2_available"] = False
        else:
            before = pd.Timestamp(day)
            st = stints[stints["game_date"] < before].drop(columns="game_date")
            bx = box[box["game_date"] < before]
            pit = live.next_game_features(st, bx, day, live.injury_statuses(snap))
            rows = rows.drop(columns=pilot.ABS).merge(pit[["player_id"] + pilot.ABS], on="player_id", how="left")
            rows["v2_available"] = True
            info["v2"] = "point-in-time absences"
        report["dates"][day] = info
        parts.append(rows)
    cur = pd.concat(parts, ignore_index=True)

    # The ledger's own pregame minutes replace the training-style minutes feature.
    minutes = graded.drop_duplicates(["date", "name_key"]).set_index(["date", "name_key"])["projected_minutes"]
    key = pd.MultiIndex.from_frame(cur[["date", "name_key"]])
    cur["minutes"] = pd.Series(minutes.reindex(key).to_numpy(), index=cur.index).fillna(cur["minutes"])

    base = [f for f in BASE_FEATURES + ["def_adj", "usage_pace"] if f in df.columns]
    for stat in engine.STATS:
        f1 = [f for f in base if f != stat]
        cur[f"v1m_{stat}"] = pilot.model().fit(train[f1], train[stat]).predict(cur[f1])
        if stat in ("pts", "reb", "ast"):
            f2 = [f for f in base + pilot.OPP + pilot.ABS if f != stat]
            cur[f"v2m_{stat}"] = pilot.model().fit(train[f2], train[stat]).predict(cur[f2])
    report["train_seasons"] = list(train_seasons)
    report["player_dates"] = int(len(cur))
    return cur, report


def v1_anchored_means(cur: pd.DataFrame, graded: pd.DataFrame, props: pd.DataFrame) -> dict:
    """Means equal to V1's recorded projections where the ledger has them (any verdict),
    otherwise the V1-feature model; PRA rescales points, rebounds and assists to the
    recorded PRA projection."""
    proj = props.assign(name_key=props["player"].map(live.norm_name),
                        projection=pd.to_numeric(props["projection"], errors="coerce"))
    proj = proj.dropna(subset=["projection"]).drop_duplicates(["date", "name_key", "stat"], keep="last")
    look = proj.set_index(["date", "name_key", "stat"])["projection"]
    key = list(zip(cur["date"], cur["name_key"]))
    get = lambda stat: np.array([look.get((d, n, stat), np.nan) for d, n in key])
    m = {s: cur[f"v1m_{s}"].to_numpy(float).copy() for s in engine.STATS}
    for s, label in (("pts", "PTS"), ("reb", "REB"), ("ast", "AST"), ("threes", "THREES")):
        rec = get(label)
        m[s] = np.where(np.isfinite(rec), rec, m[s])
    pra = get("PRA")
    total = m["pts"] + m["reb"] + m["ast"]
    k = np.where(np.isfinite(pra) & (total > 0), pra / np.maximum(total, 1e-9), 1.0)
    pra_means = {s: m[s] * k for s in ("pts", "reb", "ast")}
    return {"single": {s: np.clip(v, 0, None) for s, v in m.items()},
            "pra": {**{s: np.clip(v, 0, None) for s, v in pra_means.items()},
                    "threes": np.clip(np.minimum(m["threes"], pra_means["pts"] / 3), 0, None)}}


# ---------------------------------------------------------------- evaluation
def arm_scores(rows: pd.DataFrame, arm: str) -> dict:
    y = rows["won"].to_numpy(float)
    p = rows[f"p_{arm}"].to_numpy(float)
    out = {"rows": int(len(rows)), "brier": brier(p, y), "log_loss": log_loss(p, y),
           "mean_probability": float(p.mean()), "win_rate": float(y.mean())}
    if arm != "market":
        m = rows["p_market"].to_numpy(float)
        d = rows["date"].to_numpy()
        diff = lambda idx: brier(p[idx], y[idx]) - brier(m[idx], y[idx])
        ldiff = lambda idx: log_loss(p[idx], y[idx]) - log_loss(m[idx], y[idx])
        out["brier_minus_market"] = diff(np.arange(len(y)))
        out["brier_minus_market_ci95"] = date_bootstrap(d, diff)
        out["log_loss_minus_market"] = ldiff(np.arange(len(y)))
        out["log_loss_minus_market_ci95"] = date_bootstrap(d, ldiff)
    return out


def roi(rows: pd.DataFrame, arm: str, threshold: float) -> dict:
    ev = rows[f"ev_{arm}"].to_numpy(float)
    pick = ev >= threshold
    stake_return = np.where(rows["won"].to_numpy(bool), american_to_decimal(rows["odds"]) - 1, -1.0)
    stake_return = np.where(rows["outcome"].eq("PUSH").to_numpy(), 0.0, stake_return)
    d = rows["date"].to_numpy()
    n = int(pick.sum())
    out = {"threshold": threshold, "bets": n}
    if n:
        r = stake_return[pick]
        out.update(roi=float(r.mean()), units=float(r.sum()), win_rate=float(rows["won"].to_numpy()[pick].mean()),
                   mean_ev=float(ev[pick].mean()),
                   roi_ci95=date_bootstrap(d, lambda idx: stake_return[idx][pick[idx]].mean()
                                           if pick[idx].any() else None))
    return out


def calibration_gate(rows: pd.DataFrame, arm: str) -> dict:
    """The live gate, unchanged: shrink toward the arm fit on earlier dates, held-out
    dates must beat the market on both Brier and log loss."""
    pseudo = [{"market": "PROP", "model_family_id": "research", "date": r.date, "outcome": r.outcome,
               "market_probability": r.p_market, "raw_model_probability": getattr(r, f"p_{arm}")}
              for r in rows.itertuples() if r.outcome in ("WIN", "LOSS")]
    g = calibrate_edges.temporal_calibration(pseudo, active_ids={"PROP": "research"})["prop_price"]
    return {k: g.get(k) for k in ("status", "validation", "candidate_shrink", "training_rows", "holdout_rows",
                                  "holdout_from", "candidate_scores", "market_scores")}


def evaluate(rows: pd.DataFrame, arms) -> dict:
    out = {"rows": int(len(rows)), "dates": int(rows["date"].nunique()),
           "scores": {a: arm_scores(rows, a) for a in arms},
           "by_stat": {}, "by_book": {}, "calibration_gate": {}, "hypothetical_roi": {}}
    for stat, g in rows.groupby("stat"):
        out["by_stat"][stat] = {a: {k: v for k, v in arm_scores(g, a).items() if "ci95" not in k} for a in arms}
    for book, g in rows.groupby("sportsbook_key"):
        out["by_book"][book] = {"rows": int(len(g)),
                                **{a: {"log_loss": log_loss(g[f"p_{a}"], g["won"].to_numpy(float))} for a in arms},
                                "roi_mc_ev_ge_5pct": {a: roi(g, a, 0.05) for a in arms if a != "market"}}
    for a in arms:
        if a != "market":
            out["calibration_gate"][a] = calibration_gate(rows, a)
            out.setdefault("strict_gate", {})[a] = gate.strict_gate(rows, a)
            out["hypothetical_roi"][a] = [roi(rows, a, t) for t in EV_THRESHOLDS]
    stake = np.where(rows["won"], american_to_decimal(rows["odds"]) - 1, -1.0)
    out["every_recorded_side_roi"] = float(stake.mean())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=2026)
    ap.add_argument("--train-seasons", type=int, nargs="+", default=[2021, 2022, 2023, 2024, 2025])
    ap.add_argument("--n-sims", type=int, default=10000)
    ap.add_argument("--db", default=str(warehouse.DB_PATH))
    a = ap.parse_args()

    graded, props = ledger_props()
    cal = json.loads(CALIBRATION.read_text())
    params = engine.Params.from_dict(cal["params"])
    cur, v2_report = v2_inputs(a.season, a.train_seasons, a.db, graded)
    cur = cur.drop_duplicates(["date", "name_key"]).reset_index(drop=True)

    anchored = v1_anchored_means(cur, graded, props)
    v2m = {s: np.clip(cur[f"v2m_{s}" if s != "threes" else "v1m_threes"].to_numpy(float), 0, None)
           for s in engine.STATS}
    sims = {"mc_v1": engine.simulate(cur, anchored["single"], params, a.n_sims, seed=11),
            "mc_v1_pra": engine.simulate(cur, anchored["pra"], params, a.n_sims, seed=11),
            "mc_v2": engine.simulate(cur, v2m, params, a.n_sims, seed=11)}

    pos = {k: i for i, k in enumerate(zip(cur["date"], cur["name_key"]))}
    graded["row"] = [pos.get(k, -1) for k in zip(graded["date"], graded["name_key"])]
    matched = graded[graded["row"] >= 0].copy()
    report = {"generated_at_utc": datetime.now(timezone.utc).isoformat(), "season": a.season,
              "n_sims": a.n_sims, "mc_calibration_version": cal.get("version"),
              "ledger_graded_props": int(len(graded)), "matched_to_player_game": int(len(matched)),
              "unmatched_examples": graded.loc[graded["row"] < 0, "player"].head(10).tolist(),
              "v2_inputs": v2_report}

    idx = matched["row"].to_numpy()
    matched["won"] = matched["outcome"].eq("WIN")
    matched["p_market"] = matched["market_probability"]
    matched["p_v1"] = matched["raw_model_probability"]
    for arm in ("mc_v1", "mc_v2"):
        win = np.full(len(matched), np.nan)
        loss = np.full(len(matched), np.nan)
        for stat, g in matched.groupby("stat_key"):
            source = sims["mc_v1_pra"] if (arm == "mc_v1" and stat == "pra") else sims[arm]
            m = (matched["stat_key"] == stat).to_numpy()
            w, _, l = side_probabilities(source[stat][idx[m]], g["line"], g["side"])
            win[m], loss[m] = w, l
        # Win probability given no push, the quantity the market's no-vig number states.
        matched[f"p_{arm}"] = win / np.maximum(win + loss, 1e-9)
        matched[f"ev_{arm}"] = expected_value(win, loss, matched["odds"])
    for arm in ("market", "v1"):
        p = matched[f"p_{arm}"].to_numpy(float)
        matched[f"ev_{arm}"] = expected_value(p, 1 - p, matched["odds"])
    v2_ok = matched["row"].map(cur["v2_available"]).fillna(False).to_numpy(bool)
    report["timing_minutes_before_tip"] = (
        lambda t: {"median": float(t.median()), "p10": float(t.quantile(.1)), "p90": float(t.quantile(.9))}
        if len(t) else None)(
        ((pd.to_datetime(matched.get("commence_time"), utc=True, errors="coerce") - matched["generated"])
         .dt.total_seconds() / 60).dropna())

    books = matched[matched["sportsbook_key"].isin(SPORTSBOOKS)]
    report["sportsbooks"] = {
        "all_arms_common_rows": evaluate(books[v2_ok[matched["sportsbook_key"].isin(SPORTSBOOKS).to_numpy()]], ARMS),
        "v1_anchored_all_rows": evaluate(books, ["market", "v1", "mc_v1"])}
    ex = matched[matched["sportsbook_key"].isin(EXCHANGES)]
    report["exchanges_novig"] = (evaluate(ex, ["market", "v1", "mc_v1"]) if len(ex)
                                 else {"rows": 0, "note": "no graded Novig quotes recorded yet"})
    OUT.write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(json.dumps({k: report[k] for k in ("ledger_graded_props", "matched_to_player_game")}, indent=2))
    common = report["sportsbooks"]["all_arms_common_rows"]
    print(json.dumps({a: {k: common["scores"][a].get(k) for k in ("rows", "brier", "log_loss",
                      "log_loss_minus_market_ci95")} for a in ARMS}, indent=2))


if __name__ == "__main__":
    main()
