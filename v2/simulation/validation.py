"""
v2/simulation/validation.py
---------------------------
Historical backtest of the Monte Carlo layer against V1's probability method.

  1. Out-of-sample means: for each season S, V1-feature and V2-feature prop models
     (same HistGBR as props_model.py) are trained only on seasons before S.
  2. The simulation's spread parameters and the Normal baselines' sigmas are fit on
     the calibration season only.
  3. On later test seasons, three probability methods are scored on the same rows:
       v1_normal   V1 mean, Normal with V1's residual SD (V1's live method)
       v2_normal   V2 mean, same Normal shape (isolates the better mean)
       v2_mc       V2 mean, Monte Carlo distribution (adds the simulated shape)
     Scores: MAE of the mean, CRPS, 10-90 / 25-75 interval coverage, and over/under
     Brier and log loss at pseudo-lines (the player's prior 5-game average rounded
     to the nearest .5), which are independent of every model.

Usage:
    python -m v2.simulation.validation --calibration-season 2023 --test-seasons 2024 2025
Outputs:
    models/v2_mc_calibration.json
    data/tracking/v2_mc_validation.json
"""
from __future__ import annotations

import argparse
import json
import pickle
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from prop_features import BASE_FEATURES
from v2 import pilot, warehouse
from v2.simulation import cache, calibration, engine, probabilities as P

CACHE = Path("data/simulation/cache/oos.pkl")
STATS = engine.STATS          # pts, reb, ast, threes
V2_STATS = ["pts", "reb", "ast"]  # threes stay on V1 (V2 has no threes model yet)


def build_oos(seasons, predict_seasons, db_path) -> pd.DataFrame:
    df, _ = pilot.build_dataset(seasons, db_path)
    base = [f for f in BASE_FEATURES + ["def_adj", "usage_pace"] if f in df.columns]
    out = []
    for s in predict_seasons:
        train, test = df[df["v2_season"] < s], df[df["v2_season"] == s].copy()
        for stat in STATS:
            f1 = [f for f in base if f != stat]
            test[f"v1_{stat}"] = pilot.model().fit(train[f1], train[stat]).predict(test[f1])
            if stat in V2_STATS:
                f2 = [f for f in base + pilot.OPP + pilot.ABS if f != stat]
                test[f"v2_{stat}"] = pilot.model().fit(train[f2], train[stat]).predict(test[f2])
        out.append(test)
    keep = (["game_id", "player_id", "player", "game_date", "v2_season", "minutes", "minutes_sd_l10",
             "actual_minutes"] + STATS + [f"roll5_{s}" for s in STATS]
            + [f"v1_{s}" for s in STATS] + [f"v2_{s}" for s in V2_STATS])
    return pd.concat(out, ignore_index=True)[keep]


CACHE_SCHEMA = 2   # bump when build_oos changes what it computes


def oos_key(seasons, predict_seasons) -> str:
    return cache.cache_key(schema=CACHE_SCHEMA, seasons=list(seasons), predict=list(predict_seasons),
                           base=BASE_FEATURES, opp=pilot.OPP, abs=pilot.ABS, v2_stats=V2_STATS,
                           model=repr(pilot.model().get_params()))


def attach_team(oos: pd.DataFrame, db_path) -> pd.DataFrame:
    """Team for each player-game from the warehouse (the coupling needs teammates)."""
    con = warehouse.duckdb.connect(str(db_path), read_only=True)
    team = con.execute("SELECT CAST(game_id AS VARCHAR) game_id, CAST(athlete_id AS VARCHAR) player_id, team "
                       "FROM player_box").df()
    con.close()
    oos = oos.assign(game_id=oos["game_id"].astype(str), player_id=oos["player_id"].astype(str))
    return oos.merge(team.drop_duplicates(["game_id", "player_id"]), on=["game_id", "player_id"], how="left")


def same_game_pairs(c: pd.DataFrame, sims: dict, lines: dict, res: dict):
    """Joint 'both over' at pseudo-lines for every same-game pair: simulated joint
    probability vs the independent product of the two marginals."""
    I, J, same = calibration.game_pairs(c)
    for s in ("pts", "pra"):
        y = (c["pts"] + c["reb"] + c["ast"]).to_numpy(float) if s == "pra" else c[s].to_numpy(float)
        line = lines[s]
        over = sims[s] > line[:, None]
        p = over.mean(axis=1)
        joint = (over[I] & over[J]).mean(axis=1)
        r = res[s]
        r["joint"].append(joint); r["indep"].append(p[I] * p[J])
        r["both"].append(((y[I] > line[I]) & (y[J] > line[J])).astype(float)); r["same"].append(same)
        mu = sims[s].mean(axis=1)
        r["za"].append(calibration.residual_z(y[I], mu[I])); r["zb"].append(calibration.residual_z(y[J], mu[J]))
        sz = calibration.residual_z(sims[s], mu[:, None])
        r["sim_corr_same"].append(calibration.pair_correlation(sz[I][:, :200], sz[J][:, :200], same))


def summarize_pairs(res: dict) -> dict:
    out = {}
    for s, r in res.items():
        a = {k: np.concatenate(v) for k, v in r.items() if k != "sim_corr_same"}
        sims = np.array(r["sim_corr_same"])
        out[s] = {}
        for name, m in (("teammates", a["same"]), ("opponents", ~a["same"])):
            y = a["both"][m]
            out[s][name] = {
                "pairs": int(m.sum()), "both_over_rate": float(y.mean()),
                "actual_residual_corr": calibration.pair_correlation(a["za"], a["zb"], a["same"])[0 if name == "teammates" else 1],
                "simulated_residual_corr": float(np.nanmean(sims[:, 0 if name == "teammates" else 1])),
                "brier": {"independent": P.brier(a["indep"][m], y), "coupled_mc": P.brier(a["joint"][m], y)},
                "log_loss": {"independent": P.log_loss(a["indep"][m], y), "coupled_mc": P.log_loss(a["joint"][m], y)},
                "mean_prob": {"independent": float(a["indep"][m].mean()), "coupled_mc": float(a["joint"][m].mean())}}
    return out


def means_for(d: pd.DataFrame, prefix: str) -> dict:
    # Threes always use V1's projection (interim baseline in the design).
    return {s: np.clip(d[f"{prefix}_{s}" if s in V2_STATS else f"v1_{s}"].to_numpy(float), 0, None) for s in STATS}


def evaluate(d: pd.DataFrame, params: engine.Params, sigma: dict, n_sims: int, seed: int) -> dict:
    res = {s: {k: [] for k in ("y", "v1", "v2", "mc_mean", "crps_v1n", "crps_v2n", "crps_mc",
                                "cov80_n", "cov50_n", "cov80_mc", "cov50_mc",
                                "line", "p_v1n", "p_v2n", "p_mc", "pit")} for s in STATS + ["pra"]}
    pairs = {s: {k: [] for k in ("joint", "indep", "both", "same", "za", "zb", "sim_corr_same")}
             for s in ("pts", "pra")}
    d = d.sort_values(["game_id", "player_id"]).reset_index(drop=True)
    # Chunks end on game boundaries so same-game pairs and the coupling stay whole.
    gid = d["game_id"].to_numpy()
    cuts = [0] + [i for i in range(2000, len(d), 2000)] + [len(d)]
    cuts = sorted({c if c in (0, len(d)) else int(np.searchsorted(gid, gid[c], "left")) for c in cuts})
    for start, stop in zip(cuts[:-1], cuts[1:]):
        c = d.iloc[start:stop].reset_index(drop=True)
        m2 = means_for(c, "v2")
        sims = engine.simulate(c, m2, params, n_sims, seed + start)
        lines = {}
        for s in STATS + ["pra"]:
            if s == "pra":
                y = (c["pts"] + c["reb"] + c["ast"]).to_numpy(float)
                v1 = (c["v1_pts"] + c["v1_reb"] + c["v1_ast"]).to_numpy(float)
                v2 = m2["pts"] + m2["reb"] + m2["ast"]
                prior = (c["roll5_pts"] + c["roll5_reb"] + c["roll5_ast"]).to_numpy(float)
            else:
                y, v1, v2 = c[s].to_numpy(float), c[f"v1_{s}"].to_numpy(float), m2[s]
                prior = c[f"roll5_{s}"].to_numpy(float)
            x, sg1, sg = sims[s], sigma["v1"][s], sigma["v2"][s]
            line = np.floor(np.nan_to_num(prior, nan=0.0)) + 0.5
            lines[s] = line
            lo80, hi80 = np.percentile(x, 10, axis=1), np.percentile(x, 90, axis=1)
            lo50, hi50 = np.percentile(x, 25, axis=1), np.percentile(x, 75, axis=1)
            r = res[s]
            r["y"].append(y); r["v1"].append(v1); r["v2"].append(v2); r["mc_mean"].append(x.mean(axis=1))
            r["crps_v1n"].append(P.crps_normal(v1, sg1, y)); r["crps_v2n"].append(P.crps_normal(v2, sg, y))
            r["crps_mc"].append(P.crps_samples(x, y))
            z80, z50 = 1.2816 * sg, 0.6745 * sg
            r["cov80_n"].append((y >= v2 - z80) & (y <= v2 + z80)); r["cov50_n"].append((y >= v2 - z50) & (y <= v2 + z50))
            r["cov80_mc"].append((y >= lo80) & (y <= hi80)); r["cov50_mc"].append((y >= lo50) & (y <= hi50))
            r["line"].append(line)
            r["pit"].append(P.randomized_pit(x, y, seed + start))
            r["p_v1n"].append(P.normal_over(v1, sg1, line)); r["p_v2n"].append(P.normal_over(v2, sg, line))
            r["p_mc"].append(P.over_under_push(x, line)["over"])
        if "team" in c and c["team"].notna().all():
            same_game_pairs(c, sims, lines, pairs)
    out = {"same_game_pairs": summarize_pairs(pairs) if pairs["pts"]["joint"] else None}
    for s, r in res.items():
        a = {k: np.concatenate(v) for k, v in r.items()}
        over = (a["y"] > a["line"]).astype(float)
        out[s] = {
            "rows": int(len(a["y"])),
            "mae": {"v1": float(np.abs(a["y"] - a["v1"]).mean()), "v2": float(np.abs(a["y"] - a["v2"]).mean()),
                    "v2_mc_mean": float(np.abs(a["y"] - a["mc_mean"]).mean())},
            "crps": {"v1_normal": float(a["crps_v1n"].mean()), "v2_normal": float(a["crps_v2n"].mean()),
                     "v2_mc": float(a["crps_mc"].mean())},
            "coverage_80": {"v2_normal": float(a["cov80_n"].mean()), "v2_mc": float(a["cov80_mc"].mean())},
            "coverage_50": {"v2_normal": float(a["cov50_n"].mean()), "v2_mc": float(a["cov50_mc"].mean())},
            "pit_deciles_v2_mc": [round(float(v), 4) for v in np.histogram(a["pit"], 10, (0, 1))[0] / len(a["pit"])],
            "pseudo_line_over_rate": float(over.mean()),
            "brier": {"v1_normal": P.brier(a["p_v1n"], over), "v2_normal": P.brier(a["p_v2n"], over),
                      "v2_mc": P.brier(a["p_mc"], over)},
            "log_loss": {"v1_normal": P.log_loss(a["p_v1n"], over), "v2_normal": P.log_loss(a["p_v2n"], over),
                         "v2_mc": P.log_loss(a["p_mc"], over)},
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", type=int, nargs="+", default=[2021, 2022, 2023, 2024, 2025])
    ap.add_argument("--calibration-season", type=int, default=2023)
    ap.add_argument("--test-seasons", type=int, nargs="+", default=[2024, 2025])
    ap.add_argument("--n-sims", type=int, default=2000)
    ap.add_argument("--db", default=str(warehouse.DB_PATH))
    ap.add_argument("--rebuild", action="store_true")
    a = ap.parse_args()
    key = oos_key(a.seasons, [a.calibration_season] + a.test_seasons)
    oos = None if a.rebuild else cache.load(CACHE, key)
    if oos is None:
        oos = build_oos(a.seasons, [a.calibration_season] + a.test_seasons, a.db)
        cache.save(CACHE, key, oos)
    oos = oos[oos["minutes"].notna() & oos["actual_minutes"].notna()]
    oos = attach_team(oos, a.db)
    oos = oos[oos["team"].notna()]

    cal = oos[oos["v2_season"] == a.calibration_season].reset_index(drop=True)
    cal_means = means_for(cal, "v2")
    cal_actual = {**{s: cal[s].to_numpy(float) for s in STATS}, "minutes": cal["actual_minutes"].to_numpy(float)}
    params, fit_report = calibration.fit(cal, cal_means, cal_actual)
    params, fit_report["coupling"] = calibration.fit_coupling(cal, cal_means, cal_actual, params)
    sigma = {}
    for arm in ("v1", "v2"):
        m = means_for(cal, arm)
        sigma[arm] = {st: float(np.std(cal[st] - m[st])) for st in STATS}
        sigma[arm]["pra"] = float(np.std((cal["pts"] + cal["reb"] + cal["ast"]) - (m["pts"] + m["reb"] + m["ast"])))
    Path("models/v2_mc_calibration.json").write_text(json.dumps({
        "version": "V2_MC_3", "fit_on_season": a.calibration_season, "params": params.to_dict(),
        "normal_sigma_from_calibration_season": sigma, "fit": fit_report,
        "generated_at_utc": datetime.now(timezone.utc).isoformat()}, indent=2) + "\n")

    report = {"generated_at_utc": datetime.now(timezone.utc).isoformat(), "calibration_season": a.calibration_season,
              "test_seasons": a.test_seasons, "n_sims": a.n_sims, "params": params.to_dict(), "seasons": {}}
    for s in a.test_seasons:
        d = oos[oos["v2_season"] == s].reset_index(drop=True)
        report["seasons"][str(s)] = evaluate(d, params, sigma, a.n_sims, seed=1000 * s)
    out = Path("data/tracking/v2_mc_validation.json")
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
