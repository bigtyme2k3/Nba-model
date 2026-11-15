"""
v2/simulation/calibration.py
----------------------------
Fits the simulation's spread parameters on one calibration season by minimizing
CRPS (a proper score for whole distributions), so evaluation seasons stay untouched.

Minutes noise and early exits are fit on minutes; the game factor, player factor
and per-stat dispersion are then fit on the four stats. Coarse grids, small
simulation counts: these are shape parameters, not a search for bets.
"""
from __future__ import annotations

import itertools

import numpy as np

from v2.simulation import engine, probabilities as P

GRID_MINUTES_SCALE = [0.6, 0.8, 1.0, 1.2, 1.5]
GRID_EXIT = [0.0, 0.03, 0.06]
GRID_GAME_SD = [0.0, 0.05, 0.10]
GRID_PLAYER_SD = [0.0, 0.10, 0.20, 0.30]
GRID_DISPERSION = [2.0, 4.0, 8.0, 16.0, 50.0, 1000.0]   # 1000 is effectively Poisson
GRID_FT_SHARE = [0.15, 0.23, 0.30]
GRID_TEAM_CORR = [0.0, -0.01, -0.02, -0.03, -0.04, -0.05]
GRID_OPP_CORR = [0.0, 0.005, 0.01, 0.015, 0.02]
COUPLING_STATS = ["pts", "reb", "ast"]


def residual_z(y, mu):
    mu = np.asarray(mu, float)
    return (np.asarray(y, float) - mu) / np.sqrt(np.maximum(mu, 0.5))


def game_pairs(df):
    """Index pairs (i < j) of players in the same game, and whether they are teammates."""
    games = df["game_id"].astype(str).to_numpy()
    teams = df["team"].astype(str).to_numpy()
    I, J = [], []
    for g in np.unique(games):
        rows = np.where(games == g)[0]
        a, b = np.triu_indices(len(rows), 1)
        I.append(rows[a]); J.append(rows[b])
    I, J = np.concatenate(I), np.concatenate(J)
    return I, J, teams[I] == teams[J]


def pair_correlation(za, zb, same):
    """Pooled residual correlation for teammate pairs and opponent pairs."""
    out = []
    for m in (same, ~same):
        a, b = za[m].ravel(), zb[m].ravel()
        out.append(float(np.corrcoef(a, b)[0, 1]) if len(a) > 2 else float("nan"))
    return out


def fit_coupling(df, means: dict, actual: dict, params: engine.Params, n_games: int = 300,
                 n_sims: int = 200, seed: int = 9) -> tuple[engine.Params, dict]:
    """Teammate / opponent correlation of the reordering step, matched to the calibration
    season's actual residual correlations (whole games, so pairs stay intact)."""
    I, J, same = game_pairs(df)
    target = {s: pair_correlation(residual_z(actual[s], means[s])[I], residual_z(actual[s], means[s])[J], same)
              for s in COUPLING_STATS}
    rng = np.random.default_rng(seed)
    games = df["game_id"].astype(str)
    keep = games.isin(rng.choice(games.unique(), min(n_games, games.nunique()), replace=False)).to_numpy()
    d = df[keep].reset_index(drop=True)
    mu = {s: np.asarray(m)[keep] for s, m in means.items()}
    i, j, sm = game_pairs(d)
    best = None
    for tc, oc in itertools.product(GRID_TEAM_CORR, GRID_OPP_CORR):
        p = engine.Params(**{**params.to_dict(), "team_corr": tc, "opp_corr": oc})
        sims = engine.simulate(d, mu, p, n_sims, seed)
        got = {}
        for s in COUPLING_STATS:
            z = residual_z(sims[s], mu[s][:, None])
            got[s] = pair_correlation(z[i], z[j], sm)
        err = sum((got[s][k] - target[s][k]) ** 2 for s in COUPLING_STATS for k in (0, 1))
        if best is None or err < best[0]:
            best = (err, tc, oc, got)
    err, tc, oc, got = best
    report = {"target_teammate_opponent_corr": target, "simulated_teammate_opponent_corr": got,
              "squared_error": err, "games": int(d["game_id"].nunique())}
    return engine.Params(**{**params.to_dict(), "team_corr": tc, "opp_corr": oc}), report


def fit(df, means: dict, actual: dict, n_sims: int = 400, max_rows: int = 6000, seed: int = 7) -> tuple[engine.Params, dict]:
    rng = np.random.default_rng(seed)
    rows = rng.choice(len(df), size=min(max_rows, len(df)), replace=False)
    d = df.iloc[rows].reset_index(drop=True)
    mu = {s: np.asarray(m)[rows] for s, m in means.items()}
    y = {s: np.asarray(a)[rows] for s, a in actual.items()}
    report = {"rows": int(len(d)), "n_sims": n_sims}

    best = None
    for k, e in itertools.product(GRID_MINUTES_SCALE, GRID_EXIT):
        p = engine.Params(minutes_sd_scale=k, early_exit_prob=e)
        sims = engine.simulate(d, {}, p, n_sims, seed)
        score = float(P.crps_samples(sims["minutes"], y["minutes"]).mean())
        if best is None or score < best[0]:
            best = (score, k, e)
    report["minutes_crps"], k, e = best

    best_total, best_params = None, None
    for gsd, psd in itertools.product(GRID_GAME_SD, GRID_PLAYER_SD):
        disp, total, ft_best = {}, 0.0, engine.Params.ft_share
        # Threes first: event-built points reuse the threes path and its dispersion.
        for stat in sorted(mu, key=lambda st: st != "threes"):
            scores = []
            shares = GRID_FT_SHARE if stat == "pts" and "threes" in mu else [engine.Params.ft_share]
            for r, ft in itertools.product(GRID_DISPERSION, shares):
                p = engine.Params(minutes_sd_scale=k, early_exit_prob=e, game_sd=gsd, player_sd=psd,
                                  dispersion={**disp, stat: r}, ft_share=ft)
                m = {stat: mu[stat]}
                if stat == "pts" and "threes" in mu:
                    m["threes"] = mu["threes"]
                sims = engine.simulate(d, m, p, n_sims, seed)
                scores.append((float(P.crps_samples(sims[stat], y[stat]).mean()), r, ft))
            s, r, ft = min(scores)
            disp[stat], total = r, total + s
            if stat == "pts":
                ft_best = ft
        if best_total is None or total < best_total:
            best_total = total
            best_params = engine.Params(minutes_sd_scale=k, early_exit_prob=e, game_sd=gsd,
                                        player_sd=psd, dispersion=disp, ft_share=ft_best)
    report["stat_crps_sum"] = best_total
    return best_params, report
