"""
v2/simulation/engine.py
-----------------------
Simulates each player's game from his expected stats.

Per player-game and simulation:
  minutes   projected minutes plus player-specific noise (scaled 10-game minutes SD),
            truncated to [0, 48], with a small chance of an early exit (injury,
            foul trouble, blowout) that cuts minutes to 20-70%
  game      one pace/scoring factor shared by everyone in the game
  player    one form factor shared by that player's stats in the simulation
  stats     negative-binomial counts whose mean is the model's expected stat scaled
            by minutes, game and player factors; PRA is the sum of the same path
  points    built from scoring events, not counted one point at a time:
            3 x the simulated threes + 2 x two-point makes + free throws made, with the
            non-three points split by ft_share. Counting points as one Poisson-like
            variable made the distribution too narrow (2023-24 PIT: 14.7% of games in
            the bottom decile, 12.2% in the top); the event build puts both near 10%.

The scale factors are normalized so each player's simulated mean equals the model's
expected value: the simulation adds the distribution's shape, never a second
injury or minutes adjustment (V2's projection already contains those).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

STATS = ["pts", "reb", "ast", "threes"]


@dataclass
class Params:
    minutes_sd_scale: float = 1.0      # multiplies the player's 10-game minutes SD
    minutes_sd_default: float = 5.0    # SD when a player has no history
    early_exit_prob: float = 0.03
    game_sd: float = 0.05              # log-SD of the shared game factor
    player_sd: float = 0.15            # log-SD of the player's form factor
    dispersion: dict = field(default_factory=lambda: {"pts": 20.0, "reb": 20.0, "ast": 20.0, "threes": 20.0})
    points_model: str = "events"       # "events" (3s + 2s + FTs) or "count" (the V2_MC_1 shape)
    ft_share: float = 0.23             # share of non-three points scored at the line

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        return cls(**d)


def _lognormal(rng, sd, size):
    return np.exp(rng.normal(-0.5 * sd * sd, sd, size)) if sd > 0 else np.ones(size)


def simulate(df: pd.DataFrame, means: dict, params: Params, n_sims: int = 10000, seed: int = 0) -> dict:
    """df: one row per player-game with game_id, minutes (projected), minutes_sd_l10.
    means: stat -> array of expected values aligned with df. Returns stat -> (n, n_sims)."""
    rng = np.random.default_rng(seed)
    n = len(df)
    mu_min = np.clip(pd.to_numeric(df["minutes"], errors="coerce").fillna(0).to_numpy(float), 0, 48)
    sd = pd.to_numeric(df.get("minutes_sd_l10"), errors="coerce").to_numpy(float)
    sd = np.where(np.isfinite(sd) & (sd > 0), sd, params.minutes_sd_default) * params.minutes_sd_scale

    minutes = np.clip(mu_min[:, None] + sd[:, None] * rng.standard_normal((n, n_sims)), 0, 48)
    exits = rng.random((n, n_sims)) < params.early_exit_prob
    minutes = np.where(exits, minutes * rng.uniform(0.2, 0.7, (n, n_sims)), minutes)
    scale = np.where(mu_min[:, None] > 0, minutes / np.maximum(mu_min[:, None], 1e-9), 0.0)

    games, idx = np.unique(df["game_id"].astype(str).to_numpy(), return_inverse=True)
    scale = scale * _lognormal(rng, params.game_sd, (len(games), n_sims))[idx]
    scale = scale * _lognormal(rng, params.player_sd, (n, n_sims))
    # Keep each player's simulated mean equal to the model's expectation.
    scale = scale / np.maximum(scale.mean(axis=1, keepdims=True), 1e-9)

    def counts(mu, r):
        lam = np.maximum(np.asarray(mu, float), 0)[:, None] * scale
        return rng.poisson(lam * rng.gamma(r, 1.0 / r, (n, n_sims)))

    out = {"minutes": minutes}
    for stat, mu in means.items():
        if stat == "pts" and params.points_model == "events" and "threes" in means:
            continue
        out[stat] = counts(mu, params.dispersion.get(stat, 20.0))
    if "pts" in means and "pts" not in out:
        # Same threes path as the threes market, so points and threes move together.
        rest = np.maximum(np.asarray(means["pts"], float) - 3 * np.asarray(means["threes"], float), 0)
        r = params.dispersion.get("pts", 20.0)
        out["pts"] = (3 * out["threes"] + 2 * counts(rest * (1 - params.ft_share) / 2, r)
                      + counts(rest * params.ft_share, r))
    if all(s in out for s in ("pts", "reb", "ast")):
        out["pra"] = out["pts"] + out["reb"] + out["ast"]
    return out
