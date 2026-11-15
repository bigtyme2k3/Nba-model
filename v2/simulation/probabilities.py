"""Outcome probabilities, percentiles and proper scores from simulated or Normal distributions."""
from __future__ import annotations

import numpy as np
from scipy.stats import norm


def over_under_push(sims: np.ndarray, line) -> dict:
    """P(over), P(under), P(push) per row; push is possible only on whole-number lines."""
    line = np.asarray(line, float)[:, None]
    over = (sims > line).mean(axis=1)
    push = (sims == line).mean(axis=1)
    return {"over": over, "under": 1 - over - push, "push": push}


def percentiles(sims: np.ndarray, qs=(10, 25, 50, 75, 90)) -> dict:
    return {f"p{q}": np.percentile(sims, q, axis=1) for q in qs}


def sampling_se(p: np.ndarray, n_sims: int) -> np.ndarray:
    """Monte Carlo standard error of a simulated probability."""
    return np.sqrt(np.clip(p * (1 - p), 0, None) / n_sims)


def crps_samples(sims: np.ndarray, y: np.ndarray) -> np.ndarray:
    """CRPS = E|X - y| - 0.5 E|X - X'| per row, from sorted samples (exact for the sample)."""
    s = np.sort(sims, axis=1).astype(float)
    m = s.shape[1]
    term1 = np.abs(s - np.asarray(y, float)[:, None]).mean(axis=1)
    weights = (2 * np.arange(1, m + 1) - m - 1) / (m * m)
    term2 = (s * weights).sum(axis=1)
    return term1 - term2


def crps_normal(mu, sigma, y) -> np.ndarray:
    mu, sigma, y = (np.asarray(v, float) for v in (mu, sigma, y))
    z = (y - mu) / sigma
    return sigma * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))


def normal_over(mu, sigma, line) -> np.ndarray:
    return 1 - norm.cdf((np.asarray(line, float) - np.asarray(mu, float)) / sigma)


def brier(p, outcome) -> float:
    return float(np.mean((np.asarray(p, float) - np.asarray(outcome, float)) ** 2))


def log_loss(p, outcome) -> float:
    p = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4)
    o = np.asarray(outcome, float)
    return float(-np.mean(o * np.log(p) + (1 - o) * np.log(1 - p)))


def randomized_pit(sims, actual, seed: int = 0):
    """Probability integral transform for counts: uniform on [P(X<y), P(X<=y)].
    Calibrated forecasts give a flat histogram; a U shape means tails too thin."""
    x, y = np.asarray(sims), np.asarray(actual, float)[:, None]
    lo, hi = (x < y).mean(axis=1), (x <= y).mean(axis=1)
    return lo + np.random.default_rng(seed).random(len(lo)) * (hi - lo)
