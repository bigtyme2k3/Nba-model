"""Shared prior-game box-score rates and offensive/defensive matchup features.

Rates use pooled counts over ten completed games, never an average of game
percentages. Three-point percentages get 100 pseudo-attempts, anchored to older
team history (excluding the ten-game window), itself shrunk toward .36. These
fixed research settings are not tuned using the evaluation period.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

VERSION = "NBA_GAME_FEATURES_V3_FORM_PARITY"
VALIDATION = "expanding_whole_date_groups"
BOX_COUNTS = ("fgm", "fga", "three_pm", "three_pa", "ftm", "fta", "oreb", "dreb", "tov", "poss")
WINDOW = 10
THREE_PRIOR_ATTEMPTS = 100.0
THREE_PRIOR_RATE = .36

# Numerator counts/weights and denominator counts/weights.
RATE_DEFINITIONS = {
    "efg_pct": ({"fgm": 1., "three_pm": .5}, {"fga": 1.}),
    "tov_rate": ({"tov": 1.}, {"poss": 1.}),
    "oreb_pct": ({"oreb": 1.}, {"oreb": 1., "opp_dreb": 1.}),
    "ft_rate": ({"fta": 1.}, {"fga": 1.}),
    "ft_pct": ({"ftm": 1.}, {"fta": 1.}),
    "three_pa_rate": ({"three_pa": 1.}, {"fga": 1.}),
    "three_pct": ({"three_pm": 1.}, {"three_pa": 1.}),
}
RATE_DEFINITIONS.update({
    f"opp_{name}": ({("opp_" + k if not k.startswith("opp_") else k[4:]): v for k, v in num.items()},
                    {("opp_" + k if not k.startswith("opp_") else k[4:]): v for k, v in den.items()})
    for name, (num, den) in list(RATE_DEFINITIONS.items()) if name != "ft_pct"
})
BOX_METRICS = tuple(RATE_DEFINITIONS)
MATCHUP_GROUPS = {
    "shooting": ("home_efg_matchup", "away_efg_matchup"),
    "turnovers": ("home_tov_matchup", "away_tov_matchup"),
    "rebounds": ("home_oreb_matchup", "away_oreb_matchup"),
    "free_throws": ("home_ft_rate_matchup", "away_ft_rate_matchup", "home_ft_pct", "away_ft_pct"),
    "threes": ("home_three_pa_matchup", "away_three_pa_matchup", "home_three_pct_matchup", "away_three_pct_matchup"),
}
MATCHUP_FEATURES = tuple(f for group in MATCHUP_GROUPS.values() for f in group)


def attach_opponent_counts(frame):
    """Require exactly two distinct team rows per game; incomplete pairs stay NaN."""
    out = frame.copy()
    grouped = out.groupby("game_id", sort=False)
    paired = grouped["team"].transform("size").eq(2) & grouped["team"].transform("nunique").eq(2)
    for col in BOX_COUNTS:
        values = pd.to_numeric(out.get(col, pd.Series(np.nan, index=out.index)), errors="coerce")
        out[col] = values
        totals = values.groupby(out["game_id"]).transform(lambda s: s.sum(min_count=2))
        out[f"opp_{col}"] = (totals - values).where(paired)
    return out


def prior_box_rates(frame):
    """Return rates aligned to input rows; every rolling count is shifted once."""
    work = frame.sort_values(["team", "game_date", "game_id"]).copy()
    out = pd.DataFrame(index=work.index)
    def weighted_sum(weights):
        result = pd.Series(0., index=work.index)
        for col, weight in weights.items():
            result = result + pd.to_numeric(work.get(col, pd.Series(np.nan, index=work.index)), errors="coerce") * weight
        return result
    def rolling_sum(values):
        return values.groupby(work["team"]).transform(lambda s: s.shift(1).rolling(WINDOW, min_periods=1).sum())
    for metric, (num_cols, den_cols) in RATE_DEFINITIONS.items():
        numerator, denominator = weighted_sum(num_cols), weighted_sum(den_cols)
        # A game with a missing component must not contribute only its denominator.
        valid = np.isfinite(numerator) & np.isfinite(denominator) & numerator.ge(0) & denominator.ge(0)
        numerator, denominator = numerator.where(valid), denominator.where(valid)
        num, den = rolling_sum(numerator), rolling_sum(denominator)
        if metric in {"three_pct", "opp_three_pct"}:
            older_num = numerator.groupby(work["team"]).transform(lambda s: s.shift(WINDOW + 1).expanding(min_periods=1).sum())
            older_den = denominator.groupby(work["team"]).transform(lambda s: s.shift(WINDOW + 1).expanding(min_periods=1).sum())
            anchor = (older_num.fillna(0) + THREE_PRIOR_ATTEMPTS * THREE_PRIOR_RATE) / (older_den.fillna(0) + THREE_PRIOR_ATTEMPTS)
            out[metric] = ((num + THREE_PRIOR_ATTEMPTS * anchor) / (den + THREE_PRIOR_ATTEMPTS)).where(den.gt(0))
        else:
            out[metric] = (num / den).where(den.gt(0))
    return out.reindex(frame.index)


def latest_box_rates(frame):
    """Use the historical calculation at a synthetic next game for each team."""
    if frame.empty:
        return pd.DataFrame(columns=["team", *BOX_METRICS])
    last = frame.sort_values(["game_date", "game_id"]).groupby("team", sort=False).tail(1).copy()
    last["game_date"] = pd.to_datetime(frame["game_date"]).max() + pd.Timedelta(days=1)
    last["game_id"] = "__next__"
    combined = pd.concat([frame, last], ignore_index=True)
    rates = prior_box_rates(combined).iloc[-len(last):].reset_index(drop=True)
    return pd.concat([last[["team"]].reset_index(drop=True), rates], axis=1)


def add_matchup_features(frame):
    """Identical home/away interactions for historical training and live scoring."""
    out = frame.copy()
    for side, other in (("home", "away"), ("away", "home")):
        for label, rate in (("efg", "efg_pct"), ("tov", "tov_rate"), ("oreb", "oreb_pct"),
                            ("ft_rate", "ft_rate"), ("three_pa", "three_pa_rate"), ("three_pct", "three_pct")):
            own, allowed = f"{side}_{rate}", f"{other}_opp_{rate}"
            if own in out and allowed in out:
                out[f"{side}_{label}_matchup"] = (out[own] + out[allowed]) / 2
    return out


def live_matchup_features(home, away, required):
    required = [f for f in required if f in MATCHUP_FEATURES]
    if not required:
        return {}
    raw = {f"{side}_{key}": values.get(key, np.nan)
           for side, values in (("home", home), ("away", away)) for key in BOX_METRICS}
    row = add_matchup_features(pd.DataFrame([raw])).iloc[0]
    if any(f not in row or not np.isfinite(row[f]) for f in required):
        raise ValueError("Required prior box-score matchup context is missing")
    return {f: float(row[f]) for f in required}


def training_features(frame, baseline, include_box_matchups=False):
    features = [f for f in baseline if f in frame]
    if include_box_matchups:
        missing = [f for f in MATCHUP_FEATURES if f not in frame]
        if missing:
            raise ValueError(f"Rebuild team context and master before training box matchups: {missing}")
        coverage = frame[list(MATCHUP_FEATURES)].notna().all(axis=1).mean()
        if coverage < .95:
            raise ValueError(f"Box matchup context coverage below 95%: {coverage:.2%}")
        features += list(MATCHUP_FEATURES)
    return features


def whole_date_splits(dates, n_splits=4):
    """Keep all games on a date on the same side of each chronological split."""
    from sklearn.model_selection import TimeSeriesSplit
    days = pd.to_datetime(dates).dt.normalize()
    if days.isna().any():
        raise ValueError("Validation dates must be populated")
    unique = np.sort(days.unique())
    for train, test in TimeSeriesSplit(n_splits=n_splits).split(unique):
        yield np.flatnonzero(days.isin(unique[train])), np.flatnonzero(days.isin(unique[test]))


def save_research_candidate(bundle, path):
    """An explicit candidate destination can never overwrite an active bundle."""
    import pickle
    from pathlib import Path
    destination = Path(path)
    active = {Path("models", f"{kind}_model.pkl").resolve() for kind in ("spread", "totals")}
    if destination.resolve() in active:
        raise ValueError("Candidate output must be separate from active model paths")
    destination.parent.mkdir(parents=True, exist_ok=True)
    bundle = dict(bundle, research_only=True)
    destination.write_bytes(pickle.dumps(bundle))
