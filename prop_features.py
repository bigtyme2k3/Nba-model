"""Shared prior-only baseline feature semantics for new prop model bundles."""
import math

VERSION = "NBA_PROP_FEATURES_V2"
BASE_FEATURES = ["minutes", "roll5_minutes", "usage", "ts_pct", "opp_drtg_pos",
                 "avg_pace", "team_pace", "rest_days", "is_home", "roll5_pts",
                 "roll5_reb", "roll5_ast", "roll5_threes", "roll5_pra", "season_game_num", "month"]


def num(value, default):
    try:
        value = float(value)
        return value if math.isfinite(value) else default
    except (ValueError, TypeError):
        return default


def build_features(prior, projected_minutes=None, projected_usage=None):
    """Historical/live baselines match; injury overrides are explicit live scenarios."""
    row = {name: num(prior.get(name), float("nan")) for name in BASE_FEATURES}
    m5 = num(prior.get("roll5_minutes"), num(prior.get("minutes"), 0))
    m10 = num(prior.get("roll10_minutes"), m5)
    minutes = min(42.0, max(0.0, 0.65 * m5 + 0.35 * m10))
    rest = num(prior.get("rest_days"), 2)
    row["minutes"] = (num(projected_minutes, minutes) if projected_minutes is not None else minutes) * (0.95 if rest == 0 else 1.0)
    row["roll5_minutes"] = m5
    row["usage"] = num(projected_usage, num(prior.get("usage"), .20))
    row["rest_days"] = rest
    row["def_adj"] = 114 - num(prior.get("opp_drtg_pos"), 114)
    row["usage_pace"] = row["usage"] * num(prior.get("avg_pace"), 99) / 99
    return row
