"""
season_filters.py
-----------------
Shared NBA season-phase filtering.

The raw Sportsdataverse/hoopR schedule file is the phase manifest for historical
data. Only regular-season and postseason/play-in game IDs are eligible for model
feature stores. Preseason/offseason rows are excluded before training.

When a schedule file is unavailable or does not expose a recognizable phase
column, callers receive None and may continue with an explicit warning rather
than silently guessing from calendar dates.
"""
from __future__ import annotations

from pathlib import Path
import re
import pandas as pd

ALLOWED_NUMERIC = {2, 3}
ALLOWED_TEXT = {
    "2", "3",
    "regular", "regular-season", "regular season", "regularseason",
    "postseason", "post-season", "post season", "playoffs", "playoff",
    "play-in", "play in", "playin",
}
ID_CANDIDATES = ("game_id", "event_id", "id")
PHASE_CANDIDATES = (
    "season_type", "season_type_id", "season_type_name",
    "season_slug", "season_type_slug", "seasontype",
)


def season_from_path(path) -> int | None:
    m = re.search(r"_(20\d{2})\.csv$", str(path))
    return int(m.group(1)) if m else None


def clean_id(value) -> str:
    s = str(value or "").strip()
    return s[:-2] if s.endswith(".0") else s


def _phase_column(df: pd.DataFrame) -> str | None:
    lower = {str(c).lower(): c for c in df.columns}
    for name in PHASE_CANDIDATES:
        if name in lower:
            return lower[name]
    for col in df.columns:
        name = str(col).lower()
        if "season" in name and ("type" in name or "slug" in name):
            return col
    return None


def _id_column(df: pd.DataFrame) -> str | None:
    lower = {str(c).lower(): c for c in df.columns}
    for name in ID_CANDIDATES:
        if name in lower:
            return lower[name]
    return None


def phase_allowed(value) -> bool:
    if pd.isna(value):
        return False
    try:
        n = int(float(value))
        if n in ALLOWED_NUMERIC:
            return True
    except Exception:
        pass
    s = str(value).strip().lower().replace("_", "-")
    if s in ALLOWED_TEXT:
        return True
    return (
        "regular" in s
        or "post-season" in s
        or "postseason" in s
        or "play-in" in s
        or "playoff" in s
    )


def allowed_game_ids(raw_dir: str | Path, season: int) -> set[str] | None:
    path = Path(raw_dir) / f"hoopr_schedule_{season}.csv"
    if not path.exists():
        return None
    try:
        sched = pd.read_csv(path, low_memory=False)
    except Exception:
        return None
    if sched.empty:
        return set()

    id_col = _id_column(sched)
    phase_col = _phase_column(sched)
    if id_col is None or phase_col is None:
        return None

    mask = sched[phase_col].map(phase_allowed)
    return {clean_id(x) for x in sched.loc[mask, id_col] if clean_id(x)}


def filter_games(df: pd.DataFrame, raw_dir: str | Path, season: int, label: str = "rows"):
    """Return (filtered_df, metadata)."""
    meta = {
        "season": int(season),
        "phase_filter_available": False,
        "before": int(len(df)),
        "after": int(len(df)),
        "excluded": 0,
    }
    if df.empty or "game_id" not in df.columns:
        return df, meta

    allowed = allowed_game_ids(raw_dir, season)
    if allowed is None:
        return df, meta

    meta["phase_filter_available"] = True
    ids = df["game_id"].map(clean_id)
    out = df[ids.isin(allowed)].copy()
    meta["after"] = int(len(out))
    meta["excluded"] = int(len(df) - len(out))
    if meta["excluded"]:
        print(
            f"  Season {season}: excluded {meta['excluded']:,} preseason/offseason "
            f"{label} via hoopR schedule manifest"
        )
    return out, meta


def nba_season_start_year(value) -> int | None:
    """Canonical season key: start year (Oct 2025-Jun 2026 => 2025)."""
    dt = pd.to_datetime(value, errors="coerce")
    if pd.isna(dt):
        return None
    return int(dt.year if dt.month >= 7 else dt.year - 1)


def manifest_path_for_source(raw_dir: str | Path, source_path: str | Path) -> Path | None:
    """Find the schedule manifest that belongs to a raw box/game source file."""
    raw_dir = Path(raw_dir)
    name = Path(source_path).name
    m = re.search(r"_(20\d{2})\.csv$", name)
    if not m:
        return None
    key = m.group(1)
    if name.startswith("espn_current_"):
        return raw_dir / f"espn_current_schedule_{key}.csv"
    if name.startswith("hoopr_"):
        return raw_dir / f"hoopr_schedule_{key}.csv"
    return None


def allowed_game_ids_from_manifest(manifest_path: str | Path) -> set[str] | None:
    path = Path(manifest_path)
    if not path.exists():
        return None
    try:
        sched = pd.read_csv(path, low_memory=False)
    except Exception:
        return None
    if sched.empty:
        return set()
    id_col = _id_column(sched)
    phase_col = _phase_column(sched)
    if id_col is None or phase_col is None:
        return None
    mask = sched[phase_col].map(phase_allowed)
    return {clean_id(x) for x in sched.loc[mask, id_col] if clean_id(x)}


def filter_games_for_source(
    df: pd.DataFrame,
    raw_dir: str | Path,
    source_path: str | Path,
    label: str = "rows",
):
    """Phase-filter using the schedule manifest matching the exact source family."""
    path = manifest_path_for_source(raw_dir, source_path)
    meta = {
        "source": str(source_path),
        "manifest": str(path) if path else None,
        "phase_filter_available": False,
        "before": int(len(df)),
        "after": int(len(df)),
        "excluded": 0,
    }
    if df.empty or "game_id" not in df.columns or path is None:
        return df, meta
    allowed = allowed_game_ids_from_manifest(path)
    if allowed is None:
        return df, meta
    meta["phase_filter_available"] = True
    ids = df["game_id"].map(clean_id)
    out = df[ids.isin(allowed)].copy()
    meta["after"] = int(len(out))
    meta["excluded"] = int(len(df) - len(out))
    if meta["excluded"]:
        print(
            f"  {Path(source_path).name}: excluded {meta['excluded']:,} "
            f"preseason/offseason {label}"
        )
    return out, meta
