# NBA Model V2 (research, not deployed)

V1 predictions, qualifying bets and the forward ledger do not use anything in
this package. A V2 challenger is promoted only after it beats V1 on untouched
future games without worse calibration, drawdowns or data coverage.

## Phase 1: warehouse (`v2/warehouse.py`)
DuckDB file `data/warehouse/nba_v2.duckdb` (rebuildable, not committed):
`player_box` (all hoopR seasons), `games`, `pbp` (ESPN play-by-play, fetched and
cached in `data/raw/pbp/`), `stints`. Identifier checks are written to
`data/tracking/v2_warehouse_checks.json`.

## Phase 2 pilot (`v2/lineups.py`, `v2/features.py`, `v2/pilot.py`)
- Lineups: five-player stints from substitutions; games with an unknown or
  inconsistent substitution, or minutes more than 6 off the box score, are
  excluded, not repaired.
- Features use only the player's earlier games of the same season. Missing
  coverage stays NaN with a flag.
- Teammate absence uses box-score absence as a stand-in for a confirmed OUT; a
  live V2 needs the injury report's timestamped status.

Run: `python -m v2.pilot --season 2025 --cutoff 2026-02-01`
Result: `data/tracking/v2_pilot_<season>.json`

## Safeguards
Pregame-only features, verified ESPN game/player ids, coverage gates, and V1
untouched. NBA tracking data (stats.nba.com) is not used: it was unreachable from
this environment, and its season-level endpoints must never be used to rebuild
earlier predictions.
