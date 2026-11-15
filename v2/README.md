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

Run (one season, date split): `python -m v2.pilot --season 2025 --cutoff 2026-02-01`
Run (several seasons, each tested on a model trained on earlier seasons):
`python -m v2.pilot --seasons 2021 2022 2023 2024 2025 --test-seasons 2022 2023 2024 2025`
Results: `data/tracking/v2_pilot_<season>.json`, `data/tracking/v2_pilot_<first>_<last>.json`

## Safeguards
Pregame-only features, verified ESPN game/player ids, coverage gates, and V1
untouched. NBA tracking data (stats.nba.com) is not used: it was unreachable from
this environment, and its season-level endpoints must never be used to rebuild
earlier predictions.

## Shadow challenger (`v2/challenger.py`, `v2/live.py`, `v2/compare.py`)
- `python -m v2.challenger` trains V1 prop features + opportunity + absence on every
  season with play-by-play (manual workflow **NBA V2 Challenger Train**).
- `python -m v2.live --date D` (daily pipeline and pre-tip refresh) writes each player's
  features for his next game. Absence is an **expected** count from the morning injury
  report, weighted by how often players with each status sat (OUT 100%, QUESTIONABLE
  44%, not listed 2.2% in 2026-27), because the report alone catches only 62% of
  absences and training counts every actual absence.
- `daily_runner` records `v2_pred` next to V1's prediction (and the ledger keeps it as
  `v2_projection`). It is skipped when the features are for another date or the injury
  feed is unverified. V1 makes every signal and bet.
- `python -m v2.compare` scores both on the same graded props.
