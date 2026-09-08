# NBA Betting Model

Automated daily betting model covering spreads, totals, and player props.
Runs on GitHub Actions — no computer needed, works from any device.

Built as an NBA port of [wnba-model](https://github.com/bigtyme2k3/wnba-model),
trimmed back to the core pipeline: scrape → merge → train → predict → publish.

## Setup (one time)

1. Fork this repo
2. (Optional but recommended) Add an `ODDS_API_KEY` repo secret — get a free key at
   [the-odds-api.com](https://the-odds-api.com). Without it, spreads/totals/moneylines
   are skipped and the pipeline runs on props + scores only.
3. Go to **Actions** tab → run **Bootstrap — First Time Setup** workflow
4. Go to **Settings → Pages** → set Source to `main` branch, `/docs` folder
5. Your dashboard is live at `https://YOUR-USERNAME.github.io/nba-model`

## Daily workflow (automatic)

Every morning at 9 AM ET, GitHub Actions:
- Collects team/player box scores (hoopR release data + ESPN API)
- Scrapes lines from The Odds API
- Scrapes player props from PrizePicks
- Scrapes scores from ESPN
- Pulls injury reports and referee assignments
- Runs all three models
- Grades yesterday's picks and updates the dashboard

The workflow checks `active_slate_date.py --in-season` first and no-ops
outside roughly Oct 1 – Jun 30, so it doesn't grind through empty
off-season runs.

## Models

| Model  | Algorithm              | Notes |
|--------|-------------------------|-------|
| Spread | Ridge / Gradient Boosting / Random Forest (best picked by walk-forward CV) | predicts home margin |
| Totals | Ridge / Random Forest   | predicts combined points |
| Props  | Ridge / Gradient Boosting | pts, reb, ast, threes, pra per player |

Model performance numbers are **not** hardcoded anywhere — `daily_runner.py`
reports `null` for `cv_mae` / `hit_rate` / etc. until you've actually run
`--mode train` against real season data and can point to a real backtest.

## Manual trigger

Actions tab → **NBA Daily Pipeline** → Run workflow (optionally pass a `date`).

## Files

- `teams.py` — canonical team names, ESPN ids, conferences, timezones (single source of truth)
- `active_slate_date.py` — Eastern-time slate date resolution + season guard
- `scrape_odds.py` / `scrape_props.py` / `scrape_scores.py` / `scrape_injuries.py` / `scrape_refs.py` — data collection, each quota/network-safe (writes empty-but-valid output instead of failing the pipeline)
- `collect_stats.py` — historical + current-season box scores (hoopR release data, ESPN fallback)
- `merge_data.py` — joins box scores + odds into `data/processed/master_all.csv`
- `spread_model.py` / `totals_model.py` / `props_model.py` — training + prediction
- `kelly_sizing.py` / `betting_engine.py` — bet sizing, EV, decision scoring
- `daily_runner.py` — orchestrates a full day's predictions into `predictions/predictions_YYYY-MM-DD.json`
- `results_tracker.py` / `line_movement.py` — grading and closing-line-value tracking
- `build_dashboard.py` — bakes the day's data into `docs/index.html` (generates it from scratch on first run)

## Known limitations (by design, for a clean starting point)

- `CURRENT_TEAM_STATS` / `TEAM_ROLLING` / `PLAYER_PROPS` in `daily_runner.py` are
  **seed placeholder values**, not live stats — replace them with real
  season-to-date numbers pulled from `data/processed/` once you have a few
  weeks of games collected.
- `NBA_GAME_STD` in `kelly_sizing.py` is an approximate NBA scoring-margin
  standard deviation — recalibrate it against your own backtest.
- No trained models are committed — run **Bootstrap** first.
