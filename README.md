# NBA Betting Model

Automated daily betting model covering spreads, totals, and player props.
Runs on GitHub Actions — no computer needed, works from any device.

## Historical market and fade research (isolated from NBA V1)

The NBA historical research module preserves raw Odds API bookmaker observations,
builds early/closing market baselines, grades follow/fade offers at real quoted prices,
tests qualifying streaks and next-game mean reversion, and replays flat/capped/price-aware
progressions with cash, drawdown, exposure and stated-bankroll risk accounting.
It does not retrain NBA V1 or change its immutable forward predictions.

- [Methodology, source limits, caching and run instructions](docs/NBA_MARKET_RESEARCH.md)
- [Current research report and blockers](data/research/nba_market/REPORT.md)
- [Research dashboard artifact](docs/market-research/index.html)

Start with `python -m research.nba_market plan` (zero API calls), then
`python -m research.nba_market run`. Repeating `run` resumes from cached receipts.
The manual **NBA Historical Market Research** workflow persists paid receipts in the
`market-research-data` branch and publishes separate JSON/CSV outputs.

Historical access requires a paid `ODDS_API_KEY` Actions secret. The documented Odds
API has no historical final-score backfill, so strict one-source historical grading
also requires genuine archived Odds API score responses. Empty/blocked reports are
explicitly labeled; they are not evidence that a fade or Martingale works.

Built as an NBA port of [wnba-model](https://github.com/bigtyme2k3/wnba-model),
using the WNBA V5 lessons as the foundation: collect → leakage-safe features →
minutes/rotation context → model vs market → immutable forward evidence → grade/publish.

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
- Rebuilds real player/rotation and team rolling profiles from completed games
- Runs all three models against collected market lines
- Preserves the first prediction in an immutable forward ledger
- Grades yesterday's picks/forward evidence and updates the dashboard

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
- `build_player_features.py` — prior-only player feature store + minutes/rotation profiles
- `build_team_features.py` — rolling team pace/efficiency profiles from completed games
- `merge_data.py` — joins game box scores + odds into `data/processed/master_all.csv`
- `spread_model.py` / `totals_model.py` / `props_model.py` — training + prediction
- `kelly_sizing.py` / `betting_engine.py` — bet sizing, EV, decision scoring
- `daily_runner.py` — orchestrates a full day's predictions into `predictions/predictions_YYYY-MM-DD.json`
- `forward_ledger.py` — append-only first-observed NBA predictions, including BET/PASS
- `forward_reconciler.py` — resolves actual outcomes without changing original prediction fields
- `results_tracker.py` / `line_movement.py` — grading and closing-line-value tracking
- `build_dashboard.py` — bakes the day's data into `docs/index.html` (generates it from scratch on first run)

## Current model state / next upgrades

- Real player and team profiles now overlay the old seed dictionaries whenever the
  processed feature stores are present. The seed values remain only as fail-safe
  fallbacks for missing data; production predictions should report real-profile coverage.
- Player prop market lines currently come from PrizePicks. The next production data
  upgrade is sportsbook-specific FanDuel / DraftKings / Fanatics lines and prices so
  EV, book selection, and CLV are measured per book rather than against one prop feed.
- Injury statuses are collected and OUT players are removed/adjusted, but full
  opportunity redistribution (minutes + usage + assists/rebounds/touches vacated by an
  inactive player) is still an NBA V1 priority.
- Team defense is currently overall rolling defensive efficiency; a validated
  position/archetype matchup layer comes after the core forward evidence is stable.
- `NBA_GAME_STD` in `kelly_sizing.py` remains an approximate NBA scoring-margin
  standard deviation and must be recalibrated on genuine forward evidence.
- Model promotion should be based on the immutable forward ledger, not retrospective
  fit alone.
