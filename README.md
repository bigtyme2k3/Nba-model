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
- Pulls FanDuel / DraftKings / Fanatics player props and prices from The Odds API
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
- `scrape_odds.py` — pregame FanDuel / DraftKings / Fanatics spreads, totals and moneylines
- `scrape_sportsbook_props.py` — priced player props for points, rebounds, assists, threes and PRA
- `scrape_scores.py` / `scrape_injuries.py` / `scrape_refs.py` — scores, league-wide injury status and referee context
- `opportunity_redistribution.py` — conservative confirmed-OUT minutes/usage redistribution with rotation safeguards
- `collect_stats.py` — historical + current-season box scores (hoopR release data, ESPN fallback)
- `build_player_features.py` — prior-only player feature store + minutes/rotation profiles
- `build_team_features.py` — rolling team pace/efficiency profiles from completed games
- `merge_data.py` — joins game box scores + odds into `data/processed/master_all.csv`
- `spread_model.py` / `totals_model.py` / `props_model.py` — training + prediction
- `kelly_sizing.py` / `betting_engine.py` — bet sizing, EV, decision scoring
- `daily_runner.py` — orchestrates a full day's predictions into `predictions/predictions_YYYY-MM-DD.json`
- `forward_ledger.py` — immutable first-observed NBA forecasts, including BET/PASS
- `execution_ledger.py` — separate append-only near-tip execution snapshots
- `forward_reconciler.py` — resolves actual outcomes and publishes price-aware ROI + probability calibration
- `nba_v1_readiness.py` — governance gate separating healthy pipeline, forward-testing readiness and production readiness
- `results_tracker.py` / `line_movement.py` — grading and closing-line-value tracking
- `build_dashboard.py` — bakes the day's data into `docs/index.html` (generates it from scratch on first run)

## NBA V1 current state

NBA V1 is **research / forward-testing only**. Prediction artifacts are stamped
`research_only=true` and default Kelly sizing is capped at **1% of bankroll**.
The larger 5% production cap requires an explicit future production-mode switch.

The canonical data contract now separates Sportsdataverse historical releases
(ending-year keys, e.g. `hoopr_*_2026` = 2025-26) from live ESPN current-season
files (`espn_current_*_2026` = 2026-27). Internally every row uses a season-start
year derived from its game date. Preseason and All-Star exhibition rows are
excluded before feature/model training.

Current validated baseline performance on the corrected 6,627-game master:

| Model | Selected algorithm | Walk-forward MAE | Notes |
|---|---|---:|---|
| Spread | Ridge | 11.35 | 63.4% directional accuracy; baseline challenger |
| Total | Ridge | 15.27 | Cardinality-safe 6,627-game training set |
| Points prop | HistGBR | 4.68 | trained through 2026-10-26 |
| Rebounds prop | HistGBR | 1.97 | trained through 2026-10-26 |
| Assists prop | HistGBR | 1.37 | trained through 2026-10-26 |
| Threes prop | HistGBR | 0.91 | trained through 2026-10-26 |
| PRA prop | HistGBR | 6.19 | trained through 2026-10-26 |

Prop prices are compared per sportsbook using recorded American odds and
no-vig probabilities. Because model probabilities are not yet forward-calibrated,
the research engine shrinks probability edges 55% toward the no-vig market and
caps probability tails before EV/Kelly sizing.

### Forward-testing policy

Morning/first-observed forecasts and near-tip execution snapshots are stored in
separate append-only ledgers. A third append-only ledger captures the exact
FanDuel / DraftKings / Fanatics prop market 8–25 minutes before scheduled tip.
CLV is matched back to the **same sportsbook, player, stat and side**; positive
line CLV always means the entry beat the closing number.

Production promotion is blocked until there is a meaningful certified forward
sample (currently configured as at least 200 total priced bets and 100 prop
bets), positive price-aware ROI, acceptable Brier calibration, at least 100
matched exact-book closes, a 50%+ beat-close rate, a nonnegative aggregate CLV
signal, plus review of book/market splits and drawdowns.

### Remaining V1 work

- Build leakage-safe historical team pace/efficiency features directly into the
  spread/total training master; current game models are valid baselines, not champions.
- Accumulate forward sportsbook evidence and use it to calibrate prop probabilities.
- Accumulate exact-book closing-line evidence and compare CLV stability by sportsbook/market.
- Add richer opportunity data (potential assists, rebound chances, touches/shot
  attempts vacated) only when a trustworthy historical/live source is available.
- Keep injury/rotation uncertainty fail-closed: questionable/doubtful players do
  not donate speculative minutes, and their own prop signals are blocked.
