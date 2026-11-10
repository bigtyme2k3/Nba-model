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
- `build_dashboard.py` / `dashboard_data.py` — prepare dated display records and rebuild `docs/index.html` from the maintained `dashboard/` HTML, CSS and JavaScript on every publication

## Dashboard

The phone-friendly dashboard has Today, Games, Props and Results views. All tip and
refresh times use Eastern time. Props can be searched by player/team and filtered by
game, stat, sportsbook, model lean and qualification status. Expand a player for
projected minutes, prior-game averages, availability and exact book/line/side prices.
Projection gaps are research observations unless the final card explicitly qualified
the play. Prices shown beside a lean always belong to that same side and line.

Results separates the historical priced forward-test record from current-model
validation progress. It includes a flat-one-unit profit chart and date/book/outcome
filters, and excludes consensus-priced game entries from priced performance.

Rebuild using `TARGET=YYYY-MM-DD python build_dashboard.py`. Check display data with
`python -m unittest tests.test_dashboard_data` and filters with
`node tests/test_dashboard_client.cjs`; neither command makes API requests.

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
caps probability tails without raising low probabilities before EV/Kelly sizing. Final research cards also enforce
**one bankroll-bearing prop per player**; additional same-player signals remain
visible for research but cannot stack correlated minutes/usage exposure.

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

### Current development queue

- Leakage-safe historical pace/ORTG/DRTG/net-rating/TS context is now implemented
  with 99.76% complete-game coverage. A paired time-series ablation improved both
  spread and totals in all four folds after train/live feature semantics were aligned.
  The corrected challengers are approved for a champion-gated rebuild after the
  current live evidence window; they are not yet published champions.
- Accumulate forward sportsbook evidence and use it to calibrate prop probabilities.
- Accumulate exact-book closing-line evidence and compare CLV stability by sportsbook/market.
- A lightweight overnight reconciler now grades completed slates without rebuilding
  the entire current season or consuming Odds API credits.
- Add richer opportunity data (potential assists, rebound chances, touches/shot
  attempts vacated) only when a trustworthy historical/live source is available.
- Keep injury/rotation uncertainty fail-closed: questionable/doubtful players do
  not donate speculative minutes, and their own prop signals are blocked.

### Exact-price evidence and recovery

Spread/total consensus columns are for display and comparison. An actionable
card requires a current paired FanDuel, DraftKings, or Fanatics quote, preserving
the selected side's handicap and price. Older consensus-priced game results stay
in the ledger but are excluded from priced ROI and production promotion.

The dashboard requires a prediction payload whose slate date matches the requested
Eastern date. Missing or mismatched payloads display an awaiting-predictions state;
older cards and undated fallback CSVs cannot be relabeled as current.

Closing capture checks all scheduled games before installing dependencies, captures
quotes without waiting for the publication lock, and publishes captured evidence in
a separate job. Coverage reports list missing games and unmatched exact-book entries.
GitHub scheduling can still be delayed; capture artifacts are retained for seven days.

`reconcile_pending.py --date YYYY-MM-DD` grades the requested slate and rotates
through three older pending dates. Each collection/grading step runs independently;
partial box-score fetches preserve previously collected games. Missing statistics
remain pending, explicit DNPs are voided, and near-tip execution reports accumulate
across dates. Overnight reconciliation runs twice each morning and rebuilds today's
dashboard after publishing reconciliation status.

Every model training entry point calls the same publication gate before replacing
a champion. New prop bundles use whole-date validation folds, training-fold-only
imputation, shared baseline minutes features, and out-of-fold residual dispersion.
Historical injury snapshots are unavailable, so live injury overrides remain
explicit scenarios; their conditional accuracy still requires forward evidence.

Forecasts record model/calibration artifact hashes. Calibration fits earlier dates
and must match or improve later-date Brier score and log loss against the market;
insufficient or failed holdout evidence uses market-only probabilities. Promotion
counts only the active algorithm and feature families, preserving comparable evidence
across routine refits. Exact artifact hashes remain recorded for audits. Historical unversioned results
remain available as diagnostics and cannot certify a newly trained model.
