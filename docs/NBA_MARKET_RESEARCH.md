# NBA historical sportsbook-market and fade research

This module tests whether simple market baselines fail predictably, whether opposite-side bets earn a return at obtainable prices, and how progressions change risk. It is an isolated research system, not a replacement for NBA V1, an ML training pipeline, or a live bet recommendation engine. Negative results are first-class outputs.

## Current collection status and unavoidable dependencies

See [`analysis_status.json`](../data/research/nba_market/analysis_status.json) and the [research report](../data/research/nba_market/REPORT.md) for actual collected counts. Empty files with headers are intentional when no permitted data exists; they do not represent a completed historical backtest.

The inspected October 4, 2026 daily run (37219974714, job 111488160196) passed an **empty** `ODDS_API_KEY` to `scrape_odds.py`. The workflow already wired `secrets.ODDS_API_KEY`, and the scraper read that exact variable. This is an unavailable secret/configuration issue, not a variable-name mismatch in that workflow. The workflow now also supports `THE_ODDS_API_KEY` and `THEODDS_API_KEY` secret aliases. It never prints key values or includes keys in request receipts.

To enable ingestion, add an accessible **paid historical-access** key named `ODDS_API_KEY` under this repository's **Settings → Secrets and variables → Actions → Repository secrets**. A secret in another repository or an unattached Actions environment is not available to this job. Locally, export the same environment variable; do not place a key in code, CLI arguments, an issue, a report, or a committed `.env`.

**A key alone cannot provide historical outcomes.** The documented Odds API `/scores` endpoint exposes completed games from at most three days ago. Its historical odds/events endpoints do not provide a historical final-score backfill. Under this task's strict one-source policy, the module therefore accepts only genuine previously archived Odds API score responses with plausible original receipt times. It does not read the NBA V1 hoopR/ESPN score files. If no such archives exist, older seasons' wins, streaks, ROI, drawdowns, and empirical ruin cannot be determined with this source alone. A separate result source would require an explicit change to the user's source policy; the module does not silently make that change.

## Source and calendar scope

- **Markets:** The Odds API `/v4/historical/sports/basketball_nba/odds`, using American odds, `h2h,spreads,totals`, US region. All returned bookmakers are retained; execution preference is FanDuel, DraftKings, Fanatics, then alphabetical remaining books.
- **Primary seasons:** 2022-23, 2023-24, 2024-25, 2025-26. The config uses **start-year season labels**, independently of the existing collector's filenames.
- **Primary population:** Each selected team's first 20 observed regular-season games; opponent game numbers are computed independently. Four buckets: 1–5, 6–10, 11–15, 16–20. Exact observed numbers are retained.
- **Calendar exclusions:** Before regular-season opening day, after regular-season ending day, and NBA Cup championship dates. Cup group, quarterfinal, and semifinal games remain regular-season games. Championship dates have no other NBA regular-season games and are excluded in full.
- **Reference population:** `--full-season-reference` collects through regular-season end and enables Games 21+ comparisons. A 65-day primary catalog alone is not evidence about the entire later season.

Calendar dates are metadata, not an additional betting-line dataset. They were checked against official NBA season announcements. To add earlier seasons, add a verified start/end/exclusion entry to `research/nba_market/config.json`; no ingestion redesign is needed. The API's earlier snapshot cadence and source coverage limitations still apply.

### Schedule completeness matters

Historical odds inventory contains events listed by bookmakers, not a guaranteed complete played-game schedule. Events can be unlisted, canceled, rescheduled, or have changed identifiers. The system numbers distinct event IDs chronologically per season and team, counts only calendar-eligible events, and flags duplicate matchup/date pairs and rescheduled tips. It labels all numbers **`observed_catalog_not_independently_certified`**, reports discovery coverage and counts of teams reaching 20, and never claims those counts independently prove a complete schedule.

Scores archived from the same source can expand the catalog, but missing/canceled games must still be audited. Exact first-20 findings should not be promoted while the inventory remains uncertified. NBA V1's unrelated historical score files are deliberately not used to conceal that limitation.

## Run and resume

From the repository root (Python 3.11+; research uses the standard library):

```bash
# Zero network requests: inspect known work and the credit estimate.
python -m research.nba_market plan

# Bounded ingestion plus normalize/analyze/validate/report, even if ingestion blocks.
python -m research.nba_market run --max-requests 500 --max-credits 15000 --quota-reserve 200

# Resume with the IDENTICAL command. Successful snapshots are cache hits.
python -m research.nba_market run --max-requests 500 --max-credits 15000 --quota-reserve 200

# Target one configured season or collect the full regular-season reference.
python -m research.nba_market run --seasons 2022-23 --full-season-reference

# Regenerate everything from stored data. No API calls and no credential needed.
python -m research.nba_market analyze
python -m research.nba_market validate

# Import an unmodified archived Odds API response envelope; no network calls.
python -m research.nba_market import-archive --file /path/to/archived_response.json

# Optional collection of recent final results only (NOT historical score backfill).
python -m research.nba_market capture-scores --max-requests 1 --max-credits 2

# Tests never request real API data.
python -m unittest discover -s tests -v
```

Subcommands support `--root`, `--config`, `--no-page`, and `--page-dir`. Use a different root for experiments; never write synthetic fixtures into the real research warehouse. Exit code 2 from `run` means an ingestion/outcome dependency or budget prevented a completed historical run. Exit code 1 means a validation failure. `analyze`/`validate` can succeed operationally while explicitly reporting unavailable or partial data.

GitHub Actions: **Actions → NBA Historical Market Research → Run workflow**. The default manual `plan` mode spends zero quota. `run` fetches within supplied bounds; `analyze` uses cached data; `capture-scores` archives only recent results. The initial authorized publication also updates `.github/nba-market-research-trigger.json` to start one bounded `run` in the repository environment, where Actions can access its secrets. Ordinary code/report commits do not trigger paid fetching. No recurring paid research fetch is installed, and NBA V1's daily workflow does not import this module.

## Quota and durable caching

Historical requests cost **10 credits per region per market**: 30 for all three US game markets. A free `/sports?all=true` probe checks response quota headers before paid requests. The primary discovery plan queries 10:00 and 16:00 Eastern each day for 65 days, excluding Cup finals, then adds each observed qualifying game's T−24h and T−60s snapshots. Queries are shared across events when timestamps match; the all-sport historical endpoint returns multiple events in each response.

The first plan is a lower-bound estimate because undiscovered event-specific snapshots cannot yet be counted. Four seasons' discovery alone currently plans 514 snapshots / 15,420 credits, before cache hits and before event-specific work. The default per-run request/credit bounds intentionally require resuming for large collections. Quota reserve is enforced against the latest response headers; actual billed cost replaces the estimate when present.

Each successful HTTP response is saved as a gzip-compressed, checksum-protected envelope **before normalization**, then normalized transactionally into SQLite. Cache keys include endpoint and canonical public query parameters, exclude the key, sort market/region lists, and floor historical query times to the documented 5/10-minute cadence. Current score receipts also include collection time. Snapshots in the same requested interval are reused. Repeated ingestion is deduplicated by `(event_id, snapshot_timestamp, bookmaker)`.

- A parser failure reuses the saved response after repair, with no second purchase.
- A crash after receipt write but before SQLite commit is recovered from that receipt.
- Missing/damaged paid receipts are explicit errors, not automatic purchases.
- A network timeout may have been billed; it is marked uncertain and is not automatically retried. Check account usage before using `--retry-uncertain`.
- Requests, attempts, costs, remaining/used quota headers, cache hits, and resume plans are recorded without credentials.

The Actions workflow restores and persists purchased responses plus database state on the **`market-research-data` branch of this same repository**. The database is compressed into checksum-verified chunks below GitHub's individual-file limit. The normal `main` branch contains code and readable research exports, not the binary/raw cache. Artifact uploads provide a 90-day recovery fallback. Workflow concurrency prevents two research runs from racing; data writes are fast-forward pushes, never forced.

For local fetches, use `python scripts/research_cache.py persist` to preserve successful responses remotely with your normal Git credentials. On a fresh checkout, `python scripts/research_cache.py restore` recovers them before the next fetch. Restore is intended for a fresh runner; preserve unpushed local ingestion state before replacing it with remote state. Existing NBA V1 tracked data is outside these paths.

## Warehouse and artifacts

`data/research/nba_market/warehouse.sqlite3` stores:

| Table | Key / purpose |
| --- | --- |
| requests | Request fingerprint, endpoint, public params, payload hash, receipt path, exact API timestamp, quota headers |
| request_log | Every actual attempt, status, estimated/actual credits, remaining/used quota |
| events | Stable Odds API event ID; teams, season, Eastern date, conservative tip, schedule-change flag |
| observations | Event + timestamp + bookmaker; raw normalized two-sided ML/spread/total values and market update timestamps |
| outcomes | Event; final score, time it was available, original receipt time/hash, strict source provenance |
| issues | Deduplicated malformed lines, rejected market updates, conflicting receipts |
| plans | Configured scope and resumable task inventory |

JSON and CSV exports include:

Large tables are automatically partitioned below a 32 MiB working bound rather than exceeding GitHub's single-file limit. In that case the original JSON/CSV filename becomes an explicit index of `*-part-NNNN` files, with row counts and content hashes. `research.nba_market.report.read_exported_table(root, name)` reads either format and verifies all parts. `manifest.json` lists every part; the research page exposes CSV parts for download. A table is never silently truncated to fit publication limits.

| Artifact | Contents |
| --- | --- |
| historical_market_warehouse | Raw bookmaker values, selected snapshot labels, exact/observed team numbers, independent buckets, implied/no-vig probabilities, consensus references, disagreement, early-to-close movement, final grades when available |
| market_consensus | Every selected market/view, method inputs, dispersion and book counts |
| frozen_baseline_bets | Frozen signal ID, market direction, signal time, actual execution book/line/price, reference grade, execution grade and P&L |
| early_season_summary / early_vs_later_summary | Separate seasons plus pooled rows; team buckets and later-season comparison |
| underdog_price_bucket_summary / favorite_summary | Prices, wins/losses, expected probability, ROI, paired favorite ROI |
| ats_summary / totals_summary / totals_team_buckets | Executable ATS/total returns; independent home/away game-number views |
| streak_distribution / streak_runs | Exact 1–5, 6+ runs, maximum run and censoring |
| next_game_mean_reversion | Conditional next qualifying event; continuation/reversion P&L and skipped timing cases |
| line_movement / open_vs_close_games / open_vs_close_comparison | Direction, reference changes, flips and correction categories |
| sportsbook_disagreement_games / sportsbook_disagreement_analysis | Per-event disagreement and season/group failure comparisons |
| flat_bet_simulation / progression_simulation | Flat benchmark and every specified progression, with risk accounting |
| simulation_bet_ledger / simulation_sequences | Reproducible stake, settlement, cash, equity and sequence audit trail |
| maximum_risk_report | Observed bankroll requirement and specified-bankroll bootstrap operational-ruin estimates |
| season_validation | Replication across individual seasons; absent/negative seasons preserved |
| quality_control / manifest / analysis_status | Counts, coverage, blockers, validation, config/source/output hashes |
| REPORT.md | Human-readable findings or explicit unanswerable questions |

`docs/market-research/index.html` is a separately labeled dashboard page with CSV downloads and the report. The existing dashboard gains only a link. It does not receive research bets, overwrite prediction DATA, or change recommendation rules.

## Early/open and closing methodology

**EARLY / Model A's OPEN** is the earliest valid retrieved snapshot with at least two valid books for that market. It is explicitly **not a verified true opening line**. If the earliest sample is under six hours before tip, it is additionally labeled late. T−24h requests improve the coverage but do not establish a true opener.

**CLOSE / Model B** is the latest observed clean snapshot strictly before the event's conservative scheduled tip, at least 60 seconds before it and within 15 minutes of it. Each included closing bookmaker's market update must be no more than 15 minutes older than that snapshot. A missing clean close stays missing; an early or stale quote is never substituted and called closing. Updates after the API snapshot or at/after tip are rejected. The exact returned API snapshot timestamp, not only the requested timestamp, is stored.

When schedule times change, the earliest observed tip is used conservatively to prevent delayed schedule metadata from admitting potentially post-tip odds. This can discard valid postponed-game markets; it is a coverage loss, not a reason to relax the no-leakage boundary.

Selection is **market-specific**: h2h, spreads, and totals may have different valid timestamps/book coverage. Favorite flips and comparisons require distinct early and closing snapshots. Derived early-to-close movement attached to warehouse exports is an after-close research measurement; it is not a field used to construct earlier signals.

## Consensus and execution

For each bookmaker's complete ML pair, convert American odds to implied probabilities, divide each by their sum, and then average the home no-vig probability across books with equal weights. Away probability is its complement. The favorite is the side above 50%; exact pick'ems generate no automatic favorite bet. Consensus American reference prices are converted from averaged implied probabilities; signed American odds are never averaged directly.

Spread and total references are medians of complete two-sided lines. Disagreement includes ML probability range/std, differing favorites, spread range, and total range. Thresholds are configurable: 8 probability points, 1.5 spread points, 3 total points by default.

**Financial replay always uses an actually quoted book price and line.** Select the book nearest the spread/total consensus, then configured book priority and name to break ties. ML uses priority/name, without selecting the best eventual result. This avoids P&L at synthetic median quarter-lines or synthesized consensus prices. Each bookmaker also has its own separate baseline/flat summary. Reference correctness and execution correctness can differ and are stored separately.

## Simple market baselines and opposite results

| Baseline | Follow | Fade |
| --- | --- | --- |
| A_OPEN_FAVORITE | EARLY consensus ML favorite | EARLY consensus underdog |
| B_CLOSE_FAVORITE | CLOSE consensus ML favorite | CLOSE consensus underdog |
| C_SPREAD_FAVORITE | Side favored by the consensus spread | Opposite ATS side |
| D_OVER_REFERENCE | Over the consensus total reference | Under that reference |

A total is not inherently a directional sportsbook forecast. Model D's Over/Under convention measures total-line bias and side returns, not a claim that bookmakers predict every Over. Similarly, moneyline correctness cannot improve from OPEN to CLOSE without a favorite flip; repricing the same favorite changes expected probability/payout, not the outright winner prediction.

Underdog bands are lower-inclusive, upper-exclusive: +100–149, +150–199, +200–249, +250–299, +300–399, +400+. A consensus underdog can be negative-priced at a disagreeing execution book; those cases are separately labeled rather than dropped or forced into a positive-price band. Price-band favorite ROI uses the **same paired events**, while favorite-team early-bucket summaries use the favorite team's own game number.

## Streaks, mean reversion, and temporal validity

League-wide descriptive sequences sort unique qualifying events by scheduled tip, with event ID as a deterministic tie-break. Same-team sequences include only that team's defined role/condition: e.g. its underdog games for underdog streaks. A nonqualifying game is skipped, not treated as a loss or reset. Season boundaries reset all streaks. Unknown qualifying results censor/reset a descriptive run; pushes are neutral and break a directional hit streak. Terminal runs are right-censored and distinguished from completed runs.

The engine measures underdog wins, favorite losses/wins, underdog ATS covers, favorite ATS failures, Overs, Unders, OPEN/CLOSE failures, and fade losses separately. The next-game table conditions on prior streaks of exactly 1, 2, 3, 4, or 5+, uses the next **qualifying** event, and tests continuation and reversion, including actual-price P&L.

Prior results must have `result_available_at < signal_timestamp` before they can qualify the next bet. Simultaneous/overlapping league games are descriptive observations but cannot be used as already-known losses to raise a stake. The output counts skipped timing cases. No result changes the pre-tip baseline or frozen signal ID.

Progression triggers are always-qualifying, or after at least 1/2/3 known prior market-reference failures. Qualification is frozen before the offered snapshot; no next-game result decides whether the signal qualifies. Missing qualifying outcomes **stop that simulated lane for the season**, preserving the unresolved path rather than bridging it with favorable later data.

`qualifier=selection_team` manages the side being followed/faded through its qualifying role. For the priority mean-reversion test, `qualifier=underdog_team_opposing_favorites` additionally follows the **same qualifying underdog's identity** while betting its changing opposing favorites. The lane does not reset merely because the opponent changed. Its first-20 status and known upset/ATS streak come from that underdog; the execution book/price and actual favorite being bet remain explicit. This is separate from tracking a favorite team's own consecutive failures.

## Progression and cash definitions

The benchmark is flat 1 unit. Fixed progressions are 1→2, 1→2→4, 1→2→4→8, and 1→1.5→2.5→4. A win resets a fixed progression even if its actual payout does not recover prior losses; such winning-but-negative sequences and target-profit shortfalls are reported. A push returns the stake without advancing the step or adding a loss. Reaching the final step with a loss is a progression failure; an unfinished sequence is censored, not declared successful.

Price-aware variants have finite 4/8-step caps and a configurable maximum single stake:

```
net profit per unit = American price / 100           for positive odds
net profit per unit = 100 / abs(American price)      for negative odds
required next stake = (previous sequence losses + target profit) / net profit per unit
```

For example, after losing 1 unit, recovery plus 1 unit at −250 requires 5 units. With repeated −250 bets, the first three price-aware stakes for a 1-unit target are 2.5, 8.75, 30.625: 41.875 units of cumulative exposure before the third settles. This is a math test, **not a historical NBA observation**.

Signals lock cash at placement; settlements release stake plus profit. Same-team lanes are separately managed but share portfolio cash. League progressions wait for settlement and explicitly skip overlapping qualifying offers; flat betting can take all offers. Totals same-team replay places separate correlated lane bets for each qualifying opponent; it can stake twice on one event and is explicitly not two independent outcomes.

Reports distinguish maximum single stake, losses-plus-current-stake sequence exposure, simultaneous outstanding stakes, realized-equity drawdown, and minimum initial liquid bankroll needed to fund the observed path. A configured bankroll that cannot fund a stake stops the replay. No future winnings fund an earlier stake. Stake rounding/book limits/taxes/promotions/slippage are not modeled.

The risk table uses deterministic contiguous offer-block bootstrap within each season, preserves price/outcome pairs and overlap within each block, replays the progression, and measures inability to fund the next stake at 25/50/100/250/500/1000 units. Blocks are serialized. These estimates describe a stated resampling experiment, not a calibrated real-world ruin forecast; triggers' risk estimates remain the deterministic observed exposure rather than fabricated bootstrap claims. Config controls repetitions, block size, seed, and bankrolls.

## Validation and interpretation

Tests cover normalization, season/timezone exclusions, independent game numbering, buckets, odds/payout math, no-vig/consensus, favorite/pick'em identification, grades/pushes, snapshot freshness and no post-tip leakage, qualifying streaks and prior-result availability, fixed/price-aware stakes, cash overlap/drawdown/ruin, receipt recovery/budget/quota/dedup, chunked storage integrity, signal immutability, end-to-end synthetic multi-season replay, output idempotence, and unchanged NBA V1 model/prediction/tracking artifacts.

Quality outputs check actual counts, unique event IDs and matchup/date pairs, quote orientation, malformed values, source provenance, season exclusions, bucket consistency, no-vig sums, result/P&L math and coverage. Raw payload hashes permit manual comparison with exact response envelopes. **Real response spot-checks cannot be marked complete when no real data was collected.**

Every major summary has separate seasons plus pooled results. Samples under 100 graded games are labeled small; proposed descriptive candidates also require positive ROI in every configured season and at least 30 observations per season, plus a positive pooled exploratory ROI lower bound. This is still an exploratory screen with correlated observations and multiple testing; it never automatically justifies live integration. Absent seasons and negative findings remain visible.

No spread, totals, or props model is retrained by this task. No immutable forward-ledger prediction is altered. Frozen baseline artifacts can support a later shadow comparison of NBA V1 / market / fade / early / close / historical condition using event IDs and signal timestamps, but no production adapter or automatic betting hook is installed.

## Existing daily-pipeline repair

The inspected daily job failed in `load_team_context`: current team boxes contained both `points` and `pts`, and renaming both to `team_points` created duplicate columns. `coalesce_team_aliases` now merges aliases in a deterministic order while preserving established canonical fields and historical `team_score` priority. The existing shifted rolling feature logic is unchanged. Regression tests cover current and historical schemas and prior-only team context; full feature-store smoke checks write to a temporary directory, never to the production artifacts.

## Primary documentation used

- [The Odds API v4 documentation](https://the-odds-api.com/liveapi/guides/v4/): historical odds, quota headers/costs, snapshot cadence, and recent-scores limits.
- [2022-23 NBA schedule announcement](https://www.nba.com/news/nba-unveils-2022-23-regular-season-schedule).
- [2023-24 NBA schedule announcement](https://www.nba.com/news/2023-24-nba-regular-season-schedule).
- [2024-25 NBA schedule announcement](https://www.nba.com/news/2024-25-nba-regular-season-schedule).
- [2025-26 NBA schedule announcement](https://pr.nba.com/2025-26-nba-regular-season-schedule/).
- [2023 tournament rules](https://www.nba.com/news/2023-nba-in-season-tournament-schedule), [2025 Cup championship date](https://www.nba.com/news/tickets-on-sale-2025-emirates-nba-cup-semis-championship).
