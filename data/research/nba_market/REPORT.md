# NBA historical market and fade research

**Historical research only. No live recommended bets. NBA V1 predictions and forward ledger remain independent.**

Status: **blocked_no_historical_data**. 0 observed events; 0 bookmaker snapshots; 0 events with permitted final scores.

## Data and blockers

- Ingestion blocker: missing_api_key. Set repository Actions secret ODDS_API_KEY to a paid historical-access key, or export ODDS_API_KEY locally.
- Historical score backfill is unavailable through the documented Odds API: /scores exposes at most the past three days. Supply genuine archived Odds API scores to grade older seasons under the strict one-source rule.
- True openers are unverified; EARLY/OPEN is explicitly the earliest valid retrieved snapshot. CLOSE requires a clean snapshot within 15 minutes before scheduled tip, using a one-minute safety buffer.
- Observed team-game numbering is provisional until complete played-game inventory is independently certified. NBA Cup championship dates, preseason, and dates outside regular-season boundaries are excluded.
- No complete later-season comparison is available unless --full-season-reference ingestion has completed; early-season unusualness relative to the rest of the season is not established.
- Exploratory groups overlap. Multiple testing and correlated games can create apparent edges. All individual-season results and negative returns are retained.

Market source: The Odds API historical h2h/spreads/totals, US region. Scores must be archived Odds API responses; the module never reads existing ESPN/hoopR scores or betting lines.

API request cost: 30 credits per all-market US historical snapshot. Successful raw responses, receipts, and normalization state are cached. No retries of successful purchases.

Quota: {"credits_recorded": 0, "network_requests_recorded": 0, "remaining_last_observed": null, "successful_cached_responses": 0, "used_last_observed": null}

## Twelve research questions

1. **Are early-season underdogs unusual?** Not estimable: no permitted graded historical data.
2. **Which underdog price bands differ?** Not estimable: no permitted graded historical data.
3. **Does the effect change across Games 1–5, 6–10, 11–15, and 16–20?** Not estimable: no permitted graded historical data.
4. **How often do underdogs win consecutive qualifying games?** Not estimable: no permitted graded historical data.
5. **What is the longest qualifying upset streak?** Not estimable: no permitted graded historical data.
6. **How often does the market favorite fail consecutively?** Not estimable: no permitted graded historical data.
7. **Does a market fade have positive flat-bet expectation?** Not estimable: no permitted graded historical data.
8. **Does an apparent edge survive multiple seasons?** Not estimable: no permitted graded historical data.
9. **Do opening and closing failures differ?** Not estimable: no permitted graded historical data.
10. **Does sportsbook disagreement contain useful information?** Not estimable: no permitted graded historical data.
11. **Does a progression improve expected value or increase tail risk?** Not estimable: no permitted graded historical data. The price-aware simulator is implemented, but no empirical profitability or risk conclusion is asserted.
12. **What bankroll/exposure survives the worst observed sequence?** Not estimable: no permitted graded historical data.

## Interpretation

A total is a score reference, not a sportsbook Over prediction. Model D consistently uses Over as its reference side and Under as its opposite; this convention does not establish that bookmakers predicted Over.

A moneyline favorite is graded identically at early and closing snapshots unless its designation flips. Price movement can change return without changing correctness. Spread/total reference grades and executable-book grades are stored separately.

NBA V1 is neither retrained nor connected to these signals. No finding is promoted to production. A future shadow test requires complete, auditable market/outcome coverage, season replication, and prospectively frozen rules.

Game numbering is exact within the observed catalog, but unverified as the complete played schedule. The Odds API does not guarantee that every unlisted/canceled NBA game can be reconstructed. This limitation is carried into every export.

Bootstrap ruin estimates, when populated, replay paired price/outcome blocks within season and retain overlap inside each block. Blocks are serialized. These exploratory assumptions are not a prediction of future ruin. Historical survival is not a guarantee.

Results preserve unsuccessful bands and strategies. Hit-rate confidence intervals and exploratory ROI intervals do not correct correlated events or multiple testing.

See [methodology and run instructions](https://github.com/bigtyme2k3/Nba-model/blob/main/docs/NBA_MARKET_RESEARCH.md) and JSON/CSV files alongside this report.
