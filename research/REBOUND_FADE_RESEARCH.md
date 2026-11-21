# NBA Rebound Fade V1 — Pre-Registered Shadow Experiment

**Status:** Research only. No production picks, calibration, model weights, sportsbook accounts, bankroll settings, or auto-bets are touched.

## Hypothesis fixed November 21, 2026

The original Consistency module's **PRETIP exact-line REB OVER** signals may systematically lose more often than the best quoted **UNDER at the same line** would require. Test this even when the original Consistency recommendation is PASS, WATCH or LEAN; these are **monitored research candidates, not qualified bets**.

This hypothesis was chosen **after** examining November 19–20 records. Those dates cannot be counted as independent confirmation. The first prospective shadow date is November 21, 2026.

## Discovery evidence (November 19–20)

- 15 graded rebound Over signals; 2 original Overs won and 13 lost.
- Reversing to the exact-line Under, at the best matching recorded DraftKings, FanDuel or Fanatics pre-tip odds, returned **+10.9584 flat units (+73.06% ROI)** across those 15 examples.
- November 19: 4–1 fade, +3.0561 units (+61.12%). November 20: 9–1 fade, +7.9023 units (+79.02%).
- All selected quotes were captured before tipoff, roughly 68–128 minutes prior.
- The 15 original recommendations were 12 PASS, 2 WATCH and 1 LEAN; **zero BET** recommendations.
- Only **two dates**: outcomes are dependent within game/date, and the pattern was identified after seeing these losses. This result does **not** establish an edge.

Sources: data/history/nba_consistency_signals.jsonl (immutable quotes) and data/raw/espn_current_player_box_2026.csv (actuals). A separate consistency_report.json summary is an audit cross-check; this code grades directly from ESPN box rows, so it doesn't depend on the report's rolling 400-row history cap.

## Reproducible and leakage-safe procedure

1. Filter to kind=LINE, phase=PRETIP, category=REB, side=OVER. Take the **latest recorded PRETIP** signal for each player/date, without considering game results.
2. Require valid ISO timestamps for signal recording, quote capture, and tip. Both signal and capture must precede tip; quoted data must be no more than 10 minutes older than signal recording.
3. Require the opposing UNDER at **exactly the same numeric line**. Select the greatest American odds across FanDuel, DraftKings, and Fanatics at that snapshot. Never substitute nearby alt lines, later prices, or -110 assumptions.
4. Grade the opposite side on actual rebounds from the box-score row for player/date. DNP => VOID_DNP; integer-line ties => PUSH; unplayed => PENDING. No backfilled grade where player actuals are missing.
5. Track a one-unit *hypothetical* price-aware profit for every priced and settled candidate, not actual wagers. Report historical discovery and prospective shadow periods independently, with daily counts.
6. Never interpret a positive discovery ROI as a proven strategy. For prospective evidence, inspect multiple independent dates, time-block bootstrapped uncertainty, actual quote coverage, and performance relative to comparable no-vig market probabilities before considering any promotion.
7. Closing-line coverage is presently insufficient for meaningful CLV claims. The quote snapshots are observable offer records, not proof of available real-money fills.

## Run locally or via GitHub Actions

Run from repository root:

    python -m unittest tests.test_rebound_fade_shadow -v
    python research/rebound_fade_shadow.py --out artifacts/rebound_fade_shadow

This writes three independent outputs: `summary.json`, `ledger.csv`, and `REPORT.md`. Nothing is written to active predictions, weights, dashboard, ledger, or any other production evidence.

The `NBA Rebound Fade Shadow Research` workflow can be triggered manually and, once merged, runs daily at 16:30 UTC. It has **contents: read** permission only and uploads a research artifact, without pushing commits or consuming Odds API credits.

## Promotion policy

No automated promotions. A candidate cannot graduate based on this retrospectively selected two-day sample. Require chronological unseen-date validation with date/game clustered uncertainty and credible priced ROI, including comparisons against the sportsbook baseline and worse-case execution. If opposite-side prices are missing or timestamp integrity fails, mark the row unpriced; do not count it as a wager.
