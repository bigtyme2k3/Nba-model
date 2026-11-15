# V2 Monte Carlo probability layer

Shadow research only. V1 makes every signal, price check and bet; the Monte Carlo
layer has no bet authority until it passes the promotion review below.

## What it does
`v2/simulation/engine.py` turns V2's expected stats (V1's for threes, until V2 has a
threes model) into outcome distributions for each player-game:

- **Minutes:** projected minutes plus player-specific noise (scaled 10-game minutes
  SD), truncated to 0-48, with a small early-exit probability.
- **Game factor:** one scoring/pace factor shared by everyone in a game.
- **Player factor:** one form factor shared by a player's stats in a simulation.
- **Counts:** negative-binomial stats around the scaled expectation; PRA is the sum of
  the same path.
- **Points from scoring events:** 3 x the simulated threes + 2 x two-point makes + free
  throws (non-three points split by `ft_share`, fit 0.23). Counting points one at a time
  made the distribution too narrow; this also links points to the threes market.
- **Same-game coupling:** each player's simulations are reordered within a game so
  points ranks follow a Gaussian reference with teammate correlation -0.03 and
  opponent correlation +0.01 (fit on 2023-24 residuals). Whole simulations move
  together, so each player's distribution and every single-prop price are unchanged.
- Each player's simulated mean is held equal to the model's expectation, so the
  simulation adds shape only and never re-applies injury or minutes effects that V2's
  projection already contains.

`probabilities.py` gives over/under/push (push only on whole-number lines),
percentiles, Monte Carlo standard errors, CRPS, Brier and log loss.

## Validation (`python -m v2.simulation.validation`)
- Out-of-sample means: each season's V1 and V2 prop models are trained only on
  earlier seasons.
- Spread parameters (`models/v2_mc_calibration.json`) and the Normal baselines'
  sigmas are fit on 2023-24 only; 2024-25 and 2025-26 are untouched test seasons.
- Three methods on the same rows: `v1_normal` (V1's live method), `v2_normal` (V2's
  mean, same Normal shape: isolates the better mean) and `v2_mc` (V2's mean, simulated
  shape).
- Over/under is scored at pseudo-lines (prior 5-game average rounded to .5), which no
  model sees. Real sportsbook lines come in Phase 3.

### Result (`data/tracking/v2_mc_validation.json`, version V2_MC_2), 2025-26 test season
| Stat | Brier v1_normal | **v2_mc** | Log loss v1_normal → v2_mc | Bottom / top PIT decile |
|---|---|---|---|---|
| PTS | 0.2391 | **0.2331** | 0.671 → 0.658 | 10.8% / 10.1% |
| REB | 0.2414 | **0.2325** | 0.676 → 0.657 | 9.5% / 9.6% |
| AST | 0.2390 | **0.2276** | 0.671 → 0.647 | 9.9% / 10.3% |
| 3PM | 0.2254 | **0.2063** | 0.643 → 0.595 | 10.5% / 10.5% |
| PRA | 0.2408 | **0.2352** | 0.675 → 0.663 | 10.1% / 9.8% |

2024-25 shows the same ordering on every stat. Points tails (V2_MC_1 had 14.7% of
games in the bottom decile, 12.2% in the top) are fixed by event-built points: points
CRPS 3.229 → 3.206 and log loss 0.661 → 0.658 on 2025-26, other stats unchanged.

### Same-game dependence (V2_MC_3)
A shared game factor can only make everyone in a game move together, but real
teammates move slightly against each other (2023-24 points residuals: teammates
-0.029, opponents +0.011; rebounds and assists closer to 0). The CRPS-fit game factor
(0.10) made teammates +0.028, the wrong sign. Extra within-team usage noise could
match the sign only by making every player's points distribution too wide (bottom
decile 7.7%), so dependence is now set by reordering instead (see above); the game
factor stays as a marginal-shape parameter only.

Out of sample (all same-game pairs, "both over" at pseudo-lines, ~135k teammate and
~147k opponent pairs per season):

| | 2024-25 actual / simulated corr | 2025-26 actual / simulated | Log loss independent → coupled |
|---|---|---|---|
| PTS teammates | -0.026 / -0.028 | -0.025 / -0.028 | 0.47603 → 0.47582 ; 0.47593 → 0.47588 |
| PTS opponents | +0.014 / +0.009 | +0.012 / +0.009 | 0.48876 → 0.48889 ; 0.48865 → 0.48876 |
| PRA teammates | -0.005 / -0.023 | -0.003 / -0.023 | 0.49527 → 0.49525 ; 0.49604 → 0.49604 |
| PRA opponents | +0.008 / +0.007 | +0.005 / +0.008 | 0.50331 → 0.50340 ; 0.50398 → 0.50407 |

- The teammate points correlation holds on both untouched seasons.
- Same-game dependence between player props is tiny: joint probabilities move by
  about 0.4 percentage points, and pricing gains over independence are at the fifth
  decimal, slightly negative for opponents. PRA teammates are too negative (the
  coupling is keyed on points; rebounds and assists offset it in reality).
- Conclusion: independence is an adequate approximation for player-prop pairs; the
  coupling is not a source of edge and same-game combinations should not be priced
  as if it were. Parameters were not re-tuned on the test seasons.

## Phase 3: recorded sportsbook prices (`python -m v2.simulation.price_research`)
The ledger's graded props (2026-27, Oct 27 - Nov 13, 17 dates) at the exact quote
recorded before tip: 2,426 of 2,661 rows (the other 235 are in games whose lineups
could not be reconstructed from ESPN's play-by-play). All lines were half points, so
no pushes. Quotes were taken a median 6.5 hours before tip (10-90%: 4.0-11.2 h).
V2's absences use the injury report version scraped before that day's predictions
(from git history); every date had one.

| Arm (recorded side wins) | Brier | Log loss | Log loss − market, 95% CI |
|---|---|---|---|
| Market no-vig | **0.2476** | **0.6884** | — |
| V1 recorded probability | 0.2727 | 0.7472 | +0.038 to +0.084 |
| MC around V1's projection | 0.2626 | 0.7215 | +0.018 to +0.049 |
| MC around V2's mean | 0.2612 | 0.7190 | +0.015 to +0.049 |

- The simulated shape removes about 45% of V1's probability deficit with the same mean;
  V2's mean adds a little more. Every model arm is still clearly worse than the market
  on every stat and at DraftKings and FanDuel; at Fanatics (59 rows) MC-on-V2's log
  loss is 0.6868 vs the market's 0.6879, too few rows to mean anything. Model probabilities average 57-61% on sides that won 51%.
- Calibration gate (`calibrate_edges.temporal_calibration`, unchanged): V1 and MC-on-V1
  are rejected; MC-on-V2 technically passes with shrink 0.13, holdout log loss 0.68803
  vs the market's 0.68807 (about 485 rows). That is a 0.00004 difference on a model
  that loses to the market overall: noise, not evidence. Nothing was changed; no
  positive model weight is restored from this.
- Strict gate (`v2/simulation/gate.py`, fixed 2026-11-15 before later data was
  examined): every arm FAILS. MC-on-V2 has 4 hold-out dates (needs 7), no margin
  (holdout log loss −0.00004 vs the required −0.002), a CI upper bound of +0.002
  (must be < 0), and its first walk-forward split loses to the market by 0.011.
- Why the model arms are overconfident (diagnostics on the OVER side of each row, so
  V1's choice of side cannot shape them): the level is right (MC-on-V2's mean over
  probability 47.3% vs a 47.4% over rate) but the calibration slope is 0.16 (market
  0.95), and the data give MC only about 0.10 weight on top of the market. MC-on-V2
  differs from the market by 10.6 points on average, and that disagreement is mostly
  noise. Since the simulated shape is calibrated on historical pseudo-lines (flat
  PIT), this is a projection-vs-line problem, not a distribution-shape problem: the
  market's line carries information (late news, lineups) the projection lacks.
  Re-shaping the distribution cannot fix it.
- Coverage: the 235 unmatched rows (games without reconstructable lineups) won 55.7%
  vs 51.4% matched (+4.8% vs −5.5% ROI on every recorded side), with a lower over
  share (49% vs 56%) and similar stat and book mix. The win-rate gap is about 1.3
  standard errors on 235 rows; V1 still loses to the market on them (log loss 0.743
  vs 0.702). No sign the exclusion flatters the results, but it is watched.
- Hypothetical ROI at recorded odds, "bet when EV ≥ threshold": every recorded side
  −5.5%; MC-on-V2 −3.0% to −4.6%, MC-on-V1 −3.3% to −5.1%, V1 −5.3% to −6.1%. Every
  interval includes or lies below zero. No profitable rule is shown.
- Novig: no graded Novig quotes recorded yet; reported separately once they exist.

## Next phases
1. Daily shadow publication beside `v2_pred`, forward grading and the V2 Research Lab;
   rerun Phase 3 as the ledger grows and Novig quotes accrue.

## Promotion review
Better calibrated probabilities on untouched future games, stable historical and
forward validation, no leakage or double adjustments, adequate matched prices,
acceptable drawdowns and closing-line value, and no loss of coverage. Until all pass,
V1 stays in control.
