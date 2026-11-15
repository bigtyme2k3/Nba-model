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

### Result (`data/tracking/v2_mc_validation.json`), 2025-26 test season
| Stat | Brier v1_normal | v2_normal | **v2_mc** | Log loss v1_normal → v2_mc |
|---|---|---|---|---|
| PTS | 0.2391 | 0.2361 | **0.2342** | 0.671 → 0.661 |
| REB | 0.2414 | 0.2361 | **0.2326** | 0.676 → 0.657 |
| AST | 0.2390 | 0.2344 | **0.2276** | 0.671 → 0.647 |
| 3PM | 0.2254 | 0.2254 | **0.2062** | 0.643 → 0.595 |
| PRA | 0.2408 | 0.2362 | **0.2356** | 0.675 → 0.664 |

2024-25 shows the same ordering on every stat; CRPS improves the same way.

Calibration (randomized PIT, 8,000 player-games, 2025-26): rebounds, assists and threes
put 79-80% of outcomes in the central 80%. **Points tails are too thin**: 73.5% in the
central 80% and 14.7% in the bottom decile, so bust games are under-simulated. Raw
p10-p90 coverage looks too wide for low counts only because outcomes are whole numbers.

The fitted shared game factor is 0: across one season it did not improve individual
distributions, so teammates' outcomes are not yet linked. Same-game combinations must
not be priced from this version.

## Next phases
1. Fatter points tails (minutes/usage bust scenarios), and a game factor fit on
   same-game residual correlation rather than individual CRPS.
2. Price-aware research on recorded FanDuel/DraftKings/Fanatics quotes, Novig separate,
   with the existing calibration gate.
3. Daily shadow publication beside `v2_pred`, forward grading and the V2 Research Lab.

## Promotion review
Better calibrated probabilities on untouched future games, stable historical and
forward validation, no leakage or double adjustments, adequate matched prices,
acceptable drawdowns and closing-line value, and no loss of coverage. Until all pass,
V1 stays in control.
