# Game box-score matchup implementation

The spread and total models can consume five groups of prior-game box-score
matchups: effective shooting, turnovers, offensive rebounding, free throws, and
three-point volume/accuracy. These additions are opt-in research candidates.
Normal training retains the baseline feature list. Existing active bundles and
prediction files are not replaced by this change.

`SPREAD_FEATURES` and `TOTALS_FEATURES` remain the baseline lists, including in
`backtest_vs_close.py`. Extended lists have separate `BOX_SPREAD_FEATURES` and
`BOX_TOTALS_FEATURES` names. The spread row-coverage filter counts baseline
features only, even when research columns are present but missing.

## Foundation fixes

`build_team_features.py` now calculates 5/10-game net ratings from the raw game
ratings. Previously historical context applied a second rolling average to an
already averaged series, while live profiles used raw ratings.

The odds merge preserves the ESPN `game_id` and puts the odds-provider ID in
`game_id_odds`. Previously the merge renamed them to `game_id_x/game_id_y`,
leaving current-season `game_id` blank in the combined master.

Feature semantics are stamped `NBA_GAME_FEATURES_V3_FORM_PARITY`. Validation
uses whole-date expanding folds. Both change the model-family ID on retraining;
calibration evidence for an older family does not become evidence for the new
family. Existing betting qualification rules are unchanged.

Stored CV scores from different validation methods are not compared. On a method
migration the publication report marks relative MAE/directional comparisons as
skipped and leaves the MAE delta unset. Candidate validation method, context,
semantics, production scope, training freshness and absolute-sanity gates still
apply. Once both bundles use the same method, relative nonregression applies
again. A skipped comparison does not demonstrate improvement over the champion.

## Rate definitions

All windows use only earlier completed games, continue across season boundaries,
and aggregate ten games of counts before dividing. Missing components exclude
that game from both numerator and denominator. An unobserved denominator leaves
the rate missing, not zero. Exactly two distinct teams are required to obtain
opponent counts.

| Rate | Definition |
| --- | --- |
| Effective FG% | `(FGM + 0.5 * 3PM) / FGA` |
| Turnovers per possession | `TOV / (FGA - OREB + TOV + 0.44 * FTA)` |
| Offensive rebound percentage | `OREB / (OREB + opponent DREB)` |
| Free-throw attempt rate | `FTA / FGA` |
| Free-throw percentage | `FTM / FTA` |
| Three-point attempt share | `3PA / FGA` |
| Stabilized three-point percentage | Ten-game 3PM/3PA with 100 pseudo-attempts anchored to older history |

The three-point anchor uses older team games excluding the recent ten-game
window, with another 100 pseudo-attempts at a fixed .36. This stabilization is
also applied to opponent three-point accuracy allowed. No evaluation-period
league average is used. These are fixed initial research settings, not estimates
proven optimal for this model.

Each matchup averages the attacking team's rate and the defending team's
corresponding opponent rate. Home and away free-throw percentages are also
included. Fourteen candidate features result. `team_box_matchups.py` shares all
rate and interaction calculations between historical data and daily scoring.
Candidate scoring stops if required matchup context is missing; legacy bundles
do not require the new context. The daily runner and direct spread scorer skip
only that game; the direct totals batch retains an explicit skipped row and
continues scoring the rest of the slate.

## Reproduce the review

Use the pinned project dependencies (Python 3.11 / `requirements.txt` in CI).
Preserve the player logs when building an isolated master: they supply the
existing injury/availability features used by the baseline.

```bash
OUT=data/research/game_box_matchups/processed
python build_team_features.py --out "$OUT"
cp data/processed/player_logs.csv "$OUT/player_logs.csv"
python merge_data.py --year all --out "$OUT"
python research/game_box_matchups.py --data "$OUT/master_all.csv" \
  --holdout-season 2025 --out data/tracking/game_box_matchups_ablation.json
```

`2025` identifies the completed 2025-26 season. Older seasons select a feature
group on four whole-date folds. Only that frozen selection is compared with the
corrected baseline on the reserved season; newer games are reported separately.
All variants use the same eligible rows. Paired confidence intervals resample
whole game dates. The report is descriptive research, not automatic promotion.
Recorded consensus lines, when present, provide unpriced diagnostics only;
missing lines never become an ATS or betting-return claim.

The manual **NBA Game Box Matchup Research** workflow reproduces this analysis
with read-only repository permissions and uploads evidence without publishing.

## Isolated full-model candidates

Both training entry points support these flags. `--candidate-out` alone trains
the corrected baseline; adding `--box-matchups` trains all five groups.

```bash
python spread_model.py --data "$OUT/master_all.csv" --box-matchups \
  --candidate-out models/candidates/box_spread.pkl
python totals_model.py --data "$OUT/master_all.csv" --box-matchups \
  --candidate-out models/candidates/box_totals.pkl
```

Candidates carry `research_only=True`; the publication gate rejects that scope.
Research writes anywhere in `models/` must use `models/candidates/`; production
props, calibration and other model files are protected too. Traversal and
symlink targets are checked before training and again before saving. A later promotion needs
reviewed out-of-sample evidence and explicit removal of research-only scope.
The model comparison, forward evidence, calibration, and actual executable
FD/DK/Fanatics prices remain separate requirements.

## Initial evidence (through November 19, 2026)

The paired ablation uses 6,785 eligible games, with 1,316 reserved 2025-26 games
and 226 newer games reported separately. Three-point features were selected
using older-date CV for both targets.

| Mean absolute error, points | Corrected baseline | Selected three-point candidate |
| --- | ---: | ---: |
| Spread, reserved 2025-26 season | 11.494 | 11.389 |
| Total, reserved 2025-26 season | 15.244 | 15.222 |
| Spread, newer 2026-27 games | 12.405 | 12.427 |
| Total, newer 2026-27 games | 14.659 | 14.709 |

The spread holdout improvement has a paired date-bootstrap 95% interval of
[-0.179, -0.025] points. The total holdout interval includes zero. Neither target
improves MAE on the newer sample. This supports further spread shadow research,
not production promotion. All five groups are implemented, but adding every
group performed worse in development CV than adding the three-point group alone.

Separate full-training smoke runs successfully built isolated 48-feature spread
and 49-feature total bundles, each on 6,785 games. Both scored through the daily
runner with all fourteen matchup inputs; active model hashes stayed unchanged.
These full bundles include all groups and are distinct from the selected-group
paired experiment above. The evidence JSON records the runtime and input hash.
