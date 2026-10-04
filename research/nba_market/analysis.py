"""Season-stratified descriptive analysis. Unknown outcomes never become losses."""
from __future__ import annotations

import statistics
from collections import defaultdict

from .core import wilson
from .dataset import line_movement
from .simulation import STRATEGIES, bootstrap_ruin, simulate
from .streaks import qualify_after_failures, streak_tables


def summarize(rows, config):
    graded = [r for r in rows if r.get("result") in {"win", "loss", "push"}]
    wins = sum(r["result"] == "win" for r in graded)
    losses = sum(r["result"] == "loss" for r in graded)
    pushes = len(graded)-wins-losses
    profits = [r["net_units"] for r in graded]
    low, high = wilson(wins, wins+losses)
    mean = statistics.mean(profits) if profits else None
    se = statistics.stdev(profits)/(len(profits)**.5) if len(profits) > 1 else None
    probabilities = [r["no_vig_probability"] for r in graded if r.get("no_vig_probability") is not None]
    return {"games": len(rows), "graded_games": len(graded), "ungraded_games": len(rows)-len(graded),
            "wins": wins, "losses": losses, "pushes": pushes, "win_pct": wins/(wins+losses) if wins+losses else None,
            "win_pct_ci_low": low, "win_pct_ci_high": high, "net_units": sum(profits) if profits else None,
            "flat_bet_roi": mean, "roi_normal_ci_low": mean-1.96*se if se is not None else None,
            "roi_normal_ci_high": mean+1.96*se if se is not None else None,
            "average_price": statistics.mean(r["price"] for r in rows) if rows else None,
            "median_price": statistics.median(r["price"] for r in rows) if rows else None,
            "no_vig_expected_win_pct": statistics.mean(probabilities) if probabilities else None,
            "observed_minus_no_vig": wins/(wins+losses)-statistics.mean(probabilities) if probabilities and wins+losses else None,
            "small_sample": len(graded) < config["small_sample_threshold"],
            "ci_limitations": "Wilson hit rate; normal ROI approximation; correlated events and multiple testing are not corrected",
            "game_number_status": "observed_catalog_not_independently_certified"}


def summary_tables(bets, config):
    groups = defaultdict(list)
    price_groups = defaultdict(list)
    totals_groups = defaultdict(list)
    pairs = {(r["event_id"], r["basis"], r["snapshot_type"], r["market"], r["mode"]): r for r in bets}
    for bet in bets:
        if not bet["qualifies"]:
            continue
        for season in (bet["season"], "ALL"):
            for bucket in ("ALL_EARLY", bet["early_bucket"]) if bet["market"] != "totals" else ("ALL_EARLY",):
                key = (season, bet["basis"], bet["snapshot_type"], bet["market"], bet["mode"], bucket)
                groups[key].append(bet)
                if bet["market"] == "h2h" and bet["mode"] == "FADE":
                    price_groups[key+(bet["price_band"],)].append(bet)
            if bet["market"] == "totals":
                # Independent team classifications are retained in separate, labeled views.
                for side in ("home", "away"):
                    if bet[f"team_game_number_{side}"] <= 20:
                        totals_groups[(season, bet["basis"], bet["snapshot_type"], bet["mode"], side, bet[f"early_bucket_{side}"])].append(bet)
    fields = ("season", "basis", "snapshot_type", "market", "mode", "early_bucket")
    summaries = [dict(zip(fields, key), **summarize(rows, config)) for key, rows in sorted(groups.items())]
    underdogs = []
    for key, rows in sorted(price_groups.items()):
        paired_favorites = [pairs[(r["event_id"], r["basis"], r["snapshot_type"], "h2h", "FOLLOW")] for r in rows]
        fav = summarize(paired_favorites, config)
        dog = summarize(rows, config)
        underdogs.append(dict(zip(fields+("price_band",), key), **dog,
                              underdog_wins=dog["wins"], underdog_losses=dog["losses"],
                              underdog_flat_bet_roi=dog["flat_bet_roi"], favorite_win_pct=fav["win_pct"],
                              favorite_flat_bet_roi=fav["flat_bet_roi"],
                              paired_favorite_games=fav["graded_games"]))
    total_buckets = [dict(zip(("season", "basis", "snapshot_type", "mode", "team_side_view", "early_bucket"), key),
                         **summarize(rows, config), paired_views_not_independent=True)
                    for key, rows in sorted(totals_groups.items())]
    return summaries, underdogs, total_buckets


def early_vs_later(bets, config):
    groups = defaultdict(list)
    for bet in bets:
        early = bet["early_event"] if bet["market"] == "totals" else bet["team_game_number"] <= 20
        for season in (bet["season"], "ALL"):
            groups[(season, bet["basis"], bet["snapshot_type"], bet["market"], bet["mode"], "Games 1-20" if early else "Games 21+")].append(bet)
    rows = [dict(zip(("season", "basis", "snapshot_type", "market", "mode", "period"), key), **summarize(values, config))
            for key, values in sorted(groups.items())]
    index = {(r["season"], r["basis"], r["snapshot_type"], r["market"], r["mode"], r["period"]): r for r in rows}
    for row in rows:
        other = index.get((row["season"], row["basis"], row["snapshot_type"], row["market"], row["mode"], "Games 21+" if row["period"] == "Games 1-20" else "Games 1-20"))
        row["win_pct_minus_other_period"] = row["win_pct"]-other["win_pct"] if other and row["win_pct"] is not None and other["win_pct"] is not None else None
        row["flat_roi_minus_other_period"] = row["flat_bet_roi"]-other["flat_bet_roi"] if other and row["flat_bet_roi"] is not None and other["flat_bet_roi"] is not None else None
        row["reference_inventory_requires_coverage_audit"] = True
    return rows


def opening_comparison(bets, movements, config):
    grouped = defaultdict(dict)
    for bet in bets:
        if bet["mode"] == "FOLLOW" and bet["early_event"]:
            grouped[(bet["event_id"], bet["basis"], bet["market"])][bet["snapshot_type"]] = bet
    moves = {(m["event_id"], m["basis"], m["market"]): m for m in movements}
    details, groups = [], defaultdict(list)
    for key, snapshots in sorted(grouped.items()):
        if set(snapshots) != {"EARLY", "CLOSE"} or key not in moves:
            continue
        early, close = snapshots["EARLY"], snapshots["CLOSE"]
        er, cr = early.get("reference_result"), close.get("reference_result")
        label = f"Open {'right' if er == 'win' else 'wrong'} / Close {'right' if cr == 'win' else 'wrong'}" if er in {"win", "loss"} and cr in {"win", "loss"} else "push_or_ungraded"
        record = dict(moves[key], comparison=label, open_result=er, close_result=cr,
                      open_price=early["price"], close_price=close["price"],
                      open_reference_line=early["reference_line"], close_reference_line=close["reference_line"],
                      open_execution_line=early["execution_line"], close_execution_line=close["execution_line"],
                      open_flat_pnl=early["net_units"], close_flat_pnl=close["net_units"])
        details.append(record)
        for season in (early["season"], "ALL"):
            groups[(season, early["basis"], early["market"], label, record["designation"])].append(record)
    summary = []
    for key, rows in sorted(groups.items()):
        summary.append(dict(zip(("season", "basis", "market", "comparison", "movement"), key),
                            games=len(rows), favorite_flips=sum(r["favorite_flipped"] for r in rows),
                            open_flat_net_units=sum(r["open_flat_pnl"] for r in rows if r["open_flat_pnl"] is not None),
                            close_flat_net_units=sum(r["close_flat_pnl"] for r in rows if r["close_flat_pnl"] is not None),
                            small_sample=len(rows) < config["small_sample_threshold"]))
    return details, summary


def disagreement_tables(views, bets, config):
    lookup = {(b["event_id"], b["snapshot_type"], b["market"]): b
              for b in bets if b["basis"] == "CONSENSUS" and b["mode"] == "FOLLOW"}
    details, groups = [], defaultdict(list)
    for view in views:
        if view["basis"] != "CONSENSUS" or min(view["team_game_number_home"], view["team_game_number_away"]) > 20:
            continue
        market = view["market"]
        if market == "h2h":
            large = view.get("different_favorites", False) or view.get("ml_probability_range", 0) >= config["probability_disagreement"]
            metric = "favorite_outright_failure"
        elif market == "spreads":
            large = view.get("spreads_range", 0) >= config["spread_disagreement_points"]
            metric = "favorite_ATS_failure"
        else:
            large = view.get("totals_range", 0) >= config["total_disagreement_points"]
            metric = "Under_frequency_against_Over_reference"
        bet = lookup.get((view["event_id"], view["snapshot_type"], market))
        result = bet.get("reference_result") if bet else None
        record = dict(view, high_disagreement=large, failure_metric=metric, follow_reference_result=result,
                      follow_flat_pnl=bet.get("net_units") if bet else None)
        details.append(record)
        for season in (view["season"], "ALL"):
            groups[(season, view["snapshot_type"], market, large)].append(record)
    summary = []
    for key, rows in sorted(groups.items()):
        graded = [r for r in rows if r["follow_reference_result"] in {"win", "loss"}]
        failures = sum(r["follow_reference_result"] == "loss" for r in graded)
        low, high = wilson(failures, len(graded))
        summary.append(dict(zip(("season", "snapshot_type", "market", "high_disagreement"), key),
                            games=len(rows), graded_games=len(graded), failures=failures,
                            failure_rate=failures/len(graded) if graded else None, ci_low=low, ci_high=high,
                            small_sample=len(graded) < config["small_sample_threshold"]))
    indexed = {(r["season"], r["snapshot_type"], r["market"], r["high_disagreement"]): r for r in summary}
    for row in summary:
        other = indexed.get((row["season"], row["snapshot_type"], row["market"], not row["high_disagreement"]))
        row["failure_rate_minus_other_group"] = row["failure_rate"]-other["failure_rate"] if other and row["failure_rate"] is not None and other["failure_rate"] is not None else None
    return details, summary


def simulation_tables(bets, config):
    grouped = defaultdict(list)
    follows = {(b["event_id"], b["snapshot_type"], b["market"]): b["reference_result"]
               for b in bets if b["basis"] == "CONSENSUS" and b["mode"] == "FOLLOW"}
    underdog_actors = {(b["event_id"], b["snapshot_type"], b["market"]): b
                      for b in bets if b["basis"] == "CONSENSUS" and b["mode"] == "FADE" and b["market"] != "totals"}
    for bet in bets:
        if bet["basis"] == "CONSENSUS":
            row = dict(bet, market_follow_reference_result=follows.get((bet["event_id"], bet["snapshot_type"], bet["market"])))
            grouped[(bet["snapshot_type"], bet["market"], bet["mode"])].append(row)
    summaries, ledgers, sequences, risks = [], [], [], []
    for (snapshot, market, mode), offers in sorted(grouped.items()):
        for scope in ("league", "same_team"):
            if scope == "same_team" and market == "totals":
                team_offers = []
                for offer in offers:
                    for side in ("home", "away"):
                        team_offers.append(dict(offer, team=offer[f"{side}_team"], qualifies=offer[f"team_game_number_{side}"] <= 20,
                                                signal_id=offer["signal_id"]+"_"+side))
                scoped = team_offers
            else:
                scoped = offers
            variants = [("selection_team" if scope == "same_team" else "league_event", scoped)]
            if scope == "same_team" and market != "totals" and mode == "FOLLOW":
                actor_offers = []
                for offer in offers:
                    actor = underdog_actors[(offer["event_id"], snapshot, market)]
                    actor_offers.append(dict(offer, sequence_team=actor["team"], qualifies=actor["qualifies"],
                                             qualifying_team_game_number=actor["team_game_number"],
                                             signal_id=offer["signal_id"]+"_underdog_actor"))
                variants.append(("underdog_team_opposing_favorites", actor_offers))
            for qualifier, variant in variants:
                for threshold in (0, 1, 2, 3):
                    qualified = qualify_after_failures(variant, scope, threshold)
                    season_sets = [(s, [o for o in qualified if o["season"] == s]) for s in sorted({o["season"] for o in qualified})]
                    season_sets.append(("ALL", qualified))
                    for season, rows in season_sets:
                        for strategy in STRATEGIES:
                            meta = {"season": season, "snapshot_type": snapshot, "market": market, "mode": mode,
                                    "scope": scope, "qualifier": qualifier, "after_market_failures": threshold, "strategy": strategy}
                            summary, ledger, seq = simulate(rows, strategy, scope, config["target_profit"], config["max_progression_stake"])
                            summary.update(meta)
                            summaries.append(summary)
                            if season == "ALL":
                                ledgers.extend(dict(r, **{k: v for k, v in meta.items() if k != "season"}) for r in ledger)
                                sequences.extend(dict(r, **{k: v for k, v in meta.items() if k != "season"}) for r in seq)
                                for risk in (bootstrap_ruin(rows, strategy, scope, config) if threshold == 0 else []):
                                    risks.append(dict(meta, **risk,
                                                      observed_path_operational_failure=summary["historical_bankroll_requirement"] > risk["bankroll"] if summary["historical_bankroll_requirement"] is not None else None,
                                                      historical_bankroll_requirement=summary["historical_bankroll_requirement"]))
    return summaries, ledgers, sequences, risks


def season_validation(underdogs, summaries, config):
    groups = defaultdict(dict)
    inputs = [("underdog_band", r) for r in underdogs if r["basis"] == "CONSENSUS"]
    inputs += [("baseline", r) for r in summaries if r["basis"] == "CONSENSUS"]
    for kind, row in inputs:
        key = (kind, row["snapshot_type"], row["market"], row["mode"], row["early_bucket"], row.get("price_band", "all"))
        groups[key][row["season"]] = row
    validation = []
    for key, seasons in sorted(groups.items()):
        pooled = seasons.get("ALL")
        if not pooled:
            continue
        individual = [seasons.get(s) for s in config["seasons"]]
        positive = [r for r in individual if r and r["flat_bet_roi"] is not None and r["flat_bet_roi"] > 0]
        sufficient = all(r and r["graded_games"] >= 30 for r in individual)
        stronger = sufficient and len(positive) == len(individual) and pooled["graded_games"] >= config["small_sample_threshold"] and (pooled["roi_normal_ci_low"] or -1) > 0
        validation.append(dict(zip(("kind", "snapshot_type", "market", "mode", "early_bucket", "price_band"), key),
                               pooled_roi=pooled["flat_bet_roi"], pooled_graded_games=pooled["graded_games"],
                               positive_seasons=len(positive), requested_seasons=len(individual),
                               season_roi={s: seasons[s]["flat_bet_roi"] if s in seasons else None for s in config["seasons"]},
                               season_games={s: seasons[s]["graded_games"] if s in seasons else 0 for s in config["seasons"]},
                               repeats_across_all_requested_seasons=len(positive) == len(individual),
                               descriptive_candidate=stronger,
                               verdict="exploratory_candidate_requires_coverage_audit_and_prospective_shadow" if stronger else "insufficient_evidence",
                               automatic_live_integration=False))
    return validation


def analyze(views, bets, config):
    summary, underdogs, total_buckets = summary_tables(bets, config)
    movements = line_movement(views, config)
    open_detail, open_summary = opening_comparison(bets, movements, config)
    disagree_detail, disagree_summary = disagreement_tables(views, bets, config)
    streak_distribution, next_game, streak_runs = streak_tables(bets, config)
    simulations, ledger, sequences, risks = simulation_tables(bets, config)
    return {
        "early_season_summary": summary,
        "early_vs_later_summary": early_vs_later(bets, config),
        "underdog_price_bucket_summary": underdogs,
        "favorite_summary": [r for r in summary if r["market"] == "h2h" and r["mode"] == "FOLLOW"],
        "ats_summary": [r for r in summary if r["market"] == "spreads"],
        "totals_summary": [r for r in summary if r["market"] == "totals"],
        "totals_team_buckets": total_buckets,
        "streak_distribution": streak_distribution,
        "streak_runs": streak_runs,
        "next_game_mean_reversion": next_game,
        "line_movement": movements,
        "open_vs_close_comparison": open_summary,
        "open_vs_close_games": open_detail,
        "sportsbook_disagreement_analysis": disagree_summary,
        "sportsbook_disagreement_games": disagree_detail,
        "flat_bet_simulation": [r for r in simulations if r["strategy"] == "flat_1"],
        "progression_simulation": [r for r in simulations if r["strategy"] != "flat_1"],
        "simulation_bet_ledger": ledger,
        "simulation_sequences": sequences,
        "maximum_risk_report": risks,
        "season_validation": season_validation(underdogs, summary, config),
    }
