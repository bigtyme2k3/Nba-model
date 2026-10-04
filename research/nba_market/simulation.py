"""Event-time cash accounting, price-aware recovery, and explicit tail-risk testing."""
from __future__ import annotations

import math
import random
import statistics
from collections import defaultdict
from datetime import datetime, timedelta, timezone

from .core import flat_profit, iso, parse_time, payout

STRATEGIES = {
    "flat_1": {"stakes": [1]},
    "1_2": {"stakes": [1, 2]},
    "1_2_4": {"stakes": [1, 2, 4]},
    "1_2_4_8": {"stakes": [1, 2, 4, 8]},
    "1_1.5_2.5_4": {"stakes": [1, 1.5, 2.5, 4]},
    "price_aware_4": {"steps": 4, "price_aware": True},
    "price_aware_8": {"steps": 8, "price_aware": True},
}


def recovery_stake(losses, odds, target=1):
    """Stake to recover actual prior losses plus target at the current offered odds."""
    if losses < 0 or target <= 0:
        raise ValueError("Losses must be nonnegative and target positive")
    return (losses + target) / payout(odds)


def maximum_drawdown(profits):
    equity, peak, drawdown = 0.0, 0.0, 0.0
    for profit in profits:
        equity += profit
        peak = max(peak, equity)
        drawdown = max(drawdown, peak-equity)
    return drawdown


def simulate(offers, strategy, scope="league", target=1, max_stake=10000, bankroll=None, keep_ledger=True):
    """Signals place stakes; settlements release cash. Ties process signal before result."""
    if scope not in {"league", "same_team"}:
        raise ValueError("Invalid progression scope")
    spec = STRATEGIES[strategy] if isinstance(strategy, str) else strategy
    cap = spec.get("steps", len(spec.get("stakes", [])))
    if cap < 1:
        raise ValueError("Progression must have a finite number of steps")
    flat = not spec.get("price_aware") and spec.get("stakes") == [1]
    timeline, eligible, unknown = [], [], 0
    for offer in offers:
        if not offer.get("qualifies", True):
            continue
        signal = parse_time(offer["signal_timestamp"])
        settled = parse_time(offer["result_available_at"]) if offer.get("result_available_at") else None
        if settled and settled <= signal:
            raise ValueError("Result predates frozen betting signal")
        lane = (offer["season"], offer.get("sequence_team") or offer.get("team") or "totals") if scope == "same_team" else (offer["season"], "league")
        key = str(len(eligible))
        eligible.append((offer, lane))
        timeline.append((signal, 0, offer["event_id"], key))
        if settled and offer.get("result") in {"win", "loss", "push"}:
            timeline.append((settled, 1, offer["event_id"], key))
    timeline.sort()
    lanes, placed, ledger, sequences = {}, {}, [], []
    cash_delta, equity, peak, drawdown, outstanding = 0., 0., 0., 0., 0.
    required, max_outstanding, max_single, max_exposure, max_requested, max_requested_exposure = 0., 0., 0., 0., 0., 0.
    turnover, bets, wins, losses, pushes, overlaps, unaffordable, stake_caps = 0., 0, 0, 0, 0, 0, 0, 0
    longest_loss = 0
    sequence_counter, censored_lanes, censored_skips = 0, set(), 0
    for timestamp, kind, _, key in timeline:
        offer, lane = eligible[int(key)]
        state = lanes.setdefault(lane, {"step": 0, "losses": 0., "pending": 0, "sequence": None, "loss_streak": 0})
        if kind == 0:
            if offer.get("result") not in {"win", "loss", "push"} or not offer.get("result_available_at"):
                unknown += 1
                censored_lanes.add(lane)
                continue
            if lane in censored_lanes:
                censored_skips += 1
                continue
            if state["pending"] and not flat:
                overlaps += 1
                continue
            stake = recovery_stake(state["losses"], offer["price"], target) if spec.get("price_aware") else spec["stakes"][state["step"]]
            max_requested = max(max_requested, stake)
            max_requested_exposure = max(max_requested_exposure, state["losses"]+stake if not flat else stake)
            if not math.isfinite(stake) or stake > max_stake:
                stake_caps += 1
                if state["sequence"]:
                    state["sequence"].update(ending="stake_cap", net_units=-state["losses"], failed=True)
                state.update(step=0, losses=0., sequence=None)
                continue
            if bankroll is not None and bankroll + cash_delta < stake - 1e-9:
                unaffordable += 1
                # A bankrupt strategy cannot continue placing later bets with imaginary cash.
                break
            if flat or state["sequence"] is None:
                sequence_counter += 1
                sequence = {"sequence_id": sequence_counter, "season": offer["season"], "team": lane[1],
                            "start": iso(timestamp), "bets": 0, "turnover": 0., "exposure": 0.,
                            "net_units": 0., "ending": "right_censored", "failed": False}
                sequences.append(sequence)
                if not flat:
                    state["sequence"] = sequence
            else:
                sequence = state["sequence"]
            sequence["bets"] += 1
            sequence["turnover"] += stake
            sequence["exposure"] = max(sequence["exposure"], state["losses"]+stake if not flat else stake)
            max_exposure = max(max_exposure, sequence["exposure"])
            state["pending"] += 1
            cash_delta -= stake
            outstanding += stake
            required = max(required, -cash_delta)
            max_outstanding = max(max_outstanding, outstanding)
            max_single = max(max_single, stake)
            turnover += stake
            bets += 1
            placed[key] = (stake, sequence)
            continue
        if key not in placed:
            continue
        stake, sequence = placed.pop(key)
        profit = flat_profit(offer["result"], offer["price"], stake)
        cash_delta += stake + profit
        outstanding -= stake
        equity += profit
        peak = max(peak, equity)
        drawdown = max(drawdown, peak-equity)
        sequence["net_units"] += profit
        sequence["end"] = iso(timestamp)
        state["pending"] -= 1
        if offer["result"] == "win":
            wins += 1
            state["loss_streak"] = 0
            sequence["ending"] = "win"
            if not flat:
                state.update(step=0, losses=0., sequence=None)
        elif offer["result"] == "loss":
            losses += 1
            state["loss_streak"] += 1
            longest_loss = max(longest_loss, state["loss_streak"])
            if flat:
                sequence["ending"] = "loss"
            else:
                state["losses"] += stake
                state["step"] += 1
                if state["step"] >= cap:
                    sequence.update(ending="progression_exhausted", failed=True)
                    state.update(step=0, losses=0., sequence=None)
        else:
            pushes += 1
            if flat:
                sequence["ending"] = "push"
            # Push returns stake, preserves step/losses, and never counts as a loss.
        if keep_ledger:
            ledger.append({"event_id": offer["event_id"], "signal_id": offer.get("signal_id"), "season": offer["season"],
                           "team": lane[1], "sequence_id": sequence["sequence_id"], "signal_timestamp": offer["signal_timestamp"],
                           "settled_at": iso(timestamp), "price": offer["price"], "result": offer["result"],
                           "stake": stake, "net_units": profit, "equity": equity, "cash_delta": cash_delta,
                           "outstanding_stakes": max(0., outstanding), "drawdown": peak-equity})
    failures = sum(s["failed"] for s in sequences)
    summary = {"strategy": strategy if isinstance(strategy, str) else "custom", "scope": scope,
               "number_of_sequences": len(sequences), "bets": bets, "wins": wins, "losses": losses, "pushes": pushes,
               "win_rate": wins/(wins+losses) if wins+losses else None, "net_units": equity if bets else None,
               "roi": equity/turnover if turnover else None, "total_staked": turnover,
               "average_stake": turnover/bets if bets else None, "maximum_single_stake": max_single if bets else None,
               "maximum_sequence_exposure": max_exposure if bets else None, "maximum_outstanding_stakes": max_outstanding if bets else None,
               "maximum_requested_stake_including_rejected": max_requested if max_requested else None,
               "maximum_requested_sequence_exposure_including_rejected": max_requested_exposure if max_requested_exposure else None,
               "maximum_drawdown": drawdown if bets else None, "longest_losing_sequence": longest_loss if bets else None,
               "historical_bankroll_requirement": required if bets else None, "progression_failures": failures,
               "stake_cap_failures": stake_caps, "unaffordable_bets": unaffordable,
               "sequences_ending_win_with_net_loss": sum(s["ending"] == "win" and s["net_units"] < -1e-9 for s in sequences),
               "target_profit_shortfalls": sum(s["ending"] == "win" and s["net_units"] < target-1e-9 for s in sequences),
               "right_censored_sequences": sum(s["ending"] == "right_censored" for s in sequences),
               "skipped_overlapping_qualifying_offers": overlaps, "unknown_qualifying_outcomes": unknown,
               "censored_after_unknown_qualifying_outcome": censored_skips,
               "bankroll": bankroll, "operational_ruin": unaffordable > 0,
               "tail_risk_is_not_a_live_bet_recommendation": True}
    return summary, ledger, sequences


def bootstrap_offers(offers, rng, block_games):
    """Resample contiguous offer blocks within season, retaining price/outcome pairs."""
    seasons = defaultdict(list)
    for offer in offers:
        if offer.get("qualifies", True) and offer.get("result") in {"win", "loss", "push"}:
            seasons[offer["season"]].append(offer)
    output = []
    cursor = datetime(2000, 1, 1, tzinfo=timezone.utc)
    for season, rows in sorted(seasons.items()):
        rows.sort(key=lambda r: (r["signal_timestamp"], r["event_id"]))
        count = 0
        while count < len(rows):
            start = rng.randrange(len(rows))
            block = rows[start:min(start+block_games, len(rows))]
            block = block[:len(rows)-count]
            origin = parse_time(block[0]["signal_timestamp"])
            latest = cursor
            for offer in block:
                signal = cursor + (parse_time(offer["signal_timestamp"])-origin)
                settled = signal + (parse_time(offer["result_available_at"])-parse_time(offer["signal_timestamp"]))
                latest = max(latest, settled)
                row = dict(offer, event_id=f"bootstrap_{len(output):08d}", signal_timestamp=iso(signal),
                           result_available_at=iso(settled), commence_time=iso(signal+timedelta(minutes=5)))
                output.append(row)
            count += len(block)
            cursor = latest+timedelta(days=1)
    return output


def bootstrap_ruin(offers, strategy, scope, config):
    graded = [r for r in offers if r.get("qualifies", True) and r.get("result") in {"win", "loss", "push"}]
    if not graded:
        return [{"bankroll": b, "operational_ruin_probability": None, "repetitions": 0,
                 "method": "no_graded_data"} for b in config["bankrolls"]]
    rng = random.Random(config["bootstrap_seed"])
    requirements = []
    for _ in range(config["bootstrap_repetitions"]):
        sampled = bootstrap_offers(graded, rng, config["bootstrap_block_games"])
        summary, _, _ = simulate(sampled, strategy, scope, config["target_profit"], config["max_progression_stake"], keep_ledger=False)
        requirements.append(summary["historical_bankroll_requirement"] or 0.)
    result = []
    for bankroll in config["bankrolls"]:
        failed = sum(r > bankroll+1e-9 for r in requirements)
        n = len(requirements)
        result.append({"bankroll": bankroll, "operational_ruin_probability": failed/n if n else None,
                       "ruin_definition": "insufficient liquid cash to fund the next required stake",
                       "repetitions": n, "horizon_graded_offers": len(graded), "seed": config["bootstrap_seed"],
                       "block_games": config["bootstrap_block_games"],
                       "method": "within_season_contiguous_offer_block_bootstrap_replaying_stakes",
                       "assumptions": "blocks serialized; price/outcome pair and within-block overlap retained; exploratory, not a forecast",
                       "bootstrap_max_bankroll_requirement": max(requirements, default=None)})
    return result
