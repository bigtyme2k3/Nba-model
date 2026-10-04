"""Qualifying sequences, censored runs, and next-event conditioning without leakage."""
from __future__ import annotations

from collections import defaultdict

from .core import flat_profit, parse_time, wilson


def run_statistics(records):
    """Nonqualifying games are skipped; unknown outcomes/pushes break the run."""
    runs, next_events = [], []
    length, available = 0, None
    for record in records:
        if not record.get("qualifies", True):
            continue
        value = record.get("value")
        if length:
            known = available is not None and parse_time(available) < parse_time(record["signal_timestamp"])
            next_events.append(dict(record, previous_streak=length, prior_results_available=known))
        if value is True:
            length += 1
            settled = record.get("result_available_at")
            available = max(available, settled) if available and settled else settled
        else:
            if length:
                runs.append({"length": length, "right_censored": value is None})
            length, available = 0, None
    if length:
        runs.append({"length": length, "right_censored": True})
    return runs, next_events


def metric_specs():
    return [
        ("underdog_outright_wins", "h2h", "FADE", "win"),
        ("favorite_outright_losses", "h2h", "FOLLOW", "loss"),
        ("favorite_outright_wins", "h2h", "FOLLOW", "win"),
        ("underdog_ATS_covers", "spreads", "FADE", "win"),
        ("favorite_ATS_failures", "spreads", "FOLLOW", "loss"),
        ("Overs", "totals", "FOLLOW", "win"),
        ("Unders", "totals", "FADE", "win"),
        ("OPEN_model_failures", "h2h", "FOLLOW", "loss"),
        ("CLOSE_model_failures", "h2h", "FOLLOW", "loss"),
        ("fade_losses_ML", "h2h", "FADE", "loss"),
        ("fade_losses_ATS", "spreads", "FADE", "loss"),
        ("fade_losses_total", "totals", "FADE", "loss"),
    ]


def streak_tables(bets, config):
    consensus_bets = [b for b in bets if b["basis"] == "CONSENSUS"]
    opposite = {(b["event_id"], b["snapshot_type"], b["market"], b["mode"]): b for b in consensus_bets}
    distributions, mean_reversion, detail = [], [], []
    for metric, market, mode, wanted in metric_specs():
        candidates = [b for b in consensus_bets if b["market"] == market and b["mode"] == mode]
        if metric.startswith("OPEN_"):
            candidates = [b for b in candidates if b["snapshot_type"] == "EARLY"]
        elif metric.startswith("CLOSE_"):
            candidates = [b for b in candidates if b["snapshot_type"] == "CLOSE"]
        for scope in ("league", "same_team"):
            streams = defaultdict(list)
            for bet in candidates:
                result = bet.get("reference_result")
                value = result == wanted if result in {"win", "loss"} else None
                record = dict(bet, value=value)
                record["qualifies"] = bet["early_event"] if scope == "league" else bet["qualifies"]
                if scope == "same_team" and market == "totals":
                    for side in ("home", "away"):
                        if bet[f"team_game_number_{side}"] <= 20:
                            streams[(bet["season"], bet["snapshot_type"], bet[f"{side}_team"])].append(dict(record, team=bet[f"{side}_team"], qualifies=True))
                else:
                    team = bet["team"] if scope == "same_team" else "league"
                    streams[(bet["season"], bet["snapshot_type"], team)].append(record)
            grouped_runs, grouped_next, grouped_records = defaultdict(list), defaultdict(list), defaultdict(list)
            for (season, snapshot, team), records in streams.items():
                records.sort(key=lambda r: (r["commence_time"], r["event_id"]))
                runs, next_events = run_statistics(records)
                for run in runs:
                    detail.append(dict(run, season=season, snapshot_type=snapshot, scope=scope, metric=metric, team=team))
                for label in (season, "ALL"):
                    group = (label, snapshot)
                    grouped_runs[group].extend(runs)
                    grouped_next[group].extend(next_events)
                    grouped_records[group].extend(r for r in records if r["qualifies"])
            for (season, snapshot), runs in sorted(grouped_runs.items()):
                observations = grouped_records[(season, snapshot)]
                known_count = sum(r["value"] is not None for r in observations)
                meta = {"season": season, "snapshot_type": snapshot, "scope": scope, "metric": metric,
                        "season_boundaries_reset": True, "game_number_status": "observed_catalog_not_independently_certified",
                        "qualifying_events": len(observations), "graded_directional_events": known_count,
                        "unknown_or_push_events": len(observations)-known_count}
                for size in (1, 2, 3, 4, 5, "6+"):
                    matched = [r for r in runs if (r["length"] >= 6 if size == "6+" else r["length"] == size)]
                    distributions.append(dict(meta, streak_length=size, runs=len(matched),
                                              completed_runs=sum(not r["right_censored"] for r in matched),
                                              censored_runs=sum(r["right_censored"] for r in matched),
                                              maximum_historical_streak=max((r["length"] for r in runs), default=0) if known_count else None))
                baseline = [r for r in observations if r["value"] is not None]
                baseline_rate = sum(r["value"] for r in baseline)/len(baseline) if baseline else None
                for size in (1, 2, 3, 4, "5+"):
                    matched = [r for r in grouped_next[(season, snapshot)] if (r["previous_streak"] >= 5 if size == "5+" else r["previous_streak"] == size)]
                    usable = [r for r in matched if r["value"] is not None and r["prior_results_available"]]
                    n, wins = len(usable), sum(r["value"] for r in usable)
                    low, high = wilson(wins, n)
                    continuing, reversing = [], []
                    for record in usable:
                        other = opposite.get((record["event_id"], snapshot, market, "FOLLOW" if mode == "FADE" else "FADE"))
                        # A metric that describes failures continues by betting the opposite side.
                        same, reverse = (record, other) if wanted == "win" else (other, record)
                        if same and same.get("net_units") is not None:
                            continuing.append(same["net_units"])
                        if reverse and reverse.get("net_units") is not None:
                            reversing.append(reverse["net_units"])
                    mean_reversion.append(dict(meta, prior_streak=size, next_qualifying_events=n,
                                               continued=wins, reverted=n-wins, continuation_rate=wins/n if n else None,
                                               unconditional_rate=baseline_rate,
                                               continuation_minus_unconditional=wins/n-baseline_rate if n and baseline_rate is not None else None,
                                               win_rate_ci_low=low, win_rate_ci_high=high,
                                               continuation_flat_roi=sum(continuing)/len(continuing) if continuing else None,
                                               reversion_flat_roi=sum(reversing)/len(reversing) if reversing else None,
                                               skipped_prior_result_not_available=sum(not r["prior_results_available"] for r in matched),
                                               skipped_unknown_or_push=sum(r["value"] is None for r in matched),
                                               small_sample=n < config["small_sample_threshold"]))
    return distributions, mean_reversion, detail


def qualify_after_failures(bets, scope, threshold):
    """Freeze a trigger using settled market failures before the offered snapshot."""
    if threshold == 0:
        return [dict(b, qualifies=(b["early_event"] if scope == "league" else b["qualifies"])) for b in bets]
    lanes = defaultdict(list)
    for bet in bets:
        key = (bet["season"], "league" if scope == "league" else bet.get("sequence_team") or bet.get("team") or "totals")
        lanes[key].append(bet)
    qualified = []
    for records in lanes.values():
        records.sort(key=lambda b: (b["commence_time"], b["event_id"]))
        run, available = 0, None
        for bet in records:
            base = bet["early_event"] if scope == "league" else bet["qualifies"]
            if not base:
                continue
            known = available is not None and parse_time(available) < parse_time(bet["signal_timestamp"])
            row = dict(bet, qualifies=known and run >= threshold, trigger_prior_failures=run,
                       trigger_prior_available_at=available)
            qualified.append(row)
            result = bet.get("market_follow_reference_result")
            if result == "loss":
                run += 1
                settled = bet.get("result_available_at")
                available = max(available, settled) if available and settled else settled
            else:
                run, available = 0, None
    return qualified
