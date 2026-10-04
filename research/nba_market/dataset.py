"""Build frozen pre-tip baselines and executable follow/fade offers from the warehouse."""
from __future__ import annotations

import json
from collections import defaultdict

from .core import (BOOK_FIELDS, consensus, execution_book, favorite, fingerprint, flat_profit,
                   grade_ml, grade_spread, grade_total, implied, no_vig, number_games,
                   parse_time, price_band, select_snapshots)


def build_dataset(warehouse):
    config = warehouse.config
    events = number_games(warehouse.rows("events"))
    outcomes = {r["event_id"]: r for r in warehouse.rows("outcomes")}
    groups = defaultdict(lambda: defaultdict(list))
    raw_rows = []
    numbered = {r["event_id"]: r for r in events}
    for observation in warehouse.rows("observations"):
        row = json.loads(observation["record"])
        # Earliest scheduled tip across responses is the conservative no-leakage boundary.
        event = numbered[row["event_id"]]
        if parse_time(row["snapshot_timestamp"]) >= parse_time(event["commence_time"]):
            continue
        row.update({k: event[k] for k in ("team_game_number_home", "team_game_number_away", "early_bucket_home", "early_bucket_away")})
        row["payload_hash"] = observation["payload_hash"]
        row["game_number_status"] = "observed_catalog_not_independently_certified"
        row["snapshot_type"] = "OBSERVED"
        row["source"] = "the-odds-api"
        if row.get("home_ml") is not None:
            row["home_implied"], row["away_implied"] = implied(row["home_ml"]), implied(row["away_ml"])
            row["home_no_vig"], row["away_no_vig"] = no_vig(row["home_ml"], row["away_ml"])
            fav = favorite(row["home_no_vig"])
            row["favorite"] = event[f"{fav}_team"] if fav else None
            row["underdog"] = event[f"{'away' if fav == 'home' else 'home'}_team"] if fav else None
            row["favorite_price"] = row[f"{fav}_ml"] if fav else None
            row["underdog_price"] = row[f"{'away' if fav == 'home' else 'home'}_ml"] if fav else None
        if row["event_id"] in outcomes:
            outcome = outcomes[row["event_id"]]
            row.update({k: outcome[k] for k in ("home_score", "away_score", "result_available_at")})
            row["outcome_payload_hash"] = outcome["payload_hash"]
            row["ml_home_result"] = grade_ml(row["home_score"], row["away_score"], "home")
            row["spread_home_result"] = grade_spread(row["home_score"], row["away_score"], "home", row["home_spread"]) if row.get("home_spread") is not None else None
            row["over_result"] = grade_total(row["home_score"], row["away_score"], "over", row["total"]) if row.get("total") is not None else None
        groups[row["event_id"]][row["snapshot_timestamp"]].append(row)
        raw_rows.append(row)
    views, bets = [], []
    for event in events:
        selections = select_snapshots(event, groups[event["event_id"]], config)
        for (snapshot_type, market), (timestamp, rows, quality) in sorted(selections.items()):
            bases = [("CONSENSUS", consensus(rows, config["minimum_consensus_books"]), None)]
            for row in rows:
                bases.append((row["bookmaker"], consensus([row], 1), row))
            for basis, ref, book in bases:
                if market == "h2h" and "home_no_vig" not in ref:
                    continue
                if market == "spreads" and "home_spread" not in ref:
                    continue
                if market == "totals" and "total" not in ref:
                    continue
                view = dict(event, basis=basis, market=market, snapshot_type=snapshot_type,
                            snapshot_timestamp=timestamp, snapshot_quality=quality, **ref)
                view["game_number_status"] = "observed_catalog_not_independently_certified"
                views.append(view)
                reference = ref.get("home_spread") if market == "spreads" else ref.get("total")
                offer = book or execution_book(rows, market, reference, config["book_priority"])
                if market == "h2h":
                    fav = ref.get("favorite")
                    if not fav:
                        continue  # Pick'em has no favorite or automatic opposite-side bet.
                    sides = [("FOLLOW", fav), ("FADE", "away" if fav == "home" else "home")]
                    model = "A_OPEN_FAVORITE" if snapshot_type == "EARLY" else "B_CLOSE_FAVORITE"
                elif market == "spreads":
                    fav = ref.get("spread_favorite")
                    if not fav:
                        continue
                    sides = [("FOLLOW", fav), ("FADE", "away" if fav == "home" else "home")]
                    model = "C_SPREAD_FAVORITE"
                else:
                    # A total is a reference, not an intrinsic directional sportsbook forecast.
                    sides, model = [("FOLLOW", "over"), ("FADE", "under")], "D_OVER_REFERENCE"
                outcome = outcomes.get(event["event_id"])
                for mode, side in sides:
                    row = dict(event, basis=basis, model=model, market=market, mode=mode, side=side,
                               snapshot_type=snapshot_type, snapshot_timestamp=timestamp, snapshot_quality=quality,
                               signal_timestamp=timestamp, bookmaker=offer["bookmaker"], source="the-odds-api",
                               result=None, reference_result=None, net_units=None, result_available_at=None,
                               game_number_status="observed_catalog_not_independently_certified")
                    if market == "h2h":
                        row["price"] = offer[f"{side}_ml"]
                        row["reference_line"], row["execution_line"] = None, None
                        row["no_vig_probability"] = ref[f"{side}_no_vig"]
                        row["implied_probability"] = implied(row["price"])
                        row["team"] = event[f"{side}_team"]
                        row["team_game_number"] = event[f"team_game_number_{side}"]
                        row["early_bucket"] = event[f"early_bucket_{side}"]
                        row["price_band"] = price_band(row["price"], config["underdog_price_bands"]) if mode == "FADE" else "favorite"
                    elif market == "spreads":
                        row["price"] = offer[f"{side}_spread_price"]
                        row["reference_line"], row["execution_line"] = ref[f"{side}_spread"], offer[f"{side}_spread"]
                        row["team"] = event[f"{side}_team"]
                        row["team_game_number"] = event[f"team_game_number_{side}"]
                        row["early_bucket"] = event[f"early_bucket_{side}"]
                        row["price_band"] = "not_moneyline"
                    else:
                        row["price"] = offer[f"{side}_price"]
                        row["reference_line"], row["execution_line"] = ref["total"], offer["total"]
                        row["team"], row["team_game_number"] = None, None
                        row["early_bucket"], row["price_band"] = "event_unique", "not_moneyline"
                    frozen = {k: row[k] for k in ("event_id", "basis", "model", "mode", "side", "signal_timestamp", "bookmaker", "price", "reference_line", "execution_line")}
                    row["signal_id"] = fingerprint(frozen)
                    row["early_event"] = min(event["team_game_number_home"], event["team_game_number_away"]) <= 20
                    row["qualifies"] = row["early_event"] if market == "totals" else row["team_game_number"] <= 20
                    if outcome:
                        row.update(home_score=outcome["home_score"], away_score=outcome["away_score"],
                                   result_available_at=outcome["result_available_at"], outcome_payload_hash=outcome["payload_hash"])
                        h, a = outcome["home_score"], outcome["away_score"]
                        if market == "h2h":
                            row["result"] = row["reference_result"] = grade_ml(h, a, side)
                        elif market == "spreads":
                            row["result"] = grade_spread(h, a, side, row["execution_line"])
                            row["reference_result"] = grade_spread(h, a, side, row["reference_line"])
                        else:
                            row["result"] = grade_total(h, a, side, row["execution_line"])
                            row["reference_result"] = grade_total(h, a, side, row["reference_line"])
                        row["net_units"] = flat_profit(row["result"], row["price"])
                    bets.append(row)
    selected_types = defaultdict(lambda: defaultdict(set))
    for view in views:
        if view["basis"] != "CONSENSUS":
            selected_types[(view["event_id"], view["snapshot_timestamp"], view["basis"])][view["market"]].add(view["snapshot_type"])
    for row in raw_rows:
        ref = consensus(groups[row["event_id"]][row["snapshot_timestamp"]], config["minimum_consensus_books"])
        for field in ("home_ml", "away_ml", "home_no_vig", "away_no_vig", "home_spread", "away_spread", "total"):
            row[f"consensus_{field}"] = ref.get(field)
        for field in ("ml_probability_range", "ml_probability_std", "different_favorites", "spreads_range", "totals_range"):
            row[field] = ref.get(field)
        types = selected_types[(row["event_id"], row["snapshot_timestamp"], row["bookmaker"])]
        row["snapshot_type"] = ",".join(sorted({s for values in types.values() for s in values})) or "OBSERVED"
        for market in ("h2h", "spreads", "totals"):
            row[f"{market}_snapshot_types"] = sorted(types[market])
    movement_lookup = {(m["event_id"], m["basis"], m["market"]): m for m in line_movement(views, config)}
    for row in raw_rows:
        for market in ("h2h", "spreads", "totals"):
            movement = movement_lookup.get((row["event_id"], row["bookmaker"], market), {})
            row[f"{market}_early_to_close_movement"] = movement.get("designation")
            row[f"{market}_early_to_close_change"] = movement.get({"h2h": "home_probability_change", "spreads": "home_spread_change", "totals": "total_change"}[market])
    return events, raw_rows, views, bets


def line_movement(views, config):
    grouped = defaultdict(dict)
    for view in views:
        grouped[(view["event_id"], view["basis"], view["market"])][view["snapshot_type"]] = view
    records = []
    for (event_id, basis, market), snapshots in sorted(grouped.items()):
        if set(snapshots) != {"EARLY", "CLOSE"}:
            continue
        early, close = snapshots["EARLY"], snapshots["CLOSE"]
        if early["snapshot_timestamp"] >= close["snapshot_timestamp"]:
            continue
        row = {k: early[k] for k in ("season", "game_date", "home_team", "away_team", "team_game_number_home", "team_game_number_away")}
        row.update(event_id=event_id, basis=basis, market=market,
                   early_timestamp=early["snapshot_timestamp"], close_timestamp=close["snapshot_timestamp"],
                   favorite_flipped=False, designation="little/no movement")
        if market == "h2h":
            delta = close["home_no_vig"]-early["home_no_vig"]
            row["home_probability_change"] = delta
            row["favorite_flipped"] = early.get("favorite") != close.get("favorite") and bool(early.get("favorite") and close.get("favorite"))
            if abs(delta) >= config["probability_movement_threshold"] and early.get("favorite"):
                toward_fav = delta * (1 if early["favorite"] == "home" else -1) > 0
                row["designation"] = "toward early favorite" if toward_fav else "toward early underdog"
        elif market == "spreads":
            delta = close["home_spread"]-early["home_spread"]
            row["home_spread_change"] = delta
            row["favorite_flipped"] = early.get("spread_favorite") != close.get("spread_favorite") and bool(early.get("spread_favorite") and close.get("spread_favorite"))
            if abs(delta) >= .5 and early.get("spread_favorite"):
                toward_fav = delta * (1 if early["spread_favorite"] == "home" else -1) < 0
                row["designation"] = "toward early favorite" if toward_fav else "toward early underdog"
        else:
            delta = close["total"]-early["total"]
            row["total_change"] = delta
            if abs(delta) >= config["total_movement_threshold"]:
                row["designation"] = "toward Over" if delta > 0 else "toward Under"
        records.append(row)
    return records
