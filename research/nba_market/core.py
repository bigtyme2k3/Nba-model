"""Pure market math, conservative snapshot selection, and grading contracts."""
from __future__ import annotations

import hashlib
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from teams import TEAM_ABBR, TEAM_NAMES

UTC = timezone.utc
ET = ZoneInfo("America/New_York")
MARKET_FIELDS = {
    "h2h": ("home_ml", "away_ml"),
    "spreads": ("home_spread", "home_spread_price", "away_spread", "away_spread_price"),
    "totals": ("total", "over_price", "under_price"),
}
BOOK_FIELDS = [field for fields in MARKET_FIELDS.values() for field in fields]
CONFIG_PATH = Path(__file__).with_name("config.json")


def load_config(path=None):
    return json.loads(Path(path or CONFIG_PATH).read_text())


def parse_time(value):
    dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if dt.tzinfo is None:
        raise ValueError("Timestamp must include a timezone")
    return dt.astimezone(UTC)


def iso(value):
    dt = parse_time(value) if isinstance(value, str) else value.astimezone(UTC)
    return dt.isoformat(timespec="seconds").replace("+00:00", "Z")


def game_date(value):
    return parse_time(value).astimezone(ET).date().isoformat()


def normalize_team(value):
    aliases = {name.casefold(): name for name in TEAM_NAMES}
    aliases.update({abbr.casefold(): name for name, abbr in TEAM_ABBR.items()})
    aliases.update({"los angeles clippers": "LA Clippers", "la lakers": "Los Angeles Lakers"})
    key = " ".join(str(value or "").split()).casefold()
    if key not in aliases:
        raise ValueError(f"Unknown NBA team: {value!r}")
    return aliases[key]


def season_for(value, config):
    day = game_date(value)
    for season, spec in config["seasons"].items():
        if spec["start"] <= day <= spec["end"] and day not in spec.get("excluded_dates", []):
            return season
    return None


def early_bucket(number):
    if number is None or number < 1:
        return "unknown"
    if number > 20:
        return "Games 21+"
    start = ((number - 1) // 5) * 5 + 1
    return f"Games {start}-{start + 4}"


def number_games(events):
    """Numbers the observed regular-season catalog; completeness is separate evidence."""
    counts, seen, numbered = {}, set(), []
    for event in sorted(events, key=lambda e: (e["commence_time"], e["event_id"])):
        if event["event_id"] in seen:
            continue
        seen.add(event["event_id"])
        row = dict(event)
        for side in ("home", "away"):
            key = (event["season"], event[f"{side}_team"])
            counts[key] = counts.get(key, 0) + 1
            row[f"team_game_number_{side}"] = counts[key]
            row[f"early_bucket_{side}"] = early_bucket(counts[key])
        numbered.append(row)
    return numbered


def american(value):
    value = float(value)
    if not math.isfinite(value) or abs(value) < 100 or abs(value) > 100000:
        raise ValueError("Malformed American odds")
    return value


def payout(odds):
    odds = american(odds)
    return odds / 100 if odds > 0 else 100 / -odds


def implied(odds):
    return 1 / (1 + payout(odds))


def probability_to_american(prob):
    if not 0 < prob < 1:
        raise ValueError("Probability must be strictly between zero and one")
    return 100 * (1 - prob) / prob if prob <= .5 else -100 * prob / (1 - prob)


def no_vig(home, away):
    h, a = implied(home), implied(away)
    return h / (h + a), a / (h + a)


def favorite(home_prob, tolerance=1e-10):
    if home_prob is None or abs(home_prob - .5) <= tolerance:
        return None
    return "home" if home_prob > .5 else "away"


def price_band(price, bands):
    if price is None or price < 100:
        return "negative-priced underdog / outside bands"
    for low, high in bands:
        if low <= price and (high is None or price < high):
            return f"+{low}+" if high is None else f"+{low} to +{high-1}"
    return "outside bands"


def grade_margin(value):
    return "win" if value > 1e-8 else "loss" if value < -1e-8 else "push"


def grade_ml(home_score, away_score, side):
    return grade_margin((home_score - away_score) * (1 if side == "home" else -1))


def grade_spread(home_score, away_score, side, line):
    margin = home_score - away_score if side == "home" else away_score - home_score
    return grade_margin(margin + line)


def grade_total(home_score, away_score, side, line):
    return grade_margin((home_score + away_score - line) * (1 if side == "over" else -1))


def flat_profit(result, odds, stake=1):
    if result == "win":
        return stake * payout(odds)
    if result == "loss":
        return -stake
    if result == "push":
        return 0.0
    return None


def inverse(result):
    return {"win": "loss", "loss": "win", "push": "push"}.get(result)


def fingerprint(value):
    digest = hashlib.sha256()
    for chunk in json.JSONEncoder(sort_keys=True, separators=(",", ":")).iterencode(value):
        digest.update(chunk.encode())
    return digest.hexdigest()


def normalize_event(event, timestamp, config):
    """Reject all post-tip snapshots, including inconsistent market update timestamps."""
    if event.get("sport_key") != "basketball_nba" or not event.get("id"):
        raise ValueError("Not an identified NBA event")
    home, away = normalize_team(event["home_team"]), normalize_team(event["away_team"])
    if home == away:
        raise ValueError("Identical opponents")
    tip, snapshot = parse_time(event["commence_time"]), parse_time(timestamp)
    season = season_for(event["commence_time"], config)
    if season is None:
        return None, [], ["outside_regular_season"]
    base = {"event_id": event["id"], "season": season, "game_date": game_date(iso(tip)),
            "commence_time": iso(tip), "home_team": home, "away_team": away}
    if snapshot >= tip:
        return base, [], ["post_tip_snapshot"]
    records, errors = [], []
    for book in event.get("bookmakers", []):
        key = book.get("key")
        if not key:
            errors.append("missing_bookmaker")
            continue
        row = dict(base, bookmaker=key, snapshot_timestamp=iso(snapshot), **dict.fromkeys(BOOK_FIELDS))
        for market in book.get("markets", []):
            name = market.get("key")
            if name not in MARKET_FIELDS:
                continue
            try:
                update = parse_time(market.get("last_update") or book.get("last_update"))
                if update > snapshot or update >= tip:
                    raise ValueError("future_market_update")
                if (snapshot - update).total_seconds() > config["market_max_age_minutes"] * 60:
                    raise ValueError("stale_market_update")
                raw = market.get("outcomes", [])
                if len(raw) != 2:
                    raise ValueError("not_two_sided")
                pairs = {}
                for outcome in raw:
                    label = outcome["name"].lower() if name == "totals" else normalize_team(outcome["name"])
                    if label in pairs:
                        raise ValueError("duplicate_outcome")
                    pairs[label] = outcome
                left, right = ("over", "under") if name == "totals" else (home, away)
                p, q = american(pairs[left]["price"]), american(pairs[right]["price"])
                if not .80 <= implied(p) + implied(q) <= 1.40:
                    raise ValueError("implausible_overround")
                if name == "h2h":
                    row.update(home_ml=p, away_ml=q)
                elif name == "spreads":
                    h, a = float(pairs[left]["point"]), float(pairs[right]["point"])
                    if not math.isfinite(h + a) or abs(h + a) > 1e-8 or abs(h) > 50:
                        raise ValueError("malformed_spread_pair")
                    row.update(home_spread=h, away_spread=a, home_spread_price=p, away_spread_price=q)
                else:
                    t, u = float(pairs[left]["point"]), float(pairs[right]["point"])
                    if not math.isfinite(t + u) or t != u or not 100 <= t <= 400:
                        raise ValueError("malformed_total_pair")
                    row.update(total=t, over_price=p, under_price=q)
                row[f"{name}_last_update"] = iso(update)
            except (KeyError, TypeError, ValueError) as exc:
                errors.append(f"{key}:{name}:{exc}")
        if any(row[f] is not None for f in BOOK_FIELDS):
            records.append(row)
    return base, records, errors


def consensus(rows, minimum=2):
    """Equal book weights, pairwise no-vig ML; median spread and total references."""
    out = {"book_count": len({r["bookmaker"] for r in rows})}
    for market, fields in MARKET_FIELDS.items():
        good = {r["bookmaker"]: r for r in rows if all(r.get(f) is not None for f in fields)}
        values = list(good.values())
        out[f"{market}_books"] = len(values)
        if len(values) < minimum:
            continue
        if market == "h2h":
            probs = [no_vig(r["home_ml"], r["away_ml"])[0] for r in values]
            out.update(home_no_vig=statistics.mean(probs), away_no_vig=1-statistics.mean(probs),
                       home_implied=statistics.mean(implied(r["home_ml"]) for r in values),
                       away_implied=statistics.mean(implied(r["away_ml"]) for r in values),
                       ml_probability_range=max(probs)-min(probs),
                       ml_probability_std=statistics.pstdev(probs),
                       different_favorites=len({favorite(p) for p in probs if favorite(p)}) > 1)
            out["home_ml"] = probability_to_american(out["home_implied"])
            out["away_ml"] = probability_to_american(out["away_implied"])
            out["favorite"] = favorite(out["home_no_vig"])
        else:
            field = "home_spread" if market == "spreads" else "total"
            lines = [r[field] for r in values]
            out[field] = statistics.median(lines)
            out[f"{market}_range"] = max(lines)-min(lines)
            out[f"{market}_std"] = statistics.pstdev(lines)
            if market == "spreads":
                out["away_spread"] = -out[field]
                out["spread_favorite"] = "home" if out[field] < 0 else "away" if out[field] > 0 else None
    return out


def select_snapshots(event, groups, config):
    """Market-specific EARLY and CLOSE; absent close stays absent, not a late proxy."""
    tip = parse_time(event["commence_time"])
    selected = {}
    for market, fields in MARKET_FIELDS.items():
        eligible = []
        for timestamp, rows in groups.items():
            dt = parse_time(timestamp)
            if dt >= tip or (tip - dt).total_seconds() < config["close_buffer_seconds"]:
                continue
            valid = []
            for row in rows:
                update = row.get(f"{market}_last_update")
                if not all(row.get(f) is not None for f in fields) or not update:
                    continue
                upd = parse_time(update)
                if upd > dt or (dt-upd).total_seconds() > config["market_max_age_minutes"] * 60:
                    continue
                valid.append(row)
            if len({r["bookmaker"] for r in valid}) >= config["minimum_consensus_books"]:
                eligible.append((dt, valid))
        if not eligible:
            continue
        eligible.sort(key=lambda x: x[0])
        early = eligible[0]
        selected[("EARLY", market)] = (iso(early[0]), early[1],
                                      "earliest_observed_not_true_open" if (tip-early[0]).total_seconds() >= 21600
                                      else "late_earliest_observed_not_true_open")
        closing_candidates = []
        for dt, valid in eligible:
            if (tip-dt).total_seconds() > config["close_max_age_minutes"]*60:
                continue
            clean = [r for r in valid if (dt-parse_time(r[f"{market}_last_update"])).total_seconds() <= config.get("close_market_max_age_minutes", 15)*60]
            if len({r["bookmaker"] for r in clean}) >= config["minimum_consensus_books"]:
                closing_candidates.append((dt, clean))
        if closing_candidates:
            closing = closing_candidates[-1]
            selected[("CLOSE", market)] = (iso(closing[0]), closing[1], "final_observed_clean_pretip")
    return selected


def execution_book(rows, market, reference, priority):
    field = "home_spread" if market == "spreads" else "total" if market == "totals" else None
    return min(rows, key=lambda r: (abs(r[field]-reference) if field else 0,
                                   priority.index(r["bookmaker"]) if r["bookmaker"] in priority else len(priority),
                                   r["bookmaker"]))


def wilson(wins, n):
    if not n:
        return None, None
    z, p = 1.959963984540054, wins / n
    den = 1 + z*z/n
    center = (p + z*z/(2*n))/den
    half = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/den
    return max(0, center-half), min(1, center+half)
