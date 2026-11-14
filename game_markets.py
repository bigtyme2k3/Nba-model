"""Exact, paired sportsbook game quotes. Consensus columns are display-only."""
from __future__ import annotations

import math
from datetime import datetime, timezone

# Novig is an exchange (us_ex) with no vig; up to 10 books still cost one region.
BOOKS = {"fanduel", "draftkings", "fanatics", "novig"}
QUOTE_POLICY = "EXACT_BOOK_SIDE_V1"


def number(value):
    try:
        val = float(value)
        return val if math.isfinite(val) else None
    except (TypeError, ValueError):
        return None


def valid_price(value):
    val = number(value)
    return val is not None and abs(val) >= 100 and val.is_integer()


def timestamp(value):
    try:
        val = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return val.astimezone(timezone.utc) if val.tzinfo else None
    except (ValueError, TypeError):
        return None


def decimal(price):
    return 1 + (price / 100 if price > 0 else 100 / abs(price))


def offers_from_event(event, captured_at):
    """Retain both actual sides of each same-book two-way market."""
    offers = []
    home, away = event.get("home_team"), event.get("away_team")
    for book in event.get("bookmakers", []):
        if book.get("key") not in BOOKS:
            continue
        for market in book.get("markets", []):
            kind = {"spreads": "SPREAD", "totals": "TOTAL"}.get(market.get("key"))
            if not kind:
                continue
            names = (home, away) if kind == "SPREAD" else ("Over", "Under")
            outcomes = market.get("outcomes", [])
            if len(outcomes) != 2:
                continue
            pair = {x.get("name"): x for x in outcomes}
            if set(pair) != set(names):
                continue
            a, b = (pair[n] for n in names)
            la, lb = number(a.get("point")), number(b.get("point"))
            if la is None or lb is None or not all(valid_price(x.get("price")) for x in (a, b)):
                continue
            if abs(la + lb if kind == "SPREAD" else la - lb) > 1e-8:
                continue
            prices = [int(x["price"]) for x in (a, b)]
            implied = [1 / decimal(p) for p in prices]
            for i, outcome in enumerate((a, b)):
                offers.append({
                    "event_id": event.get("id"), "market": kind,
                    "side": (("HOME", "AWAY") if kind == "SPREAD" else ("OVER", "UNDER"))[i],
                    "line": number(outcome["point"]), "price": prices[i],
                    "book_key": book["key"], "book": book.get("title", book["key"]),
                    "no_vig_probability": implied[i] / sum(implied),
                    "last_update": market.get("last_update") or book.get("last_update"),
                    "captured_at_utc": captured_at, "commence_time": event.get("commence_time"),
                    "quote_policy": QUOTE_POLICY,
                    # Betslip link when the book supplies one (includeLinks=true).
                    "link": outcome.get("link") or market.get("link") or book.get("link"),
                })
    return offers


def select_offer(projection, market, offers, now=None, require_value=True):
    """Choose positive-EV exact quotes; never turn a synthetic consensus into a bet."""
    from kelly_sizing import load_calibration, outcome_std
    projection = number(projection)
    if projection is None:
        return None
    now = now or datetime.now(timezone.utc)
    evaluated = []
    for offer in offers:
        if offer.get("market") != market or offer.get("book_key") not in BOOKS:
            continue
        line, price = number(offer.get("line")), number(offer.get("price"))
        cap, tip, update = (timestamp(offer.get(k)) for k in
                            ("captured_at_utc", "commence_time", "last_update"))
        if (offer.get("quote_policy") != QUOTE_POLICY or line is None or not valid_price(price)
                or not cap or not tip or not update or tip <= now or update > cap
                or not 0 <= (now - cap).total_seconds() <= 1800
                or not 0 <= (cap - update).total_seconds() <= 1800):
            continue
        side = offer.get("side")
        if market == "SPREAD" and side in {"HOME", "AWAY"}:
            edge = (projection if side == "HOME" else -projection) + line
        elif market == "TOTAL" and side in {"OVER", "UNDER"}:
            edge = (projection - line) * (1 if side == "OVER" else -1)
        else:
            continue
        kind = "spread" if market == "SPREAD" else "totals"
        market_p = number(offer.get("no_vig_probability"))
        if market_p is None or not 0 < market_p < 1:
            continue
        raw = .5 * (1 + math.erf(edge / (outcome_std(kind) * math.sqrt(2))))
        shrink = load_calibration()["edge"][kind]
        prob = min(.60, market_p + shrink * (raw-market_p))
        ev = prob * decimal(price) - 1
        qualified = edge >= 2 and ev >= .025
        if qualified or not require_value:
            evaluated.append({**offer, "edge": edge, "model_probability": prob,
                              "raw_model_probability": raw, "ev": ev, "qualified": qualified})
    return max(evaluated, key=lambda x: (x["ev"], x["edge"], x["book_key"])) if evaluated else None


def priced_evidence(row):
    """Legacy prop quotes remain valid; consensus game rows stay diagnostic only."""
    book = row.get("sportsbook_key") or row.get("book_key")
    if book not in BOOKS or not valid_price(row.get("odds")):
        return False
    if str(row.get("market")).upper() == "PROP":
        return row.get("side") in {"OVER", "UNDER"}
    return (row.get("quote_policy") == QUOTE_POLICY and number(row.get("line")) is not None
            and ((row.get("market") == "SPREAD" and row.get("side") in {"HOME", "AWAY"})
                 or (row.get("market") == "TOTAL" and row.get("side") in {"OVER", "UNDER"})))
