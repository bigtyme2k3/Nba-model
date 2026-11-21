"""Read-only rebound-Over fade research from immutable NBA Consistency quotes.

This module does not create wagers, alter active picks, or infer missing prices.
The rule was fixed on 2026-11-21 after reviewing 2026-11-19/20 outcomes.
Anything earlier is a discovery sample; 2026-11-21 onward is shadow evidence.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path

SIGNALS = Path("data/history/nba_consistency_signals.jsonl")
PLAYER_BOX = Path("data/raw/espn_current_player_box_2026.csv")
OUT = Path("artifacts/rebound_fade_shadow")
FROZEN_FROM = "2026-11-21"
BOOKS = ("fanduel", "draftkings", "fanatics")
MAX_QUOTE_AGE_SECONDS = 600


def timestamp(value):
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return parsed.astimezone(timezone.utc) if parsed.tzinfo else None
    except (TypeError, ValueError, OverflowError):
        return None


def number(value):
    try:
        v = float(value)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def american_profit(price):
    price = number(price)
    if price is None or abs(price) < 100:
        raise ValueError("not an American price")
    return price / 100 if price > 0 else 100 / (-price)


def implied(price):
    price = number(price)
    if price is None or abs(price) < 100:
        return None
    return (100 / (price + 100)) if price > 0 else (-price) / (-price + 100)


def read_signals(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_player_box(path):
    """Match the Consistency grader's latest player/date row behavior."""
    index = {}
    with Path(path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("game_date") and row.get("player_id"):
                index[(row["game_date"], str(row["player_id"]))] = row
    return index


def latest_pretip_signals(signals):
    """Pre-register REB Over, keep one latest pre-tip signal per player/date."""
    latest = {}
    for s in signals:
        if (s.get("kind"), s.get("phase"), s.get("category"), s.get("side")) != (
            "LINE", "PRETIP", "REB", "OVER"
        ):
            continue
        if not s.get("date") or s.get("player_id") is None:
            continue
        key = (s["date"], str(s["player_id"]))
        stamp = timestamp(s.get("recorded_at_utc"))
        if key not in latest or (stamp or datetime.min.replace(tzinfo=timezone.utc)) > (
            timestamp(latest[key].get("recorded_at_utc"))
            or datetime.min.replace(tzinfo=timezone.utc)
        ):
            latest[key] = s
    return [latest[k] for k in sorted(latest)]


def quote_selection(s):
    """Use only pre-tip, fresh, same-line sportsbook Under offers."""
    recorded = timestamp(s.get("recorded_at_utc"))
    captured = timestamp(s.get("quote_scraped_at"))
    tip = timestamp(s.get("tip"))
    if not (recorded and captured and tip):
        return None, "NO_VERIFIABLE_TIMESTAMPS"
    age = (recorded - captured).total_seconds()
    if not (0 <= age <= MAX_QUOTE_AGE_SECONDS and captured < tip and recorded < tip):
        return None, "STALE_OR_POST_TIP_QUOTE"
    line = number(s.get("line"))
    if line is None:
        return None, "NO_EXACT_LINE"
    offers = []
    for book in BOOKS:
        b = (s.get("books") or {}).get(book) or {}
        odd = number(b.get("under"))
        if number(b.get("line")) != line or odd is None or abs(odd) < 100:
            continue
        over = number(b.get("over"))
        p_over, p_under = implied(over), implied(odd)
        no_vig = (
            p_under / (p_over + p_under)
            if p_over is not None and p_under is not None else None
        )
        offers.append({"book": book, "odds": odd, "no_vig_under": no_vig})
    if not offers:
        return None, "NO_SAME_LINE_UNDER_QUOTE"
    # Max American odds is always the most favorable quote at the same line.
    best = max(offers, key=lambda item: item["odds"])
    best["minutes_to_tip"] = round((tip - captured).total_seconds() / 60, 2)
    best["quote_scraped_at"] = s["quote_scraped_at"]
    return best, None


def grade(s, box_row):
    if box_row is None:
        return "PENDING", None
    minutes = number(box_row.get("minutes"))
    if str(box_row.get("did_not_play", "")).strip().lower() == "true" or (
        minutes is not None and minutes <= 0
    ):
        return "VOID_DNP", None
    actual = number(box_row.get("reb"))
    line = number(s.get("line"))
    if actual is None or line is None or minutes is None:
        return "MISSING_ACTUAL", actual
    if actual == line:
        return "PUSH", actual
    # A fade of an Over is an Under at exactly the same line.
    return ("WIN" if actual < line else "LOSS"), actual


def build_rows(signals, player_box, frozen_from=FROZEN_FROM):
    result = []
    for s in latest_pretip_signals(signals):
        quote, reason = quote_selection(s)
        box = player_box.get((s["date"], str(s["player_id"])))
        outcome, actual = grade(s, box)
        period = "PROSPECTIVE_SHADOW" if s["date"] >= frozen_from else "DISCOVERY"
        row = {
            "date": s["date"], "period": period, "signal_key": s.get("signal_key"),
            "player_id": s["player_id"], "player": s.get("player"),
            "category": "REB", "original_side": "OVER", "fade_side": "UNDER",
            "line": number(s.get("line")), "recommendation": s.get("recommendation"),
            "version": s.get("version"), "game_event_id": s.get("event_id"),
            "snapshot_at_utc": s.get("recorded_at_utc"),
            "tip_at_utc": s.get("tip"), "quote_scraped_at": s.get("quote_scraped_at"),
            "price_status": "MATCHED" if quote else reason,
            "fade_book": quote["book"] if quote else None,
            "fade_odds": quote["odds"] if quote else None,
            "no_vig_under_same_book": quote["no_vig_under"] if quote else None,
            "minutes_to_tip": quote["minutes_to_tip"] if quote else None,
            "actual_rebounds": actual,
            "outcome": outcome,
            "profit_units": (
                round(american_profit(quote["odds"]), 8) if quote and outcome == "WIN"
                else -1 if quote and outcome == "LOSS"
                else 0 if quote and outcome == "PUSH"
                else None
            ),
            "is_real_wager": False,
        }
        result.append(row)
    return result


def summarize(rows):
    graded = [r for r in rows if r["price_status"] == "MATCHED" and r["outcome"] in ("WIN", "LOSS", "PUSH")]
    wins = sum(r["outcome"] == "WIN" for r in graded)
    losses = sum(r["outcome"] == "LOSS" for r in graded)
    pushes = sum(r["outcome"] == "PUSH" for r in graded)
    days = sorted({r["date"] for r in graded})
    units = sum(r["profit_units"] for r in graded)
    return {
        "candidates": len(rows), "priced": sum(r["price_status"] == "MATCHED" for r in rows),
        "unpriced_or_invalid": sum(r["price_status"] != "MATCHED" for r in rows),
        "pending": sum(r["price_status"] == "MATCHED" and r["outcome"] == "PENDING" for r in rows),
        "graded": len(graded), "wins": wins, "losses": losses, "pushes": pushes,
        "hit_rate": round(wins / (wins + losses), 6) if wins + losses else None,
        "profit_units": round(units, 6),
        "roi_flat_stake": round(units / len(graded), 6) if graded else None,
        "graded_dates": len(days), "date_range": [days[0], days[-1]] if days else None,
        "status": "INSUFFICIENT_INDEPENDENT_DATES" if len(days) < 10 else "SHADOW_RESEARCH_ONLY",
        "bet_qualified": 0,
    }


def make_report(rows, frozen_from=FROZEN_FROM):
    discovery = [r for r in rows if r["period"] == "DISCOVERY"]
    forward = [r for r in rows if r["period"] == "PROSPECTIVE_SHADOW"]
    return {
        "title": "Rebound-Over fade, price matched, shadow only",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "frozen_rule_date": frozen_from,
        "frozen_rule": "Latest PRETIP Consistency LINE with REB OVER, regardless of PASS/WATCH/LEAN/BET; fade exact same line UNDER at best recorded FD/DK/Fanatics pre-tip quote.",
        "quote_policy": "SAME_LINE_AND_CAPTURE; MAX_10_MIN_QUOTE_AGE; NO_PRICE_IMPUTATION; NO_POSTTIP",
        "source_files": [str(SIGNALS), str(PLAYER_BOX)],
        "note": "Discovery is selected retrospectively; only later unseen dates can test repeatability. No live wagering, no prices invented, no bet or promotion recommendation.",
        "discovery": summarize(discovery),
        "prospective_shadow": summarize(forward),
        "by_discovery_date": {d: summarize([r for r in discovery if r["date"] == d]) for d in sorted({r["date"] for r in discovery})},
        "by_prospective_date": {d: summarize([r for r in forward if r["date"] == d]) for d in sorted({r["date"] for r in forward})},
    }


def export(rows, report, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    fields = list(rows[0]) if rows else ["date", "period", "price_status", "outcome"]
    with (out / "ledger.csv").open("w", newline="", encoding="utf-8") as handle:
        w = csv.DictWriter(handle, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    lines = [
        "# Rebound Over fade — research-only shadow audit",
        "",
        "Rule fixed on " + report["frozen_rule_date"] + ". Preceding dates are **discovery only**.",
        "",
        "Sources: immutable Consistency PRETIP LINE signals and ESPN player box scores.",
        "Opposite-side prices: only matching line and pre-tip quote on DraftKings, FanDuel, Fanatics.",
        "This is neither a betting recommendation nor evidence of positive forward EV.",
        "",
    ]
    for name in ("discovery", "prospective_shadow"):
        s = report[name]
        lines += [
            "## " + name.replace("_", " ").title(), "",
            "Candidates: " + str(s["candidates"]) + "; priced: " + str(s["priced"]) +
            "; graded: " + str(s["graded"]) + "; wins/losses: " +
            str(s["wins"]) + "/" + str(s["losses"]) + ".",
            "Observed flat-stake ROI: " + (
                f'{100 * s["roi_flat_stake"]:.2f}%' if s["roi_flat_stake"] is not None else "unavailable"
            ) + "; eligible BETs: 0 (shadow-only).",
            "Number of graded dates: " + str(s["graded_dates"]) + ".",
            "",
        ]
    lines += [
        "## Limitations", "",
        "- Historical discovery was chosen after observing poor rebound Over results.",
        "- Multiple players on the same date are dependent; per-row intervals overstate certainty.",
        "- Quotes are captured, not guaranteed fills; provider-level quote timestamps may differ.",
        "- Only evaluated signal snapshots and exact same-line quotes are included.",
        "- The prospective period must be evaluated on later dates before claiming an edge.",
        "",
    ]
    (out / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--signals", type=Path, default=SIGNALS)
    p.add_argument("--box", type=Path, default=PLAYER_BOX)
    p.add_argument("--out", type=Path, default=OUT)
    p.add_argument("--frozen-from", default=FROZEN_FROM)
    args = p.parse_args()
    datetime.strptime(args.frozen_from, "%Y-%m-%d")
    rows = build_rows(read_signals(args.signals), read_player_box(args.box), args.frozen_from)
    report = make_report(rows, args.frozen_from)
    export(rows, report, args.out)
    print(json.dumps({"discovery": report["discovery"], "prospective_shadow": report["prospective_shadow"]}, indent=2))


if __name__ == "__main__":
    main()
