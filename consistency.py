"""
consistency.py
--------------
Player Consistency (V2 tab): the 20 most statistically consistent players this season,
a reserve of 10, and which of their trends deserve further betting analysis.

Pregame only. Every number for date D comes from games played before D (box scores with
game_date < D), sportsbook quotes scraped before tip, and the injury feed.

Categories   PTS AST REB 3PM STL BLK PRA PA PR RA. Combined stats come from per-game
             totals, never from adding separate averages or hit rates.
Thresholds   "N+" (the stat reaches N, the same as over N-0.5). The best meaningful
             threshold is the highest N at or above the category floor that the player
             reached in 80%+ of games; failing that, the highest at 70%+; failing that,
             the floor.
Games        DNPs (did_not_play or 0 minutes) are excluded from every denominator.
             Games under LOW_MINUTES stay in and are counted as flagged low-minute games.
Score        category score = 100 x (0.50 Wilson lower bound of the season hit rate
             + 0.25 Wilson lower bound of the last-10 hit rate + 0.25 x (1 - CV)),
             so small samples and volatile stats are shrunk. Player score = mean of the
             player's three best category scores, minus 5 for each preference missed
             (28+ MPG, stable minutes, last 10 at 70%+ in the best category, 2+
             qualifying categories).
Qualified    10+ games played and at least one category at 80%+ of games at a meaningful
             threshold. Top 20 by score are the active list, the next 10 the reserve;
             unqualified players never fill empty slots. Manual overrides
             (data/consistency_overrides.json) are allowed and always marked.
Tonight      exact sportsbook lines (FanDuel, DraftKings, Fanatics; latest pregame quote
             per book) for PTS REB AST 3PM PRA. Hit counts are at the exact line; an
             alternate threshold is never substituted for it. An edge is shown only
             when the prop price calibration is validated (models/edge_calibration.json);
             without it there is no edge and no BET.
             BET   calibrated probability beats break-even by EDGE_MARGIN, all checks pass
             LEAN  all checks pass and the exact-line record is strong, not validated
             WATCH needs more information (questionable status, no projection, ...)
             PASS  fails a check
Records      data/history/nba_consistency_signals.jsonl, one immutable row per (date,
             phase, kind, player, category), written before tip with a version and the
             game's event id. kind TREND = a monitored threshold (consistency tracking,
             alternate lines, never a wager); kind LINE = an exact sportsbook line with
             its recommendation. data/history/nba_consistency_rankings.jsonl keeps each
             day's versioned qualification decisions.

Usage:
    python consistency.py build --date 2026-11-19 --phase EARLY
    python consistency.py build --date 2026-11-19 --phase PRETIP --predictions predictions/pretip_2026-11-19.json
    python consistency.py grade
Output:
    data/consistency/consistency_<date>.json, data/tracking/consistency_report.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import date as Date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from teams import normalize_team

VERSION = "consistency-v2.0"
BOX = Path("data/raw/espn_current_player_box_2026.csv")
SCHEDULE = Path("data/raw/espn_current_schedule_2026.csv")
RAW = Path("data/raw")
PRED_DIR = Path("predictions")
OUT_DIR = Path("data/consistency")
SIGNALS = Path("data/history/nba_consistency_signals.jsonl")
RANKINGS = Path("data/history/nba_consistency_rankings.jsonl")
REPORT = Path("data/tracking/consistency_report.json")
OVERRIDES = Path("data/consistency_overrides.json")
CALIBRATION = Path("models/edge_calibration.json")

CATS = {"PTS": ("pts",), "AST": ("ast",), "REB": ("reb",), "3PM": ("threes_made",), "STL": ("stl",),
        "BLK": ("blk",), "PRA": ("pts", "reb", "ast"), "PA": ("pts", "ast"), "PR": ("pts", "reb"),
        "RA": ("reb", "ast")}
COMBINED = {"PRA", "PA", "PR", "RA"}
FLOORS = {"PTS": 10, "AST": 3, "REB": 4, "3PM": 1, "STL": 1, "BLK": 1, "PRA": 15, "PA": 12, "PR": 12, "RA": 6}
PROP_STAT = {"PTS": "pts", "REB": "reb", "AST": "ast", "3PM": "threes", "PRA": "pra"}
V1_BOOKS = ("fanduel", "draftkings", "fanatics")

MIN_GP = 10
QUALIFY_RATE = 0.80
FALLBACK_RATE = 0.70
L10_PREFERRED = 0.70
MPG_PREFERRED = 28.0
MINUTES_CV_STABLE = 0.20
LOW_MINUTES = 15.0
ACTIVE_N, RESERVE_N = 20, 10
EDGE_MARGIN = 0.03
STALE_BOX_DAYS = 3
RULES = {"min_gp": MIN_GP, "qualify_rate": QUALIFY_RATE, "fallback_rate": FALLBACK_RATE,
         "l10_preferred": L10_PREFERRED, "mpg_preferred": MPG_PREFERRED, "minutes_cv_stable": MINUTES_CV_STABLE,
         "floors": FLOORS, "active": ACTIVE_N, "reserve": RESERVE_N, "edge_margin": EDGE_MARGIN,
         "low_minutes": LOW_MINUTES, "score": "0.50 wilson(season) + 0.25 wilson(l10) + 0.25 (1-cv); top-3 mean - 5/missed preference"}
RULES_ID = hashlib.sha256(json.dumps(RULES, sort_keys=True).encode()).hexdigest()[:12]
OUT_STATUSES = {"OUT", "DOUBTFUL", "SUSPENSION", "INACTIVE"}


def _name(v) -> str:
    return " ".join(str(v).lower().replace(".", "").replace("'", "").split())


def _num(v):
    try:
        v = float(v)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def wilson_lower(hits: int, n: int, z: float = 1.2816) -> float:
    if n <= 0:
        return 0.0
    p = hits / n
    centre = p + z * z / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, (centre - margin) / (1 + z * z / n))


def implied(odds) -> float | None:
    o = _num(odds)
    if o is None or o == 0:
        return None
    return 100 / (o + 100) if o > 0 else -o / (-o + 100)


# ---------------------------------------------------------------- games

def load_games(target: str, path=None) -> pd.DataFrame:
    """Every player-game before `target`, one row per (game_id, player_id), with category
    totals and game context. DNPs are kept (flagged) so callers can count them."""
    b = pd.read_csv(path or BOX, low_memory=False)
    b["game_date"] = b["game_date"].astype(str)
    b = b[b["game_date"] < str(target)].copy()
    b = b.drop_duplicates(["game_id", "player_id"], keep="last")
    for c in ("minutes", "pts", "reb", "ast", "stl", "blk", "threes_made"):
        b[c] = pd.to_numeric(b[c], errors="coerce")
    b["dnp"] = b["did_not_play"].astype(str).str.lower().eq("true") | b["minutes"].fillna(0).le(0)
    for cat, parts in CATS.items():
        b[cat] = b[list(parts)].sum(axis=1, min_count=len(parts))
    team_pts = b.groupby(["game_id", "team_abbr"])["pts"].sum().rename("team_pts").reset_index()
    opp = team_pts.merge(team_pts, on="game_id", suffixes=("", "_opp"))
    opp = opp[opp["team_abbr"] != opp["team_abbr_opp"]].rename(columns={"team_abbr_opp": "opp", "team_pts_opp": "opp_pts"})
    b = b.merge(opp, on=["game_id", "team_abbr"], how="left")
    b["margin"] = b["team_pts"] - b["opp_pts"]
    sched = b[["team_abbr", "game_date"]].drop_duplicates().sort_values(["team_abbr", "game_date"])
    sched["rest"] = sched.groupby("team_abbr")["game_date"].transform(
        lambda s: pd.to_datetime(s).diff().dt.days)
    b = b.merge(sched, on=["team_abbr", "game_date"], how="left")
    b["b2b"] = b["rest"].eq(1)
    # Opponent defense: season-to-date points allowed per game (all games before target).
    allowed = opp.groupby("team_abbr")["opp_pts"].mean().rank(ascending=True, method="min")
    b["opp_def_rank"] = b["opp"].map(allowed)
    b["player_key"] = b["player_name"].map(_name)
    return b.sort_values(["player_id", "game_date"])


def category_profile(values: np.ndarray, cat: str) -> dict:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]   # a played game with a missing stat is left out of that category only
    n = len(v)
    floor = FLOORS[cat]
    top = int(np.nanmax(v)) if n else floor
    best, qualifies = floor, False
    for rate_needed, flag in ((QUALIFY_RATE, True), (FALLBACK_RATE, False)):
        ok = [t for t in range(floor, max(top, floor) + 1) if n and (v >= t).mean() >= rate_needed]
        if ok:
            best, qualifies = max(ok), flag
            break
    hit = v >= best
    l10, l5 = hit[-10:], hit[-5:]
    streak = 0
    for h in hit[::-1]:
        if not h:
            break
        streak += 1
    mean = float(v.mean()) if n else None
    std = float(v.std(ddof=1)) if n > 1 else None
    cv = (std / mean) if (mean and std is not None and mean > 0) else 1.0
    score = 100 * (0.5 * wilson_lower(int(hit.sum()), n) + 0.25 * wilson_lower(int(l10.sum()), len(l10))
                   + 0.25 * max(0.0, 1 - min(cv, 1.0)))
    return {"threshold": best, "qualifies": bool(qualifies and n >= MIN_GP), "gp": n,
            "season_hits": int(hit.sum()), "season_rate": round(float(hit.mean()), 4) if n else None,
            "l10_hits": int(l10.sum()), "l10_n": int(len(l10)), "l10_rate": round(float(l10.mean()), 4) if len(l10) else None,
            "l5_hits": int(l5.sum()), "l5_n": int(len(l5)), "l5_rate": round(float(l5.mean()), 4) if len(l5) else None,
            "streak": streak, "mean": round(mean, 2) if mean is not None else None,
            "std": round(std, 2) if std is not None else None, "median": float(np.median(v)) if n else None,
            "l5_avg": round(float(v[-5:].mean()), 2) if n else None, "l10_avg": round(float(v[-10:].mean()), 2) if n else None,
            "cv": round(cv, 3), "score": round(score, 1)}


def player_profile(g: pd.DataFrame) -> dict:
    played = g[~g["dnp"]]
    mins = played["minutes"].astype(float)
    cats = {cat: category_profile(played[cat].to_numpy(), cat) for cat in CATS}
    mpg = float(mins.mean()) if len(mins) else 0.0
    min_cv = float(mins.std(ddof=1) / mpg) if len(mins) > 1 and mpg else 1.0
    q = [c for c, p in cats.items() if p["qualifies"]]
    ranked = sorted(cats.items(), key=lambda kv: -kv[1]["score"])
    best_cat = max(q, key=lambda c: cats[c]["score"]) if q else ranked[0][0]
    misses = []
    if mpg < MPG_PREFERRED:
        misses.append("MPG under 28")
    if min_cv > MINUTES_CV_STABLE:
        misses.append("minutes unstable")
    if (cats[best_cat]["l10_rate"] or 0) < L10_PREFERRED:
        misses.append("last 10 under 70% in best category")
    if len(q) < 2:
        misses.append("fewer than 2 qualifying categories")
    score = float(np.mean([p["score"] for _, p in ranked[:3]])) - 5 * len(misses)
    last = g.iloc[-1]
    return {"player_id": int(last["player_id"]), "player": last["player_name"], "player_key": last["player_key"],
            "team": last["team_abbr"], "team_name": f"{last['team_location']} {last['team_name']}",
            "position": last.get("position"), "gp": int(len(played)), "dnp": int(g["dnp"].sum()),
            "low_minute_games": int((mins < LOW_MINUTES).sum()), "mpg": round(mpg, 1), "minutes_cv": round(min_cv, 3),
            "last_game": str(played["game_date"].iloc[-1]) if len(played) else None,
            "qualified": bool(len(played) >= MIN_GP and q), "qualifying_categories": q, "best_category": best_cat,
            "preferences_missed": misses, "score": round(score, 1), "categories": cats}


# ---------------------------------------------------------------- deep dive

def _split(df: pd.DataFrame, mask, cat: str, threshold: float) -> list:
    """[games, average, hit rate at the category's threshold]"""
    s = df.loc[mask, cat].astype(float)
    return [int(len(s)), round(float(s.mean()), 1) if len(s) else None,
            round(float((s >= threshold).mean()), 3) if len(s) else None]


def splits(g: pd.DataFrame, all_games: pd.DataFrame, profile: dict) -> dict:
    played = g[~g["dnp"]].copy()
    # Top teammate: the teammate with the most points per game this season, with or without him.
    mates = all_games[(all_games["team_abbr"] == profile["team"]) & (all_games["player_id"] != profile["player_id"])
                      & ~all_games["dnp"]]
    mate = None
    if len(mates):
        ppg = mates.groupby(["player_id", "player_name"])["pts"].agg(["mean", "count"])
        ppg = ppg[ppg["count"] >= 5]
        if len(ppg):
            mate_id, mate_name = ppg["mean"].idxmax()
            mate = {"player": mate_name, "games": set(mates.loc[mates["player_id"] == mate_id, "game_id"])}
    order = np.arange(len(played))
    out = {}
    for cat, p in profile["categories"].items():
        t = p["threshold"]
        rows = {"Home": played["home_away"].eq("home"), "Away": played["home_away"].eq("away"),
                "Last 5": order >= len(played) - 5, "Last 10": order >= len(played) - 10,
                "Last 20": order >= len(played) - 20, "Season": np.ones(len(played), bool),
                "Rest (1+ days)": ~played["b2b"], "Back-to-back": played["b2b"],
                "vs top-10 defense": played["opp_def_rank"].le(10), "vs bottom-10 defense": played["opp_def_rank"].ge(21),
                "30+ minutes": played["minutes"].ge(30), "Under 30 minutes": played["minutes"].lt(30),
                "Blowouts (15+)": played["margin"].abs().ge(15), "Competitive": played["margin"].abs().lt(15)}
        if mate:
            with_mate = played["game_id"].isin(mate["games"])
            rows[f"With {mate['player']}"] = with_mate
            rows[f"Without {mate['player']}"] = ~with_mate
        out[cat] = {k: _split(played, np.asarray(m, bool), cat, t) for k, m in rows.items()}
    return out


def last_games(g: pd.DataFrame, n: int = 20) -> list[dict]:
    rows = []
    for r in g[~g["dnp"]].tail(n).to_dict("records"):
        rows.append({"game_id": str(r["game_id"]), "date": r["game_date"], "opp": r["opp"], "home": r["home_away"] == "home",
                     "b2b": bool(r["b2b"]), "minutes": _num(r["minutes"]),
                     "low_minutes": bool((_num(r["minutes"]) or 0) < LOW_MINUTES),
                     "margin": _num(r["margin"]), **{cat: _num(r[cat]) for cat in CATS}})
    return rows


def combined_stability(played: pd.DataFrame) -> dict:
    out = {}
    for cat in COMBINED:
        parts = CATS[cat]
        total = played[cat].astype(float)
        if len(total) < 2 or not total.mean():
            continue
        sum_sd = float(sum(played[p].astype(float).std(ddof=1) for p in parts))
        sd = float(total.std(ddof=1))
        out[cat] = {"mean": round(float(total.mean()), 2), "std": round(sd, 2), "cv": round(sd / float(total.mean()), 3),
                    "component_std_sum": round(sum_sd, 2),
                    "offset_ratio": round(sd / sum_sd, 3) if sum_sd else None}   # < 1: components offset each other
    return out


# ---------------------------------------------------------------- tonight

def tonight_context(data: dict) -> dict:
    """team (normalized full name) -> game context from a prediction file, and player props."""
    teams, players = {}, {}
    for g in data.get("games", []):
        home, away = g.get("home", {}), g.get("away", {})
        for side, me, them in (("home", home, away), ("away", away, home)):
            teams[normalize_team(me.get("name", ""))] = {
                "tip": g.get("tip"), "event_id": g.get("event_id"), "opp": them.get("abbr") or them.get("name"),
                "home": side == "home", "game": f"{away.get('name')} @ {home.get('name')}"}
        for p in g.get("props", []):
            players[_name(p.get("player"))] = p
    return {"teams": teams, "players": players}


def injury_map(target: str) -> dict:
    path = RAW / f"injuries_{target}.csv"
    if not path.exists():
        path = RAW / "injuries_today.csv"
    if not path.exists():
        return {}
    inj = pd.read_csv(path, low_memory=False)
    if "game_date" in inj and str(target) not in set(inj["game_date"].astype(str)):
        return {}
    return {_name(r.player): {"status": str(r.status), "severity": str(getattr(r, "severity", "") or ""),
                              "detail": str(getattr(r, "detail", "") or "")} for r in inj.itertuples()}


def latest_quotes(target: str, now: datetime) -> pd.DataFrame:
    path = RAW / f"sportsbook_props_{target}.csv"
    if not path.exists():
        return pd.DataFrame()
    q = pd.read_csv(path, low_memory=False)
    q = q[q["book_key"].isin(V1_BOOKS)].copy()
    scraped = pd.to_datetime(q["scraped_at"], utc=True, errors="coerce")
    tip = pd.to_datetime(q["commence_time"], utc=True, errors="coerce")
    q = q[(scraped < tip) & (scraped <= pd.Timestamp(now))]   # pregame quotes only
    q["player_key"] = q["player"].map(_name)
    last = q.groupby(["player_key", "stat", "book_key"])["scraped_at"].transform("max")
    q = q[q["scraped_at"] == last]
    # One market line per book: if a scrape ever holds several (alternates), keep the one priced closest to even.
    q = q.assign(_gap=(q["price"].abs() - 110).abs())
    main = q.groupby(["player_key", "stat", "book_key", "line"])["_gap"].mean().reset_index() \
        .sort_values("_gap").groupby(["player_key", "stat", "book_key"]).head(1)[["player_key", "stat", "book_key", "line"]]
    return q.merge(main, on=["player_key", "stat", "book_key", "line"]).drop(columns="_gap")


def exact_line(values: np.ndarray, line: float) -> dict:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    def counts(x):
        return {"n": int(len(x)), "over": int((x > line).sum()), "under": int((x < line).sum()), "push": int((x == line).sum())}
    return {"season": counts(v), "l10": counts(v[-10:]), "l5": counts(v[-5:])}


def calibration_state(path=None) -> dict:
    try:
        m = json.loads(Path(path or CALIBRATION).read_text())["markets"]["prop_price"]
    except (OSError, KeyError, ValueError):
        return {"calibrated": False, "status": "UNAVAILABLE"}
    ok = m.get("status") == "VALIDATED" and m.get("validation") == "PASSED" and (m.get("shrink") or 0) > 0
    return {"calibrated": bool(ok), "status": m.get("status"), "validation": m.get("validation"), "shrink": m.get("shrink")}


def evaluate_line(player: dict, cat: str, values: np.ndarray, quotes: pd.DataFrame, prop: dict | None,
                  injury: dict | None, calibration: dict, stale: bool) -> dict | None:
    stat = PROP_STAT.get(cat)
    if stat is None or quotes.empty:
        return None
    q = quotes[(quotes["player_key"] == player["player_key"]) & (quotes["stat"] == stat)]
    if q.empty:
        return None
    overs = q[q["side"] == "OVER"]
    if overs.empty:
        return None
    lines = overs["line"].round(1).value_counts()
    line = float(lines[lines == lines.max()].index.min()) if len(lines) else None
    books = {}
    for book in V1_BOOKS:
        bq = q[q["book_key"] == book]
        at = bq[bq["line"].round(1) == line]
        books[book] = {"line": float(bq["line"].iloc[0]) if len(bq) else None,
                       "over": int(at.loc[at["side"] == "OVER", "price"].iloc[0]) if (at["side"] == "OVER").any() else None,
                       "under": int(at.loc[at["side"] == "UNDER", "price"].iloc[0]) if (at["side"] == "UNDER").any() else None}
    hits = exact_line(values, line)
    s = hits["season"]
    side = "OVER" if s["over"] >= s["under"] else "UNDER"
    key = side.lower()
    priced = {b: v[key] for b, v in books.items() if v[key] is not None}
    best_book = max(priced, key=lambda b: priced[b]) if priced else None
    best_odds = priced.get(best_book)
    break_even = implied(best_odds)
    rate = lambda h: (h[key] / (h["n"] - h["push"])) if (h["n"] - h["push"]) > 0 else None
    pred = prob = cal_prob = None
    if prop:
        pr = (prop.get("props") or {}).get(stat) or {}
        pred = _num(pr.get("pred"))
        same_line = _num(pr.get("line")) == line
        raw = _num(pr.get("raw_model_probability"))
        if raw is not None and same_line:
            prob = raw if pr.get("offer_side") == side else 1 - raw
        mp = _num(pr.get("model_probability"))
        if calibration["calibrated"] and mp is not None and same_line:
            cal_prob = mp if pr.get("offer_side") == side else 1 - mp
    edge = round(cal_prob - break_even, 4) if (cal_prob is not None and break_even is not None) else None
    status = str((injury or {}).get("status") or prop and prop.get("injury_status") or "").upper()
    checks = {"price_available": best_odds is not None,
              "not_out": not any(s_ in status for s_ in OUT_STATUSES),
              "not_questionable": "QUESTIONABLE" not in status and "GTD" not in status and "DAY-TO-DAY" not in status,
              "no_minutes_restriction": not (prop or {}).get("minutes_guard", False),
              "fresh_data": not stale,
              "season_rate_60": (rate(s) or 0) >= 0.60,
              "l10_rate_60": (rate(hits["l10"]) or 0) >= 0.60,
              "record_beats_break_even": break_even is not None and (rate(s) or 0) >= break_even + EDGE_MARGIN,
              "projection_available": pred is not None,
              "projection_agrees": pred is not None and ((pred > line) if side == "OVER" else (pred < line))}
    hard = ("price_available", "not_out", "no_minutes_restriction", "fresh_data", "season_rate_60")
    if not all(checks[c] for c in hard) or (pred is not None and not checks["projection_agrees"]):
        rec = "PASS"
    elif all(checks.values()) and edge is not None and edge >= EDGE_MARGIN:
        rec = "BET"
    elif all(checks.values()) and (rate(s) or 0) >= 0.70 and (rate(hits["l10"]) or 0) >= 0.70:
        rec = "LEAN"
    else:
        rec = "WATCH"
    failed = [c for c, ok in checks.items() if not ok]
    return {"category": cat, "line": line, "side": side, "books": books, "best_book": best_book, "best_odds": best_odds,
            "break_even": round(break_even, 4) if break_even is not None else None, "hits": hits,
            "season_rate": round(rate(s), 4) if rate(s) is not None else None,
            "l10_rate": round(rate(hits["l10"]), 4) if rate(hits["l10"]) is not None else None,
            "l5_rate": round(rate(hits["l5"]), 4) if rate(hits["l5"]) is not None else None,
            "projection": pred, "model_probability": prob, "calibrated_probability": cal_prob, "edge": edge,
            "recommendation": rec, "failed_checks": failed, "line_spread": sorted({v["line"] for v in books.values() if v["line"] is not None})}


# ---------------------------------------------------------------- build

def box_is_stale(target: str, last_box: str | None, schedule=None) -> bool:
    """Stale when a game finished before `target` but after the last box score we hold
    (the schedule says it was played; we don't have it). Without a schedule, fall back
    to a gap of more than STALE_BOX_DAYS days."""
    if last_box is None:
        return True
    path = Path(schedule) if schedule else SCHEDULE
    if path.exists():
        s = pd.read_csv(path, low_memory=False)
        s["game_date"] = s["game_date"].astype(str)
        done = s[s["status"].astype(str).str.contains("FINAL", case=False) & (s["game_date"] < str(target))]
        return bool((done["game_date"] > str(last_box)).any())
    return (Date.fromisoformat(target) - Date.fromisoformat(last_box)).days > STALE_BOX_DAYS


def load_overrides(path=None) -> dict:
    try:
        d = json.loads(Path(path or OVERRIDES).read_text())
    except (OSError, ValueError):
        return {"include": [], "exclude": []}
    return {"include": d.get("include", []), "exclude": d.get("exclude", [])}


def rank(profiles: list[dict], overrides: dict) -> tuple[list[dict], list[dict], list[dict]]:
    excluded = {_name(o.get("player")): o.get("reason", "") for o in overrides.get("exclude", [])}
    included = {_name(o.get("player")): o.get("reason", "") for o in overrides.get("include", [])}
    removed = []
    pool = []
    for p in profiles:
        if p["player_key"] in excluded:
            removed.append({"player": p["player"], "reason": excluded[p["player_key"]], "override": "MANUAL_EXCLUDE"})
            continue
        if p["qualified"]:
            pool.append(p)
    pool.sort(key=lambda p: (-p["score"], -p["gp"], p["player"]))
    active, reserve = pool[:ACTIVE_N], pool[ACTIVE_N:ACTIVE_N + RESERVE_N]
    in_active = {p["player_key"] for p in active}
    reserve = [p for p in reserve if p["player_key"] not in included]   # an included reserve player moves up, marked
    for p in profiles:
        if p["player_key"] in included and p["player_key"] not in in_active:
            active.append({**p, "override": "MANUAL_INCLUDE", "override_reason": included[p["player_key"]]})
    for i, p in enumerate(active, 1):
        p["rank"] = i
    for i, p in enumerate(reserve, 1):
        p["rank"] = f"R{i}"
    return active, reserve, removed


def next_games(target: str) -> dict:
    """team abbr -> first scheduled date on or after target (from prediction files)."""
    out = {}
    for f in sorted(PRED_DIR.glob("predictions_*.json")):
        d = f.stem.split("_")[-1]
        if d < target:
            continue
        try:
            data = json.loads(f.read_text())
        except ValueError:
            continue
        for g in data.get("games", []):
            for t in (g.get("home", {}), g.get("away", {})):
                out.setdefault(normalize_team(t.get("name", "")), d)
    return out


def build(target: str, phase: str = "EARLY", predictions: str | None = None, now: datetime | None = None,
          box=None, record: bool = True) -> dict:
    now = now or datetime.now(timezone.utc)
    games = load_games(target, box or BOX)
    daily_path = PRED_DIR / f"predictions_{target}.json"
    pred_path = Path(predictions) if predictions else daily_path
    data = json.loads(daily_path.read_text()) if daily_path.exists() else {"games": []}
    ctx = tonight_context(data)
    window = None   # a pre-tip file covers only the games in its window: record signals for those only
    if phase == "PRETIP" and (pred_path == daily_path or not pred_path.exists()):
        window = set()   # no fresh window predictions: refresh the display, record no signals
    elif pred_path != daily_path and pred_path.exists():
        latest = json.loads(pred_path.read_text())
        win = tonight_context(latest)
        ctx["teams"].update(win["teams"])
        ctx["players"].update(win["players"])
        window = set(win["teams"])
        data = {**data, "injury_feed": latest.get("injury_feed") or data.get("injury_feed"),
                "generated": latest.get("generated") or data.get("generated")}
    injuries = injury_map(target)
    quotes = latest_quotes(target, now)
    calibration = calibration_state()
    upcoming = next_games(target)
    last_box = games["game_date"].max() if len(games) else None
    stale = box_is_stale(target, last_box)
    profiles = [player_profile(g) for _, g in games.groupby("player_id") if (~g["dnp"]).sum() > 0]
    active, reserve, removed = rank(profiles, load_overrides())
    tonight, signals = [], []
    for p in active + reserve:
        g = games[games["player_id"] == p["player_id"]]
        played = g[~g["dnp"]]
        team_full = normalize_team(p["team_name"])
        game = ctx["teams"].get(team_full)
        prop = ctx["players"].get(p["player_key"])
        inj = injuries.get(p["player_key"])
        p["injury"] = inj["status"] if inj else ((prop or {}).get("injury_status") if prop and str(prop.get("injury_status", "")).upper() in OUT_STATUSES | {"QUESTIONABLE", "PROBABLE"} else None)
        p["injury_flag"] = bool(p["injury"]) and any(s in str(p["injury"]).upper() for s in OUT_STATUSES | {"QUESTIONABLE", "DOUBTFUL", "DAY-TO-DAY"})
        p["next_game"] = ({"date": target, "tip": game["tip"], "opp": game["opp"], "home": game["home"]} if game
                          else ({"date": upcoming.get(team_full)} if upcoming.get(team_full) else None))
        p["playing_tonight"] = bool(game)
        p["projected_minutes"] = _num((prop or {}).get("proj_min"))
        p["minutes_restriction"] = bool((prop or {}).get("minutes_guard", False))
        p["last20"] = last_games(g)
        p["splits"] = splits(g, games, p)
        p["combined_stability"] = combined_stability(played)
        p["lines"] = {}
        if game:
            tip = datetime.fromisoformat(str(game["tip"]).replace("Z", "+00:00"))
            pregame = tip > now and (window is None or team_full in window)
            for cat in CATS:
                if not game.get("event_id") and len(quotes):
                    ids = quotes.loc[quotes["player_key"] == p["player_key"], "event_id"].dropna()
                    if len(ids):
                        game = {**game, "event_id": str(ids.iloc[0])}
                ev = evaluate_line(p, cat, played[cat].to_numpy(), quotes, prop, inj, calibration, stale)
                if ev:
                    p["lines"][cat] = ev
                    tonight.append({"game": game["game"], "player": p["player"], "player_id": p["player_id"], "team": p["team"],
                                    "opp": game["opp"], "tip": game["tip"], "event_id": game["event_id"],
                                    "home": game["home"], "rank": p["rank"], "projected_minutes": p["projected_minutes"],
                                    "injury": p["injury"], "injury_adjustment": (prop or {}).get("injury_status"),
                                    "opportunity_minutes_delta": _num((prop or {}).get("opportunity_minutes_delta")),
                                    "qualified_category": cat in p["qualifying_categories"], **ev})
                    if pregame:
                        signals.append(_signal(target, phase, "LINE", p, cat, game, now, ev))
                c = p["categories"][cat]
                if pregame and cat in p["qualifying_categories"] and isinstance(p["rank"], int):
                    signals.append(_signal(target, phase, "TREND", p, cat, game, now,
                                           {"threshold": c["threshold"], "season_rate": c["season_rate"],
                                            "l10_rate": c["l10_rate"], "streak": c["streak"], "recommendation": "MONITOR"}))
        statuses = [v["recommendation"] for v in p["lines"].values()]
        p["status"] = next((s for s in ("BET", "LEAN", "WATCH") if s in statuses), "PASS")
        p["status_reason"] = None if statuses else ("No game today" if not game else "No sportsbook line yet")
        best = p["lines"].get(p["best_category"]) or next(iter(p["lines"].values()), None)
        p["best_line"] = {k: best.get(k) for k in ("category", "line", "side", "best_book", "best_odds", "edge",
                                                    "model_probability", "calibrated_probability")} if best else None
        p.pop("lines")   # the full rows live in snapshot["tonight"]
    snapshot = {"date": target, "phase": phase, "version": VERSION, "rules_id": RULES_ID, "rules": RULES,
                "generated_at_utc": now.isoformat(), "predictions": str(pred_path) if pred_path.exists() else None,
                "signal_window": sorted(window) if window is not None else "full slate",
                "calibration": calibration,
                "freshness": {"last_box_score_date": last_box, "stale_box_scores": stale,
                              "props_scraped_at": quotes["scraped_at"].max() if len(quotes) else None,
                              "injury_feed": data.get("injury_feed"), "prediction_generated": data.get("generated"),
                              "missing_feeds": missing_feeds(quotes, pred_path, injuries, stale)},
                "pool": {"players_evaluated": len(profiles), "qualified": sum(p["qualified"] for p in profiles)},
                "active": active, "reserve": reserve, "removed_by_override": removed,
                "tonight": sorted(tonight, key=lambda r: (str(r["tip"]), str(r["player"]), r["category"]))}
    snapshot["kpis"] = kpis(snapshot)
    snapshot = clean(snapshot)
    signals = clean(signals)
    if record:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        (OUT_DIR / f"consistency_{target}.json").write_text(json.dumps(snapshot, default=_json, allow_nan=False) + "\n")
        n = append_signals(signals)
        append_ranking(snapshot)
        snapshot["signals_recorded"] = n
    return snapshot


def clean(o):
    """NaN/inf -> None throughout, so a missing box-score field can never break the write."""
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (float, np.floating)):
        return float(o) if math.isfinite(float(o)) else None
    return o


def _json(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o) if math.isfinite(float(o)) else None
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, set):
        return sorted(o)
    return str(o)


def missing_feeds(quotes, pred_path, injuries, stale) -> list[str]:
    out = []
    if quotes.empty:
        out.append("Sportsbook props (no pregame FanDuel/DraftKings/Fanatics quotes)")
    if not Path(pred_path).exists():
        out.append("Model predictions")
    if not injuries:
        out.append("Injury report")
    if stale:
        out.append("Box scores (stale)")
    out.append("STL, BLK, PA, PR, RA sportsbook lines (not collected)")
    return out


def _signal(target, phase, kind, p, cat, game, now, payload) -> dict:
    key = "|".join([target, phase, kind, str(p["player_id"]), cat])
    return {"signal_key": key, "date": target, "phase": phase, "kind": kind, "version": VERSION, "rules_id": RULES_ID,
            "recorded_at_utc": now.isoformat(), "player_id": p["player_id"], "player": p["player"], "team": p["team"],
            "rank": p["rank"], "category": cat, "combined": cat in COMBINED, "event_id": game.get("event_id"),
            "tip": game.get("tip"), "opp": game.get("opp"), "injury": p.get("injury"),
            **{k: v for k, v in payload.items() if k not in ("hits", "books")},
            "books": payload.get("books")}


def read_jsonl(path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def append_signals(rows: list[dict], path=None) -> int:
    path = path or SIGNALS
    have = {r["signal_key"] for r in read_jsonl(path)}
    new = [r for r in rows if r["signal_key"] not in have]
    if new:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with Path(path).open("a", encoding="utf-8") as fh:
            for r in new:
                fh.write(json.dumps(r, default=_json, allow_nan=False) + "\n")
    return len(new)


def append_ranking(snapshot: dict, path=None) -> bool:
    path = path or RANKINGS
    key = f"{snapshot['date']}|{snapshot['phase']}"
    if any(r.get("ranking_key") == key for r in read_jsonl(path)):
        return False
    row = {"ranking_key": key, "date": snapshot["date"], "phase": snapshot["phase"], "version": VERSION,
           "rules_id": RULES_ID, "recorded_at_utc": snapshot["generated_at_utc"],
           "active": [{"rank": p["rank"], "player_id": p["player_id"], "player": p["player"], "score": p["score"],
                       "qualifying_categories": p["qualifying_categories"], "override": p.get("override")}
                      for p in snapshot["active"]],
           "reserve": [{"rank": p["rank"], "player_id": p["player_id"], "player": p["player"], "score": p["score"]}
                       for p in snapshot["reserve"]],
           "removed_by_override": snapshot["removed_by_override"]}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, default=_json) + "\n")
    return True


def kpis(s: dict) -> dict:
    active = s["active"]
    report = {}
    try:
        report = json.loads(REPORT.read_text())
    except (OSError, ValueError):
        pass
    return {"active_players": len(active), "reserve_players": len(s["reserve"]),
            "qualified_trends": sum(len(p["qualifying_categories"]) for p in active),
            "active_streaks": sum(1 for p in active for c in p["qualifying_categories"] if p["categories"][c]["streak"] >= 5),
            "tonight_players": sum(p["playing_tonight"] for p in active),
            "potential_edges": sum(1 for r in s["tonight"] if r["recommendation"] in ("BET", "LEAN")),
            "bets": sum(1 for r in s["tonight"] if r["recommendation"] == "BET"),
            "results_7d": (report.get("rolling_7d") or {})}


# ---------------------------------------------------------------- grade

def grade(box=None, path=None, out=None) -> dict:
    box, path, out = box or BOX, path or SIGNALS, out or REPORT
    rows = read_jsonl(path)
    b = pd.read_csv(box, low_memory=False)
    b["game_date"] = b["game_date"].astype(str)
    b = b.drop_duplicates(["game_id", "player_id"], keep="last")
    for c in ("minutes", "pts", "reb", "ast", "stl", "blk", "threes_made"):
        b[c] = pd.to_numeric(b[c], errors="coerce")
    for cat, parts in CATS.items():
        b[cat] = b[list(parts)].sum(axis=1, min_count=len(parts))
    b["dnp"] = b["did_not_play"].astype(str).str.lower().eq("true") | b["minutes"].fillna(0).le(0)
    idx = {(r["game_date"], int(r["player_id"])): r for r in b.to_dict("records")}
    # One graded row per (date, kind, player, category): the latest pregame phase.
    latest = {}
    for r in rows:
        k = (r["date"], r["kind"], r["player_id"], r["category"])
        if k not in latest or (r["phase"] == "PRETIP" and latest[k]["phase"] != "PRETIP"):
            latest[k] = r
    graded = []
    for r in latest.values():
        g = idx.get((r["date"], int(r["player_id"])))
        if g is None:
            continue
        out_row = {k: r.get(k) for k in ("date", "phase", "kind", "player", "player_id", "team", "opp", "category",
                                         "combined", "threshold", "line", "side", "best_book", "best_odds",
                                         "recommendation", "edge", "event_id", "version", "rank")}
        out_row["game_id"] = str(g["game_id"])
        if g["dnp"]:
            out_row["outcome"] = "VOID_DNP"
            graded.append(out_row)
            continue
        actual = float(g[r["category"]])
        out_row["actual"] = actual
        out_row["minutes"] = float(g["minutes"])
        if r["kind"] == "TREND":
            out_row["outcome"] = "HIT" if actual >= r["threshold"] else "MISS"
        else:
            line = float(r["line"])
            if actual == line:
                out_row["outcome"] = "PUSH"
            else:
                won = actual > line if r["side"] == "OVER" else actual < line
                out_row["outcome"] = "WIN" if won else "LOSS"
                o = _num(r.get("best_odds"))
                if o is not None:
                    out_row["units_if_followed"] = round((o / 100 if o > 0 else 100 / -o) if won else -1.0, 3)
        graded.append(out_row)
    G = pd.DataFrame(graded)
    report = {"generated_at_utc": datetime.now(timezone.utc).isoformat(), "version": VERSION,
              "note": "TREND rows are monitored thresholds (consistency tracking, alternate lines), never wagers. "
                      "LINE rows are exact sportsbook lines; units are what following would have returned at the "
                      "recorded best price. Only BET rows are qualified selections."}
    if G.empty:
        report.update({"graded": 0, "rolling_7d": {}})
    else:
        trend = G[G["kind"] == "TREND"]
        line = G[G["kind"] == "LINE"]
        report.update({
            "graded": int(len(G)),
            "individual_trends": summarize(trend[~trend["combined"].astype(bool)]),
            "combined_trends": summarize(trend[trend["combined"].astype(bool)]),
            "alternate_thresholds": summarize(trend),
            "exact_lines": summarize(line),
            "exact_lines_individual": summarize(line[~line["combined"].astype(bool)]),
            "exact_lines_combined": summarize(line[line["combined"].astype(bool)]),
            "by_recommendation": {k: summarize(v) for k, v in line.groupby("recommendation")},
            "bet_qualified": summarize(line[line["recommendation"] == "BET"]),
            "by_category": {k: summarize(v) for k, v in G.groupby("category")},
        })
        last = sorted(G["date"].unique())[-7:]
        recent = G[G["date"].isin(last)]
        report["rolling_7d"] = {"dates": [last[0], last[-1]] if last else None,
                                "trends": summarize(recent[recent["kind"] == "TREND"]),
                                "exact_lines": summarize(recent[recent["kind"] == "LINE"]),
                                "bet_qualified": summarize(recent[(recent["kind"] == "LINE") & (recent["recommendation"] == "BET")])}
        report["history"] = G.sort_values(["date", "player", "category"], ascending=[False, True, True]) \
            .head(400).replace({np.nan: None}).to_dict("records")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(report, indent=2, default=_json) + "\n")
    return report


def summarize(df: pd.DataFrame) -> dict:
    if df is None or df.empty:
        return {"n": 0}
    hit = df["outcome"].isin(["HIT", "WIN"])
    miss = df["outcome"].isin(["MISS", "LOSS"])
    n = int(hit.sum() + miss.sum())
    out = {"n": n, "hits": int(hit.sum()), "misses": int(miss.sum()), "pushes": int(df["outcome"].eq("PUSH").sum()),
           "void_dnp": int(df["outcome"].eq("VOID_DNP").sum()), "hit_rate": round(float(hit.sum() / n), 4) if n else None}
    if "units_if_followed" in df and df["units_if_followed"].notna().any():
        u = df["units_if_followed"].dropna()
        out.update(units_if_followed=round(float(u.sum()), 2), roi_if_followed=round(float(u.mean()), 4))
    return out


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--date", required=True)
    b.add_argument("--phase", default="EARLY", choices=["EARLY", "PRETIP"])
    b.add_argument("--predictions", default=None)
    sub.add_parser("grade")
    a = ap.parse_args()
    if a.cmd == "grade":
        r = grade()
        print(json.dumps({k: r.get(k) for k in ("graded", "rolling_7d", "exact_lines", "alternate_thresholds")}, indent=2, default=_json))
        return
    s = build(a.date, a.phase, a.predictions)
    print(json.dumps({"date": s["date"], "phase": s["phase"], "kpis": s["kpis"], "pool": s["pool"],
                      "signals_recorded": s.get("signals_recorded"),
                      "active": [f"{p['rank']}. {p['player']} ({p['team']}) {p['score']} {p['qualifying_categories']} {p['status']}"
                                 for p in s["active"]],
                      "reserve": [f"{p['rank']}. {p['player']} {p['score']}" for p in s["reserve"]],
                      "freshness": s["freshness"]}, indent=2, default=_json))


if __name__ == "__main__":
    main()
