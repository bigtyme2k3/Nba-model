"""
opportunity_redistribution.py
-----------------------------
NBA V1 opportunity redistribution layer.

Consumes current player profiles plus the live injury report and estimates:
- minutes vacated by unavailable / limited players
- who is most likely to absorb those minutes
- usage-load redistribution
- conservative stat-specific multipliers for points, rebounds, assists, threes, PRA

This is deliberately transparent and capped. It does NOT claim to know coach intent
or true tracking-stat "potential assists/rebound chances"; rebounds and assists use
recent box-score production as opportunity proxies until tracking data is added.

Outputs
-------
data/raw/opportunity_adjustments_today.csv
data/raw/opportunity_adjustments_YYYY-MM-DD.csv
data/raw/opportunity_redistribution_status.json
"""
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

PROFILE_PATH = Path("data/processed/player_profiles_latest.csv")
INJURY_PATH = Path("data/raw/injuries_today.csv")
INJURY_STATUS_PATH = Path("data/raw/injuries_status.json")
OUT_DIR = Path("data/raw")

STATUS_VACANCY = {
    "OUT": 1.00,
    # Unconfirmed availability is uncertainty, not an assumed vacancy.
    "DOUBTFUL": 0.00,
    "QUESTIONABLE": 0.00,
    "PROBABLE": 0.00,
}
ROLE_WEIGHT = {"BENCH": 1.25, "MIXED": 1.10, "STARTER": 0.90, "UNKNOWN": 1.0}
POSITION_AFFINITY = {
    "G": {"G": 1.40, "F": 0.90, "C": 0.35},
    "F": {"G": 0.90, "F": 1.30, "C": 0.90},
    "C": {"G": 0.35, "F": 1.00, "C": 1.50},
}
MAX_MINUTES = 40.0
MAX_USAGE_BUMP = 0.065
USAGE_CAPTURE = 0.70


def f(value, default=0.0):
    try:
        v = float(value)
        return default if math.isnan(v) else v
    except Exception:
        return default


def norm(value):
    return " ".join(str(value or "").strip().lower().replace("’", "'").split())


def pos_group(value):
    s = str(value or "").upper()
    if "C" in s:
        return "C"
    if "F" in s:
        return "F"
    return "G"


def clamp(value, lo, hi):
    return max(lo, min(hi, value))


def load_json(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default
    except Exception:
        return default


def load_inputs(profile_path=PROFILE_PATH, injury_path=INJURY_PATH):
    if not profile_path.exists():
        return pd.DataFrame(), pd.DataFrame()
    profiles = pd.read_csv(profile_path)
    injuries = pd.read_csv(injury_path) if injury_path.exists() else pd.DataFrame()
    return profiles, injuries


def days_since_last_game(row, target_date):
    last = pd.to_datetime(row.get("last_game_date"), errors="coerce")
    target = pd.to_datetime(target_date, errors="coerce")
    if pd.isna(last) or pd.isna(target):
        return None
    return int((target - last).days)


def recent_recipient(row, target_date):
    days = days_since_last_game(row, target_date)
    # Expected minute recipients should look like the current rotation, not everyone
    # who appeared during camp or months earlier.
    return days is not None and 0 <= days <= 14


def allocation_weight(out_row, rec_row):
    out_pos = pos_group(out_row.get("pos"))
    rec_pos = pos_group(rec_row.get("pos"))
    affinity = POSITION_AFFINITY[out_pos][rec_pos]
    role = ROLE_WEIGHT.get(str(rec_row.get("projected_role", "UNKNOWN")).upper(), 1.0)
    minutes = f(rec_row.get("projected_minutes"), f(rec_row.get("mpg"), 0.0))
    minute_factor = clamp(0.55 + minutes / 35.0, 0.65, 1.55)
    stability = f(rec_row.get("rotation_stability_score"), 60.0)
    stability_factor = clamp(0.75 + stability / 200.0, 0.80, 1.25)
    return affinity * role * minute_factor * stability_factor


def distribute_minutes(vacated, out_row, recipients, current_minutes):
    remaining = max(0.0, vacated)
    received = {norm(r["player"]): 0.0 for _, r in recipients.iterrows()}
    # Recompute shares after capped players fill.
    for _ in range(6):
        if remaining <= 0.05:
            break
        eligible = []
        total_weight = 0.0
        for _, r in recipients.iterrows():
            key = norm(r["player"])
            base = current_minutes.get(key, f(r.get("projected_minutes"), f(r.get("mpg"), 0.0)))
            capacity = max(0.0, MAX_MINUTES - base)
            if capacity <= 0.05:
                continue
            w = allocation_weight(out_row, r)
            if w <= 0:
                continue
            eligible.append((r, key, w, capacity))
            total_weight += w
        if not eligible or total_weight <= 0:
            break

        spent = 0.0
        for r, key, w, capacity in eligible:
            share = remaining * (w / total_weight)
            add = min(share, capacity)
            if add <= 0:
                continue
            current_minutes[key] = current_minutes.get(
                key, f(r.get("projected_minutes"), f(r.get("mpg"), 0.0))
            ) + add
            received[key] += add
            spent += add
        if spent <= 0.01:
            break
        remaining -= spent
    return received, remaining


def infer_slate_teams(target_date: str, raw_dir: Path = OUT_DIR):
    """Infer today's teams from Odds API commence times, not scrape-date labels."""
    paths = [raw_dir / f"odds_{target_date}.csv", raw_dir / "odds_today.csv"]
    path = next((p for p in paths if p.exists()), None)
    if path is None:
        return set()
    try:
        df = pd.read_csv(path)
    except Exception:
        return set()
    et = ZoneInfo("America/New_York")
    teams = set()
    for _, r in df.iterrows():
        commence = str(r.get("commence_time", "") or "")
        try:
            dt = datetime.fromisoformat(commence.replace("Z", "+00:00")).astimezone(et)
        except Exception:
            continue
        if dt.date().isoformat() != target_date:
            continue
        for col in ("home_team", "away_team"):
            team = str(r.get(col, "") or "").strip()
            if team and team.lower() != "nan":
                teams.add(team)
    return teams


def build_opportunity_adjustments(
    target_date: str,
    profile_path: Path = PROFILE_PATH,
    injury_path: Path = INJURY_PATH,
    out_dir: Path = OUT_DIR,
    slate_teams: set[str] | None = None,
):
    profiles, injuries = load_inputs(profile_path, injury_path)
    injury_status = load_json(INJURY_STATUS_PATH, {})
    out_dir.mkdir(parents=True, exist_ok=True)
    slate_teams = set(slate_teams or infer_slate_teams(target_date, out_dir))

    status = {
        "status": "unknown",
        "target_date": target_date,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "injury_feed_status": injury_status.get("status"),
        "injury_rows": int(len(injuries)),
        "slate_teams": sorted(slate_teams),
        "teams_affected": 0,
        "absences_modeled": 0,
        "recipient_rows": 0,
        "unallocated_minutes": 0.0,
        "method": "POSITION_ROLE_WEIGHTED_CAPPED_REDISTRIBUTION",
        "coach_intent_inferred": False,
        "tracking_opportunity_data_used": False,
    }

    if profiles.empty:
        status["status"] = "WAITING_FOR_PLAYER_PROFILES"
        _write([], status, target_date, out_dir)
        return {}

    if injuries.empty:
        status["status"] = (
            "WAITING_FOR_VERIFIED_INJURY_FEED"
            if injury_status.get("status") in {None, "empty_unverified", "error"}
            else "NO_REPORTED_INJURIES"
        )
        _write([], status, target_date, out_dir)
        return {}

    profiles = profiles.copy()
    profiles["_key"] = profiles["player"].map(norm)
    pmap = {r["_key"]: r for _, r in profiles.iterrows()}

    injuries = injuries.copy()
    injuries["severity"] = injuries.get("severity", "").astype(str).str.upper()
    injuries["_key"] = injuries.get("player", "").map(norm)
    actionable = injuries[injuries["severity"].isin(STATUS_VACANCY)].copy()
    if slate_teams:
        actionable = actionable[
            actionable["_key"].map(
                lambda k: str(pmap.get(k, {}).get("team", "")) in slate_teams if k in pmap else False
            )
        ].copy()
    if actionable.empty:
        status["status"] = "NO_ACTIONABLE_INJURIES"
        _write([], status, target_date, out_dir)
        return {}

    # Final expected minutes begin at the profile projection.
    current_minutes = {
        r["_key"]: f(r.get("projected_minutes"), f(r.get("mpg"), 0.0))
        for _, r in profiles.iterrows()
    }
    received_minutes = {k: 0.0 for k in current_minutes}
    usage_load_received = {k: 0.0 for k in current_minutes}
    cause_players = {k: [] for k in current_minutes}
    own_reduction = {}
    teams_affected = set()
    unallocated_total = 0.0
    modeled = 0

    injury_by_key = {
        r["_key"]: str(r.get("severity", "")).upper()
        for _, r in actionable.iterrows()
    }

    for _, inj in actionable.iterrows():
        key = inj["_key"]
        severity = str(inj["severity"]).upper()
        vacancy_factor = STATUS_VACANCY.get(severity, 0.0)
        out_row = pmap.get(key)
        if out_row is None:
            continue

        team_raw = inj.get("team")
        team = "" if pd.isna(team_raw) else str(team_raw or "").strip()
        profile_team = str(out_row.get("team") or "").strip()
        if not team or team.lower() == "nan" or (slate_teams and team not in slate_teams):
            team = profile_team
        if not team or (slate_teams and team not in slate_teams):
            continue
        base_min = f(out_row.get("projected_minutes"), f(out_row.get("mpg"), 0.0))
        last_days = days_since_last_game(out_row, target_date)

        # An OUT player creates a NEW vacancy only if he was part of the recent
        # rotation. Otherwise the recent baseline already reflects his absence.
        vacancy_eligible = (
            severity == "OUT"
            and last_days is not None
            and 0 <= last_days <= 10
            and base_min >= 6.0
        )
        vacated = base_min * vacancy_factor if vacancy_eligible else 0.0

        # Never project unavailable players into a market even when their old minutes
        # are not redistributed because the baseline has already adapted.
        if severity == "OUT":
            own_reduction[key] = own_reduction.get(key, 0.0) + base_min
            current_minutes[key] = 0.0
        else:
            own_reduction[key] = own_reduction.get(key, 0.0)
            current_minutes[key] = current_minutes.get(key, base_min)

        if vacated <= 0:
            continue

        teams_affected.add(team)
        modeled += 1

        team_rows = profiles[
            profiles["team"].astype(str).eq(team)
            & profiles["_key"].ne(key)
        ].copy()
        if not team_rows.empty:
            team_rows = team_rows[
                team_rows.apply(lambda r: recent_recipient(r, target_date), axis=1)
            ]
            team_rows = team_rows[
                team_rows["projected_minutes"].fillna(team_rows["mpg"]).fillna(0).ge(8.0)
            ]
            # Do not redistribute into another unavailable/uncertain player.
            team_rows = team_rows[
                ~team_rows["_key"].map(
                    lambda k: injury_by_key.get(k) in {"OUT","DOUBTFUL","QUESTIONABLE"}
                )
            ]
            team_rows["_rotation_minutes"] = pd.to_numeric(
                team_rows["projected_minutes"], errors="coerce"
            ).fillna(pd.to_numeric(team_rows["mpg"], errors="coerce")).fillna(0)
            team_rows = team_rows.sort_values("_rotation_minutes", ascending=False).head(10)

        if team_rows.empty:
            unallocated_total += vacated
            continue

        received, remaining = distribute_minutes(vacated, out_row, team_rows, current_minutes)
        unallocated_total += remaining

        assigned = sum(received.values())
        usage_load = f(out_row.get("usage"), 0.20) * assigned * USAGE_CAPTURE

        weights = {}
        total_usage_weight = 0.0
        for _, rec in team_rows.iterrows():
            rkey = rec["_key"]
            mins_added = received.get(rkey, 0.0)
            if mins_added <= 0:
                continue
            received_minutes[rkey] += mins_added
            cause_players[rkey].append(str(out_row.get("player", "")))
            # Favor recipients already trusted with ball-handling / usage.
            w = max(0.05, f(rec.get("usage"), 0.20)) * max(1.0, current_minutes.get(rkey, 0.0))
            weights[rkey] = w
            total_usage_weight += w

        if total_usage_weight > 0:
            for rkey, w in weights.items():
                usage_load_received[rkey] += usage_load * (w / total_usage_weight)

    rows = []
    mapping = {}
    for _, r in profiles.iterrows():
        key = r["_key"]
        base_min = f(r.get("projected_minutes"), f(r.get("mpg"), 0.0))
        new_min = current_minutes.get(key, base_min)
        severity = injury_by_key.get(key)
        affected = abs(new_min - base_min) > 0.05 or usage_load_received.get(key, 0.0) > 0
        if not affected and severity not in STATUS_VACANCY:
            continue

        base_usage = f(r.get("usage"), 0.20)
        usage_delta = 0.0
        if new_min > 0 and usage_load_received.get(key, 0.0) > 0:
            usage_delta = usage_load_received[key] / new_min
            usage_delta = clamp(usage_delta, 0.0, MAX_USAGE_BUMP)
        new_usage = clamp(base_usage + usage_delta, 0.05, 0.42) if new_min > 0 else 0.0

        min_mult = (new_min / base_min) if base_min > 0 else 1.0
        usage_mult = (new_usage / base_usage) if base_usage > 0 else 1.0
        # Conservative role-aware stat scaling. Minute effects dominate.
        pts_mult = clamp(min_mult * (1.0 + 0.75 * (usage_mult - 1.0)), 0.0, 1.60)
        ast_mult = clamp(min_mult * (1.0 + 0.55 * (usage_mult - 1.0)), 0.0, 1.50)
        threes_mult = clamp(min_mult * (1.0 + 0.50 * (usage_mult - 1.0)), 0.0, 1.50)

        # Rebounds are less usage-sensitive. Bigs/forwards receive a small extra role
        # capture when their minutes rise, capped to avoid double counting.
        big_boost = 0.0
        if new_min > base_min and pos_group(r.get("pos")) in {"F", "C"}:
            big_boost = min(0.08, (new_min - base_min) / 100.0)
        reb_mult = clamp(min_mult * (1.0 + big_boost), 0.0, 1.45)

        p = f(r.get("roll5_pts"), 0.0)
        rb = f(r.get("roll5_reb"), 0.0)
        a = f(r.get("roll5_ast"), 0.0)
        base_pra = p + rb + a
        adj_pra = p * pts_mult + rb * reb_mult + a * ast_mult
        pra_mult = (adj_pra / base_pra) if base_pra > 0 else min_mult

        row = {
            "game_date": target_date,
            "player": r.get("player"),
            "team": r.get("team"),
            "position": r.get("pos"),
            "injury_severity": severity,
            "base_minutes": round(base_min, 3),
            "adjusted_minutes": round(new_min, 3),
            "minutes_delta": round(new_min - base_min, 3),
            "base_usage": round(base_usage, 5),
            "adjusted_usage": round(new_usage, 5),
            "usage_delta": round(new_usage - base_usage, 5),
            "minutes_received": round(received_minutes.get(key, 0.0), 3),
            "minutes_reduced": round(own_reduction.get(key, 0.0), 3),
            "pts_multiplier": round(pts_mult, 5),
            "reb_multiplier": round(reb_mult, 5),
            "ast_multiplier": round(ast_mult, 5),
            "threes_multiplier": round(threes_mult, 5),
            "pra_multiplier": round(pra_mult, 5),
            "vacancy_sources": "|".join(sorted(set(cause_players.get(key, [])))),
            "basis": "INJURY_STATUS_PLUS_PRIOR_ROTATION_PROFILE",
        }
        rows.append(row)
        mapping[str(r.get("player"))] = row

    status.update({
        "status": "READY",
        "teams_affected": len(teams_affected),
        "absences_modeled": modeled,
        "recipient_rows": sum(1 for r in rows if r["minutes_received"] > 0),
        "adjustment_rows": len(rows),
        "unallocated_minutes": round(unallocated_total, 3),
    })
    _write(rows, status, target_date, out_dir)
    return mapping


def _write(rows, status, target_date, out_dir):
    cols = [
        "game_date","player","team","position","injury_severity",
        "base_minutes","adjusted_minutes","minutes_delta",
        "base_usage","adjusted_usage","usage_delta",
        "minutes_received","minutes_reduced",
        "pts_multiplier","reb_multiplier","ast_multiplier",
        "threes_multiplier","pra_multiplier","vacancy_sources","basis"
    ]
    df = pd.DataFrame(rows, columns=cols)
    df.to_csv(out_dir / "opportunity_adjustments_today.csv", index=False)
    df.to_csv(out_dir / f"opportunity_adjustments_{target_date}.csv", index=False)
    (out_dir / "opportunity_redistribution_status.json").write_text(
        json.dumps(status, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(status, indent=2))
    if not df.empty:
        print(df.sort_values("minutes_delta", ascending=False).head(25).to_string(index=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True)
    parser.add_argument("--profiles", default=str(PROFILE_PATH))
    parser.add_argument("--injuries", default=str(INJURY_PATH))
    parser.add_argument("--out", default=str(OUT_DIR))
    args = parser.parse_args()
    build_opportunity_adjustments(
        args.date,
        Path(args.profiles),
        Path(args.injuries),
        Path(args.out),
    )


if __name__ == "__main__":
    main()
