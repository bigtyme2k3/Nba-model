#!/usr/bin/env python3
"""Jev / TypeSafe shadow decision harness.

Research-only: reads an existing model artifact, asks Jev for typed shadow
judgments, and writes a separate artifact + immutable ledger. It never mutates
production BET/LEAN/WATCH/PASS labels, bankroll sizing, or model predictions.

TypeSafe API contract:
  POST https://api.typesafe.ai/v1/systemone
  Authorization: Bearer $TYPESAFE_API_KEY
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

API_URL = "https://api.typesafe.ai/v1/systemone"
DEFAULT_MODEL = "jev-latest"
QUESTION_VERSION = "jev_shadow_v2"

CANDIDATE_KEYS = (
    "best_bets",
    "qualified_bets",
    "ranked_cards",
    "top_decisions",
    "decisions",
    "signals",
)

# Intentionally excludes existing production action/rank/score fields so Jev cannot
# simply echo the current policy. The shadow layer sees the underlying model/market
# evidence, then makes an independent typed judgment.
KEEP_FIELDS = (
    "type", "game", "play", "player", "team", "opponent", "stat", "market",
    "side", "signal", "line", "posted_line", "model_line",
    "odds", "american_odds", "price", "sportsbook", "best_book",
    "best_book_title", "edge", "edge_abs", "edge_pct", "projection_edge",
    "model_projection", "projection", "pred", "simulation_probability",
    "model_prob", "model_prob_pct", "market_probability",
    "market_implied_probability", "implied_probability", "implied_prob",
    "implied_prob_pct", "expected_value", "ev", "ev_pct", "raw_ev_pct",
    "confidence", "confidence_score", "conf", "stars", "data_quality",
    "book_count", "available_books", "history_games", "prior_games",
    "injury_status", "injury_detail", "risks", "reasons", "flags",
    "fair_odds", "tip",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def fnum(value: Any) -> float | None:
    try:
        out = float(value)
        return out if math.isfinite(out) else None
    except (TypeError, ValueError):
        return None


def compact_candidate(row: dict[str, Any]) -> dict[str, Any]:
    out = {k: row.get(k) for k in KEEP_FIELDS if k in row and row.get(k) is not None}

    trend = row.get("trend")
    if isinstance(trend, dict):
        small_trend: dict[str, Any] = {}
        for split in ("last5", "last10", "last20", "season", "opponent", "location"):
            block = trend.get(split)
            if isinstance(block, dict):
                small_trend[split] = {
                    k: block.get(k)
                    for k in ("sample", "hits", "hit_rate", "average")
                    if block.get(k) is not None
                }
        if small_trend:
            out["trend"] = small_trend

    votes = row.get("votes")
    if isinstance(votes, list):
        out["votes"] = [
            {k: v.get(k) for k in ("engine", "score", "agree") if v.get(k) is not None}
            for v in votes[:12]
            if isinstance(v, dict)
        ]
    return out


def select_candidates(payload: dict[str, Any], limit: int) -> tuple[list[dict[str, Any]], str | None]:
    for key in CANDIDATE_KEYS:
        rows = payload.get(key)
        if isinstance(rows, list) and rows:
            return [compact_candidate(r) for r in rows[:limit] if isinstance(r, dict)], key

    current = payload.get("current")
    if isinstance(current, dict):
        rows = current.get("shadow_bet_rows")
        if isinstance(rows, list) and rows:
            return [compact_candidate(r) for r in rows[:limit] if isinstance(r, dict)], "current.shadow_bet_rows"

    return [], None


def parse_iso(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        text = str(value).strip().replace("Z", "+00:00")
        dt = datetime.fromisoformat(text)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except ValueError:
        return None


def freshness_problem(payload: dict[str, Any], requested_date: str | None) -> str | None:
    target = str(payload.get("target_date") or payload.get("date") or "")[:10]
    if requested_date and target and target != requested_date:
        return f"artifact target_date={target} does not match requested date={requested_date}"

    target_for_age = requested_date or target
    source_ts = parse_iso(payload.get("source_generated_at_utc"))
    if source_ts and target_for_age:
        try:
            target_dt = datetime.fromisoformat(target_for_age).replace(tzinfo=timezone.utc)
            age_days = (target_dt.date() - source_ts.date()).days
            if age_days > 2:
                return (
                    f"source_generated_at_utc={source_ts.isoformat()} is {age_days} days "
                    f"older than target_date={target_for_age}"
                )
        except ValueError:
            pass
    return None


def source_meta(payload: dict[str, Any], input_path: Path, candidate_key: str | None) -> dict[str, Any]:
    keep = (
        "target_date", "date", "generated_at", "generated_at_utc",
        "source_generated_at_utc", "source", "status", "schema_version",
        "research_only", "scoring_scale", "scoring_note",
    )
    meta = {k: payload.get(k) for k in keep if payload.get(k) is not None}
    meta["input_path"] = str(input_path)
    meta["candidate_key"] = candidate_key
    if isinstance(payload.get("guardrails"), dict):
        meta["guardrails"] = payload["guardrails"]
    if isinstance(payload.get("model_stats"), dict):
        meta["model_stats"] = payload["model_stats"]
    return meta


def make_shadow_id(league: str, target_date: str, candidate: dict[str, Any]) -> str:
    identity = {
        "question_version": QUESTION_VERSION,
        "league": league,
        "target_date": target_date,
        "type": candidate.get("type"),
        "game": candidate.get("game"),
        "play": candidate.get("play"),
        "player": candidate.get("player"),
        "stat": candidate.get("stat") or candidate.get("market"),
        "side": candidate.get("side") or candidate.get("signal") or candidate.get("recommendation"),
        "line": candidate.get("line"),
        "odds": candidate.get("odds") or candidate.get("american_odds") or candidate.get("price"),
        "sportsbook": candidate.get("sportsbook") or candidate.get("best_book"),
    }
    raw = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def build_questions(candidate_count: int) -> dict[str, Any]:
    q: dict[str, Any] = {
        "batch_directional_concentration": {
            "type": "noul",
            "instructions": (
                "Does the full `candidates` set show suspicious directional concentration, "
                "repetition, or a one-sided pattern that could indicate a systemic model or "
                "data artifact rather than a naturally varied slate?"
            ),
            "criteria": {
                "true": "The batch is suspiciously concentrated or repetitive enough to warrant slate-level caution.",
                "false": "The batch composition is not suspiciously concentrated from the supplied state.",
            },
        },
        "batch_systemic_integrity_risk": {
            "type": "noul",
            "instructions": (
                "Across the full `candidates` set, is there evidence of systemic data/model "
                "integrity risk such as implausible projections, impossible values, repeated "
                "constant metrics, contradictory fields, or other patterns that should block "
                "blind automated use?"
            ),
            "criteria": {
                "true": "At least one systemic integrity pattern is materially concerning.",
                "false": "No material systemic integrity pattern is apparent from the supplied state.",
            },
        },
    }
    action_criteria = {
        "BET": "The candidate is coherent, sufficiently supported, and suitable to advance through the existing model's normal action gates. This is a shadow research label, not permission to place a wager.",
        "LEAN": "The candidate has meaningful support, but uncertainty or evidence quality materially reduces conviction.",
        "WATCH": "The candidate is interesting but should wait for better evidence, freshness, price, lineup, injury, sample, or model confirmation.",
        "PASS": "The candidate is not sufficiently supported or reliable to advance.",
    }
    trust_criteria = {
        "HIGH": "The supplied signals and data quality are mutually coherent with little material uncertainty.",
        "MEDIUM": "The candidate is usable but has one or more meaningful uncertainties.",
        "LOW": "Important evidence is weak, sparse, contradictory, stale, or missing.",
        "REJECT": "The candidate should not be trusted as an actionable model signal from the supplied state.",
    }

    for i in range(candidate_count):
        ref = f"candidates[{i}]"
        q[f"c{i}_action"] = {
            "type": "choice",
            "instructions": {
                "question": f"Which shadow action best describes whether `{ref}` should advance?",
                "rules": [
                    "Judge only from the supplied state; do not invent missing statistics.",
                    "Do not recalculate or override exact numeric model/EV computations unless the supplied fields are internally inconsistent.",
                    "Treat this as an independent research decision; production rules remain authoritative.",
                ],
            },
            "criteria": action_criteria,
        }
        q[f"c{i}_trust"] = {
            "type": "choice",
            "instructions": f"How much should the prediction/recommendation in `{ref}` be trusted given its supplied evidence, uncertainty, data quality, and internal consistency?",
            "criteria": trust_criteria,
        }
        q[f"c{i}_anomaly"] = {
            "type": "noul",
            "instructions": f"Does `{ref}` contain a material anomaly, contradiction, suspicious value, stale evidence, or missing support that should reduce trust?",
            "criteria": {
                "true": "There is a material quality or consistency problem.",
                "false": "No material quality or consistency problem is apparent from the supplied state.",
            },
        }
        q[f"c{i}_review"] = {
            "type": "noul",
            "instructions": f"Should `{ref}` be sent to manual or deeper-reasoning review before any automated action?",
            "criteria": {
                "true": "Additional review is warranted before action.",
                "false": "No additional review is warranted from the supplied state.",
            },
        }
    return q


def call_typesafe(payload: dict[str, Any], api_key: str, retries: int = 4) -> dict[str, Any]:
    body = json.dumps(payload, allow_nan=False).encode("utf-8")
    for attempt in range(retries):
        req = Request(
            API_URL,
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "User-Agent": "nba-wnba-model-jev-shadow/1.0",
            },
        )
        try:
            with urlopen(req, timeout=60) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:1200]
            if exc.code in (429, 529) and attempt < retries - 1:
                time.sleep(2 ** attempt)
                continue
            raise RuntimeError(f"TypeSafe HTTP {exc.code}: {detail}") from exc
        except URLError as exc:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
                continue
            raise RuntimeError(f"TypeSafe network error: {exc}") from exc
    raise RuntimeError("TypeSafe request failed after retries")


def decision_row(
    league: str,
    target_date: str,
    candidate: dict[str, Any],
    idx: int,
    answers: dict[str, Any],
    issued_at: str,
) -> dict[str, Any]:
    action = answers.get(f"c{idx}_action", {}) or {}
    trust = answers.get(f"c{idx}_trust", {}) or {}
    anomaly = answers.get(f"c{idx}_anomaly", {}) or {}
    review = answers.get(f"c{idx}_review", {}) or {}
    return {
        "shadow_id": make_shadow_id(league, target_date, candidate),
        "question_version": QUESTION_VERSION,
        "league": league.upper(),
        "target_date": target_date,
        "issued_at_utc": issued_at,
        "candidate": candidate,
        "jev": {
            "action": action.get("choice"),
            "action_probabilities": action.get("probabilities"),
            "action_confidence": fnum(action.get("confidence")),
            "trust": trust.get("choice"),
            "trust_probabilities": trust.get("probabilities"),
            "trust_confidence": fnum(trust.get("confidence")),
            "anomaly_probability": fnum(anomaly.get("noul")),
            "review_probability": fnum(review.get("noul")),
        },
        "research_only": True,
        "production_mutation": False,
        "outcome": "PENDING",
    }


def append_new_ledger(path: Path, rows: list[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing: set[str] = set()
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
                if row.get("shadow_id"):
                    existing.add(str(row["shadow_id"]))
            except json.JSONDecodeError:
                continue

    new_rows = [r for r in rows if r["shadow_id"] not in existing]
    if new_rows:
        with path.open("a", encoding="utf-8") as fh:
            for row in new_rows:
                fh.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
    return len(new_rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--league", choices=("nba", "wnba"), required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--ledger", required=True)
    ap.add_argument("--date", default="")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--max-candidates", type=int, default=20)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    ledger_path = Path(args.ledger)
    requested_date = args.date.strip() or None
    issued_at = utc_now()

    if not input_path.exists():
        raise SystemExit(f"Input does not exist: {input_path}")

    payload = load_json(input_path)
    problem = freshness_problem(payload, requested_date)
    target_date = requested_date or str(payload.get("target_date") or payload.get("date") or "")[:10]
    candidates, candidate_key = select_candidates(payload, max(1, args.max_candidates))
    meta = source_meta(payload, input_path, candidate_key)

    base = {
        "version": QUESTION_VERSION,
        "league": args.league.upper(),
        "generated_at_utc": issued_at,
        "target_date": target_date or None,
        "source": meta,
        "research_only": True,
        "production_mutation": False,
    }

    if problem:
        write_json(output_path, {**base, "status": "STALE_SOURCE", "reason": problem, "decisions": []})
        print(f"[JEV] STALE_SOURCE: {problem}")
        return

    if not candidates:
        write_json(output_path, {**base, "status": "NO_CANDIDATES", "candidate_count": 0, "decisions": []})
        print("[JEV] No eligible candidates in source artifact.")
        return

    state = {
        "league": args.league.upper(),
        "target_date": target_date,
        "source": meta,
        "candidates": candidates,
    }
    request_payload = {
        "state": state,
        "model": args.model,
        "questions": build_questions(len(candidates)),
    }

    if args.dry_run:
        write_json(
            output_path,
            {
                **base,
                "status": "DRY_RUN",
                "candidate_count": len(candidates),
                "request_preview": request_payload,
                "decisions": [],
            },
        )
        print(f"[JEV] Dry run prepared {len(candidates)} candidates / {len(request_payload['questions'])} questions.")
        return

    api_key = os.getenv("TYPESAFE_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("TYPESAFE_API_KEY is not set. Add it as a GitHub Actions secret or run with --dry-run.")

    response = call_typesafe(request_payload, api_key)
    answers = response.get("answers")
    if not isinstance(answers, dict):
        raise RuntimeError(f"TypeSafe response missing answers map: {response}")

    decisions = [
        decision_row(args.league, target_date, candidate, i, answers, issued_at)
        for i, candidate in enumerate(candidates)
    ]
    batch = {
        "directional_concentration_probability": fnum(
            (answers.get("batch_directional_concentration") or {}).get("noul")
        ),
        "systemic_integrity_risk_probability": fnum(
            (answers.get("batch_systemic_integrity_risk") or {}).get("noul")
        ),
    }
    new_rows = append_new_ledger(ledger_path, decisions)

    summary = {
        "candidates": len(decisions),
        "new_ledger_rows": new_rows,
        "action_counts": {},
        "trust_counts": {},
        "review_ge_050": 0,
        "anomaly_ge_050": 0,
        "batch_directional_concentration_ge_050": (
            (batch.get("directional_concentration_probability") or 0) >= 0.5
        ),
        "batch_systemic_integrity_risk_ge_050": (
            (batch.get("systemic_integrity_risk_probability") or 0) >= 0.5
        ),
    }
    for row in decisions:
        j = row["jev"]
        a = str(j.get("action") or "UNKNOWN")
        t = str(j.get("trust") or "UNKNOWN")
        summary["action_counts"][a] = summary["action_counts"].get(a, 0) + 1
        summary["trust_counts"][t] = summary["trust_counts"].get(t, 0) + 1
        if (j.get("review_probability") or 0) >= 0.5:
            summary["review_ge_050"] += 1
        if (j.get("anomaly_probability") or 0) >= 0.5:
            summary["anomaly_ge_050"] += 1

    write_json(
        output_path,
        {
            **base,
            "status": "OK",
            "model": response.get("model"),
            "usage": response.get("usage"),
            "summary": summary,
            "batch": batch,
            "decisions": decisions,
            "safety": (
                "Shadow only. Jev judgments do not alter production predictions, "
                "Best Bets, bankroll sizing, or action labels."
            ),
        },
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
