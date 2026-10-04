"""Reproducible machine artifacts and a separately labeled historical research page."""
from __future__ import annotations

import csv
import html
import io
import json
from collections import Counter, defaultdict
from pathlib import Path

from . import VERSION
from .core import (BOOK_FIELDS, american, early_bucket, fingerprint, normalize_event,
                   parse_time, season_for)
from .ingest import atomic_write, discovery_tasks, write_json

EMPTY_COLUMNS = {
    "historical_market_warehouse": ["season", "game_date", "commence_time", "event_id", "home_team", "away_team", "team_game_number_home", "team_game_number_away", "early_bucket_home", "early_bucket_away", "bookmaker", "snapshot_type", "snapshot_timestamp"] + BOOK_FIELDS + ["favorite", "underdog", "favorite_price", "underdog_price", "home_implied", "away_implied", "home_no_vig", "away_no_vig", "game_number_status", "source"],
    "market_consensus": ["event_id", "season", "snapshot_type", "snapshot_timestamp", "market", "basis", "home_no_vig", "away_no_vig", "home_spread", "total"],
    "frozen_baseline_bets": ["signal_id", "event_id", "season", "model", "snapshot_type", "signal_timestamp", "basis", "bookmaker", "market", "mode", "side", "price", "reference_line", "execution_line", "team_game_number", "early_bucket", "qualifies", "result", "net_units"],
    "events": ["event_id", "season", "commence_time", "home_team", "away_team", "team_game_number_home", "team_game_number_away"],
    "streak_distribution": ["season", "snapshot_type", "scope", "metric", "streak_length", "runs", "completed_runs", "censored_runs", "maximum_historical_streak"],
    "next_game_mean_reversion": ["season", "snapshot_type", "scope", "metric", "prior_streak", "next_qualifying_events", "continued", "reverted", "continuation_rate", "continuation_flat_roi", "reversion_flat_roi", "small_sample"],
    "maximum_risk_report": ["season", "snapshot_type", "market", "mode", "strategy", "scope", "bankroll", "operational_ruin_probability", "historical_bankroll_requirement"],
    "progression_simulation": ["season", "snapshot_type", "market", "mode", "scope", "strategy", "bets", "net_units", "roi", "maximum_single_stake", "maximum_sequence_exposure", "maximum_drawdown", "historical_bankroll_requirement", "progression_failures"],
}
DEFAULT_COLUMNS = ["season", "snapshot_type", "market", "mode", "basis", "early_bucket", "games", "graded_games", "wins", "losses", "pushes", "flat_bet_roi", "small_sample"]
MAX_EXPORT_BYTES = 32 * 1024 * 1024


def export_table(root, name, rows):
    root = Path(root)
    base = EMPTY_COLUMNS.get(name, DEFAULT_COLUMNS)
    columns = list(dict.fromkeys(base + sorted({k for row in rows for k in row})))
    chunks, current, size = [], [], 0
    for row in rows:
        estimate = len(json.dumps([row], indent=2, sort_keys=True).encode()) + len(json.dumps(row, sort_keys=True).encode())
        if current and size+estimate > MAX_EXPORT_BYTES:
            chunks.append(current)
            current, size = [], 0
        current.append(row)
        size += estimate
    chunks.append(current)
    for old in list(root.glob(f"{name}-part-*.csv"))+list(root.glob(f"{name}-part-*.json")):
        old.unlink()
    artifacts = []
    for index, chunk in enumerate(chunks):
        stem = name if len(chunks) == 1 else f"{name}-part-{index:04d}"
        stream = io.StringIO(newline="")
        writer = csv.DictWriter(stream, columns, lineterminator="\n")
        writer.writeheader()
        for row in chunk:
            writer.writerow({k: json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else v for k, v in row.items()})
        atomic_write(root / f"{stem}.csv", stream.getvalue().encode())
        write_json(root / f"{stem}.json", chunk)
        artifacts.append({"csv": f"{stem}.csv", "json": f"{stem}.json", "rows": len(chunk), "content_hash": fingerprint(chunk)})
    if len(chunks) > 1:
        # Original file names become explicit part indexes, never silently truncated tables.
        write_json(root / f"{name}.json", {"format": "partitioned_table", "total_rows": len(rows), "columns": columns, "parts": artifacts})
        stream = io.StringIO(newline="")
        writer = csv.DictWriter(stream, ["csv", "json", "rows", "content_hash"], lineterminator="\n")
        writer.writeheader();writer.writerows(artifacts)
        atomic_write(root / f"{name}.csv", stream.getvalue().encode())
    return artifacts


def read_exported_table(root, name):
    value = json.loads((Path(root)/f"{name}.json").read_text())
    if isinstance(value, list):
        return value
    if value.get("format") != "partitioned_table":
        raise ValueError("Unknown export format")
    result = []
    for part in value["parts"]:
        rows = json.loads((Path(root)/part["json"]).read_text())
        if len(rows) != part["rows"] or fingerprint(rows) != part["content_hash"]:
            raise ValueError("Partitioned export checksum mismatch")
        result.extend(rows)
    if len(result) != value["total_rows"]:
        raise ValueError("Partitioned export row count mismatch")
    return result


def quality_control(warehouse, events, raw, views, bets):
    errors, config = [], warehouse.config
    ids = [e["event_id"] for e in events]
    if len(ids) != len(set(ids)):
        errors.append("duplicate_event_ids")
    pairs = [(e["season"], e["game_date"], e["home_team"], e["away_team"]) for e in events]
    if len(pairs) != len(set(pairs)):
        errors.append("duplicate_matchup_date_with_different_event_ids_requires_resolution")
    lookup = {e["event_id"]: e for e in events}
    keys = [(r["event_id"], r["snapshot_timestamp"], r["bookmaker"]) for r in raw]
    if len(keys) != len(set(keys)):
        errors.append("duplicate_book_observations")
    for event in events:
        if season_for(event["commence_time"], config) != event["season"]:
            errors.append("preseason_playoff_or_cup_final_contamination")
        for side in ("home", "away"):
            if early_bucket(event[f"team_game_number_{side}"]) != event[f"early_bucket_{side}"]:
                errors.append("incorrect_early_bucket")
    for row in raw:
        tip = parse_time(lookup[row["event_id"]]["commence_time"])
        snapshot = parse_time(row["snapshot_timestamp"])
        if snapshot >= tip:
            errors.append("post_tip_odds")
        for market in ("h2h", "spreads", "totals"):
            if row.get(f"{market}_last_update") and parse_time(row[f"{market}_last_update"]) > snapshot:
                errors.append("future_book_update")
        for field in ("home_ml", "away_ml", "home_spread_price", "away_spread_price", "over_price", "under_price"):
            if row.get(field) is not None:
                try:
                    american(row[field])
                except ValueError:
                    errors.append("bad_odds_sign_or_magnitude")
        if row.get("home_spread") is not None and abs(row["home_spread"]+row["away_spread"]) > 1e-8:
            errors.append("asymmetric_spread")
        if row.get("home_no_vig") is not None and abs(row["home_no_vig"]+row["away_no_vig"]-1) > 1e-8:
            errors.append("bad_no_vig_pair")
    for bet in bets:
        if bet["source"] != "the-odds-api":
            errors.append("unauthorized_market_source")
        if bet["result"] is not None:
            h, a = bet["home_score"], bet["away_score"]
            if bet["market"] == "h2h":
                signed = (h-a) if bet["side"] == "home" else (a-h)
            elif bet["market"] == "spreads":
                signed = ((h-a) if bet["side"] == "home" else (a-h)) + bet["execution_line"]
            else:
                signed = (h+a-bet["execution_line"])*(1 if bet["side"] == "over" else -1)
            expected = "push" if abs(signed) < 1e-8 else "win" if signed > 0 else "loss"
            if bet["result"] != expected:
                errors.append("wrong_grade_orientation")
            price = bet["price"]
            profit = (price/100 if price > 0 else 100/-price) if expected == "win" else -1 if expected == "loss" else 0
            if abs(bet["net_units"]-profit) > 1e-8:
                errors.append("wrong_ml_or_juice_payout")
    counts = Counter(e["season"] for e in events)
    requests = warehouse.rows("requests")
    by_request = {r["request_key"]: r for r in requests}
    coverage = []
    for season in config["seasons"]:
        planned = [r for r in discovery_tasks(config, [season])]
        from .ingest import ODDS_PATH, canonical_params, market_params
        done = sum(fingerprint([ODDS_PATH, canonical_params(market_params(config, r["timestamp"]))]) in by_request for r in planned)
        team_counts = Counter(team for e in events if e["season"] == season for team in (e["home_team"], e["away_team"]))
        coverage.append({"season": season, "events_observed": counts[season], "discovery_snapshots_expected": len(planned),
                         "discovery_snapshots_cached": done, "teams_observed": len(team_counts),
                         "teams_with_at_least_20_observed_games": sum(n >= 20 for n in team_counts.values()),
                         "complete_schedule_independently_certified": False,
                         "observed_game_numbering_provisional": True,
                         "limitation": "Odds API inventory covers bookmaker-listed events; missing/canceled games can affect numbering"})
    serious = [r for r in warehouse.rows("issues") if r["reason"] == "conflicting_duplicate_observation"]
    if serious:
        errors.append("conflicting_book_snapshot_receipts")
    return {"status": "failed" if errors else "passed_available_rows" if raw else "no_historical_rows_to_validate",
            "errors": sorted(set(errors)), "event_count": len(events), "sportsbook_observations": len(raw),
            "selected_consensus_views": sum(v["basis"] == "CONSENSUS" for v in views),
            "frozen_bets": len(bets), "graded_bets": sum(b["result"] is not None for b in bets),
            "no_post_tip_leakage_in_available_rows": not any("tip" in e or "future" in e for e in errors),
            "coverage": coverage, "normalization_issues": len(warehouse.rows("issues")),
            "independent_raw_response_spot_checks": "pending_real_responses" if not raw else "inspect cache receipt hashes listed in warehouse exports",
            "full_historical_analysis_complete": False}


def md_table(rows, columns):
    if not rows:
        return "No graded historical observations available."
    def cell(value):
        if value is None:
            return "—"
        if isinstance(value, float):
            return f"{value:.4f}"
        return str(value).replace("|", "/")
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    lines += ["| " + " | ".join(cell(r.get(c)) for c in columns) + " |" for r in rows]
    return "\n".join(lines)


def research_report(status, tables, config):
    graded = status["graded_events"]
    lines = ["# NBA historical market and fade research", "", "**Historical research only. No live recommended bets. NBA V1 predictions and forward ledger remain independent.**", "",
             f"Status: **{status['status']}**. {status['events']} observed events; {status['sportsbook_observations']} bookmaker snapshots; {graded} events with permitted final scores.", "",
             "## Data and blockers", "", *[f"- {s}" for s in status["limitations"]], "",
             "Market source: The Odds API historical h2h/spreads/totals, US region. Scores must be archived Odds API responses; the module never reads existing ESPN/hoopR scores or betting lines.", "",
             "API request cost: 30 credits per all-market US historical snapshot. Successful raw responses, receipts, and normalization state are cached. No retries of successful purchases.", "",
             f"Quota: {json.dumps(status['api_quota'], sort_keys=True)}", "", "## Twelve research questions", ""]
    if not graded:
        questions = [
            "Are early-season underdogs unusual?", "Which underdog price bands differ?", "Does the effect change across Games 1–5, 6–10, 11–15, and 16–20?",
            "How often do underdogs win consecutive qualifying games?", "What is the longest qualifying upset streak?", "How often does the market favorite fail consecutively?",
            "Does a market fade have positive flat-bet expectation?", "Does an apparent edge survive multiple seasons?", "Do opening and closing failures differ?",
            "Does sportsbook disagreement contain useful information?", "Does a progression improve expected value or increase tail risk?", "What bankroll/exposure survives the worst observed sequence?"
        ]
        for i, question in enumerate(questions, 1):
            answer = "Not estimable: no permitted graded historical data."
            if i == 11:
                answer += " The price-aware simulator is implemented, but no empirical profitability or risk conclusion is asserted."
            lines.append(f"{i}. **{question}** {answer}")
    else:
        dog = [r for r in tables["early_season_summary"] if r["basis"] == "CONSENSUS" and r["market"] == "h2h" and r["mode"] == "FADE" and r["snapshot_type"] == "CLOSE"]
        lines += ["1–3. Closing underdog results by season and team-game bucket (ROI uses actual execution-book prices):", "", md_table(dog, ["season", "early_bucket", "graded_games", "win_pct", "flat_bet_roi", "no_vig_expected_win_pct", "small_sample"]), "",
                  "Price bands:", "", md_table([r for r in tables["underdog_price_bucket_summary"] if r["basis"] == "CONSENSUS" and r["snapshot_type"] == "CLOSE" and r["early_bucket"] == "ALL_EARLY"], ["season", "price_band", "graded_games", "underdog_flat_bet_roi", "favorite_flat_bet_roi", "small_sample"]), "",
                  "4. Next qualifying game after upset streaks; previous outcomes must already be available:", "", md_table([r for r in tables["next_game_mean_reversion"] if r["metric"] == "underdog_outright_wins" and r["scope"] == "same_team" and r["snapshot_type"] == "CLOSE"], ["season", "prior_streak", "next_qualifying_events", "continuation_rate", "continuation_flat_roi", "reversion_flat_roi", "small_sample"]), "",
                  "5–6. Maximum streaks (season boundaries reset; missing results/pushes censor or end a run):", "", md_table([r for r in tables["streak_distribution"] if r["streak_length"] == "6+" and r["metric"] in {"underdog_outright_wins", "favorite_outright_losses", "favorite_outright_wins"}], ["season", "snapshot_type", "scope", "metric", "maximum_historical_streak"]), "",
                  "7–8. Every candidate is exploratory; pooled positive ROI alone is insufficient. Seasonal validation includes negative and absent seasons:", "", md_table(tables["season_validation"], ["kind", "snapshot_type", "market", "mode", "early_bucket", "price_band", "pooled_roi", "positive_seasons", "verdict"]), "",
                  "9. Open/close results; OPEN means earliest retrieved, not a verified true opener:", "", md_table(tables["open_vs_close_comparison"], ["season", "basis", "market", "comparison", "games", "favorite_flips"]), "",
                  "10. Disagreement is evaluated against the lower-disagreement group, not presumed useful:", "", md_table(tables["sportsbook_disagreement_analysis"], ["season", "snapshot_type", "market", "high_disagreement", "graded_games", "failure_rate", "failure_rate_minus_other_group", "small_sample"]), "",
                  "11–12. Progression, flat benchmark, and worst liquidity requirement (units):", "", md_table([r for r in tables["flat_bet_simulation"]+tables["progression_simulation"] if r["season"] == "ALL" and r["after_market_failures"] == 0], ["snapshot_type", "market", "mode", "scope", "strategy", "bets", "roi", "net_units", "maximum_single_stake", "maximum_sequence_exposure", "maximum_drawdown", "historical_bankroll_requirement", "progression_failures"])]
    lines += ["", "## Interpretation", "",
              "A total is a score reference, not a sportsbook Over prediction. Model D consistently uses Over as its reference side and Under as its opposite; this convention does not establish that bookmakers predicted Over.", "",
              "A moneyline favorite is graded identically at early and closing snapshots unless its designation flips. Price movement can change return without changing correctness. Spread/total reference grades and executable-book grades are stored separately.", "",
              "NBA V1 is neither retrained nor connected to these signals. No finding is promoted to production. A future shadow test requires complete, auditable market/outcome coverage, season replication, and prospectively frozen rules.", "",
              "Game numbering is exact within the observed catalog, but unverified as the complete played schedule. The Odds API does not guarantee that every unlisted/canceled NBA game can be reconstructed. This limitation is carried into every export.", "",
              "Bootstrap ruin estimates, when populated, replay paired price/outcome blocks within season and retain overlap inside each block. Blocks are serialized. These exploratory assumptions are not a prediction of future ruin. Historical survival is not a guarantee.", "",
              "Results preserve unsuccessful bands and strategies. Hit-rate confidence intervals and exploratory ROI intervals do not correct correlated events or multiple testing.", "",
              "See [methodology and run instructions](https://github.com/bigtyme2k3/Nba-model/blob/main/docs/NBA_MARKET_RESEARCH.md) and JSON/CSV files alongside this report.", ""]
    return "\n".join(lines)


def publish_outputs(warehouse, events, raw, views, bets, tables, page_dir=None):
    root, config = warehouse.root, warehouse.config
    qc = quality_control(warehouse, events, raw, views, bets)
    logs = warehouse.rows("request_log")
    quota_row = max(logs, key=lambda r: r["attempt_id"]) if logs else {}
    ingestion_path = root / "ingestion_status.json"
    ingestion = json.loads(ingestion_path.read_text()) if ingestion_path.exists() else {}
    graded_ids = {b["event_id"] for b in bets if b["result"] is not None}
    selected_ids = {b["event_id"] for b in bets if b["early_event"]}
    limitations = [
        "Historical score backfill is unavailable through the documented Odds API: /scores exposes at most the past three days. Supply genuine archived Odds API scores to grade older seasons under the strict one-source rule.",
        "True openers are unverified; EARLY/OPEN is explicitly the earliest valid retrieved snapshot. CLOSE requires a clean snapshot within 15 minutes before scheduled tip, using a one-minute safety buffer.",
        "Observed team-game numbering is provisional until complete played-game inventory is independently certified. NBA Cup championship dates, preseason, and dates outside regular-season boundaries are excluded.",
        "No complete later-season comparison is available unless --full-season-reference ingestion has completed; early-season unusualness relative to the rest of the season is not established.",
        "Exploratory groups overlap. Multiple testing and correlated games can create apparent edges. All individual-season results and negative returns are retained.",
    ]
    if ingestion.get("status") and ingestion["status"] != "ok":
        limitations.insert(0, f"Ingestion blocker: {ingestion['status']}. {ingestion.get('message', '')}")
    missing_scores = len({b["event_id"] for b in bets})-len(graded_ids)
    status = {"module_version": VERSION, "status": "validation_failed" if qc["errors"] else "blocked_no_historical_data" if not raw else "blocked_historical_outcomes" if not graded_ids else "partial_exploratory_results",
              "requested_seasons": list(config["seasons"]), "seasons_collected": sorted({r["season"] for r in raw}),
              "events": len(events), "early_events_with_frozen_bets": len(selected_ids), "sportsbook_observations": len(raw),
              "graded_events": len(graded_ids), "ungraded_events_with_bets": missing_scores,
              "source": "the-odds-api", "outcome_policy": "archived_odds_api_scores_only",
              "api_quota": {"network_requests_recorded": len(logs), "credits_recorded": sum(r["actual_cost"] if r["actual_cost"] is not None else r["estimated_cost"] for r in logs),
                            "remaining_last_observed": quota_row.get("quota_remaining"), "used_last_observed": quota_row.get("quota_used"), "successful_cached_responses": len(warehouse.rows("requests"))},
              "config_fingerprint": fingerprint(config), "limitations": limitations, "production_integration": False,
              "live_shadow_test_warranted": False, "completion_claim": "implementation_available; historical conclusions require permitted data"}
    all_tables = {"events": events, "historical_market_warehouse": raw, "market_consensus": views, "frozen_baseline_bets": bets, **tables}
    exported = {name: export_table(root, name, rows) for name, rows in all_tables.items()}
    write_json(root / "quality_control.json", qc)
    write_json(root / "analysis_status.json", status)
    report = research_report(status, tables, config)
    atomic_write(root / "REPORT.md", report.encode())
    manifest = {"module_version": VERSION, "config": config, "rows": {name: len(rows) for name, rows in all_tables.items()},
                "payload_hashes": sorted({r["payload_hash"] for r in warehouse.rows("requests")}),
                "output_hashes": {name: fingerprint(rows) for name, rows in all_tables.items()}, "export_files": exported}
    write_json(root / "manifest.json", manifest)
    if page_dir:
        page_dir = Path(page_dir)
        page_dir.mkdir(parents=True, exist_ok=True)
        safe_report = html.escape(report)
        page = '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>NBA historical market research</title><style>body{max-width:1100px;margin:2rem auto;padding:0 1rem;background:#111827;color:#e5e7eb;font:16px system-ui}a{color:#93c5fd}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:14px/1.6 ui-monospace,monospace}.notice{background:#312e81;padding:1rem;border-radius:8px}li{padding:.3rem}</style><h1>NBA historical market research</h1><p class="notice">Historical research only · No live recommended bets · NBA V1 remains independent</p><p><a href="../index.html">NBA V1 dashboard</a></p><ul>'
        for name in all_tables:
            atomic_write(page_dir / f"{name}.csv", (root / f"{name}.csv").read_bytes())
            page += f'<li><a href="{name}.csv">{html.escape(name.replace("_", " "))}</a></li>'
            for part in exported[name] if len(exported[name]) > 1 else []:
                atomic_write(page_dir/part["csv"], (root/part["csv"]).read_bytes())
                page += f'<li><a href="{part["csv"]}">{html.escape(part["csv"])} ({part["rows"]} rows)</a></li>'
        atomic_write(page_dir / "analysis_status.json", (root / "analysis_status.json").read_bytes())
        atomic_write(page_dir / "REPORT.md", report.encode())
        page += '</ul><h2>Research report</h2><pre>' + safe_report + '</pre></html>'
        atomic_write(page_dir / "index.html", page.encode())
    return status, qc
