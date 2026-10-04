"""python -m research.nba_market --help"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .analysis import analyze
from .core import load_config
from .dataset import build_dataset
from .ingest import Client, SCORES_PATH, Warehouse, build_plan, fetch_history, write_json
from .report import publish_outputs

DEFAULT_ROOT = "data/research/nba_market"


def main(argv=None):
    parser = argparse.ArgumentParser(description="Isolated NBA historical Odds API market/fade research")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("plan", "fetch", "analyze", "validate", "run", "import-archive", "capture-scores"):
        p = sub.add_parser(name)
        p.add_argument("--root", default=DEFAULT_ROOT)
        p.add_argument("--config")
        p.add_argument("--seasons", nargs="+", help="Defaults to every season in config")
        p.add_argument("--full-season-reference", action="store_true")
        p.add_argument("--page-dir", default="docs/market-research")
        p.add_argument("--no-page", action="store_true")
        p.add_argument("--max-requests", type=int, default=500)
        p.add_argument("--max-credits", type=int, default=15000)
        p.add_argument("--quota-reserve", type=int, default=200)
        p.add_argument("--retry-uncertain", action="store_true")
        if name == "import-archive":
            p.add_argument("--file", required=True, type=Path)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    seasons = args.seasons or list(config["seasons"])
    if set(seasons)-set(config["seasons"]):
        parser.error("Unknown season: add a verified calendar entry to config first")
    if min(args.max_requests, args.max_credits, args.quota_reserve) < 0:
        parser.error("Request/credit limits and reserve must be nonnegative")
    if args.seasons:
        config["seasons"] = {s: config["seasons"][s] for s in seasons}
    with Warehouse(args.root, config) as warehouse:
        result = None
        client = Client(warehouse, args.max_requests, args.max_credits, args.quota_reserve, retry_uncertain=args.retry_uncertain)
        if args.command == "plan":
            plan = build_plan(warehouse, seasons, args.full_season_reference)
            write_json(Path(args.root) / "fetch_plan.json", plan)
            print(json.dumps({k: v for k, v in plan.items() if k != "tasks"}, indent=2))
            return 0
        if args.command == "import-archive":
            key = warehouse.import_archive(args.file)
            print(json.dumps({"imported_response_receipt": key, "network_requests": 0}))
        elif args.command in {"fetch", "run"}:
            result = fetch_history(warehouse, client, seasons, args.full_season_reference)
            print(json.dumps(result, indent=2))
            if args.command == "fetch":
                return 0 if result["status"] == "ok" else 2
        elif args.command == "capture-scores":
            from .ingest import Blocked
            try:
                client.fetch(SCORES_PATH, {"daysFrom": "3", "dateFormat": "iso"}, 2)
            except Blocked as exc:
                print(json.dumps({"status": exc.status, "message": str(exc)}))
                return 2
        events, raw, views, bets = build_dataset(warehouse)
        tables = analyze(views, bets, config)
        status, qc = publish_outputs(warehouse, events, raw, views, bets, tables,
                                     None if args.no_page else args.page_dir)
        print(json.dumps({"analysis_status": status["status"], "events": status["events"],
                          "sportsbook_observations": status["sportsbook_observations"], "graded_events": status["graded_events"],
                          "validation_status": qc["status"], "validation_errors": qc["errors"]}, indent=2))
        if qc["errors"]:
            return 1
        return 2 if args.command == "run" and (result["status"] != "ok" or not status["graded_events"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
