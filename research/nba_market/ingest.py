"""Quota-bounded, durable and resumable Odds API ingestion using only stdlib."""
from __future__ import annotations

import gzip
import json
import os
import sqlite3
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from .core import ET, fingerprint, iso, normalize_event, normalize_team, parse_time, season_for

SOURCE = "the-odds-api"
HOST = "https://api.the-odds-api.com"
ODDS_PATH = "/v4/historical/sports/basketball_nba/odds"
SCORES_PATH = "/v4/sports/basketball_nba/scores"
PUBLIC_PARAMS = {"date", "regions", "markets", "oddsFormat", "dateFormat", "daysFrom", "all"}


class Blocked(RuntimeError):
    def __init__(self, status, message):
        self.status = status
        super().__init__(message)


def atomic_write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with open(temp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


def write_json(path, value):
    atomic_write(path, (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode())


def resolve_key():
    for name in ("ODDS_API_KEY", "THE_ODDS_API_KEY", "THEODDS_API_KEY"):
        if os.environ.get(name, "").strip():
            return os.environ[name].strip(), name
    return None, None


def canonical_params(params):
    if set(params) - PUBLIC_PARAMS:
        raise ValueError("Unsupported or secret request parameter")
    values = {k: str(v) for k, v in params.items()}
    for field in ("markets", "regions"):
        if field in values:
            values[field] = ",".join(sorted(set(values[field].split(","))))
    if "date" in values:
        dt = parse_time(values["date"])
        minutes = 5 if dt >= parse_time("2022-09-01T00:00:00Z") else 10
        dt = dt.replace(minute=dt.minute // minutes * minutes, second=0, microsecond=0)
        values["date"] = iso(dt)
    return values


def transport(path, params, key):
    query = urllib.parse.urlencode(dict(params, apiKey=key))
    try:
        with urllib.request.urlopen(HOST + path + "?" + query, timeout=30) as response:
            return response.status, dict(response.headers), response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, dict(exc.headers), exc.read()
    except (urllib.error.URLError, TimeoutError, OSError):
        # Exception text/URL can include the API key; never propagate it.
        raise Blocked("uncertain_request", "Network request failed. Receipt may be uncertain; use --retry-uncertain only after checking quota.") from None


class Warehouse:
    def __init__(self, root, config):
        self.root, self.config = Path(root), config
        self.root.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.root / "warehouse.sqlite3", timeout=30)
        self.db.row_factory = sqlite3.Row
        self.db.executescript("""
        PRAGMA foreign_keys=ON;
        CREATE TABLE IF NOT EXISTS requests (
          request_key TEXT PRIMARY KEY, path TEXT NOT NULL, params TEXT NOT NULL,
          payload_hash TEXT NOT NULL, cache_path TEXT NOT NULL, snapshot_timestamp TEXT,
          fetched_at TEXT NOT NULL, headers TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS request_log (
          attempt_id INTEGER PRIMARY KEY, request_key TEXT NOT NULL, path TEXT NOT NULL,
          attempted_at TEXT NOT NULL, status TEXT NOT NULL, http_status INTEGER,
          estimated_cost INTEGER NOT NULL, actual_cost INTEGER, quota_remaining INTEGER, quota_used INTEGER);
        CREATE TABLE IF NOT EXISTS events (
          event_id TEXT PRIMARY KEY, season TEXT NOT NULL, commence_time TEXT NOT NULL,
          home_team TEXT NOT NULL, away_team TEXT NOT NULL, game_date TEXT NOT NULL,
          first_observed TEXT NOT NULL, last_observed TEXT NOT NULL, schedule_changed INTEGER NOT NULL DEFAULT 0);
        CREATE TABLE IF NOT EXISTS observations (
          event_id TEXT NOT NULL REFERENCES events(event_id), snapshot_timestamp TEXT NOT NULL,
          bookmaker TEXT NOT NULL, payload_hash TEXT NOT NULL, record TEXT NOT NULL,
          PRIMARY KEY(event_id, snapshot_timestamp, bookmaker));
        CREATE TABLE IF NOT EXISTS outcomes (
          event_id TEXT PRIMARY KEY REFERENCES events(event_id), home_score INTEGER NOT NULL,
          away_score INTEGER NOT NULL, result_available_at TEXT NOT NULL, source TEXT NOT NULL,
          payload_hash TEXT NOT NULL, fetched_at TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS issues (
          issue_key TEXT PRIMARY KEY, request_key TEXT, event_id TEXT, reason TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS plans (
          plan_key TEXT PRIMARY KEY, plan TEXT NOT NULL);
        """)
        self.db.commit()

    def close(self):
        self.db.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def rows(self, table):
        if table not in {"events", "observations", "outcomes", "requests", "request_log", "issues", "plans"}:
            raise ValueError("Unknown table")
        return [dict(r) for r in self.db.execute(f"SELECT * FROM {table}")]

    def issue(self, request_key, event_id, reason):
        key = fingerprint([request_key, event_id, reason])
        self.db.execute("INSERT OR IGNORE INTO issues VALUES (?, ?, ?, ?)", (key, request_key, event_id, reason))

    def event(self, event, timestamp):
        previous = self.db.execute("SELECT * FROM events WHERE event_id=?", (event["event_id"],)).fetchone()
        if previous:
            if any(previous[k] != event[k] for k in ("home_team", "away_team", "season")):
                raise ValueError("Stable event identifier changed opponents/season")
            tip = min(previous["commence_time"], event["commence_time"])
            self.db.execute("UPDATE events SET commence_time=?,game_date=?,last_observed=?,schedule_changed=? WHERE event_id=?",
                            (tip, min(previous["game_date"], event["game_date"]), max(previous["last_observed"], timestamp),
                             int(previous["schedule_changed"] or previous["commence_time"] != event["commence_time"]), event["event_id"]))
        else:
            self.db.execute("INSERT INTO events VALUES (?,?,?,?,?,?,?,?,0)",
                            tuple(event[k] for k in ("event_id", "season", "commence_time", "home_team", "away_team", "game_date")) + (timestamp, timestamp))

    def ingest(self, envelope, request_key):
        if envelope.get("source") != SOURCE or envelope["path"] not in {ODDS_PATH, SCORES_PATH, "/v4/sports"}:
            raise ValueError("Only unmodified Odds API response archives are accepted")
        params = canonical_params(envelope["params"])
        data = envelope["payload"]
        payload_hash = fingerprint(data)
        if envelope.get("payload_hash", payload_hash) != payload_hash:
            raise ValueError("Archive checksum does not match")
        fetched = iso(envelope["fetched_at"])
        path = envelope["path"]
        if path == ODDS_PATH:
            if not isinstance(data, dict) or not isinstance(data.get("data"), list):
                raise ValueError("Malformed historical odds response structure")
            timestamp = iso(data["timestamp"])
            if parse_time(timestamp) > parse_time(params["date"]):
                raise ValueError("API returned a snapshot after requested time")
            if (parse_time(params["date"])-parse_time(timestamp)).total_seconds() > 15*60:
                raise ValueError("Historical API snapshot too far from request")
            for raw in data["data"]:
                if not isinstance(raw, dict):
                    self.issue(request_key, None, "malformed_event_object")
                    continue
                try:
                    event, records, errors = normalize_event(raw, timestamp, self.config)
                    for error in errors:
                        self.issue(request_key, raw.get("id"), error)
                    if event is None:
                        continue
                    self.event(event, timestamp)
                    for row in records:
                        previous = self.db.execute("SELECT record FROM observations WHERE event_id=? AND snapshot_timestamp=? AND bookmaker=?",
                                                   (event["event_id"], timestamp, row["bookmaker"])).fetchone()
                        record = json.dumps(row, sort_keys=True, allow_nan=False)
                        if previous and previous["record"] != record:
                            self.issue(request_key, event["event_id"], "conflicting_duplicate_observation")
                        self.db.execute("INSERT OR IGNORE INTO observations VALUES (?,?,?,?,?)",
                                        (event["event_id"], timestamp, row["bookmaker"], payload_hash, record))
                except (ValueError, KeyError, TypeError) as exc:
                    self.issue(request_key, raw.get("id"), str(exc))
        elif path == SCORES_PATH:
            timestamp = None
            if not isinstance(data, list):
                raise ValueError("Scores archive must contain API response list")
            for raw in data:
                if not raw.get("completed"):
                    continue
                event, _, _ = normalize_event(raw, raw["commence_time"], self.config)
                if event is None:
                    continue
                update = iso(raw["last_update"])
                if not parse_time(raw["commence_time"]) < parse_time(update) <= parse_time(fetched):
                    raise ValueError("Score availability time is inconsistent")
                if (parse_time(fetched)-parse_time(raw["commence_time"])).total_seconds() > 4*86400:
                    raise ValueError("Receipt is outside Odds API recent-scores coverage; not a credible archived response")
                scores = {normalize_team(s["name"]): int(s["score"]) for s in raw["scores"]}
                h, a = scores[event["home_team"]], scores[event["away_team"]]
                if len(scores) != 2 or not 0 < h <= 300 or not 0 < a <= 300 or h == a:
                    raise ValueError("Impossible completed NBA score")
                self.event(event, update)
                prev = self.db.execute("SELECT home_score,away_score FROM outcomes WHERE event_id=?", (event["event_id"],)).fetchone()
                if prev and (prev[0], prev[1]) != (h, a):
                    raise ValueError("Conflicting final score: resolve archive before grading")
                self.db.execute("INSERT OR IGNORE INTO outcomes VALUES (?,?,?,?,?,?,?)",
                                (event["event_id"], h, a, update, SOURCE, payload_hash, fetched))
        else:
            timestamp = None
        cache_path = f"cache/{request_key}.json.gz"
        self.db.execute("INSERT OR IGNORE INTO requests VALUES (?,?,?,?,?,?,?,?)",
                        (request_key, path, json.dumps(params, sort_keys=True), payload_hash, cache_path, timestamp,
                         fetched, json.dumps(envelope.get("headers", {}), sort_keys=True)))

    def import_archive(self, archive):
        envelope = json.loads(Path(archive).read_text())
        params = canonical_params(envelope["params"])
        # Live scores requests at different dates must not share a historical cache key.
        key = fingerprint([envelope["path"], params, iso(envelope["fetched_at"])]) if envelope["path"] == SCORES_PATH else fingerprint([envelope["path"], params])
        previous = self.db.execute("SELECT payload_hash FROM requests WHERE request_key=?", (key,)).fetchone()
        if previous:
            if previous[0] != fingerprint(envelope["payload"]):
                raise ValueError("Conflicting archive for the same historical request; cached receipt is immutable")
            return key
        # Validate inside transaction before writing an accepted receipt.
        with self.db:
            self.ingest(envelope, key)
        atomic_write(self.root / f"cache/{key}.json.gz", gzip.compress(json.dumps(envelope, sort_keys=True).encode(), mtime=0))
        return key


class Client:
    def __init__(self, warehouse, max_requests=500, max_credits=15000, reserve=200,
                 key=None, send=transport, retry_uncertain=False):
        self.warehouse, self.send = warehouse, send
        self.key = key if key is not None else resolve_key()[0]
        self.max_requests, self.max_credits, self.reserve = max_requests, max_credits, reserve
        self.retry_uncertain = retry_uncertain
        self.requests, self.credits, self.cache_hits = 0, 0, 0
        self.remaining = None
        self.used = None

    def fetch(self, path, params, cost):
        params = canonical_params(params)
        now = iso(datetime.now(timezone.utc))
        request_key = fingerprint([path, params, now]) if path in {SCORES_PATH, "/v4/sports"} else fingerprint([path, params])
        cached = self.warehouse.db.execute("SELECT cache_path FROM requests WHERE request_key=?", (request_key,)).fetchone()
        cache_path = self.warehouse.root / (cached[0] if cached else f"cache/{request_key}.json.gz")
        if cache_path.exists():
            try:
                envelope = json.loads(gzip.decompress(cache_path.read_bytes()))
            except (OSError, ValueError):
                raise Blocked("corrupt_cache", "Historical response receipt cannot be decoded; restore it before resuming.") from None
            identity = [envelope["path"], canonical_params(envelope["params"])]
            if path in {SCORES_PATH, "/v4/sports"}:
                identity.append(iso(envelope["fetched_at"]))
            if fingerprint(identity) != request_key:
                raise Blocked("corrupt_cache", "Historical cache identity mismatch; no automatic refetch.")
            payload_hash = fingerprint(envelope["payload"])
            if envelope.get("payload_hash", payload_hash) != payload_hash:
                raise Blocked("corrupt_cache", "Historical payload checksum mismatch; restore the original receipt before continuing.")
            if not cached:
                with self.warehouse.db:
                    self.warehouse.ingest(envelope, request_key)
            self.cache_hits += 1
            return envelope["payload"]
        if cached:
            raise Blocked("missing_cache_file", "Paid response receipt exists but its cache file is missing. Restore it before fetching.")
        last = self.warehouse.db.execute("SELECT status FROM request_log WHERE request_key=? ORDER BY attempt_id DESC LIMIT 1", (request_key,)).fetchone()
        if last and last[0] in {"in_flight", "uncertain_request"} and not self.retry_uncertain:
            raise Blocked("uncertain_request", "Previous request may have consumed credits. Check account usage then resume with --retry-uncertain.")
        if not self.key:
            raise Blocked("missing_api_key", "Set repository Actions secret ODDS_API_KEY to a paid historical-access key, or export ODDS_API_KEY locally.")
        if self.requests >= self.max_requests or self.credits + cost > self.max_credits:
            raise Blocked("budget_limit", "This run reached its request/credit budget. All completed requests are cached; rerun to resume.")
        if self.remaining is not None and self.remaining - cost < self.reserve:
            raise Blocked("quota_reserve", "Stopping before consuming the configured quota reserve.")
        with self.warehouse.db:
            cursor = self.warehouse.db.execute("INSERT INTO request_log(request_key,path,attempted_at,status,estimated_cost) VALUES (?,?,?,?,?)",
                                               (request_key, path, now, "in_flight", cost))
        attempt = cursor.lastrowid
        self.requests += 1
        try:
            status, headers, raw = self.send(path, params, self.key)
        except Blocked:
            with self.warehouse.db:
                self.warehouse.db.execute("UPDATE request_log SET status='uncertain_request' WHERE attempt_id=?", (attempt,))
            self.credits += cost
            raise
        safe_headers = {k.lower(): str(v) for k, v in headers.items() if k.lower().startswith("x-requests-")}
        def header_num(name):
            try:
                return int(float(safe_headers[name]))
            except (KeyError, TypeError, ValueError):
                return None
        last_cost = header_num("x-requests-last")
        self.credits += last_cost if last_cost is not None else cost
        self.remaining, self.used = header_num("x-requests-remaining"), header_num("x-requests-used")
        reason = {401: "invalid_api_key", 402: "quota_exhausted", 403: "historical_plan_unavailable",
                  429: "rate_limited"}.get(status, "ok" if status == 200 else f"http_{status}")
        with self.warehouse.db:
            self.warehouse.db.execute("UPDATE request_log SET status=?,http_status=?,actual_cost=?,quota_remaining=?,quota_used=? WHERE attempt_id=?",
                                      (reason, status, last_cost, self.remaining, self.used, attempt))
        if status != 200:
            raise Blocked(reason, f"Odds API returned HTTP {status}. Fix credential/plan/quota or retry a transient error; no key or response error text is logged.")
        try:
            data = json.loads(raw)
            envelope = {"source": SOURCE, "path": path, "params": params, "headers": safe_headers,
                        "fetched_at": now, "payload": data, "payload_hash": fingerprint(data)}
            # Cache before normalization, so a parser defect never requires another paid request.
            atomic_write(cache_path, gzip.compress(json.dumps(envelope, sort_keys=True).encode(), mtime=0))
            with self.warehouse.db:
                self.warehouse.ingest(envelope, request_key)
        except (ValueError, KeyError, TypeError):
            raise Blocked("response_validation_failed", "Response is preserved. Repair validation and resume from cache without refetching.") from None
        return data

    def status(self):
        return {"network_requests_this_run": self.requests, "credits_this_run": self.credits,
                "cache_hits_this_run": self.cache_hits, "api_requests_remaining": self.remaining,
                "api_requests_used": self.used, "max_requests": self.max_requests,
                "max_credits": self.max_credits, "quota_reserve": self.reserve,
                "credential_present": bool(self.key)}


def discovery_tasks(config, seasons, reference=False):
    tasks = []
    for season in seasons:
        spec = config["seasons"][season]
        start, end = date.fromisoformat(spec["start"]), date.fromisoformat(spec["end"])
        if not reference:
            end = min(end, start + timedelta(days=config["discovery_days"]-1))
        day = start
        while day <= end:
            if day.isoformat() not in spec.get("excluded_dates", []):
                for hour in config["discovery_hours_et"]:
                    snapshot = datetime(day.year, day.month, day.day, hour, tzinfo=ET)
                    tasks.append({"season": season, "stage": "DISCOVER", "timestamp": iso(snapshot)})
            day += timedelta(days=1)
    return tasks


def market_params(config, timestamp):
    return {"date": timestamp, "regions": ",".join(config["regions"]), "markets": ",".join(config["markets"]),
            "oddsFormat": "american", "dateFormat": "iso"}


def build_plan(warehouse, seasons, reference=False):
    tasks = discovery_tasks(warehouse.config, seasons, reference)
    from .core import number_games
    events = number_games(warehouse.rows("events"))
    eligible = [e for e in events if e["season"] in seasons and (reference or min(e["team_game_number_home"], e["team_game_number_away"]) <= 20)]
    seen = {canonical_params(market_params(warehouse.config, t["timestamp"]))["date"] for t in tasks}
    for event in eligible:
        tip = parse_time(event["commence_time"])
        for stage, offset in (("EARLY", timedelta(hours=warehouse.config["early_hours"])),
                              ("CLOSE", timedelta(seconds=warehouse.config["close_buffer_seconds"]))):
            timestamp = canonical_params({"date": iso(tip-offset)})["date"]
            if timestamp not in seen:
                seen.add(timestamp)
                tasks.append({"season": event["season"], "stage": stage, "timestamp": timestamp})
    existing = {r["request_key"] for r in warehouse.rows("requests")}
    cost = 10*len(warehouse.config["regions"])*len(warehouse.config["markets"])
    for task in tasks:
        task["request_key"] = fingerprint([ODDS_PATH, canonical_params(market_params(warehouse.config, task["timestamp"]))])
        task["cached"] = task["request_key"] in existing
    tasks.sort(key=lambda t: (0 if t["stage"] == "DISCOVER" else 1, t["timestamp"]))
    return {"seasons": list(seasons), "reference_full_season": reference, "tasks": tasks,
            "planned_requests": len(tasks), "uncached_requests": sum(not t["cached"] for t in tasks),
            "estimated_remaining_credits": sum(not t["cached"] for t in tasks)*cost,
            "estimate_excludes_undiscovered_event_specific_snapshots": True, "credits_per_snapshot": cost}


def fetch_history(warehouse, client, seasons, reference=False):
    status, message = "ok", "All planned discovery and event snapshots completed."
    try:
        if client.key:
            # Free endpoint provides quota metadata before spending historical credits.
            client.fetch("/v4/sports", {"all": "true"}, 0)
        # First discover calendar; then construct individual early/close tasks. Reruns use receipts.
        for stage in ("DISCOVER", "EVENTS"):
            plan = build_plan(warehouse, seasons, reference)
            write_json(warehouse.root / "fetch_plan.json", plan)
            with warehouse.db:
                warehouse.db.execute("INSERT OR REPLACE INTO plans VALUES (?,?)", (fingerprint([seasons, reference]), json.dumps(plan)))
            for task in plan["tasks"]:
                if (task["stage"] == "DISCOVER") != (stage == "DISCOVER"):
                    continue
                client.fetch(ODDS_PATH, market_params(warehouse.config, task["timestamp"]), plan["credits_per_snapshot"])
    except Blocked as exc:
        status, message = exc.status, str(exc)
    result = dict(client.status(), status=status, message=message, requested_seasons=list(seasons))
    write_json(warehouse.root / "ingestion_status.json", result)
    # Updated after the latest successfully ingested snapshot, even when budget/key blocked.
    write_json(warehouse.root / "fetch_plan.json", build_plan(warehouse, seasons, reference))
    return result
