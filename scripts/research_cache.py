"""Restore/persist paid historical receipts in a dedicated durable Git data branch.

Actions caches/artifacts are fallbacks, not the only copy of purchased snapshots.
Never rewrites main, never force-pushes, and never handles API credentials.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import shutil
import sqlite3
import subprocess
import tempfile
from pathlib import Path

BRANCH = "market-research-data"
ROOT = Path("data/research/nba_market")
CHUNK_BYTES = 32 * 1024 * 1024


def database_identity(database):
    with sqlite3.connect(database) as db:
        db.row_factory = sqlite3.Row
        content = {}
        for table in ("events", "observations", "outcomes", "requests", "request_log", "issues"):
            content[table] = sorted([dict(r) for r in db.execute(f"SELECT * FROM {table}")],
                                    key=lambda r: json.dumps(r, sort_keys=True))
    return hashlib.sha256(json.dumps(content, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def pack_database(database, destination):
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="nba-market-pack-") as temp:
        packed = Path(temp)/"state.gz"
        with packed.open("wb") as output:
            with gzip.GzipFile(filename="", mode="wb", fileobj=output, mtime=0) as compressed:
                with database.open("rb") as source:
                    shutil.copyfileobj(source, compressed)
        parts = []
        for old in destination.glob("state-*.gzpart"):
            old.unlink()
        with packed.open("rb") as source:
            index = 0
            while chunk := source.read(CHUNK_BYTES):
                name = f"state-{index:04d}.gzpart"
                (destination/name).write_bytes(chunk)
                parts.append({"name": name, "sha256": hashlib.sha256(chunk).hexdigest(), "bytes": len(chunk)})
                index += 1
        manifest = {"parts": parts, "logical_state_hash": database_identity(database),
                    "database_bytes": database.stat().st_size, "chunk_limit_bytes": CHUNK_BYTES}
        (destination/"manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=2)+"\n")


def unpack_database(source, destination):
    manifest = json.loads((source/"manifest.json").read_text())
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="nba-market-unpack-") as temp:
        packed = Path(temp)/"state.gz"
        with packed.open("wb") as output:
            for part in manifest["parts"]:
                name = part["name"]
                if Path(name).name != name or not name.startswith("state-") or not name.endswith(".gzpart"):
                    raise ValueError("Invalid state chunk name")
                chunk = (source/name).read_bytes()
                if len(chunk) != part["bytes"] or hashlib.sha256(chunk).hexdigest() != part["sha256"]:
                    raise ValueError("Research state chunk checksum mismatch")
                output.write(chunk)
        restored = destination.with_suffix(".restoring")
        with gzip.open(packed,"rb") as archive, restored.open("wb") as output:
            shutil.copyfileobj(archive,output)
        if restored.stat().st_size != manifest["database_bytes"] or database_identity(restored) != manifest["logical_state_hash"]:
            restored.unlink()
            raise ValueError("Research state database checksum mismatch")
        os.replace(restored,destination)


def git(*args, cwd=None, check=True):
    return subprocess.run(["git", *args], cwd=cwd, check=check, capture_output=True, text=True)


def restore():
    result = git("ls-remote", "--heads", "origin", BRANCH)
    if not result.stdout.strip():
        print("No paid research data branch yet; starting with local receipts.")
        return
    git("fetch", "--depth", "1", "origin", BRANCH)
    with tempfile.TemporaryDirectory(prefix="nba-market-state-") as temp:
        tree = Path(temp)/"state"
        git("worktree", "add", "--detach", str(tree), "FETCH_HEAD")
        try:
            source = tree/ROOT
            if source.exists():
                ROOT.mkdir(parents=True, exist_ok=True)
                for name in ("cache", "warehouse.sqlite3"):
                    path = source/name
                    if path.is_dir():
                        shutil.copytree(path, ROOT/name, dirs_exist_ok=True)
                    elif path.is_file():
                        shutil.copy2(path, ROOT/name)
                if (source/"cache_state/manifest.json").exists():
                    unpack_database(source/"cache_state", ROOT/"warehouse.sqlite3")
                print("Restored durable historical receipts and ingestion state.")
        finally:
            git("worktree", "remove", "--force", str(tree))


def persist():
    database = ROOT/"warehouse.sqlite3"
    if not database.exists():
        print("No ingestion database to persist.")
        return
    with sqlite3.connect(database) as db:
        if db.execute("SELECT COUNT(*) FROM requests").fetchone()[0] == 0:
            print("No successful API responses yet; no paid cache branch needed.")
            return
    remote = git("ls-remote", "--heads", "origin", BRANCH)
    if remote.stdout.strip():
        git("fetch", "--depth", "1", "origin", BRANCH)
        base = "FETCH_HEAD"
    else:
        base = "HEAD"
    with tempfile.TemporaryDirectory(prefix="nba-market-state-") as temp:
        tree = Path(temp)/"state"
        git("worktree", "add", "--detach", str(tree), base)
        try:
            destination = tree/ROOT
            destination.mkdir(parents=True, exist_ok=True)
            if (ROOT/"cache").exists():
                shutil.copytree(ROOT/"cache", destination/"cache", dirs_exist_ok=True)
            existing_manifest = destination/"cache_state/manifest.json"
            if existing_manifest.exists() and json.loads(existing_manifest.read_text())["logical_state_hash"] == database_identity(database):
                print("Historical responses and state already persisted; no duplicate commit.")
                return
            pack_database(database, destination/"cache_state")
            git("config", "user.name", "NBA Research Bot", cwd=tree)
            git("config", "user.email", "actions@github.com", cwd=tree)
            git("add", "-f", str(ROOT/"cache"), str(ROOT/"cache_state"), cwd=tree)
            changed = git("diff", "--cached", "--quiet", cwd=tree, check=False)
            if changed.returncode == 0:
                print("Historical receipts already persisted; no duplicate commit.")
                return
            git("commit", "-m", "Preserve cached NBA historical Odds API responses and resume state", cwd=tree)
            git("push", "origin", f"HEAD:refs/heads/{BRANCH}", cwd=tree)
            print("Historical responses persisted to the research data branch.")
        finally:
            git("worktree", "remove", "--force", str(tree))


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("action",choices=("restore","persist"))
    args=parser.parse_args()
    restore() if args.action == "restore" else persist()
