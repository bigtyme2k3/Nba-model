import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from research.nba_market.core import load_config
from research.nba_market.ingest import Warehouse

path=Path(__file__).resolve().parents[1]/"scripts/research_cache.py"
spec=importlib.util.spec_from_file_location("research_cache_storage",path)
cache=importlib.util.module_from_spec(spec);spec.loader.exec_module(cache)


class CacheStorageTests(unittest.TestCase):
    def test_compressed_chunk_restore_and_checksums(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            with Warehouse(root/"original",load_config()):
                pass
            database=root/"original/warehouse.sqlite3"
            old_size=cache.CHUNK_BYTES
            try:
                cache.CHUNK_BYTES=100
                cache.pack_database(database,root/"packed")
                manifest=json.loads((root/"packed/manifest.json").read_text())
                self.assertGreater(len(manifest["parts"]),1)
                self.assertTrue(all(p["bytes"] <= 100 for p in manifest["parts"]))
                cache.unpack_database(root/"packed",root/"restored.sqlite3")
                self.assertEqual(cache.database_identity(database),cache.database_identity(root/"restored.sqlite3"))
            finally:
                cache.CHUNK_BYTES=old_size

    def test_damaged_chunk_never_overwrites_database(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            with Warehouse(root/"original",load_config()):
                pass
            cache.pack_database(root/"original/warehouse.sqlite3",root/"packed")
            part=next((root/"packed").glob("*.gzpart"));part.write_bytes(b"corrupted")
            with self.assertRaises(ValueError):
                cache.unpack_database(root/"packed",root/"restored.sqlite3")
            self.assertFalse((root/"restored.sqlite3").exists())
