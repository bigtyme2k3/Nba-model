import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from research.nba_market.report import export_table,read_exported_table


class ResearchExportPartitionTests(unittest.TestCase):
    def test_large_table_partitions_without_losing_rows(self):
        rows=[{"event_id":str(i),"value":"x"*100} for i in range(30)]
        with tempfile.TemporaryDirectory() as temp, patch('research.nba_market.report.MAX_EXPORT_BYTES',1000):
            parts=export_table(Path(temp),'partition_test',rows)
            self.assertGreater(len(parts),1)
            self.assertEqual(read_exported_table(temp,'partition_test'),rows)
            index=json.loads((Path(temp)/'partition_test.json').read_text())
            self.assertEqual(index['total_rows'],30)
            self.assertTrue(all((Path(temp)/p['json']).stat().st_size < 1000 for p in parts))

    def test_rerun_to_small_table_removes_obsolete_parts(self):
        with tempfile.TemporaryDirectory() as temp, patch('research.nba_market.report.MAX_EXPORT_BYTES',1000):
            export_table(Path(temp),'partition_test',[{"i":i,"value":"x"*100} for i in range(30)])
            export_table(Path(temp),'partition_test',[{"i":1}])
            self.assertFalse(list(Path(temp).glob('*-part-*')))
            self.assertEqual(read_exported_table(temp,'partition_test'),[{"i":1}])
