import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from research.nba_market.__main__ import main
from research.nba_market.analysis import analyze, opening_comparison, summary_tables
from research.nba_market.core import load_config
from research.nba_market.dataset import build_dataset, line_movement
from research.nba_market.ingest import Warehouse
from research.nba_market.report import publish_outputs, quality_control
from tests.market_fixtures import event, odds_envelope, score_envelope


class EndToEndTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.config=load_config();self.config["bootstrap_repetitions"]=2
        self.w=Warehouse(self.root,self.config)
        self.counter=0

    def tearDown(self):
        self.w.close();self.temp.cleanup()

    def ingest(self,envelope):
        self.counter+=1
        path=self.root/f"fixture_{self.counter}.json";path.write_text(json.dumps(envelope))
        self.w.import_archive(path)

    def game(self,day="2022-10-20",event_id="synthetic_event_1",scores=True,flip=False):
        tip=day+"T23:00:00Z"
        early=day+"T10:00:00Z";close=day+"T22:55:00Z"
        raw=event(timestamp=early,tip=tip,event_id=event_id)
        self.ingest(odds_envelope(self.config,raw,early))
        final=event(timestamp=close,tip=tip,event_id=event_id,home_ml=170 if flip else -200,away_ml=-200 if flip else 170)
        self.ingest(odds_envelope(self.config,final,close))
        if scores:
            self.ingest(score_envelope(raw))

    def test_pipeline_grades_actual_quotes_and_frozen_signals(self):
        self.game()
        events,raw,views,bets=build_dataset(self.w)
        self.assertEqual((len(events),len(raw)),(1,4))
        self.assertNotEqual(raw[0]["payload_hash"],raw[0]["outcome_payload_hash"])
        self.assertIn("EARLY",raw[0]["snapshot_type"])
        self.assertIsNotNone(raw[0]["consensus_home_no_vig"])
        spread=next(b for b in bets if b["basis"]=="CONSENSUS" and b["market"]=="spreads" and b["snapshot_type"]=="CLOSE" and b["mode"]=="FOLLOW")
        self.assertEqual(spread["execution_line"],-4.5)
        self.assertEqual(spread["reference_line"],-5)
        self.assertEqual(spread["result"],"loss")
        self.assertEqual(spread["net_units"],-1)
        ml=next(b for b in bets if b["basis"]=="CONSENSUS" and b["market"]=="h2h" and b["snapshot_type"]=="CLOSE" and b["mode"]=="FOLLOW")
        self.assertEqual(ml["net_units"],.5)
        self.assertFalse(quality_control(self.w,events,raw,views,bets)["errors"])

    def test_adding_final_scores_does_not_change_signal_ids(self):
        self.game(scores=False)
        before=build_dataset(self.w)[3]
        ids={r["signal_id"] for r in before}
        self.assertTrue(all(r["result"] is None for r in before))
        self.ingest(score_envelope(event()))
        after=build_dataset(self.w)[3]
        self.assertEqual(ids,{r["signal_id"] for r in after})

    def test_four_season_analysis_and_idempotent_exports(self):
        for season,day in [("2022-23","2022-10-20"),("2023-24","2023-10-26"),("2024-25","2024-10-24"),("2025-26","2025-10-23")]:
            self.game(day,season)
        data=build_dataset(self.w)
        tables=analyze(data[2],data[3],self.config)
        status,qc=publish_outputs(self.w,*data,tables)
        self.assertEqual(status["graded_events"],4)
        self.assertEqual(status["seasons_collected"],list(self.config["seasons"]))
        self.assertFalse(qc["errors"])
        self.assertTrue(tables["progression_simulation"])
        self.assertTrue(tables["maximum_risk_report"])
        before=json.loads((self.root/"manifest.json").read_text())["output_hashes"]
        publish_outputs(self.w,*build_dataset(self.w),analyze(data[2],data[3],self.config))
        after=json.loads((self.root/"manifest.json").read_text())["output_hashes"]
        self.assertEqual(before,after)
        self.assertTrue(all(not r["automatic_live_integration"] for r in tables["season_validation"]))

    def test_open_favorite_flips_and_market_correctness(self):
        self.game(flip=True)
        _,_,views,bets=build_dataset(self.w)
        movements=line_movement(views,self.config)
        details,summary=opening_comparison(bets,movements,self.config)
        row=next(r for r in details if r["basis"]=="CONSENSUS" and r["market"]=="h2h")
        self.assertTrue(row["favorite_flipped"])
        self.assertEqual(row["comparison"],"Open right / Close wrong")

    def test_same_favorite_does_not_manufacture_correction(self):
        self.game()
        _,_,views,bets=build_dataset(self.w)
        details,_=opening_comparison(bets,line_movement(views,self.config),self.config)
        row=next(r for r in details if r["basis"]=="CONSENSUS" and r["market"]=="h2h")
        self.assertEqual(row["comparison"],"Open right / Close right")

    def test_underdog_band_uses_paired_favorite_roi(self):
        self.game()
        bets=build_dataset(self.w)[3]
        _,dogs,_=summary_tables(bets,self.config)
        row=next(r for r in dogs if r["basis"]=="CONSENSUS" and r["season"]=="ALL" and r["snapshot_type"]=="CLOSE" and r["early_bucket"]=="ALL_EARLY")
        self.assertEqual(row["underdog_flat_bet_roi"],-1)
        self.assertEqual(row["favorite_flat_bet_roi"],.5)

    def test_empty_outputs_are_ungraded_not_profitable(self):
        data=build_dataset(self.w);tables=analyze(data[2],data[3],self.config)
        status,qc=publish_outputs(self.w,*data,tables,self.root/"page")
        self.assertEqual(status["status"],"blocked_no_historical_data")
        self.assertFalse(status["live_shadow_test_warranted"])
        self.assertIn("Not estimable",(self.root/"REPORT.md").read_text())
        self.assertTrue((self.root/"progression_simulation.csv").read_text().startswith("season,"))
        self.assertIn("No live recommended bets",(self.root/"page/index.html").read_text())

    def test_existing_production_artifacts_not_changed(self):
        repo=Path(__file__).resolve().parents[1]
        paths=list((repo/"models").glob("*"))+list((repo/"predictions").glob("*"))+list((repo/"data/tracking").glob("*"))
        before={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.is_file()}
        self.game();data=build_dataset(self.w);publish_outputs(self.w,*data,analyze(data[2],data[3],self.config))
        self.assertEqual(before,{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.is_file()})

    def test_duplicate_matchup_ids_fail_quality(self):
        raw=event(event_id="one")
        envelope=odds_envelope(self.config,raw,"2022-10-19T23:00:00Z")
        envelope["payload"]["data"].append(event(event_id="two"))
        self.ingest(envelope)
        qc=quality_control(self.w,*build_dataset(self.w))
        self.assertIn("duplicate_matchup_date_with_different_event_ids_requires_resolution",qc["errors"])

    def test_conservative_rescheduled_tip_filters_later_odds(self):
        self.game(scores=False)
        raw=event(timestamp="2022-10-20T23:25:00Z",tip="2022-10-20T23:30:00Z")
        self.ingest(odds_envelope(self.config,raw,"2022-10-20T23:25:00Z"))
        events,rows,_,_=build_dataset(self.w)
        self.assertTrue(events[0]["schedule_changed"])
        self.assertTrue(all(r["snapshot_timestamp"] < events[0]["commence_time"] for r in rows))


if __name__ == "__main__":
    unittest.main()
