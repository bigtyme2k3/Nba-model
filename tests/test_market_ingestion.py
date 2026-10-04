import gzip
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from research.nba_market.core import fingerprint, load_config
from research.nba_market.ingest import *
from tests.market_fixtures import event, odds_envelope, score_envelope


class IngestionTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name)
        self.config=load_config()
        self.w=Warehouse(self.root,self.config)
        self.calls=[]

    def tearDown(self):
        self.w.close();self.temp.cleanup()

    def send(self,path,params,key):
        self.calls.append((path,dict(params)))
        self.assertNotIn("apiKey",params)
        stamp=params.get("date","2022-10-19T23:00:00Z")
        payload=[] if path == "/v4/sports" else odds_envelope(self.config,event(timestamp=stamp),stamp)["payload"]
        cost="0" if path == "/v4/sports" else "30"
        return 200,{"X-Requests-Last":cost,"x-requests-remaining":"970","x-requests-used":"30"},json.dumps(payload).encode()

    def params(self,date="2022-10-19T23:00:00Z"):
        return market_params(self.config,date)

    def archive(self,envelope,name="archive.json"):
        path=self.root/name;path.write_text(json.dumps(envelope));return path

    def test_secret_alias_resolution(self):
        with patch.dict(os.environ,{"THE_ODDS_API_KEY":"fake-fixture-key"},clear=True):
            self.assertEqual(resolve_key(),("fake-fixture-key","THE_ODDS_API_KEY"))

    def test_key_priority(self):
        with patch.dict(os.environ,{"ODDS_API_KEY":"primary-fixture","THE_ODDS_API_KEY":"alias-fixture"},clear=True):
            self.assertEqual(resolve_key()[1],"ODDS_API_KEY")

    def test_missing_key_is_exact_blocker(self):
        c=Client(self.w,key="",send=self.send)
        with self.assertRaises(Blocked) as context:
            c.fetch(ODDS_PATH,self.params(),30)
        self.assertEqual(context.exception.status,"missing_api_key")
        self.assertFalse(self.calls)

    def test_cache_prevents_repeat_purchase(self):
        c=Client(self.w,key="fixture",send=self.send)
        first=c.fetch(ODDS_PATH,self.params(),30)
        second=c.fetch(ODDS_PATH,self.params(),30)
        self.assertEqual(first,second)
        self.assertEqual(len(self.calls),1)
        self.assertEqual(c.cache_hits,1)
        self.assertEqual(len(self.w.rows("observations")),2)

    def test_same_five_minute_interval_uses_one_purchase(self):
        c=Client(self.w,key="fixture",send=self.send)
        c.fetch(ODDS_PATH,self.params(),30)
        c.fetch(ODDS_PATH,self.params("2022-10-19T23:02:00Z"),30)
        self.assertEqual(len(self.calls),1)

    def test_request_parameter_order_is_idempotent(self):
        c=Client(self.w,key="fixture",send=self.send)
        c.fetch(ODDS_PATH,self.params(),30)
        params=self.params();params["markets"]="totals,spreads,h2h"
        c.fetch(ODDS_PATH,params,30)
        self.assertEqual(len(self.calls),1)

    def test_cache_usable_without_key(self):
        Client(self.w,key="fixture",send=self.send).fetch(ODDS_PATH,self.params(),30)
        Client(self.w,key="",send=self.send).fetch(ODDS_PATH,self.params(),30)
        self.assertEqual(len(self.calls),1)

    def test_resume_after_budget_stop(self):
        c=Client(self.w,max_requests=1,max_credits=30,reserve=0,key="fixture",send=self.send)
        c.fetch(ODDS_PATH,self.params(),30)
        with self.assertRaises(Blocked) as context:
            c.fetch(ODDS_PATH,self.params("2022-10-20T22:55:00Z"),30)
        self.assertEqual(context.exception.status,"budget_limit")
        resumed=Client(self.w,max_requests=1,max_credits=30,reserve=0,key="fixture",send=self.send)
        resumed.fetch(ODDS_PATH,self.params(),30)
        resumed.fetch(ODDS_PATH,self.params("2022-10-20T22:55:00Z"),30)
        self.assertEqual(len(self.calls),2)
        self.assertEqual(len(self.w.rows("observations")),4)

    def test_actual_quota_recorded(self):
        c=Client(self.w,key="fixture",send=self.send)
        c.fetch(ODDS_PATH,self.params(),30)
        self.assertEqual(c.credits,30)
        self.assertEqual(c.remaining,970)
        log=self.w.rows("request_log")[0]
        self.assertEqual((log["actual_cost"],log["quota_used"]),(30,30))

    def test_reserve_stops_next_purchase(self):
        c=Client(self.w,reserve=960,key="fixture",send=self.send)
        c.fetch(ODDS_PATH,self.params(),30)
        with self.assertRaises(Blocked) as context:
            c.fetch(ODDS_PATH,self.params("2022-10-20T22:55:00Z"),30)
        self.assertEqual(context.exception.status,"quota_reserve")
        self.assertEqual(len(self.calls),1)

    def test_disk_receipt_recovers_crash_before_database_commit(self):
        envelope=odds_envelope(self.config,event(),"2022-10-19T23:00:00Z")
        key=fingerprint([ODDS_PATH,canonical_params(self.params())])
        atomic_write(self.root/f"cache/{key}.json.gz",gzip.compress(json.dumps(envelope).encode()))
        c=Client(self.w,key="",send=self.send)
        c.fetch(ODDS_PATH,self.params(),30)
        self.assertFalse(self.calls)
        self.assertEqual(len(self.w.rows("observations")),2)

    def test_missing_cache_not_automatically_rebought(self):
        c=Client(self.w,key="fixture",send=self.send)
        c.fetch(ODDS_PATH,self.params(),30)
        cache=self.root/self.w.rows("requests")[0]["cache_path"];cache.unlink()
        with self.assertRaises(Blocked) as context:
            c.fetch(ODDS_PATH,self.params(),30)
        self.assertEqual(context.exception.status,"missing_cache_file")
        self.assertEqual(len(self.calls),1)

    def test_uncertain_network_request_not_automatically_retried(self):
        def failed(*args):
            raise Blocked("uncertain_request","safe network failure")
        c=Client(self.w,key="fixture",send=failed)
        with self.assertRaises(Blocked):
            c.fetch(ODDS_PATH,self.params(),30)
        with self.assertRaises(Blocked) as context:
            Client(self.w,key="fixture",send=self.send).fetch(ODDS_PATH,self.params(),30)
        self.assertEqual(context.exception.status,"uncertain_request")
        self.assertFalse(self.calls)

    def test_explicit_uncertain_retry(self):
        key=fingerprint([ODDS_PATH,canonical_params(self.params())])
        with self.w.db:
            self.w.db.execute("INSERT INTO request_log(request_key,path,attempted_at,status,estimated_cost) VALUES (?,?,?,?,?)",(key,ODDS_PATH,"2026-10-04T19:00:00Z","uncertain_request",30))
        Client(self.w,key="fixture",send=self.send,retry_uncertain=True).fetch(ODDS_PATH,self.params(),30)
        self.assertEqual(len(self.calls),1)

    def test_http_error_does_not_expose_key(self):
        secret="SYNTHETIC_DO_NOT_LOG"
        def denied(*args):
            return 401,{},json.dumps({"message":"apiKey="+secret}).encode()
        c=Client(self.w,key=secret,send=denied)
        with self.assertRaises(Blocked) as context:
            c.fetch(ODDS_PATH,self.params(),30)
        self.assertNotIn(secret,str(context.exception))
        for path in self.root.rglob("*.json*"):
            self.assertNotIn(secret,path.read_bytes().decode(errors="ignore"))

    def test_key_not_in_successful_receipt_or_log(self):
        secret="SYNTHETIC_DO_NOT_LOG"
        Client(self.w,key=secret,send=self.send).fetch(ODDS_PATH,self.params(),30)
        receipt=gzip.decompress(next((self.root/"cache").glob("*.gz")).read_bytes()).decode()
        self.assertNotIn(secret,receipt)
        self.assertNotIn("apiKey",receipt)
        self.assertNotIn(secret,json.dumps(self.w.rows("request_log")))

    def test_duplicate_archive_prevention(self):
        envelope=odds_envelope(self.config,event(),"2022-10-19T23:00:00Z")
        path=self.archive(envelope)
        self.w.import_archive(path);self.w.import_archive(path)
        self.assertEqual(len(self.w.rows("events")),1)
        self.assertEqual(len(self.w.rows("observations")),2)
        self.assertEqual(len(self.w.rows("requests")),1)

    def test_conflicting_archive_cannot_replace_paid_receipt(self):
        envelope=odds_envelope(self.config,event(),"2022-10-19T23:00:00Z")
        self.w.import_archive(self.archive(envelope))
        envelope["payload"]["data"][0]["id"]="changed-id"
        with self.assertRaises(ValueError):
            self.w.import_archive(self.archive(envelope,"conflict.json"))

    def test_archived_scores_join_stable_id(self):
        raw=event()
        self.w.import_archive(self.archive(odds_envelope(self.config,raw,"2022-10-19T23:00:00Z")))
        self.w.import_archive(self.archive(score_envelope(raw),"scores.json"))
        self.assertEqual(self.w.rows("outcomes")[0]["home_score"],114)
        self.assertEqual(self.w.rows("outcomes")[0]["source"],SOURCE)

    def test_different_score_archive_dates_not_collapsed(self):
        first=event();second=event(event_id="second",tip="2022-10-22T23:00:00Z")
        self.w.import_archive(self.archive(score_envelope(first),"scores1.json"))
        self.w.import_archive(self.archive(score_envelope(second),"scores2.json"))
        self.assertEqual(len(self.w.rows("outcomes")),2)

    def test_other_source_not_accepted(self):
        envelope=score_envelope(event());envelope["source"]="espn"
        with self.assertRaises(ValueError):
            self.w.import_archive(self.archive(envelope))
        self.assertFalse(self.w.rows("outcomes"))

    def test_stale_score_receipt_not_claimed_historical_api(self):
        envelope=score_envelope(event());envelope["fetched_at"]="2026-10-04T19:00:00Z"
        with self.assertRaises(ValueError):
            self.w.import_archive(self.archive(envelope))

    def test_tied_completed_nba_score_rejected(self):
        with self.assertRaises(ValueError):
            self.w.import_archive(self.archive(score_envelope(event(),110,110)))

    def test_future_score_update_rejected(self):
        envelope=score_envelope(event());envelope["payload"][0]["last_update"]="2022-10-22T05:00:00Z"
        with self.assertRaises(ValueError):
            self.w.import_archive(self.archive(envelope))

    def test_archive_checksum_validation(self):
        envelope=score_envelope(event());envelope["payload_hash"]="wrong"
        with self.assertRaises(ValueError):
            self.w.import_archive(self.archive(envelope))

    def test_snapshot_after_requested_time_rejected(self):
        envelope=odds_envelope(self.config,event(),"2022-10-19T23:00:00Z")
        envelope["payload"]["timestamp"]="2022-10-19T23:05:00Z"
        with self.assertRaises(ValueError):
            self.w.import_archive(self.archive(envelope))

    def test_no_preseason_in_observed_events(self):
        raw=event(tip="2022-10-10T23:00:00Z",timestamp="2022-10-09T23:00:00Z")
        self.w.import_archive(self.archive(odds_envelope(self.config,raw,"2022-10-09T23:00:00Z")))
        self.assertFalse(self.w.rows("events"))

    def test_plan_cost_no_network(self):
        plan=build_plan(self.w,["2022-23"])
        self.assertEqual(plan["planned_requests"],130)
        self.assertEqual(plan["estimated_remaining_credits"],3900)
        self.assertFalse(self.calls)

    def test_fetch_missing_key_still_saves_resume_plan(self):
        status=fetch_history(self.w,Client(self.w,key=""),["2022-23"])
        self.assertEqual(status["status"],"missing_api_key")
        self.assertTrue((self.root/"fetch_plan.json").exists())


if __name__ == "__main__":
    unittest.main()
