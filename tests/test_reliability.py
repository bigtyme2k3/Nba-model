"""Regression tests for executable quotes, stale slates, and durable evidence."""
import contextlib
import copy
import io
import json
import os
import pickle
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pandas as pd

import build_dashboard
import collect_result_box
import execution_reconciler
import forward_reconciler
from calibrate_edges import temporal_calibration
from clv_reconciler import select_close, closing_coverage
from forward_ledger import EVIDENCE_EPOCH, flatten
from game_markets import offers_from_event, select_offer, QUOTE_POLICY
from model_publication import publish_model_bundle
from model_publication import model_family_id, artifact_id
from prop_features import VERSION, build_features
from reconcile_pending import pending_dates


@contextlib.contextmanager
def workspace():
    previous = Path.cwd()
    with tempfile.TemporaryDirectory() as temp:
        os.chdir(temp)
        try:
            yield Path(temp)
        finally:
            os.chdir(previous)


def event():
    return {"id": "event-1", "home_team": "Home", "away_team": "Away",
            "commence_time": "2026-11-10T23:30:00Z", "bookmakers": [{
                "key": "fanduel", "title": "FanDuel", "last_update": "2026-11-10T23:00:00Z",
                "markets": [
                    {"key": "spreads", "outcomes": [{"name": "Home", "point": -5.5, "price": -120},
                                                     {"name": "Away", "point": 5.5, "price": -105}]},
                    {"key": "totals", "outcomes": [{"name": "Over", "point": 230.5, "price": -105},
                                                    {"name": "Under", "point": 230.5, "price": -115}]}]}]}


class ExactQuoteTests(unittest.TestCase):
    def setUp(self):
        # Quote-selection tests use fixed calibration, not the changing live artifact.
        calibration = patch('kelly_sizing.load_calibration', return_value={
            'edge': {'spread': .25, 'totals': .25, 'props': .25}, 'prop_price': .45})
        calibration.start()
        self.addCleanup(calibration.stop)
        self.now = datetime(2026, 11, 10, 23, 1, tzinfo=timezone.utc)
        self.quotes = offers_from_event(event(), "2026-11-10T23:00:00Z")

    def test_away_and_under_use_their_own_prices(self):
        away = select_offer(0, "SPREAD", self.quotes, self.now)
        under = select_offer(200, "TOTAL", self.quotes, self.now)
        self.assertEqual((away["side"], away["line"], away["price"]), ("AWAY", 5.5, -105))
        self.assertEqual((under["side"], under["line"], under["price"]), ("UNDER", 230.5, -115))

    def test_missing_or_stale_quote_is_not_executable(self):
        self.assertIsNone(select_offer(0, "SPREAD", [], self.now))
        later = self.now.replace(hour=23, minute=31)
        self.assertIsNone(select_offer(0, "SPREAD", self.quotes, later))
        stale = copy.deepcopy(self.quotes)
        for quote in stale:
            quote["last_update"] = "2026-11-10T22:00:00Z"
        self.assertIsNone(select_offer(0, "SPREAD", stale, self.now))

    def test_fractional_american_price_and_unpaired_lines_are_rejected(self):
        bad = event()
        bad["bookmakers"][0]["markets"][0]["outcomes"][0]["price"] = -112.5
        self.assertFalse(any(q["market"] == "SPREAD" for q in offers_from_event(bad, self.now.isoformat())))
        bad = event()
        bad["bookmakers"][0]["markets"][0]["outcomes"][1]["point"] = 6.5
        self.assertFalse(any(q["market"] == "SPREAD" for q in offers_from_event(bad, self.now.isoformat())))

    def test_consensus_results_are_excluded_without_rewriting_the_rows(self):
        row = {"market": "SPREAD", "decision": "BET", "outcome": "WIN", "odds": -112.5}
        original = dict(row)
        result = forward_reconciler.performance([row])
        self.assertEqual(result["bet_rows"], 0)
        self.assertEqual(result["excluded_unpriced_bets"], 1)
        self.assertEqual(row, original)

    def test_exact_away_handicap_grades_in_selected_side_orientation(self):
        row = {"date": "2026-11-10", "game": "Away @ Home", "market": "SPREAD", "side": "AWAY",
               "line": 5.5, "line_basis": "SELECTED_SIDE", "outcome": "PENDING"}
        scores = {"away @ home": {"margin": 5, "total": 225}}
        self.assertTrue(forward_reconciler.resolve_row(row, scores, {}, "2026-11-10", "now"))
        self.assertEqual(row["outcome"], "WIN")

    def test_nonpositive_ev_cannot_create_a_game_bet(self):
        quotes = copy.deepcopy(self.quotes)
        for quote in quotes:
            quote["price"] = -1000
        self.assertIsNone(select_offer(0, "SPREAD", quotes, self.now))

    def test_market_only_does_not_manufacture_value_at_plus_money(self):
        quotes = copy.deepcopy(self.quotes)
        for quote in quotes:
            quote.update(price=110, no_vig_probability=.46)
        with patch("kelly_sizing.load_calibration", return_value={"edge": {"spread": 0}}):
            self.assertIsNone(select_offer(0, "SPREAD", quotes, self.now))
            self.assertIsNotNone(select_offer(0, "SPREAD", quotes, self.now, require_value=False))

    def test_pipeline_to_ledger_preserves_exact_away_and_under_quotes(self):
        import daily_runner
        schedule = [{"home": "Home", "away": "Away", "tip": "2026-11-10T23:30:00Z",
                     "posted_spread": -6, "posted_total": 231, "game_offers": self.quotes}]
        with workspace(), patch.object(daily_runner, "load_team_profiles_into_context", return_value=0), \
                patch.object(daily_runner, "load_models", return_value=(None, None, None)), \
                patch.object(daily_runner, "run_spread", return_value=0), \
                patch.object(daily_runner, "run_totals", return_value=200), \
                patch.object(daily_runner, "run_props", return_value=[]), \
                patch.object(daily_runner, "OPPORTUNITY_AVAILABLE", False), \
                patch.object(daily_runner, "select_offer", side_effect=lambda p,m,o,**kw: select_offer(p,m,o,self.now,**kw)), \
                contextlib.redirect_stdout(io.StringIO()):
            result = daily_runner.run_pipeline(self.now.date(), schedule, {})
        self.assertEqual(len(result["best_bets"]), 2)
        rows = {row["market"]: row for row in flatten(result)}
        self.assertEqual((rows["SPREAD"]["side"], rows["SPREAD"]["line"], rows["SPREAD"]["odds"]), ("AWAY", 5.5, -105))
        self.assertEqual((rows["TOTAL"]["side"], rows["TOTAL"]["odds"]), ("UNDER", -115))


class FreshSlateTests(unittest.TestCase):
    def test_missing_today_keeps_old_cards_off_today(self):
        with workspace():
            Path("predictions").mkdir()
            Path("predictions/predictions_2026-11-09.json").write_text(json.dumps({
                "date": "2026-11-09", "generated": "2026-11-09T15:00:00Z", "best_bets": [{"play": "OLD"}]}))
            with patch.object(build_dashboard, "target_today", return_value="2026-11-10"):
                result = build_dashboard.find_predictions()
                self.assertEqual(result["best_bets"], [])
                self.assertEqual(result["slate_status"], "AWAITING_PREDICTIONS")
                self.assertEqual(result["last_successful_slate_date"], "2026-11-09")

    def test_payload_date_must_match_even_when_filename_matches(self):
        with workspace():
            Path("predictions").mkdir()
            Path("predictions/predictions_2026-11-10.json").write_text('{"date":"2026-11-09","games":[{}]}')
            with patch.object(build_dashboard, "target_today", return_value="2026-11-10"):
                self.assertEqual(build_dashboard.find_predictions()["games"], [])
                self.assertEqual(build_dashboard.enrich_data({"date": "2026-11-09", "games": [{}]})["games"], [])

    def test_undated_and_wrong_date_csvs_are_not_today_data(self):
        with workspace():
            pd.DataFrame([{"player": "Old", "game_date": "2026-11-09"}]).to_csv("stale.csv", index=False)
            pd.DataFrame([{"player": "Undated"}]).to_csv("undated.csv", index=False)
            self.assertEqual(build_dashboard.load_csv_records(["stale.csv", "undated.csv"], "2026-11-10"), [])

    def test_freshness_banner_is_visible_and_idempotent(self):
        with workspace(), patch.object(build_dashboard, "target_today", return_value="2026-11-10"):
            data = build_dashboard.empty_dashboard_data()
            data["slate_status"] = "AWAITING_PREDICTIONS"
            build_dashboard.build_html(data)
            build_dashboard.build_html(data)
            html = Path("docs/index.html").read_text()
            self.assertEqual(html.count('id="slate-freshness"'), 1)
            self.assertIn("Awaiting today", html)


class DurableEvidenceTests(unittest.TestCase):
    def test_partial_boxes_survive_and_zero_shooting_attempts_are_valid(self):
        events = [{"id": str(i), "status": {"type": {"name": "STATUS_FINAL"}}} for i in (1, 2)]
        row = {"game_id": "1", "player_id": "p1", "player_name": "Player", "game_date": "2026-11-10",
               "minutes": 3, "fg_att": 0, "pts": 0, "reb": 1, "ast": 0, "threes_made": 0}
        def box(game_id):
            if game_id == "2":
                raise RuntimeError("temporary failure")
            return {}
        with workspace(), patch.object(collect_result_box, "fetch_espn_scoreboard", return_value=events), \
                patch.object(collect_result_box, "fetch_espn_boxscore", side_effect=box), \
                patch.object(collect_result_box, "parse_espn_player_box", return_value=[row]):
            report = collect_result_box.collect("2026-11-10")
            self.assertEqual(report["status"], "PARTIAL")
            self.assertEqual(report["shooting_attempt_coverage"], 1)
            self.assertEqual(report["player_rows"], 1)
            with patch.object(collect_result_box, "fetch_espn_scoreboard", side_effect=RuntimeError("offline")):
                self.assertEqual(collect_result_box.collect("2026-11-10")["player_rows"], 1)

    def test_missing_stat_and_nonfinal_game_remain_pending(self):
        base = {"date": "2026-11-10", "game": "Away @ Home", "market": "PROP", "player": "P",
                "stat": "PTS", "side": "UNDER", "line": 1.5, "outcome": "PENDING"}
        row = dict(base)
        self.assertFalse(forward_reconciler.resolve_row(row, {}, {"p": {"PTS": 0}}, "2026-11-10", "now"))
        self.assertFalse(forward_reconciler.resolve_row(row, {"away @ home": {}}, {"p": {"PTS": None}}, "2026-11-10", "now"))

    def test_void_player_is_not_a_zero_stat_under_win(self):
        row = {"date": "2026-11-10", "game": "Away @ Home", "market": "PROP", "player": "P",
               "stat": "PTS", "side": "UNDER", "line": 1.5, "outcome": "PENDING"}
        forward_reconciler.resolve_row(row, {"away @ home": {"margin": 5}},
                                       {"p": {"PTS": None, "did_not_play": True}}, "2026-11-10", "now")
        self.assertEqual(row["outcome"], "VOID")

    def test_closing_snapshot_cannot_predate_entry(self):
        close = {"minutes_to_tip": 20, "captured_at_utc": "2026-11-10T23:00:00Z",
                 "book_last_update": "2026-11-10T22:59:00Z", "commence_time": "2026-11-10T23:20:00Z"}
        self.assertIsNone(select_close([close], "2026-11-10T23:05:00Z"))
        self.assertEqual(select_close([close], "2026-11-10T22:55:00Z"), close)

    def test_closing_coverage_reports_missing_games(self):
        with workspace():
            report = closing_coverage([{"date": "2026-11-10", "game": "Away @ Home", "evidence_epoch": EVIDENCE_EPOCH}], [])
            self.assertEqual(report["2026-11-10"]["missing_games"], ["away @ home"])

    def test_retry_queue_rotates_older_unresolved_dates(self):
        with workspace():
            Path("data/history").mkdir(parents=True)
            rows = [{"date": f"2026-11-0{i}", "evidence_epoch": EVIDENCE_EPOCH, "outcome": "PENDING"} for i in (1, 2)]
            Path("data/history/nba_forward_predictions.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
            Path("data/tracking").mkdir()
            Path("data/tracking/reconciliation_status.json").write_text(json.dumps({"last_attempt_by_date": {"2026-11-01": "recent"}}))
            self.assertEqual(pending_dates("2026-11-10", 1)[0], ["2026-11-10", "2026-11-02"])

    def test_execution_regrading_retains_other_dates(self):
        rows = [{"date": d, "evidence_epoch": EVIDENCE_EPOCH, "ranking_key": d, "minutes_to_tip": 40,
                 "market": "TOTAL", "game": "Away @ Home", "side": "OVER", "line": 200,
                 "decision": "PASS", "outcome": "PENDING"} for d in ("2026-11-08", "2026-11-09")]
        with workspace(), patch.object(execution_reconciler, "read_jsonl", return_value=rows), \
                patch.object(execution_reconciler, "load_scores", return_value={"away @ home": {"total": 220}}), \
                patch.object(execution_reconciler, "load_player_actuals", return_value={}), contextlib.redirect_stdout(io.StringIO()):
            for day in ("2026-11-08", "2026-11-09"):
                with patch("sys.argv", ["execution_reconciler.py", "--date", day]):
                    execution_reconciler.main()
            certified = pd.read_csv("data/tracking/nba_execution_certified.csv")
            self.assertEqual(set(certified["date"]), {"2026-11-08", "2026-11-09"})


class ModelGovernanceTests(unittest.TestCase):
    def test_routine_refits_keep_family_but_record_distinct_artifacts(self):
        with workspace():
            path = Path("model.pkl")
            path.write_bytes(pickle.dumps({"model_name": "Ridge", "feature_names": ["a"], "trained_rows": 10}))
            family, exact = model_family_id(path), artifact_id(path)
            path.write_bytes(pickle.dumps({"model_name": "Ridge", "feature_names": ["a"], "trained_rows": 20}))
            self.assertEqual(model_family_id(path), family)
            self.assertNotEqual(artifact_id(path), exact)
            path.write_bytes(pickle.dumps({"model_name": "Ridge", "feature_names": ["a", "b"]}))
            self.assertNotEqual(model_family_id(path), family)

    def test_probability_floor_does_not_create_a_longshot_edge(self):
        from kelly_sizing import prop_model_probability
        self.assertAlmostEqual(prop_model_probability(.9, .10, shrink=0), .10)

    def test_new_prop_training_keeps_complete_dates_out_of_training_folds(self):
        from props_model import prepare_data, train_prop_model
        rows = []
        for day in range(1, 21):
            for player in ("A", "B", "C"):
                rows.append({"game_date": f"2026-10-{day:02d}", "player": player, "season": 2026,
                             "pts": 10 + day % 4, "roll5_pts": 11, "roll5_minutes": 30,
                             "roll10_minutes": 28, "season_game_num": day})
        bundle = train_prop_model(prepare_data(pd.DataFrame(rows)), "pts")
        self.assertTrue(all(fold["train_through"] < fold["test_from"] for fold in bundle["date_folds"]))
        self.assertGreater(bundle["residual_std"], 0)

    def test_promotion_counts_only_the_active_algorithm_and_feature_family(self):
        from nba_v1_readiness import current_model_rows
        rows = [{"market": "PROP", "model_family_id": fingerprint} for fingerprint in (None, "old", "active")]
        self.assertEqual(current_model_rows(rows, {"PROP": "active"}), [rows[2]])

    def test_baseline_minutes_are_identical_for_training_and_live(self):
        prior = {"roll5_minutes": 35, "roll10_minutes": 30, "rest_days": 0, "usage": .2}
        training = build_features(prior)
        live = build_features(prior, projected_minutes=33.25, projected_usage=.2)
        self.assertAlmostEqual(training["minutes"], live["minutes"])
        self.assertEqual(training["roll5_minutes"], 35)

    def test_rejected_candidate_cannot_overwrite_champion(self):
        champion = {"status": "trained", "trained_through": "2026-11-08",
                    "feature_semantics_version": VERSION, "validation": "expanding_date_groups_fold_imputation",
                    "models": {s: {"cv_mae": 1.0} for s in ("pts", "reb", "ast", "threes", "pra")}}
        candidate = copy.deepcopy(champion)
        candidate["models"]["pts"]["cv_mae"] = 5
        with workspace():
            Path("models").mkdir()
            path = Path("models/props_models.pkl")
            path.write_bytes(pickle.dumps(champion))
            original = path.read_bytes()
            self.assertFalse(publish_model_bundle(candidate, path, "props"))
            self.assertEqual(path.read_bytes(), original)

    def test_later_date_failure_rejects_apparent_training_edge(self):
        rows = []
        for day in range(1, 11):
            for i in range(30):
                rows.append({"date": f"2026-10-{day:02d}", "market": "PROP", "model_family_id": "active",
                             "raw_model_probability": .9, "market_probability": .5,
                             "outcome": "WIN" if day <= 8 else "LOSS"})
        result = temporal_calibration(rows, 200, {"PROP": "active"})["prop_price"]
        self.assertEqual(result["status"], "MARKET_ONLY")
        self.assertEqual(result["validation"], "REJECTED_HOLDOUT")
        self.assertLess(result["train_through"], result["holdout_from"])

    def test_calibration_cannot_pool_different_algorithm_and_feature_families(self):
        rows = [{"date": "2026-11-01", "market": "PROP", "model_family_id": "old",
                 "raw_model_probability": .9, "market_probability": .5, "outcome": "WIN"}]
        result = temporal_calibration(rows, 1, {"PROP": "new"})["prop_price"]
        self.assertEqual(result["graded"], 0)
        self.assertEqual(result["shrink"], 0)


if __name__ == "__main__":
    unittest.main()
