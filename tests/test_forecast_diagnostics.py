"""Regression tests for pregame benchmarking, stale-injury fail-closed, and minute guards."""
import csv
import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import prediction_diagnostics
import slate_injuries
from qualification_audit import summarize_qualification


def write_csv(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class InjuryFeedGuards(unittest.TestCase):
    def test_fresh_dated_report_required(self):
        with tempfile.TemporaryDirectory() as temp:
            raw = Path(temp)
            status = {"target_date": "2026-11-10", "status": "ok",
                      "generated_at_utc": "2026-11-10T18:00:00Z"}
            (raw / "injuries_status.json").write_text(json.dumps(status))
            write_csv(raw / "injuries_today.csv", [
                {"game_date": "2026-11-10", "player": "Test Player", "severity": "OUT"}
            ])
            now = datetime(2026, 11, 10, 20, 0, tzinfo=timezone.utc)
            valid = slate_injuries.assess_injury_feed("2026-11-10", raw, now)
            self.assertTrue(valid["healthy"])
            self.assertEqual(valid["reason"], "OK")
            self.assertEqual(slate_injuries.assess_injury_feed("2026-11-11", raw, now)["reason"], "WRONG_SLATE_DATE")
            write_csv(raw / "injuries_today.csv", [
                {"game_date": "2026-11-09", "player": "Test Player", "severity": "OUT"}
            ])
            self.assertEqual(slate_injuries.assess_injury_feed("2026-11-10", raw, now)["reason"], "REPORT_DATE_MISMATCH")
            write_csv(raw / "injuries_today.csv", [
                {"game_date": "2026-11-10", "player": "Test Player", "severity": "OUT"}
            ])
            late = datetime(2026, 11, 11, 17, 0, tzinfo=timezone.utc)
            self.assertEqual(slate_injuries.assess_injury_feed("2026-11-10", raw, late)["reason"], "STALE_REPORT")


class YesterdayBenchmarks(unittest.TestCase):
    def test_uses_pretip_prediction_and_final_matching_game_only(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "predictions").mkdir()
            (root / "raw").mkdir()
            forecast = {
                "date": "2026-11-10", "generated": "2026-11-10T20:00:00Z",
                "best_bets": [], "games": [{
                    "tip": "2026-11-11T00:00:00Z",
                    "home": {"name": "Home"}, "away": {"name": "Away"},
                    "spread": {"pred": 7, "posted_line": -5},
                    "totals": {"pred": 220, "line": 225},
                    "props": [{
                        "player": "A Player", "proj_min": 30,
                        "props": {"pts": {"pred": 20, "line": 18.5, "odds": -110},
                                  "reb": {"pred": 5, "line": 6.5, "odds": -110},
                                  "pra": {"pred": 28, "line": 29.5, "odds": -110}}
                    }]
                }]
            }
            path = root / "predictions/predictions_2026-11-10.json"
            path.write_text(json.dumps(forecast))
            write_csv(root / "raw/scores_2026-11-10.csv", [
                {"game_date": "2026-11-10", "home_team": "Home", "away_team": "Away",
                 "game_id": "1", "status": "STATUS_FINAL", "is_final": "True",
                 "actual_spread": "10", "actual_total": "228"},
                {"game_date": "2026-11-09", "home_team": "Other", "away_team": "Away",
                 "game_id": "2", "status": "STATUS_FINAL", "is_final": "True",
                 "actual_spread": "100", "actual_total": "999"}
            ])
            write_csv(root / "raw/result_player_box_2026-11-10.csv", [
                {"game_date": "2026-11-10", "game_id": "1", "player_name": "A Player",
                 "did_not_play": "False", "minutes": "25", "pts": "24", "reb": "4",
                 "ast": "3", "threes_made": "2"}
            ])
            result = prediction_diagnostics.evaluate_day("2026-11-10", root / "predictions", root / "raw")
            self.assertEqual(result["status"], "COMPLETE")
            self.assertEqual(result["summary"]["completed_games"], 1)
            self.assertEqual(result["summary"]["margin_mae_model"], 3)
            self.assertEqual(result["summary"]["margin_mae_market"], 5)
            self.assertEqual(result["summary"]["margin_mae_shadow"], 4.3)
            self.assertEqual(result["summary"]["total_mae_model"], 8)
            self.assertEqual(result["summary"]["total_mae_market"], 3)
            self.assertEqual(result["stats"]["PTS"]["directions_correct"], 1)
            self.assertEqual(result["stats"]["REB"]["directions_correct"], 1)
            self.assertEqual(result["stats"]["MINUTES"]["mae"], 5)
            self.assertEqual(result["qualified_bets"], 0)
            forecast["generated"] = "2026-11-11T03:00:00Z"
            path.write_text(json.dumps(forecast))
            self.assertEqual(prediction_diagnostics.evaluate_day("2026-11-10", root / "predictions", root / "raw")["status"], "PREGAME_SNAPSHOT_INVALID")


class RejectionAudit(unittest.TestCase):
    def test_market_only_research_cannot_be_reclassified_as_bet(self):
        games = [{
            "spread": {"reference_quote": {"price": -110}, "selected_quote": None},
            "totals": {"reference_quote": {"price": -110}, "selected_quote": None},
            "props": [{
                "props": {
                    "pts": {"line": 20.5, "odds": -110, "probability_calibration": "0.00_market_only"},
                    "reb": {"availability_block": True, "block_reason": "UNSTABLE_MINUTES"},
                    "ast": {"line": None, "odds": None},
                }
            }],
        }]
        result = summarize_qualification(games, [])
        self.assertEqual(result["final_bets"], 0)
        self.assertEqual(result["rejection_counts"]["PROP_CALIBRATION_MARKET_ONLY"], 1)
        self.assertEqual(result["rejection_counts"]["UNSTABLE_MINUTES"], 1)
        self.assertEqual(result["rejection_counts"]["MISSING_EXACT_PROP_PRICE"], 1)
        self.assertEqual(result["rejection_counts"]["GAME_PRICE_OR_CALIBRATION_GATE"], 2)


class MinutesEligibility(unittest.TestCase):
    def test_large_unexplained_minutes_change_is_research_only(self):
        import daily_runner

        class FakeModel:
            def predict(self, x):
                return [25.0]

        bundle = {"models": {"pts": {"features": ["minutes"], "model": FakeModel(), "residual_std": 2}},
                  "thresholds": {"pts": 2.5}}
        home, away = "Boston Celtics", "Milwaukee Bucks"
        ctx = {"home_b2b": False, "away_b2b": False, "home_rest_days": 2, "away_rest_days": 2,
               "season_game_num": 10, "month": 11, "home_game_num": 10, "away_game_num": 10}
        player = {"team": home, "pos": "G", "projected_minutes": 30,
                  "roll5_minutes": 19, "mpg": 28, "usage": .2}
        posted = {"Test Player": {"__team": home, "__offers": {"pts": [{
            "side": "OVER", "line": 18.5, "price": -110, "no_vig_probability": .5,
            "book": "DraftKings", "book_key": "draftkings"}]}}}
        with patch.object(daily_runner, "load_player_profiles", return_value={"Test Player": player}), \
                patch.object(daily_runner, "prop_model_probability", return_value=.75):
            data = daily_runner.run_props(home, away, ctx, bundle, posted, availability_ready=True)
            self.assertEqual(len(data), 1)
            self.assertTrue(data[0]["minutes_guard"])
            self.assertEqual(data[0]["props"]["pts"]["block_reason"], "UNSTABLE_MINUTES")
            self.assertIsNone(data[0]["props"]["pts"]["signal"])
            player["roll5_minutes"] = 28
            data = daily_runner.run_props(home, away, ctx, bundle, posted, availability_ready=True)
            self.assertEqual(data[0]["props"]["pts"]["signal"], "OVER")
            data = daily_runner.run_props(home, away, ctx, bundle, posted, availability_ready=False)
            self.assertEqual(data[0]["props"]["pts"]["block_reason"], "UNVERIFIED_INJURY_FEED")
            self.assertIsNone(data[0]["props"]["pts"]["signal"])


if __name__ == "__main__":
    unittest.main()
