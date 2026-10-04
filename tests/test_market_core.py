import unittest
from copy import deepcopy

from research.nba_market.core import *
from tests.market_fixtures import event


class MarketCoreTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config()

    def test_team_normalization(self):
        for raw in ("Los Angeles Clippers", " LA  Clippers ", "lac", "LA CLIPPERS"):
            self.assertEqual(normalize_team(raw), "LA Clippers")
        self.assertEqual(normalize_team("bOs"), "Boston Celtics")

    def test_unknown_and_ambiguous_team_rejected(self):
        for raw in ("LA", "New York", None, "Seattle Supersonics"):
            with self.assertRaises(ValueError):
                normalize_team(raw)

    def test_season_assignment_uses_eastern_day(self):
        self.assertEqual(season_for("2022-10-19T01:00:00Z", self.config), "2022-23")
        self.assertIsNone(season_for("2022-10-18T01:00:00Z", self.config))
        self.assertEqual(season_for("2023-04-10T01:00:00Z", self.config), "2022-23")

    def test_preseason_and_playoff_excluded(self):
        self.assertIsNone(season_for("2022-10-10T22:00:00Z", self.config))
        self.assertIsNone(season_for("2023-04-15T22:00:00Z", self.config))

    def test_cup_final_excluded(self):
        for day in ("2023-12-09", "2024-12-17", "2025-12-16"):
            self.assertIsNone(season_for(day+"T23:00:00Z", self.config))

    def test_timezone_required(self):
        with self.assertRaises(ValueError):
            parse_time("2022-10-20T23:00:00")

    def test_independent_team_game_numbering(self):
        events = [dict(event_id="a",season="2022-23",commence_time="2022-10-20T00:00:00Z",home_team="Boston Celtics",away_team="LA Clippers"),
                  dict(event_id="b",season="2022-23",commence_time="2022-10-21T00:00:00Z",home_team="Boston Celtics",away_team="Miami Heat")]
        rows = number_games(list(reversed(events))+[events[0]])
        self.assertEqual(len(rows), 2)
        self.assertEqual((rows[1]["team_game_number_home"],rows[1]["team_game_number_away"]), (2,1))

    def test_numbering_resets_at_season(self):
        events = [dict(event_id=str(i),season=season,commence_time=time,home_team="Boston Celtics",away_team="Miami Heat")
                  for i,season,time in [(1,"2022-23","2022-11-01T23:00:00Z"),(2,"2023-24","2023-11-01T23:00:00Z")]]
        self.assertEqual([e["team_game_number_home"] for e in number_games(events)], [1,1])

    def test_early_buckets_all_boundaries(self):
        expected={1:"Games 1-5",5:"Games 1-5",6:"Games 6-10",10:"Games 6-10",11:"Games 11-15",15:"Games 11-15",16:"Games 16-20",20:"Games 16-20",21:"Games 21+",0:"unknown"}
        for n,bucket in expected.items():
            self.assertEqual(early_bucket(n),bucket)

    def test_american_payouts(self):
        self.assertEqual(payout(250),2.5)
        self.assertEqual(payout(-250),.4)
        self.assertEqual(payout(-100),1)
        self.assertEqual(payout(100),1)

    def test_malformed_odds(self):
        for odds in (0,99,-99,float("nan"),float("inf"),100001):
            with self.assertRaises(ValueError):
                american(odds)

    def test_implied_probabilities(self):
        self.assertAlmostEqual(implied(-250),5/7)
        self.assertAlmostEqual(implied(250),2/7)

    def test_no_vig_pair(self):
        h,a=no_vig(-200,170)
        self.assertAlmostEqual(h,9/14)
        self.assertAlmostEqual(h+a,1)

    def test_probability_to_american_round_trip(self):
        for odds in (-400,-110,100,150,400):
            self.assertAlmostEqual(probability_to_american(implied(odds)), odds)

    def test_favorite_and_pickem(self):
        self.assertEqual(favorite(.6),"home")
        self.assertEqual(favorite(.4),"away")
        self.assertIsNone(favorite(.5))

    def test_price_band_boundaries(self):
        for price,expected in [(100,"+100 to +149"),(149,"+100 to +149"),(150,"+150 to +199"),(399,"+300 to +399"),(400,"+400+")]:
            self.assertEqual(price_band(price,self.config["underdog_price_bands"]),expected)
        self.assertIn("negative-priced",price_band(-105,self.config["underdog_price_bands"]))

    def test_moneyline_grades(self):
        self.assertEqual(grade_ml(110,100,"home"),"win")
        self.assertEqual(grade_ml(110,100,"away"),"loss")
        self.assertEqual(grade_ml(100,100,"home"),"push")

    def test_spread_orientation(self):
        self.assertEqual(grade_spread(114,110,"home",-4.5),"loss")
        self.assertEqual(grade_spread(114,110,"away",4.5),"win")
        self.assertEqual(grade_spread(114,110,"home",-4),"push")
        self.assertEqual(grade_spread(114,110,"away",4),"push")

    def test_total_grading_and_push(self):
        self.assertEqual(grade_total(114,110,"over",223.5),"win")
        self.assertEqual(grade_total(114,110,"under",223.5),"loss")
        self.assertEqual(grade_total(114,110,"over",224),"push")
        self.assertEqual(grade_total(114,110,"under",224),"push")

    def test_flat_payout_positive_negative_and_push(self):
        self.assertEqual(flat_profit("win",200,3),6)
        self.assertEqual(flat_profit("win",-250,5),2)
        self.assertEqual(flat_profit("loss",200,3),-3)
        self.assertEqual(flat_profit("push",-250,5),0)
        self.assertIsNone(flat_profit(None,-250))

    def normalize(self,raw=None,timestamp="2022-10-19T23:00:00Z"):
        return normalize_event(raw or event(timestamp=timestamp),timestamp,self.config)

    def test_normalization_retains_books_and_updates(self):
        base,rows,errors=self.normalize()
        self.assertEqual(base["season"],"2022-23")
        self.assertEqual(len(rows),2)
        self.assertEqual(errors,[])
        self.assertEqual(rows[0]["home_spread"],-4.5)
        self.assertEqual(rows[0]["away_spread"],4.5)

    def test_post_tip_snapshot_rejected(self):
        _,rows,errors=self.normalize(event(),"2022-10-20T23:01:00Z")
        self.assertFalse(rows)
        self.assertIn("post_tip_snapshot",errors)

    def test_at_tip_is_not_pretip(self):
        _,rows,errors=self.normalize(event(),"2022-10-20T23:00:00Z")
        self.assertFalse(rows)

    def test_future_update_market_rejected(self):
        raw=event()
        raw["bookmakers"][0]["markets"][0]["last_update"]="2022-10-19T23:01:00Z"
        _,rows,errors=self.normalize(raw)
        self.assertIsNone(rows[0]["home_ml"])
        self.assertTrue(any("future_market_update" in e for e in errors))

    def test_stale_update_rejected(self):
        raw=event()
        raw["bookmakers"][0]["markets"][0]["last_update"]="2022-10-19T19:00:00Z"
        _,rows,errors=self.normalize(raw)
        self.assertIsNone(rows[0]["home_ml"])
        self.assertTrue(any("stale" in e for e in errors))

    def test_asymmetric_spread_rejected(self):
        raw=event();raw["bookmakers"][0]["markets"][1]["outcomes"][1]["point"]=5
        _,rows,errors=self.normalize(raw)
        self.assertIsNone(rows[0]["home_spread"])
        self.assertTrue(any("malformed_spread" in e for e in errors))

    def test_mismatched_total_rejected(self):
        raw=event();raw["bookmakers"][0]["markets"][2]["outcomes"][1]["point"]=223
        _,rows,errors=self.normalize(raw)
        self.assertIsNone(rows[0]["total"])

    def test_duplicate_outcome_rejected(self):
        raw=event();m=raw["bookmakers"][0]["markets"][0];m["outcomes"][1]=deepcopy(m["outcomes"][0])
        _,rows,errors=self.normalize(raw)
        self.assertIsNone(rows[0]["home_ml"])

    def test_consensus_pairwise_probability_not_american_average(self):
        _,rows,_=self.normalize()
        result=consensus(rows,2)
        expected=sum(no_vig(r["home_ml"],r["away_ml"])[0] for r in rows)/2
        self.assertAlmostEqual(result["home_no_vig"],expected)
        self.assertAlmostEqual(result["home_no_vig"]+result["away_no_vig"],1)
        self.assertEqual(result["home_spread"],-5)
        self.assertEqual(result["total"],221.5)

    def test_consensus_deduplicates_books(self):
        _,rows,_=self.normalize()
        self.assertEqual(consensus(rows+[rows[0]])["h2h_books"],2)

    def test_consensus_minimum_books(self):
        _,rows,_=self.normalize()
        self.assertNotIn("favorite",consensus(rows[:1],2))

    def test_book_disagreement_detects_different_favorites(self):
        _,rows,_=self.normalize()
        rows[1].update(home_ml=170,away_ml=-200)
        result=consensus(rows,2)
        self.assertTrue(result["different_favorites"])
        self.assertGreater(result["ml_probability_range"],.2)

    def test_execution_uses_actual_book_line(self):
        _,rows,_=self.normalize()
        book=execution_book(rows,"spreads",-5,["fanduel","draftkings"])
        self.assertEqual(book["bookmaker"],"fanduel")
        self.assertEqual(book["home_spread"],-4.5)

    def test_snapshot_early_and_close_selection(self):
        base,early,_=self.normalize()
        _,close,_=self.normalize(timestamp="2022-10-20T22:55:00Z")
        selected=select_snapshots(base,{early[0]["snapshot_timestamp"]:early,close[0]["snapshot_timestamp"]:close},self.config)
        self.assertEqual(selected[("EARLY","h2h")][0],"2022-10-19T23:00:00Z")
        self.assertEqual(selected[("CLOSE","h2h")][0],"2022-10-20T22:55:00Z")

    def test_stale_snapshot_not_called_close(self):
        base,rows,_=self.normalize()
        selected=select_snapshots(base,{rows[0]["snapshot_timestamp"]:rows},self.config)
        self.assertNotIn(("CLOSE","h2h"),selected)
        self.assertIn(("EARLY","h2h"),selected)

    def test_old_book_updates_cannot_qualify_as_clean_close(self):
        base,rows,_=self.normalize(timestamp="2022-10-20T22:55:00Z")
        for row in rows:
            row["h2h_last_update"]="2022-10-20T22:00:00Z"
        selected=select_snapshots(base,{rows[0]["snapshot_timestamp"]:rows},self.config)
        self.assertNotIn(("CLOSE","h2h"),selected)
        self.assertIn(("CLOSE","spreads"),selected)

    def test_late_earliest_labeled(self):
        base,rows,_=self.normalize(timestamp="2022-10-20T22:55:00Z")
        selected=select_snapshots(base,{rows[0]["snapshot_timestamp"]:rows},self.config)
        self.assertIn("late_earliest",selected[("EARLY","h2h")][2])

    def test_safety_buffer_excludes_final_seconds(self):
        base,rows,_=self.normalize(timestamp="2022-10-20T22:59:30Z")
        self.assertFalse(select_snapshots(base,{rows[0]["snapshot_timestamp"]:rows},self.config))

    def test_wilson_zero_and_known(self):
        self.assertEqual(wilson(0,0),(None,None))
        low,high=wilson(50,100)
        self.assertLess(low,.5);self.assertGreater(high,.5)


if __name__ == "__main__":
    unittest.main()
