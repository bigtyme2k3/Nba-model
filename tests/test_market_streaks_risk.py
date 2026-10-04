import unittest

from research.nba_market.core import load_config
from research.nba_market.simulation import *
from research.nba_market.streaks import qualify_after_failures, run_statistics
from tests.market_fixtures import offer


class StreakTests(unittest.TestCase):
    def records(self,values):
        return [dict(offer(i,"win" if value else "loss"),value=value) for i,value in enumerate(values)]

    def test_run_lengths_and_terminal_censoring(self):
        runs,next_events=run_statistics(self.records([True,True,False,True,True,True]))
        self.assertEqual(runs,[{"length":2,"right_censored":False},{"length":3,"right_censored":True}])
        self.assertEqual([r["previous_streak"] for r in next_events],[1,2,1,2])

    def test_nonqualifying_games_do_not_advance_or_break(self):
        records=self.records([True,False,True,False]);records[1]["qualifies"]=False
        runs,_=run_statistics(records)
        self.assertEqual(runs,[{"length":2,"right_censored":False}])

    def test_unknown_result_censors_and_resets(self):
        records=self.records([True,True,None,True,False]);records[2]["value"]=None
        runs,_=run_statistics(records)
        self.assertEqual(runs,[{"length":2,"right_censored":True},{"length":1,"right_censored":False}])

    def test_next_event_requires_previous_result_availability(self):
        records=self.records([True,False]);records[1]["signal_timestamp"]="2022-10-20T23:00:00Z"
        _,next_events=run_statistics(records)
        self.assertFalse(next_events[0]["prior_results_available"])

    def test_same_team_qualifying_continuation(self):
        offers=[offer(0,"loss"),offer(1,"loss",qualifies=False),offer(2,"win")]
        qualified=qualify_after_failures(offers,"same_team",1)
        self.assertEqual(len(qualified),2)
        self.assertFalse(qualified[0]["qualifies"])
        self.assertTrue(qualified[1]["qualifies"])
        self.assertEqual(qualified[1]["trigger_prior_failures"],1)

    def test_trigger_reset_when_market_wins(self):
        rows=qualify_after_failures([offer(0,"loss"),offer(1,"win"),offer(2,"loss")],"same_team",1)
        self.assertEqual([r["qualifies"] for r in rows],[False,True,False])

    def test_streak_signal_not_frozen_using_future_outcome(self):
        a,b=offer(0,"loss"),offer(1,"loss",signal="2022-10-20T23:00:00Z")
        rows=qualify_after_failures([a,b],"same_team",1)
        self.assertFalse(rows[1]["qualifies"])

    def test_exact_availability_timestamp_is_not_before_signal(self):
        a=offer(0,"loss");b=offer(1,"win",signal=a["result_available_at"])
        self.assertFalse(qualify_after_failures([a,b],"same_team",1)[1]["qualifies"])

    def test_team_lanes_do_not_mix(self):
        rows=qualify_after_failures([offer(0,"loss",team="Boston Celtics"),offer(1,"win",team="Miami Heat")],"same_team",1)
        self.assertFalse(any(r["qualifies"] for r in rows))

    def test_opposing_favorites_follow_qualifying_underdog_identity(self):
        first=offer(0,"loss",team="Boston Celtics")
        second=offer(1,"win",team="Miami Heat")
        for row in (first,second):
            row["sequence_team"]="Los Angeles Lakers"
        qualified=qualify_after_failures([first,second],"same_team",1)
        self.assertTrue(qualified[1]["qualifies"])
        result,ledger,_=simulate([first,second],"1_2_4",scope="same_team")
        self.assertEqual([r["stake"] for r in ledger],[1,2])
        self.assertEqual(ledger[0]["sequence_id"],ledger[1]["sequence_id"])

    def test_season_boundaries_reset_trigger(self):
        a,b=offer(0,"loss"),offer(1,"win");b["season"]="2023-24"
        self.assertFalse(any(r["qualifies"] for r in qualify_after_failures([a,b],"league",1)))


class RiskTests(unittest.TestCase):
    def test_price_aware_negative_odds_recovery(self):
        self.assertEqual(recovery_stake(1,-250,1),5)
        self.assertEqual(recovery_stake(3,-400,1),16)

    def test_price_aware_positive_odds_recovery(self):
        self.assertEqual(recovery_stake(3,200,1),2)

    def test_price_aware_stakes_and_net_target(self):
        rows=[offer(0,"loss",-250),offer(1,"loss",-250),offer(2,"win",-250)]
        result,ledger,seq=simulate(rows,"price_aware_4")
        self.assertEqual([r["stake"] for r in ledger],[2.5,8.75,30.625])
        self.assertAlmostEqual(result["net_units"],1)
        self.assertEqual(result["maximum_sequence_exposure"],41.875)
        self.assertEqual(result["historical_bankroll_requirement"],41.875)
        self.assertEqual(result["maximum_drawdown"],11.25)

    def test_doubling_negative_favorite_can_still_lose(self):
        result,_,_=simulate([offer(0,"loss",-250),offer(1,"win",-250)],"1_2")
        self.assertAlmostEqual(result["net_units"],-.2)
        self.assertEqual(result["sequences_ending_win_with_net_loss"],1)
        self.assertEqual(result["target_profit_shortfalls"],1)

    def test_flat_benchmark_profit_and_roi(self):
        result,ledger,_=simulate([offer(0,"win",200),offer(1,"loss",-250),offer(2,"push",-110)],"flat_1")
        self.assertEqual(result["net_units"],1)
        self.assertAlmostEqual(result["roi"],1/3)
        self.assertEqual((result["wins"],result["losses"],result["pushes"]),(1,1,1))

    def test_fixed_progression_failure_and_exposure(self):
        result,ledger,seq=simulate([offer(i,"loss",100) for i in range(3)],"1_2_4")
        self.assertEqual([r["stake"] for r in ledger],[1,2,4])
        self.assertEqual(result["net_units"],-7)
        self.assertEqual(result["maximum_sequence_exposure"],7)
        self.assertEqual(result["progression_failures"],1)
        self.assertEqual(seq[0]["ending"],"progression_exhausted")

    def test_push_does_not_advance_progression(self):
        result,ledger,_=simulate([offer(0,"loss",100),offer(1,"push",100),offer(2,"win",100)],"1_2_4")
        self.assertEqual([r["stake"] for r in ledger],[1,2,2])
        self.assertEqual(result["net_units"],1)
        self.assertEqual(result["maximum_sequence_exposure"],3)

    def test_nonqualifying_game_is_not_progression_loss(self):
        rows=[offer(0,"loss",100),offer(1,"loss",100,qualifies=False),offer(2,"win",100)]
        result,ledger,_=simulate(rows,"1_2_4")
        self.assertEqual([r["stake"] for r in ledger],[1,2])
        self.assertEqual(result["net_units"],1)

    def test_unknown_qualifying_game_stops_lane(self):
        unknown=offer(1,None);unknown["result_available_at"]=None
        result,ledger,_=simulate([offer(0,"loss",100),unknown,offer(2,"win",100)],"1_2_4")
        self.assertEqual(result["bets"],1)
        self.assertEqual(result["unknown_qualifying_outcomes"],1)
        self.assertEqual(result["censored_after_unknown_qualifying_outcome"],1)

    def test_drawdown_independent_known_path(self):
        self.assertEqual(maximum_drawdown([2,-1,-4,3,-2]),5)

    def test_overlapping_flat_bets_need_total_liquidity(self):
        a=offer(0,"loss",100);b=offer(1,"win",100,team="Miami Heat",signal=a["signal_timestamp"],settle=a["result_available_at"])
        result,_,_=simulate([a,b],"flat_1",scope="same_team")
        self.assertEqual(result["historical_bankroll_requirement"],2)
        self.assertEqual(result["maximum_outstanding_stakes"],2)

    def test_league_progression_waits_for_settlement(self):
        a=offer(0,"loss",100);b=offer(1,"win",100,signal=a["signal_timestamp"],settle=a["result_available_at"])
        result,_,_=simulate([a,b],"1_2",scope="league")
        self.assertEqual(result["bets"],1)
        self.assertEqual(result["skipped_overlapping_qualifying_offers"],1)

    def test_simultaneous_same_team_lanes_and_portfolio_exposure(self):
        a=offer(0,"loss",100);b=offer(1,"loss",100,team="Miami Heat",signal=a["signal_timestamp"],settle=a["result_available_at"])
        c=offer(2,"loss",100);d=offer(3,"loss",100,team="Miami Heat",signal=c["signal_timestamp"],settle=c["result_available_at"])
        result,_,_=simulate([a,b,c,d],"1_2_4",scope="same_team")
        self.assertEqual(result["historical_bankroll_requirement"],6)
        self.assertEqual(result["maximum_sequence_exposure"],3)
        self.assertEqual(result["maximum_outstanding_stakes"],4)

    def test_bankroll_cannot_fund_large_recovery_stake(self):
        result,_,_=simulate([offer(0,"loss",-250),offer(1,"loss",-250),offer(2,"win",-250)],"price_aware_4",bankroll=20)
        self.assertTrue(result["operational_ruin"])
        self.assertEqual(result["bets"],2)
        self.assertEqual(result["unaffordable_bets"],1)

    def test_stake_cap_prevents_unbounded_exposure(self):
        result,_,_=simulate([offer(i,"loss",-250) for i in range(3)],"price_aware_8",max_stake=10)
        self.assertEqual(result["stake_cap_failures"],1)
        self.assertEqual(result["progression_failures"],1)
        self.assertEqual(result["maximum_single_stake"],8.75)
        self.assertEqual(result["maximum_requested_stake_including_rejected"],30.625)

    def test_unfinished_sequence_is_censored_not_failure(self):
        result,_,_=simulate([offer(0,"loss",100)],"1_2_4")
        self.assertEqual(result["right_censored_sequences"],1)
        self.assertEqual(result["progression_failures"],0)

    def test_results_not_available_at_signal(self):
        row=offer(0);row["result_available_at"]=row["signal_timestamp"]
        with self.assertRaises(ValueError):
            simulate([row],"flat_1")

    def test_deterministic_bootstrap_and_high_bankroll_zero_operational_ruin(self):
        config=load_config();config["bootstrap_repetitions"]=10;config["bankrolls"]=[1,1000]
        rows=[offer(i,"loss" if i%2 else "win",100) for i in range(12)]
        first=bootstrap_ruin(rows,"1_2_4","same_team",config)
        self.assertEqual(first,bootstrap_ruin(rows,"1_2_4","same_team",config))
        self.assertEqual(first[1]["operational_ruin_probability"],0)
        self.assertGreater(first[0]["operational_ruin_probability"],0)

    def test_no_data_no_ruin_estimate(self):
        self.assertTrue(all(r["operational_ruin_probability"] is None for r in bootstrap_ruin([],"flat_1","league",load_config())))
        summary,_,_=simulate([],"flat_1")
        self.assertIsNone(summary["maximum_drawdown"])
        self.assertIsNone(summary["historical_bankroll_requirement"])
        self.assertIsNone(summary["net_units"])


if __name__ == "__main__":
    unittest.main()
