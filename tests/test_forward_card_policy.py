import unittest

from forward_ledger import flatten, EVIDENCE_EPOCH


class TestForwardCardDecisionPolicy(unittest.TestCase):
    def test_only_final_card_counts_as_bet_across_markets(self):
        data={
            "date":"2026-10-27",
            "generated":"2026-10-27T20:00:00-04:00",
            "model_version":"NBA_V1.0",
            "best_bets":[
                {"type":"TOTAL","game":"Away @ Home","play":"UNDER 230.5"},
                {
                    "type":"PROP","game":"Away @ Home","player":"Player A",
                    "stat":"PTS","play":"Player A PTS OVER 20.5","stars":3,
                },
            ],
            "games":[{
                "home":{"name":"Home"},
                "away":{"name":"Away"},
                "spread":{
                    "pred":4.0,"posted_line":-1.5,"edge":2.5,
                    "play":"Home -1.5","stars":3,"juice":-110,
                },
                "totals":{
                    "pred":225.0,"line":230.5,"edge":-5.5,
                    "play":"UNDER","stars":3,"juice":-110,
                    "selected_quote":{"book":"FanDuel","book_key":"fanduel","side":"UNDER",
                                      "line":230.5,"price":-110,"quote_policy":"EXACT_BOOK_SIDE_V1"},
                },
                "props":[{
                    "player":"Player A","proj_min":34,
                    "props":{
                        "pts":{
                            "pred":25.0,"line":20.5,"edge":4.5,"signal":"OVER",
                            "book":"FanDuel","book_key":"fanduel","odds":-110,
                            "market_source":"the-odds-api",
                        },
                        "pra":{
                            "pred":35.0,"line":30.5,"edge":4.5,"signal":"OVER",
                            "book":"DraftKings","book_key":"draftkings","odds":-110,
                            "market_source":"the-odds-api",
                        },
                    },
                }],
            }],
        }
        rows=flatten(data)
        decisions={(r["market"],r.get("stat")):r["decision"] for r in rows}
        self.assertEqual(decisions[("SPREAD",None)],"PASS")
        self.assertEqual(decisions[("TOTAL",None)],"BET")
        self.assertEqual(decisions[("PROP","PTS")],"BET")
        self.assertEqual(decisions[("PROP","PRA")],"PASS")
        self.assertTrue(all(r.get("evidence_epoch")==EVIDENCE_EPOCH for r in rows))

    def test_card_selection_sizing_passed_on_is_not_a_bet(self):
        data={
            "date":"2026-11-04","generated":"g","model_version":"NBA_V1.0",
            "best_bets":[
                {"type":"SPREAD","game":"Away @ Home","play":"Away +11.5","verdict":"BET"},
                {"type":"TOTAL","game":"Away @ Home","play":"UNDER 233.5","verdict":"PASS"},
            ],
            "games":[{
                "home":{"name":"Home"},"away":{"name":"Away"},
                "spread":{"pred":1.6,"posted_line":-11.5,"edge":-9.9,"play":"Away +11.5",
                          "selected_quote":{"side":"AWAY","line":11.5,"price":-110,"book":"FanDuel"}},
                "totals":{"pred":227.0,"line":233.5,"edge":-6.5,"play":"UNDER",
                          "selected_quote":{"side":"UNDER","line":233.5,"price":-110,"book":"FanDuel"}},
                "props":[],
            }],
        }
        decisions={r["market"]:r["decision"] for r in flatten(data)}
        self.assertEqual(decisions["SPREAD"],"BET")
        self.assertEqual(decisions["TOTAL"],"PASS")


if __name__=="__main__":
    unittest.main()
