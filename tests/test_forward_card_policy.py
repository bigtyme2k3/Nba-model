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


if __name__=="__main__":
    unittest.main()
