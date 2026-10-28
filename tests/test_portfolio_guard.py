import unittest

from daily_runner import collect_best_bets


class TestPortfolioGuard(unittest.TestCase):
    def test_only_one_final_prop_per_player(self):
        games=[{
            "tip":"2026-10-28T03:00:00+00:00",
            "home":{"name":"Home"},
            "away":{"name":"Away"},
            "spread":{"edge":None},
            "totals":{"edge":None},
            "props":[
                {
                    "player":"Player A",
                    "proj_min":34,
                    "opportunity_direction":"UNCHANGED",
                    "vacancy_sources":"",
                    "props":{
                        "pts":{
                            "signal":"OVER","edge":5.0,"ev":0.20,"probability_edge":0.10,
                            "line":20.5,"odds":-110,"book":"FanDuel","book_key":"fanduel",
                            "pred":25.5,"model_probability":0.62,"market_probability":0.52,
                            "market_source":"the-odds-api",
                        },
                        "pra":{
                            "signal":"OVER","edge":8.0,"ev":0.18,"probability_edge":0.09,
                            "line":30.5,"odds":-110,"book":"DraftKings","book_key":"draftkings",
                            "pred":38.5,"model_probability":0.61,"market_probability":0.52,
                            "market_source":"the-odds-api",
                        },
                    },
                },
                {
                    "player":"Player B",
                    "proj_min":30,
                    "opportunity_direction":"UNCHANGED",
                    "vacancy_sources":"",
                    "props":{
                        "ast":{
                            "signal":"UNDER","edge":-3.0,"ev":0.12,"probability_edge":0.07,
                            "line":6.5,"odds":-105,"book":"Fanatics","book_key":"fanatics",
                            "pred":3.5,"model_probability":0.59,"market_probability":0.52,
                            "market_source":"the-odds-api",
                        }
                    },
                },
            ],
        }]
        bets=collect_best_bets(games)
        props=[b for b in bets if b["type"]=="PROP"]
        self.assertEqual(len(props),2)
        self.assertEqual(len({b["player"] for b in props}),2)
        a=[b for b in props if b["player"]=="Player A"]
        self.assertEqual(len(a),1)
        self.assertEqual(a[0]["stat"],"PTS")  # higher EV beats larger raw PRA edge
        self.assertEqual(a[0]["portfolio_guard"],"ONE_PROP_PER_PLAYER")


if __name__=="__main__":
    unittest.main()
