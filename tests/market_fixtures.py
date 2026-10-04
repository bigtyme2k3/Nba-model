"""Synthetic API-shaped fixtures, never historical NBA evidence."""
from copy import deepcopy
from datetime import timedelta

from research.nba_market.core import iso, parse_time
from research.nba_market.ingest import ODDS_PATH, SCORES_PATH, SOURCE, market_params


def event(timestamp="2022-10-19T23:00:00Z", tip="2022-10-20T23:00:00Z", event_id="synthetic_event_1",
          home="Boston Celtics", away="Los Angeles Lakers", home_ml=-200, away_ml=170,
          spread=-4.5, total=220.5):
    books = []
    for i, name in enumerate(("fanduel", "draftkings")):
        books.append({"key": name, "title": name, "last_update": timestamp, "markets": [
            {"key": "h2h", "last_update": timestamp, "outcomes": [{"name": home, "price": home_ml-10*i}, {"name": away, "price": away_ml+10*i}]},
            {"key": "spreads", "last_update": timestamp, "outcomes": [{"name": home, "point": spread-i, "price": -110}, {"name": away, "point": -spread+i, "price": -110}]},
            {"key": "totals", "last_update": timestamp, "outcomes": [{"name": "Over", "point": total+2*i, "price": -110}, {"name": "Under", "point": total+2*i, "price": -110}]},
        ]})
    return {"id": event_id, "sport_key": "basketball_nba", "commence_time": tip,
            "home_team": home, "away_team": away, "bookmakers": books}


def odds_envelope(config, raw, timestamp):
    return {"source": SOURCE, "path": ODDS_PATH, "params": market_params(config, timestamp),
            "headers": {"x-requests-last": "30", "x-requests-remaining": "970"},
            "fetched_at": "2026-10-04T19:00:00Z", "payload": {"timestamp": timestamp,
            "previous_timestamp": iso(parse_time(timestamp)-timedelta(minutes=5)),
            "next_timestamp": iso(parse_time(timestamp)+timedelta(minutes=5)), "data": [deepcopy(raw)]}}


def score_envelope(raw, home_score=114, away_score=110):
    settled = iso(parse_time(raw["commence_time"])+timedelta(hours=3))
    return {"source": SOURCE, "path": SCORES_PATH, "params": {"daysFrom": "3", "dateFormat": "iso"},
            "headers": {}, "fetched_at": iso(parse_time(settled)+timedelta(minutes=10)),
            "payload": [{"id": raw["id"], "sport_key": "basketball_nba", "home_team": raw["home_team"],
                         "away_team": raw["away_team"], "commence_time": raw["commence_time"], "completed": True,
                         "last_update": settled, "scores": [{"name": raw["home_team"], "score": str(home_score)},
                                                            {"name": raw["away_team"], "score": str(away_score)}]}]}


def offer(index, result="loss", odds=-110, team="Boston Celtics", signal=None, settle=None, qualifies=True):
    start = parse_time("2022-10-20T22:55:00Z")+timedelta(days=index)
    return {"event_id": f"synthetic_offer_{index}", "signal_id": f"signal_{index}", "season": "2022-23",
            "team": team, "price": odds, "result": result, "signal_timestamp": signal or iso(start),
            "commence_time": iso(start+timedelta(minutes=5)), "result_available_at": settle or iso(start+timedelta(hours=3)),
            "qualifies": qualifies, "early_event": qualifies, "market_follow_reference_result": result}
