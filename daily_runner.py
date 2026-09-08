"""
daily_runner.py — NBA Daily Pipeline Orchestrator
Runs all three models and writes predictions JSON.

Usage:
    python daily_runner.py
    python daily_runner.py --date 2026-11-05
    python daily_runner.py --date 2026-11-05 --out predictions/today.json
"""

import os, sys, json, pickle, argparse, warnings
import numpy as np
import pandas as pd
from datetime import date, datetime
warnings.filterwarnings("ignore")

# ── Paths (relative — works on GitHub Actions and any machine) ─────────────────
MODEL_DIR  = "models"
DATA_DIR   = "data/processed"
OUTPUT_DIR = "predictions"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR,  exist_ok=True)

# ── Import local modules ───────────────────────────────────────────────────────
sys.path.insert(0, ".")
from teams import EASTERN_CONFERENCE as EAST, WESTERN_CONFERENCE as WEST, TEAM_ABBR

try:
    from totals_model import engineer_totals_features, TOTALS_FEATURES
except ImportError:
    print("[WARN] totals_model.py not found")
    engineer_totals_features = None
    TOTALS_FEATURES = []

try:
    from props_model import PROP_TARGETS
except ImportError:
    print("[WARN] props_model.py not found")
    PROP_TARGETS = ["pts","reb","ast","threes","pra"]

try:
    from kelly_sizing import size_all_bets
    KELLY_AVAILABLE = True
except ImportError:
    print("[WARN] kelly_sizing.py not found — no unit sizing")
    KELLY_AVAILABLE = False

try:
    from scrape_refs import get_crew_stats
    REFS_AVAILABLE = True
except ImportError:
    REFS_AVAILABLE = False

try:
    from scrape_injuries import apply_injury_adjustments
    INJURIES_AVAILABLE = True
except ImportError:
    INJURIES_AVAILABLE = False


# ── Team metadata ──────────────────────────────────────────────────────────────
# Seed values only — replace with real season-to-date stats once collect_stats.py /
# merge_data.py have built up data/processed/master_all.csv. Net rating tiers here
# are illustrative, not live numbers.
CURRENT_TEAM_STATS = {
    "Boston Celtics":          {"net_rtg": 7.8,"ortg":119,"drtg":111,"pace":98,"ts_pct":0.594},
    "Oklahoma City Thunder":   {"net_rtg": 8.5,"ortg":120,"drtg":111,"pace":100,"ts_pct":0.590},
    "Denver Nuggets":          {"net_rtg": 6.9,"ortg":121,"drtg":114,"pace":98,"ts_pct":0.598},
    "Cleveland Cavaliers":     {"net_rtg": 5.2,"ortg":119,"drtg":114,"pace":97,"ts_pct":0.585},
    "New York Knicks":         {"net_rtg": 4.6,"ortg":117,"drtg":112,"pace":96,"ts_pct":0.582},
    "Minnesota Timberwolves":  {"net_rtg": 4.0,"ortg":115,"drtg":111,"pace":98,"ts_pct":0.570},
    "Los Angeles Lakers":      {"net_rtg": 3.6,"ortg":117,"drtg":113,"pace":99,"ts_pct":0.578},
    "Milwaukee Bucks":         {"net_rtg": 3.2,"ortg":117,"drtg":114,"pace":99,"ts_pct":0.575},
    "Houston Rockets":         {"net_rtg": 2.4,"ortg":114,"drtg":112,"pace":99,"ts_pct":0.565},
    "Orlando Magic":           {"net_rtg": 2.1,"ortg":112,"drtg":110,"pace":97,"ts_pct":0.556},
    "Indiana Pacers":          {"net_rtg": 1.8,"ortg":118,"drtg":116,"pace":101,"ts_pct":0.583},
    "Golden State Warriors":   {"net_rtg": 1.5,"ortg":115,"drtg":114,"pace":99,"ts_pct":0.573},
    "LA Clippers":             {"net_rtg": 1.2,"ortg":114,"drtg":113,"pace":97,"ts_pct":0.567},
    "Dallas Mavericks":        {"net_rtg": 0.9,"ortg":114,"drtg":113,"pace":98,"ts_pct":0.568},
    "Sacramento Kings":        {"net_rtg": 0.6,"ortg":115,"drtg":115,"pace":99,"ts_pct":0.572},
    "Miami Heat":              {"net_rtg":-0.5,"ortg":113,"drtg":114,"pace":97,"ts_pct":0.562},
    "Phoenix Suns":            {"net_rtg":-0.8,"ortg":114,"drtg":115,"pace":98,"ts_pct":0.566},
    "Philadelphia 76ers":      {"net_rtg":-1.0,"ortg":113,"drtg":114,"pace":98,"ts_pct":0.560},
    "Atlanta Hawks":           {"net_rtg":-1.4,"ortg":114,"drtg":116,"pace":101,"ts_pct":0.563},
    "Chicago Bulls":           {"net_rtg":-1.8,"ortg":113,"drtg":115,"pace":100,"ts_pct":0.558},
    "San Antonio Spurs":       {"net_rtg":-2.0,"ortg":112,"drtg":114,"pace":99,"ts_pct":0.555},
    "Memphis Grizzlies":       {"net_rtg":-2.3,"ortg":112,"drtg":114,"pace":100,"ts_pct":0.554},
    "New Orleans Pelicans":    {"net_rtg":-3.2,"ortg":110,"drtg":113,"pace":99,"ts_pct":0.548},
    "Toronto Raptors":         {"net_rtg":-3.6,"ortg":110,"drtg":114,"pace":98,"ts_pct":0.546},
    "Brooklyn Nets":           {"net_rtg":-4.1,"ortg":109,"drtg":113,"pace":98,"ts_pct":0.542},
    "Portland Trail Blazers":  {"net_rtg":-4.5,"ortg":109,"drtg":114,"pace":99,"ts_pct":0.540},
    "Detroit Pistons":         {"net_rtg":-2.8,"ortg":111,"drtg":114,"pace":98,"ts_pct":0.552},
    "Utah Jazz":               {"net_rtg":-6.5,"ortg":109,"drtg":116,"pace":100,"ts_pct":0.538},
    "Charlotte Hornets":       {"net_rtg":-6.8,"ortg":108,"drtg":115,"pace":98,"ts_pct":0.534},
    "Washington Wizards":      {"net_rtg":-7.5,"ortg":107,"drtg":115,"pace":99,"ts_pct":0.530},
}

TEAM_ROLLING = {t: {"roll5": s["net_rtg"] * 1.1, "roll10": s["net_rtg"] * 0.9,
                     "roll_pts": s["ortg"], "roll_allowed": s["drtg"]}
                 for t, s in CURRENT_TEAM_STATS.items()}

# A handful of recognizable current-rotation stars as demo props seeds.
# Real usage/roll5 numbers should come from data/processed/player_logs.csv once populated.
PLAYER_PROPS = {
    "Nikola Jokic":              {"team":"Denver Nuggets",         "pos":"C","mpg":34,"usage":0.300,"ts":0.640,
                                  "roll5_pts":28.0,"roll5_reb":12.5,"roll5_ast":9.5,"roll5_threes":1.0},
    "Shai Gilgeous-Alexander":   {"team":"Oklahoma City Thunder",  "pos":"G","mpg":34,"usage":0.320,"ts":0.620,
                                  "roll5_pts":31.5,"roll5_reb":5.2,"roll5_ast":6.2,"roll5_threes":1.6},
    "Giannis Antetokounmpo":     {"team":"Milwaukee Bucks",        "pos":"F","mpg":33,"usage":0.335,"ts":0.615,
                                  "roll5_pts":30.5,"roll5_reb":11.5,"roll5_ast":6.0,"roll5_threes":0.4},
    "Luka Doncic":               {"team":"Los Angeles Lakers",     "pos":"G","mpg":35,"usage":0.335,"ts":0.585,
                                  "roll5_pts":32.0,"roll5_reb":8.8,"roll5_ast":9.0,"roll5_threes":3.2},
    "Jayson Tatum":              {"team":"Boston Celtics",         "pos":"F","mpg":36,"usage":0.300,"ts":0.585,
                                  "roll5_pts":27.5,"roll5_reb":8.4,"roll5_ast":4.8,"roll5_threes":3.0},
    "Anthony Davis":             {"team":"Dallas Mavericks",       "pos":"F","mpg":34,"usage":0.290,"ts":0.590,
                                  "roll5_pts":24.5,"roll5_reb":11.5,"roll5_ast":3.2,"roll5_threes":0.4},
    "Victor Wembanyama":         {"team":"San Antonio Spurs",      "pos":"C","mpg":31,"usage":0.290,"ts":0.585,
                                  "roll5_pts":23.5,"roll5_reb":10.8,"roll5_ast":3.6,"roll5_threes":2.0},
    "Anthony Edwards":           {"team":"Minnesota Timberwolves", "pos":"G","mpg":36,"usage":0.310,"ts":0.575,
                                  "roll5_pts":27.0,"roll5_reb":5.6,"roll5_ast":4.6,"roll5_threes":3.4},
    "Donovan Mitchell":          {"team":"Cleveland Cavaliers",    "pos":"G","mpg":35,"usage":0.310,"ts":0.585,
                                  "roll5_pts":27.0,"roll5_reb":4.6,"roll5_ast":5.2,"roll5_threes":3.6},
    "Jalen Brunson":             {"team":"New York Knicks",        "pos":"G","mpg":35,"usage":0.305,"ts":0.585,
                                  "roll5_pts":27.5,"roll5_reb":3.6,"roll5_ast":7.2,"roll5_threes":2.4},
    "Devin Booker":              {"team":"Phoenix Suns",           "pos":"G","mpg":35,"usage":0.300,"ts":0.585,
                                  "roll5_pts":26.0,"roll5_reb":4.6,"roll5_ast":6.6,"roll5_threes":2.6},
    "Domantas Sabonis":          {"team":"Sacramento Kings",       "pos":"C","mpg":34,"usage":0.245,"ts":0.615,
                                  "roll5_pts":19.5,"roll5_reb":13.5,"roll5_ast":6.8,"roll5_threes":0.4},
    "Tyrese Haliburton":         {"team":"Indiana Pacers",         "pos":"G","mpg":33,"usage":0.235,"ts":0.590,
                                  "roll5_pts":18.5,"roll5_reb":3.8,"roll5_ast":9.8,"roll5_threes":2.6},
    "Kevin Durant":              {"team":"Houston Rockets",        "pos":"F","mpg":35,"usage":0.290,"ts":0.615,
                                  "roll5_pts":26.5,"roll5_reb":6.2,"roll5_ast":4.2,"roll5_threes":2.0},
    "Paolo Banchero":            {"team":"Orlando Magic",          "pos":"F","mpg":34,"usage":0.295,"ts":0.560,
                                  "roll5_pts":25.5,"roll5_reb":7.4,"roll5_ast":4.4,"roll5_threes":1.6},
}

OPP_DEF_POS = {t: {"G":114,"F":114,"C":114} for t in CURRENT_TEAM_STATS}
OPP_DEF_POS.update({
    "Oklahoma City Thunder": {"G":109,"F":110,"C":111},
    "Boston Celtics":        {"G":110,"F":110,"C":112},
    "Orlando Magic":         {"G":109,"F":109,"C":111},
    "Cleveland Cavaliers":   {"G":111,"F":111,"C":113},
    "Minnesota Timberwolves":{"G":110,"F":110,"C":112},
})


# ── Get today's schedule from ESPN ────────────────────────────────────────────

def fetch_todays_schedule(target_date: date) -> list:
    """Pull today's NBA games from ESPN public API."""
    import requests
    url = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
    params = {"dates": target_date.strftime("%Y%m%d"), "limit": 20}
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"  [WARN] Could not fetch schedule: {e}")
        return []

    games = []
    for event in data.get("events", []):
        comps = event.get("competitions", [{}])[0]
        competitors = comps.get("competitors", [])
        if len(competitors) < 2:
            continue
        home = next((c for c in competitors if c.get("homeAway") == "home"), competitors[0])
        away = next((c for c in competitors if c.get("homeAway") == "away"), competitors[1])
        home_name = home.get("team", {}).get("displayName","")
        away_name = away.get("team", {}).get("displayName","")
        tip_time  = event.get("date","")

        odds_list = comps.get("odds",[{}])
        posted_total = odds_list[0].get("overUnder") if odds_list else None

        games.append({
            "home":         home_name,
            "away":         away_name,
            "tip":          tip_time,
            "posted_total": posted_total,
            "home_rest":    2,
            "away_rest":    2,
        })

    print(f"  Fetched {len(games)} games from ESPN for {target_date}")
    return games


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_models():
    models = {}
    for key, fname in [("spread","spread_model.pkl"),
                        ("totals","totals_model.pkl"),
                        ("props", "props_models.pkl")]:
        path = os.path.join(MODEL_DIR, fname)
        if os.path.exists(path):
            with open(path,"rb") as f:
                models[key] = pickle.load(f)
            print(f"  ✅ {fname} loaded")
        else:
            print(f"  [WARN] {fname} not found — skipping {key} predictions")
            models[key] = None
    return models.get("spread"), models.get("totals"), models.get("props")


def build_context(home, away, target_date, game):
    long_travel = (home in WEST and away in EAST) or (home in EAST and away in WEST)
    return {
        "home_rest_days":  game.get("home_rest", 2),
        "away_rest_days":  game.get("away_rest", 2),
        "home_b2b":        game.get("home_b2b", False),
        "away_b2b":        game.get("away_b2b", False),
        "home_3in4":       False,
        "away_3in4":       False,
        "long_travel":     long_travel,
        "east_to_west":    int(away in EAST and home in WEST),
        "west_to_east":    int(away in WEST and home in EAST),
        "season_game_num": game.get("game_num", 20),
        "month":           target_date.month,
    }


# ── Predictions ────────────────────────────────────────────────────────────────

def run_spread(home, away, ctx, bundle):
    if bundle is None: return 0.0
    model = bundle["model"]
    feats = bundle["feature_names"]
    hs, as_ = CURRENT_TEAM_STATS.get(home,{}), CURRENT_TEAM_STATS.get(away,{})
    hr, ar  = TEAM_ROLLING.get(home,{}), TEAM_ROLLING.get(away,{})
    row = {f: 0 for f in feats}
    row.update({
        "home_rolling_5g":   hr.get("roll5",0),
        "away_rolling_5g":   ar.get("roll5",0),
        "rolling_diff_5g":   hr.get("roll5",0) - ar.get("roll5",0),
        "home_rolling_10g":  hr.get("roll10",0),
        "away_rolling_10g":  ar.get("roll10",0),
        "rolling_diff_10g":  hr.get("roll10",0) - ar.get("roll10",0),
        "home_net_rtg":      hs.get("net_rtg",0),
        "away_net_rtg":      as_.get("net_rtg",0),
        "net_rtg_diff":      hs.get("net_rtg",0) - as_.get("net_rtg",0),
        "home_ortg":         hs.get("ortg",114),
        "away_ortg":         as_.get("ortg",114),
        "home_drtg":         hs.get("drtg",114),
        "away_drtg":         as_.get("drtg",114),
        "home_pace":         hs.get("pace",99),
        "away_pace":         as_.get("pace",99),
        "avg_pace":          (hs.get("pace",99)+as_.get("pace",99))/2,
        "home_rest_days":    ctx["home_rest_days"],
        "away_rest_days":    ctx["away_rest_days"],
        "rest_diff":         ctx["home_rest_days"]-ctx["away_rest_days"],
        "home_back_to_back": int(ctx["home_b2b"]),
        "away_back_to_back": int(ctx["away_b2b"]),
        "home_three_in_four":int(ctx["home_3in4"]),
        "away_three_in_four":int(ctx["away_3in4"]),
        "long_travel":       int(ctx["long_travel"]),
        "east_to_west":      ctx["east_to_west"],
        "west_to_east":      ctx["west_to_east"],
        "season_game_num":   ctx["season_game_num"],
        "month":             ctx["month"],
        "is_playoff":        int(ctx["month"] in (4, 5, 6)),
    })
    try:
        X = pd.DataFrame([row])[feats]
        return round(float(model.predict(X)[0]), 1)
    except Exception as e:
        print(f"  [WARN] Spread prediction error: {e}")
        return 0.0


def run_totals(home, away, ctx, bundle):
    if bundle is None: return 224.0
    model = bundle["model"]
    feats = bundle["feature_names"]
    hs, as_ = CURRENT_TEAM_STATS.get(home,{}), CURRENT_TEAM_STATS.get(away,{})
    hr, ar  = TEAM_ROLLING.get(home,{}), TEAM_ROLLING.get(away,{})
    avg_p = (hs.get("pace",99)+as_.get("pace",99))/2
    row = {f: 0 for f in feats}
    row.update({
        "avg_pace": avg_p, "home_pace": hs.get("pace",99), "away_pace": as_.get("pace",99),
        "pace_sum": hs.get("pace",99)+as_.get("pace",99),
        "home_ortg": hs.get("ortg",114), "away_ortg": as_.get("ortg",114),
        "combined_ortg": hs.get("ortg",114)+as_.get("ortg",114),
        "home_drtg": hs.get("drtg",114), "away_drtg": as_.get("drtg",114),
        "combined_drtg": hs.get("drtg",114)+as_.get("drtg",114),
        "home_ts_pct": hs.get("ts_pct",0.575), "away_ts_pct": as_.get("ts_pct",0.575),
        "combined_ts": hs.get("ts_pct",0.575)+as_.get("ts_pct",0.575),
        "pace_x_ortg": avg_p*(hs.get("ortg",114)+as_.get("ortg",114))/200,
        "def_mismatch": abs((hs.get("ortg",114)-as_.get("drtg",114))-(as_.get("ortg",114)-hs.get("drtg",114))),
        "home_rolling_total_5g": hr.get("roll_pts",114),
        "away_rolling_total_5g": ar.get("roll_pts",114),
        "rolling_total_sum_5g": hr.get("roll_pts",114)+ar.get("roll_pts",114),
        "home_back_to_back": int(ctx["home_b2b"]),
        "away_back_to_back": int(ctx["away_b2b"]),
        "both_b2b": int(ctx["home_b2b"] and ctx["away_b2b"]),
        "home_rest_days": ctx["home_rest_days"], "away_rest_days": ctx["away_rest_days"],
        "long_travel": int(ctx["long_travel"]),
        "month": ctx["month"], "is_playoff": int(ctx["month"] in (4, 5, 6)),
        "season_game_num": ctx["season_game_num"],
    })
    try:
        X = pd.DataFrame([row])[feats]
        return round(float(model.predict(X)[0]), 1)
    except Exception as e:
        print(f"  [WARN] Totals prediction error: {e}")
        return 224.0


def run_props(home, away, ctx, bundle, posted_lines=None):
    if bundle is None: return []
    models     = bundle["models"]
    thresholds = bundle["thresholds"]
    posted     = posted_lines or {}
    results    = []
    avg_pace = (CURRENT_TEAM_STATS.get(home,{}).get("pace",99) +
                CURRENT_TEAM_STATS.get(away,{}).get("pace",99)) / 2

    for player, pdata in PLAYER_PROPS.items():
        if pdata["team"] not in [home, away]:
            continue
        opp     = away if pdata["team"] == home else home
        opp_def = OPP_DEF_POS.get(opp,{}).get(pdata["pos"],114)
        is_home = int(pdata["team"] == home)
        b2b     = ctx["home_b2b"] if is_home else ctx["away_b2b"]
        rest    = ctx["home_rest_days"] if is_home else ctx["away_rest_days"]
        proj_min= pdata["mpg"] * (0.88 if b2b else 1.0)

        row = {
            "minutes": proj_min, "roll5_minutes": pdata["mpg"],
            "usage": pdata["usage"], "ts_pct": pdata["ts"],
            "opp_drtg_pos": opp_def, "avg_pace": avg_pace,
            "team_pace": CURRENT_TEAM_STATS.get(pdata["team"],{}).get("pace",99),
            "rest_days": rest, "is_home": is_home,
            "roll5_pts": pdata.get("roll5_pts",18), "roll5_reb": pdata.get("roll5_reb",6),
            "roll5_ast": pdata.get("roll5_ast",4),  "roll5_threes": pdata.get("roll5_threes",1),
            "roll5_pra": pdata.get("roll5_pts",18)+pdata.get("roll5_reb",6)+pdata.get("roll5_ast",4),
            "season_game_num": ctx["season_game_num"], "month": ctx["month"],
            "def_adj": 114-opp_def,
            "usage_pace": pdata["usage"]*avg_pace/99.0,
        }

        player_result = {"player": player, "team": pdata["team"],
                         "opp": opp, "pos": pdata["pos"],
                         "proj_min": round(proj_min,1), "b2b": b2b, "props": {}}

        for target in PROP_TARGETS:
            if target not in models: continue
            m     = models[target]
            feats = [f for f in m["features"] if f in row]
            try:
                X    = pd.DataFrame([row])[feats]
                pred = float(m["model"].predict(X)[0])
            except:
                pred = m.get("mean_stat", 12.0)
            thresh = thresholds.get(target, 2.0)
            line   = posted.get(player,{}).get(target)
            edge   = round(pred-line,1) if line else None
            signal = None
            if edge is not None:
                if edge > thresh:  signal = "OVER"
                elif edge < -thresh: signal = "UNDER"
            player_result["props"][target] = {
                "pred": round(pred,1), "line": line, "edge": edge, "signal": signal}

        results.append(player_result)
    return results


def score_confidence(edge, model_type):
    t = {"spread":{"HIGH":5.0,"MED":3.0},"totals":{"HIGH":5.0,"MED":2.5},"props":{"HIGH":3.5,"MED":2.0}}.get(model_type,{"HIGH":5,"MED":3})
    ae = abs(edge) if edge else 0
    if ae >= t["HIGH"]: return "HIGH", 3
    if ae >= t["MED"]:  return "MED",  2
    return "LOW", 1


def collect_best_bets(games_output):
    bets = []
    for g in games_output:
        home, away = g["home"]["name"], g["away"]["name"]
        matchup = f"{away} @ {home}"
        sp = g["spread"]
        if sp.get("edge"):
            conf, stars = score_confidence(sp["edge"], "spread")
            if stars >= 2:
                bets.append({"type":"SPREAD","game":matchup,"play":sp["model_line"],
                              "edge":sp["edge"],"conf":conf,"stars":stars,"tip":g["tip"]})
        tot = g["totals"]
        if tot.get("edge"):
            conf, stars = score_confidence(tot["edge"], "totals")
            if stars >= 2:
                bets.append({"type":"TOTAL","game":matchup,
                              "play":f"{tot['play']} {tot['line']}",
                              "edge":tot["edge"],"conf":conf,"stars":stars,"tip":g["tip"]})
        for pr in g.get("props",[]):
            for pt, pd_data in pr["props"].items():
                if pd_data["signal"] and pd_data["edge"] is not None:
                    conf, stars = score_confidence(pd_data["edge"], "props")
                    if stars >= 2:
                        bets.append({"type":"PROP","game":matchup,
                                     "play":f"{pr['player']} {pt.upper()} {pd_data['signal']} {pd_data['line']}",
                                     "edge":pd_data["edge"],"conf":conf,"stars":stars,"tip":g["tip"]})
    bets.sort(key=lambda b: (-b["stars"], -abs(b["edge"])))
    for i, b in enumerate(bets): b["rank"] = i+1
    return bets[:8]


def run_pipeline(target_date: date, schedule: list, posted_props: dict = None):
    print(f"\n═══ NBA DAILY PIPELINE — {target_date} ═══\n")
    spread_bundle, totals_bundle, props_bundle = load_models()

    # Load injury adjustments
    injury_adjustments = {}
    injuries_path = "data/raw/injuries_today.csv"
    if INJURIES_AVAILABLE and os.path.exists(injuries_path):
        import pandas as pd
        inj_df = pd.read_csv(injuries_path)
        from scrape_injuries import reallocate_minutes
        injury_adjustments = reallocate_minutes(inj_df, schedule)
        out_count = len(inj_df[inj_df["is_out"]]) if "is_out" in inj_df.columns else 0
        print(f"  Injuries loaded: {out_count} OUT players")

    # Load referee stats
    ref_stats_df = None
    ref_path = "data/raw/ref_stats.csv"
    if REFS_AVAILABLE and os.path.exists(ref_path):
        import pandas as pd
        ref_stats_df = pd.read_csv(ref_path)
        print(f"  Ref stats loaded: {len(ref_stats_df)} officials")

    # Load today's ref assignments
    ref_assignments = {}
    ref_assign_path = "data/raw/ref_assignments_today.csv"
    if os.path.exists(ref_assign_path):
        import pandas as pd
        ra = pd.read_csv(ref_assign_path)
        if "refs" in ra.columns and "home_team" in ra.columns:
            for _, row in ra.iterrows():
                key = f"{row.get('away_team','')}@{row.get('home_team','')}"
                ref_assignments[key] = str(row.get("refs",""))

    posted_props  = posted_props or {}
    games_output  = []

    if not schedule:
        print("  No games scheduled today.")

    for game in schedule:
        home = game.get("home","")
        away = game.get("away","")
        tip  = game.get("tip","TBD")
        if not home or not away:
            continue

        print(f"  Processing: {away} @ {home}")
        ctx = build_context(home, away, target_date, game)

        # Add referee crew stats to context
        game_key_ref = f"{away}@{home}"
        refs_str = ref_assignments.get(game_key_ref, "")
        if refs_str and ref_stats_df is not None and REFS_AVAILABLE:
            crew = get_crew_stats(refs_str, ref_stats_df)
            ctx.update({
                "crew_avg_total": crew.get("crew_avg_total", 224.0),
                "crew_total_adj": crew.get("crew_total_adj", 0.0),
                "crew_over_rate": crew.get("crew_over_rate", 0.5),
                "crew_foul_rate": crew.get("crew_foul_rate", 42.0),
                "crew_refs":      refs_str,
            })

        spread_pred   = run_spread(home, away, ctx, spread_bundle)
        posted_spread = game.get("posted_spread")
        spread_edge   = round(spread_pred-(-posted_spread),1) if posted_spread else None
        spread_conf, spread_stars = score_confidence(spread_edge or 0, "spread")

        total_pred  = run_totals(home, away, ctx, totals_bundle)
        posted_total= game.get("posted_total")
        total_edge  = round(total_pred-posted_total,1) if posted_total else None
        total_play  = ("OVER" if total_edge and total_edge>0 else "UNDER") if total_edge else None
        total_conf, total_stars = score_confidence(total_edge or 0, "totals")

        props_out = run_props(home, away, ctx, props_bundle,
                              posted_props.get(f"{away}@{home}",{}))

        flags = []
        if ctx["away_b2b"]:    flags.append(f"B2B ({TEAM_ABBR.get(away, away[:3].upper())})")
        if ctx["home_b2b"]:    flags.append(f"B2B ({TEAM_ABBR.get(home, home[:3].upper())})")
        if ctx["long_travel"]: flags.append("Travel")

        def fmt_line(pred, h, a):
            if abs(pred) < 0.5: return "Pick'em"
            return f"{h} -{abs(pred):.1f}" if pred > 0 else f"{a} -{abs(pred):.1f}"

        hs  = CURRENT_TEAM_STATS.get(home,{})
        as_ = CURRENT_TEAM_STATS.get(away,{})

        games_output.append({
            "tip": tip,
            "home": {"name":home,"abbr":TEAM_ABBR.get(home, home[:3].upper()),"record":game.get("home_record","?-?"),"net_rtg":hs.get("net_rtg",0)},
            "away": {"name":away,"abbr":TEAM_ABBR.get(away, away[:3].upper()),"record":game.get("away_record","?-?"),"net_rtg":as_.get("net_rtg",0)},
            "spread": {"pred":spread_pred,"model_line":fmt_line(spread_pred,home,away),
                       "posted_line":posted_spread,"edge":spread_edge,
                       "conf":spread_conf,"stars":spread_stars},
            "totals": {"pred":total_pred,"line":posted_total,"edge":total_edge,
                       "play":total_play,"conf":total_conf,"stars":total_stars},
            "props": props_out, "flags": flags,
            "ctx": {k:v for k,v in ctx.items() if not isinstance(v, np.bool_)},
        })

    best_bets = collect_best_bets(games_output)

    # Apply Kelly sizing to best bets
    if KELLY_AVAILABLE and best_bets:
        best_bets = size_all_bets(best_bets, bankroll=1000.0)
        print(f"  Kelly sizing applied: {sum(1 for b in best_bets if b.get('units',0)>0)} bets sized")

    return {
        "date":       str(target_date),
        "generated":  datetime.now().isoformat(),
        "games":      games_output,
        "best_bets":  best_bets,
        "model_stats":{"spread":{"algo":"Ridge / GBR","cv_mae":None,"dir_acc":None,"strong_ats":None,"n":0},
                       "totals":{"algo":"Ridge / RF","cv_mae":None,"ou_acc":None,"strong_ou":None,"n":0},
                       "props": {"algo":"Ridge / GBR","cv_mae":None,"hit_rate":None,"strong_hr":None,"n":0}},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=str(date.today()))
    parser.add_argument("--out",  default=None)
    args = parser.parse_args()

    target   = datetime.strptime(args.date, "%Y-%m-%d").date()
    schedule = fetch_todays_schedule(target)

    result   = run_pipeline(target, schedule)
    out_path = args.out or os.path.join(OUTPUT_DIR, f"predictions_{target}.json")

    def convert(o):
        if isinstance(o, (np.integer,)):  return int(o)
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, (np.bool_,)):    return bool(o)
        raise TypeError(f"Not serializable: {type(o)}")

    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=convert)

    print(f"\n✅ Predictions written → {out_path}")
    print(f"   Games: {len(result['games'])} | Best bets: {len(result['best_bets'])}")
    for b in result["best_bets"][:3]:
        e = f"+{b['edge']}" if b['edge'] > 0 else str(b['edge'])
        print(f"  #{b['rank']} [{b['type']}] {b['play']}  {e} pts  {'★'*b['stars']}")
