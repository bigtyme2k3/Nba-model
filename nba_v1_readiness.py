"""
nba_v1_readiness.py
-------------------
Research governance gate for NBA V1.

A technically healthy pipeline is not automatically a production betting model.
This report separates:
  1) DATA_MODEL_HEALTHY
  2) READY_FOR_FORWARD_TESTING
  3) PRODUCTION_READY

Production readiness is deliberately conservative and requires a meaningful
immutable forward sample plus price-aware ROI and probability calibration.
"""
from __future__ import annotations

import json
import pickle
from datetime import date, datetime
from pathlib import Path

DATA=Path("data/processed")
MODELS=Path("models")
TRACK=Path("data/tracking")
OUT=TRACK/"nba_v1_readiness.json"

MIN_FORWARD_BETS=200
MIN_PROP_FORWARD_BETS=100
MIN_ROI=0.01
MAX_BRIER=0.245
MIN_CLV_MATCHED=100
MIN_BEAT_CLOSE_RATE=0.50
MAX_MODEL_LAG_DAYS=14


def load_json(path, default=None):
    default={} if default is None else default
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return default


def model_meta(name):
    path=MODELS/f"{name}_model.pkl" if name!="props" else MODELS/"props_models.pkl"
    if not path.exists():
        return {"name":name,"status":"missing","path":str(path)}
    try:
        bundle=pickle.load(path.open("rb"))
    except Exception as exc:
        return {"name":name,"status":"unreadable","error":str(exc),"path":str(path)}
    return {
        "name":name,
        "status":bundle.get("status","legacy_unstamped"),
        "trained_through":bundle.get("trained_through"),
        "trained_rows":bundle.get("trained_rows"),
        "validation":bundle.get("validation"),
        "markets":sorted(bundle.get("models",{}).keys()) if name=="props" else None,
        "model_name":bundle.get("model_name") or bundle.get("best_name"),
    }


def lag_days(trained_through, feature_date):
    try:
        a=datetime.strptime(str(trained_through),"%Y-%m-%d").date()
        b=datetime.strptime(str(feature_date),"%Y-%m-%d").date()
        return (b-a).days
    except Exception:
        return None


def main():
    player=load_json(DATA/"player_features_status.json")
    team=load_json(DATA/"team_features_status.json")
    perf=load_json(TRACK/"nba_forward_performance.json")
    clv=load_json(TRACK/"nba_prop_clv_report.json")

    models={name:model_meta(name) for name in ("spread","totals","props")}
    models["spread"]["lag_days"]=lag_days(models["spread"].get("trained_through"),team.get("max_game_date"))
    models["totals"]["lag_days"]=lag_days(models["totals"].get("trained_through"),team.get("max_game_date"))
    models["props"]["lag_days"]=lag_days(models["props"].get("trained_through"),player.get("max_game_date"))

    checks={
        "player_features_ready": player.get("status")=="READY",
        "team_features_ready": team.get("status")=="READY",
        "all_30_teams_profiled": int(team.get("teams") or 0)==30,
        "all_models_trained": all(m.get("status")=="trained" for m in models.values()),
        "all_models_fresh": all(
            m.get("lag_days") is not None and m["lag_days"] <= MAX_MODEL_LAG_DAYS
            for m in models.values()
        ),
        "all_prop_markets_trained": set(models["props"].get("markets") or []) >= {"pts","reb","ast","threes","pra"},
    }

    data_model_healthy=all(checks.values())
    ready_for_forward_testing=data_model_healthy

    forward_bets=int(perf.get("bet_rows") or 0)
    prop_perf=(perf.get("by_market") or {}).get("PROP") or {}
    prop_bets=int(prop_perf.get("bets") or 0)
    roi=perf.get("roi")
    calibration=perf.get("probability_calibration") or {}
    brier=calibration.get("brier")
    clv_matched=int(clv.get("matched_closes") or 0)
    beat_close_rate=clv.get("beat_close_rate")
    avg_line_clv=clv.get("avg_line_clv")
    avg_price_clv=clv.get("avg_same_line_price_clv_probability")

    evidence_checks={
        "forward_bets_at_least_200": forward_bets >= MIN_FORWARD_BETS,
        "prop_forward_bets_at_least_100": prop_bets >= MIN_PROP_FORWARD_BETS,
        "positive_price_aware_roi": roi is not None and float(roi) >= MIN_ROI,
        "acceptable_brier": brier is not None and float(brier) <= MAX_BRIER,
        "clv_rows_at_least_100": clv_matched >= MIN_CLV_MATCHED,
        "beat_close_rate_at_least_50pct": (
            beat_close_rate is not None and float(beat_close_rate) >= MIN_BEAT_CLOSE_RATE
        ),
        "nonnegative_clv_signal": (
            (avg_line_clv is not None and float(avg_line_clv) >= 0.0)
            or (avg_price_clv is not None and float(avg_price_clv) >= 0.0)
        ),
    }

    production_ready=ready_for_forward_testing and all(evidence_checks.values())

    report={
        "version":"NBA_V1.0",
        "generated_at_utc":datetime.utcnow().isoformat()+"Z",
        "data_model_healthy":data_model_healthy,
        "ready_for_forward_testing":ready_for_forward_testing,
        "production_ready":production_ready,
        "status":(
            "PRODUCTION_READY" if production_ready
            else "READY_FOR_FORWARD_TESTING" if ready_for_forward_testing
            else "FOUNDATION_NOT_READY"
        ),
        "checks":checks,
        "models":models,
        "feature_stores":{
            "player":{"status":player.get("status"),"max_game_date":player.get("max_game_date"),"rows":player.get("rows"),"seasons":player.get("seasons")},
            "team":{"status":team.get("status"),"max_game_date":team.get("max_game_date"),"teams":team.get("teams"),"seasons":team.get("seasons")},
        },
        "forward_evidence":{
            "bet_rows":forward_bets,
            "prop_bets":prop_bets,
            "roi":roi,
            "brier":brier,
            "clv_matched":clv_matched,
            "beat_close_rate":beat_close_rate,
            "avg_line_clv":avg_line_clv,
            "avg_same_line_price_clv_probability":avg_price_clv,
            "by_sportsbook":perf.get("by_sportsbook",{}),
            "by_prop_stat":perf.get("by_prop_stat",{}),
            "clv_by_sportsbook":clv.get("by_sportsbook",{}),
            "clv_by_stat":clv.get("by_stat",{}),
            "evidence_checks":evidence_checks,
        },
        "promotion_policy":{
            "min_forward_bets":MIN_FORWARD_BETS,
            "min_prop_forward_bets":MIN_PROP_FORWARD_BETS,
            "min_price_aware_roi":MIN_ROI,
            "max_brier":MAX_BRIER,
            "min_matched_closes":MIN_CLV_MATCHED,
            "min_beat_close_rate":MIN_BEAT_CLOSE_RATE,
            "note":"Thresholds are governance gates, not proof of future profitability. Promotion still requires review of calibration stability, book/market splits, CLV distribution and drawdowns.",
        },
    }
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    main()
