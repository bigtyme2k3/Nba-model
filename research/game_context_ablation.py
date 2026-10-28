"""
research/game_context_ablation.py
---------------------------------
Research-only paired ablation for NBA game context.

Compares identical chronological folds using Ridge models:
- BASE: existing form/rest/travel/season features
- FULL: BASE + leakage-safe prior pace/ORTG/DRTG/net-rating/TS context

No models are published.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from spread_model import SPREAD_FEATURES, engineer_features as engineer_spread
from totals_model import TOTALS_FEATURES, engineer_totals_features

TEAM_CONTEXT_SPREAD = {
    "home_net_rtg","away_net_rtg","net_rtg_diff",
    "home_pace","away_pace","avg_pace",
    "home_ortg","away_ortg","home_drtg","away_drtg",
}
TEAM_CONTEXT_TOTALS = {
    "avg_pace","home_pace","away_pace","pace_sum",
    "home_ortg","away_ortg","combined_ortg",
    "home_drtg","away_drtg","combined_drtg","def_mismatch",
    "home_ts_pct","away_ts_pct","combined_ts","pace_x_ortg",
}


def ridge():
    return Pipeline([
        ("sc", StandardScaler()),
        ("m", Ridge(alpha=10.0)),
    ])


def paired_cv(df, target, full_features, removed_features, directional=False):
    available=[f for f in full_features if f in df.columns]
    base=[f for f in available if f not in removed_features]
    full=available

    X_full=df[full].fillna(0)
    X_base=df[base].fillna(0)
    y=pd.to_numeric(df[target],errors="coerce")
    valid=y.notna()
    X_full,X_base,y=X_full.loc[valid].reset_index(drop=True),X_base.loc[valid].reset_index(drop=True),y.loc[valid].reset_index(drop=True)

    tscv=TimeSeriesSplit(n_splits=4)
    folds=[]
    for fold,(tr,te) in enumerate(tscv.split(X_full),start=1):
        mb,mf=ridge(),ridge()
        mb.fit(X_base.iloc[tr],y.iloc[tr])
        mf.fit(X_full.iloc[tr],y.iloc[tr])
        pb=mb.predict(X_base.iloc[te])
        pf=mf.predict(X_full.iloc[te])
        rec={
            "fold":fold,
            "n_train":len(tr),
            "n_test":len(te),
            "base_mae":float(mean_absolute_error(y.iloc[te],pb)),
            "full_mae":float(mean_absolute_error(y.iloc[te],pf)),
        }
        rec["mae_delta_full_minus_base"]=rec["full_mae"]-rec["base_mae"]
        if directional:
            rec["base_directional"]=float(((pb>0)==(y.iloc[te].values>0)).mean())
            rec["full_directional"]=float(((pf>0)==(y.iloc[te].values>0)).mean())
        folds.append(rec)

    deltas=[x["mae_delta_full_minus_base"] for x in folds]
    return {
        "rows":len(y),
        "base_feature_count":len(base),
        "full_feature_count":len(full),
        "added_features":[f for f in full if f in removed_features],
        "folds":folds,
        "base_mean_mae":float(np.mean([x["base_mae"] for x in folds])),
        "full_mean_mae":float(np.mean([x["full_mae"] for x in folds])),
        "mean_mae_delta_full_minus_base":float(np.mean(deltas)),
        "folds_improved":int(sum(d<0 for d in deltas)),
        "folds_worsened":int(sum(d>0 for d in deltas)),
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--data",default="data/processed/master_all.csv")
    p.add_argument("--out",default="data/tracking/game_context_ablation.json")
    args=p.parse_args()

    df=pd.read_csv(args.data,parse_dates=["game_date"])
    if "team_context_complete" not in df.columns:
        raise SystemExit("team_context_complete missing; rebuild master with prior team context")
    df=df[pd.to_numeric(df["team_context_complete"],errors="coerce").fillna(0).eq(1)].copy()
    df=df.sort_values(["season","game_date"]).reset_index(drop=True)

    spread=engineer_spread(df.copy())
    totals=engineer_totals_features(df.copy())

    report={
        "status":"READY",
        "policy":"PAIRED_IDENTICAL_TIME_SERIES_FOLDS_RESEARCH_ONLY",
        "spread":paired_cv(
            spread,"actual_spread",SPREAD_FEATURES,TEAM_CONTEXT_SPREAD,directional=True
        ),
        "totals":paired_cv(
            totals,"actual_total",TOTALS_FEATURES,TEAM_CONTEXT_TOTALS,directional=False
        ),
    }
    report["spread"]["context_helped_mean_mae"]=report["spread"]["mean_mae_delta_full_minus_base"]<0
    report["totals"]["context_helped_mean_mae"]=report["totals"]["mean_mae_delta_full_minus_base"]<0

    out=Path(args.out)
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    main()
