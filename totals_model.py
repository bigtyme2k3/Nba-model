"""
totals_model.py — NBA Over/Under Model
Usage:
    python totals_model.py --mode train --data data/processed/master_all.csv
    python totals_model.py --mode predict
"""
import os, pickle, warnings, argparse
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
warnings.filterwarnings("ignore")

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)
TARGET = "actual_total"

TOTALS_FEATURES = [
    "avg_pace","home_pace","away_pace","pace_sum",
    "home_ortg","away_ortg","combined_ortg",
    "home_drtg","away_drtg","combined_drtg","def_mismatch",
    "home_ts_pct","away_ts_pct","combined_ts","pace_x_ortg",
    "home_rolling_total_5g","away_rolling_total_5g",
    "home_rolling_allowed_5g","away_rolling_allowed_5g","rolling_total_sum_5g",
    "home_back_to_back","away_back_to_back","home_rest_days","away_rest_days","both_b2b",
    "long_travel","month","is_playoff","season_game_num",
]

def engineer_totals_features(df):
    df = df.sort_values(["season","game_date"]).reset_index(drop=True)
    df["pace_sum"]      = df.get("home_pace", 99) + df.get("away_pace", 99)
    df["combined_ortg"] = df.get("home_ortg", 114) + df.get("away_ortg", 114)
    df["combined_drtg"] = df.get("home_drtg", 114) + df.get("away_drtg", 114)
    df["combined_ts"]   = df.get("home_ts_pct", 0.575) + df.get("away_ts_pct", 0.575)
    df["avg_pace"]      = df.get("pace_sum", 198) / 2
    df["pace_x_ortg"]   = df["avg_pace"] * df["combined_ortg"] / 200
    df["def_mismatch"]  = abs(
        (df.get("home_ortg", 114) - df.get("away_drtg", 114)) -
        (df.get("away_ortg", 114) - df.get("home_drtg", 114))
    )
    df["both_b2b"] = (
        (df.get("home_back_to_back", pd.Series(0, index=df.index)).astype(int) == 1) &
        (df.get("away_back_to_back", pd.Series(0, index=df.index)).astype(int) == 1)
    ).astype(int)

    df = add_rolling_scoring(df)

    for col in ["home_back_to_back","away_back_to_back","long_travel"]:
        if col in df.columns:
            df[col] = df[col].astype(int)
    if "game_date" in df.columns:
        df["month"] = pd.to_datetime(df["game_date"]).dt.month
    if "month" in df.columns:
        # NBA playoffs run April - June.
        df["is_playoff"] = df["month"].isin([4, 5, 6]).astype(int)
    return df

def add_rolling_scoring(df, window=5):
    df = df.sort_values(["season","game_date"]).reset_index(drop=True)
    if "home_pts" not in df.columns or "away_pts" not in df.columns:
        for col in [f"home_rolling_total_{window}g", f"away_rolling_total_{window}g",
                    f"home_rolling_allowed_{window}g", f"away_rolling_allowed_{window}g",
                    f"rolling_total_sum_{window}g"]:
            df[col] = np.nan
        return df

    home_rec = df[["game_date","season","home_team","home_pts","away_pts"]].copy()
    home_rec.columns = ["game_date","season","team","pts_scored","pts_allowed"]
    away_rec = df[["game_date","season","away_team","away_pts","home_pts"]].copy()
    away_rec.columns = ["game_date","season","team","pts_scored","pts_allowed"]
    all_rec = pd.concat([home_rec, away_rec]).sort_values(["team","season","game_date"])
    all_rec = all_rec[all_rec["pts_scored"].notna()].copy()

    for col in ["pts_scored","pts_allowed"]:
        all_rec[f"rolling_{window}g_{col}"] = (
            all_rec.groupby(["team","season"])[col]
            .transform(lambda x: x.shift(1).rolling(window, min_periods=2).mean())
        )

    for side, prefix in [("home_team","home"), ("away_team","away")]:
        merge = all_rec[["game_date","season","team",
                          f"rolling_{window}g_pts_scored",
                          f"rolling_{window}g_pts_allowed"]].copy()
        merge.columns = ["game_date","season",side,
                         f"{prefix}_rolling_total_{window}g",
                         f"{prefix}_rolling_allowed_{window}g"]
        df = df.merge(merge, on=["game_date","season",side], how="left")

    df[f"rolling_total_sum_{window}g"] = (
        df.get(f"home_rolling_total_{window}g", pd.Series(114, index=df.index)).fillna(114) +
        df.get(f"away_rolling_total_{window}g", pd.Series(114, index=df.index)).fillna(114)
    )
    return df

def train_totals_model(data_path):
    print("\n═══ NBA TOTALS MODEL — TRAINING ═══\n")
    df = pd.read_csv(data_path, parse_dates=["game_date"])
    df = df.sort_values(["season","game_date"]).reset_index(drop=True)
    df = engineer_totals_features(df)
    df = df[df[TARGET].notna()].copy()

    available = [f for f in TOTALS_FEATURES if f in df.columns]
    X = df[available].fillna(0)
    y = df[TARGET]
    print(f"Features: {len(available)} | Games: {len(X)}\n")

    tscv = TimeSeriesSplit(n_splits=4)
    models = {
        "Ridge":  Pipeline([("sc", StandardScaler()), ("m", Ridge(alpha=10.0))]),
        "RF":     Pipeline([("m", RandomForestRegressor(n_estimators=200, max_depth=5,
                                                         min_samples_leaf=8, random_state=42))]),
    }

    cv_results = {}
    for name, model in models.items():
        maes, ous = [], []
        for tr, te in tscv.split(X):
            model.fit(X.iloc[tr], y.iloc[tr])
            preds = model.predict(X.iloc[te])
            maes.append(mean_absolute_error(y.iloc[te], preds))
            if "avg_total_point" in df.columns:
                line = df["avg_total_point"].iloc[te].values
                valid = ~np.isnan(line)
                if valid.sum() > 0:
                    ous.append(((preds[valid] > line[valid]) == (y.iloc[te].values[valid] > line[valid])).mean())
        cv_results[name] = {"mae": np.mean(maes), "ou": np.mean(ous) if ous else None}
        print(f"  {name}: MAE={cv_results[name]['mae']:.2f}" +
              (f" | O/U={cv_results[name]['ou']:.1%}" if cv_results[name]['ou'] else ""))

    best_name = min(cv_results, key=lambda k: cv_results[k]["mae"])
    print(f"\n✅ Best: {best_name}")
    final = models[best_name]
    final.fit(X, y)

    ou_results = {}
    if "avg_total_point" in df.columns:
        preds = final.predict(X)
        line  = df["avg_total_point"].values
        valid = ~np.isnan(line)
        if valid.sum() > 0:
            edge = preds[valid] - line[valid]
            real_ov = y.values[valid] > line[valid]
            strong  = np.abs(edge) >= 2
            if strong.sum() > 0:
                won = ((edge[strong] > 0) & real_ov[strong]) | ((edge[strong] < 0) & ~real_ov[strong])
                ou_results = {"strong_n": int(strong.sum()), "strong_ou": float(won.mean()),
                              "model_mae": float(mean_absolute_error(y.values[valid], preds[valid])),
                              "line_mae":  float(mean_absolute_error(y.values[valid], line[valid]))}

    path = os.path.join(MODEL_DIR, "totals_model.pkl")
    with open(path, "wb") as f:
        pickle.dump({"model": final, "feature_names": available,
                     "model_name": best_name, "cv_results": cv_results,
                     "ou_results": ou_results}, f)
    print(f"\n✅ Saved → {path}")
    print(f"\n═══ SUMMARY ═══")
    print(f"  Model: {best_name} | CV MAE: {cv_results[best_name]['mae']:.2f} pts")
    if ou_results:
        print(f"  Strong plays O/U: {ou_results['strong_ou']:.1%} ({ou_results['strong_n']} games)")

def predict_totals(games, model_path="models/totals_model.pkl"):
    with open(model_path, "rb") as f:
        bundle = pickle.load(f)
    model, feats = bundle["model"], bundle["feature_names"]
    rows = []
    for g in games:
        hs, as_ = g["home_stats"], g["away_stats"]
        row = {f: 0 for f in feats}
        avg_p = (hs.get("pace",99) + as_.get("pace",99)) / 2
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
            "home_rolling_total_5g": hs.get("rolling_pts",114),
            "away_rolling_total_5g": as_.get("rolling_pts",114),
            "rolling_total_sum_5g": hs.get("rolling_pts",114)+as_.get("rolling_pts",114),
            "home_back_to_back": int(g["ctx"].get("home_b2b",False)),
            "away_back_to_back": int(g["ctx"].get("away_b2b",False)),
            "both_b2b": int(bool(g["ctx"].get("home_b2b")) and bool(g["ctx"].get("away_b2b"))),
            "home_rest_days": g["ctx"].get("home_rest",2),
            "away_rest_days": g["ctx"].get("away_rest",2),
            "long_travel": int(g["ctx"].get("long_travel",False)),
            "month": g["ctx"].get("month",11), "is_playoff": 0,
            "season_game_num": g["ctx"].get("game_num",10),
        })
        X = pd.DataFrame([row])[feats]
        pred = float(model.predict(X)[0])
        posted = g.get("posted_total")
        edge   = round(pred - posted, 1) if posted else None
        rows.append({"game": f"{g['away']} @ {g['home']}", "pred_total": round(pred,1),
                     "posted_total": posted, "edge": edge,
                     "model_play": ("OVER" if edge and edge>0 else "UNDER") if edge else "N/A",
                     "confidence": ("⭐⭐⭐ HIGH" if edge and abs(edge)>=5 else
                                    "⭐⭐   MED" if edge and abs(edge)>=2.5 else "⭐     LOW")})
    return pd.DataFrame(rows)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train","predict"], default="train")
    parser.add_argument("--data", default="data/processed/master_all.csv")
    args = parser.parse_args()
    if args.mode == "train":
        train_totals_model(args.data)
