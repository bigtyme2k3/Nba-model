"""
props_model.py — NBA Player Props Model
Predicts pts, reb, ast, threes, pra per player per game.
Usage:
    python props_model.py --mode train
    python props_model.py --mode predict
"""

from model_publication import publish_model_bundle
import os, pickle, warnings, argparse
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
from sklearn.impute import SimpleImputer
from prop_features import BASE_FEATURES, VERSION as FEATURE_VERSION, build_features
warnings.filterwarnings("ignore")

MODEL_DIR  = "models"
DATA_PATH  = "data/processed/player_logs.csv"
os.makedirs(MODEL_DIR, exist_ok=True)

PROP_TARGETS = ["pts","reb","ast","threes","pra"]

def prepare_data(df):
    """Prepare player rows for genuinely chronological validation.

    build_player_features.py already shifts rolling inputs by one game, so this
    function must not recreate target-game features. We compute bookkeeping in
    player order, then return rows in global game-date order for TimeSeriesSplit.
    """
    df = df.copy()
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    df = df[df["game_date"].notna()].copy()
    df = df.sort_values(["player","season","game_date"]).reset_index(drop=True)
    df["month"] = df["game_date"].dt.month
    features = pd.DataFrame([build_features(row) for row in df.to_dict("records")], index=df.index)
    for name in features.columns:
        df[name] = features[name]
    return df.sort_values(["game_date","player"]).reset_index(drop=True)

def train_prop_model(df, target):
    feats = [f for f in BASE_FEATURES + ["def_adj","usage_pace"] if f in df.columns and f != target]
    sub   = df[df["roll5_pts"].notna()].copy()
    if sub.empty or len(sub) < 20:
        return None
    sub = sub[sub[target].notna()].copy()
    if len(sub) < 20:
        return None
    X = sub[feats].apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(sub[target], errors="coerce")
    valid = y.notna()
    X, y = X.loc[valid], y.loc[valid]
    if len(y) < 20:
        return None
    tscv = TimeSeriesSplit(n_splits=4)
    dates = sub.loc[valid, "game_date"]
    unique_dates = sorted(dates.unique())
    if len(unique_dates) < 5:
        return None
    folds = [(np.flatnonzero(dates.isin(np.array(unique_dates)[tr])),
              np.flatnonzero(dates.isin(np.array(unique_dates)[te])))
             for tr, te in tscv.split(unique_dates)]
    models = {
        "Ridge": Pipeline([("impute", SimpleImputer(keep_empty_features=True)), ("sc", StandardScaler()), ("m", Ridge(alpha=5.0))]),
        "HistGBR": Pipeline([("impute", SimpleImputer(keep_empty_features=True)), ("m", HistGradientBoostingRegressor(
                    max_iter=180, learning_rate=0.05, max_leaf_nodes=15,
                    min_samples_leaf=30, l2_regularization=1.0, random_state=42))]),
    }
    cv, residuals = {}, {}
    for name, model in models.items():
        maes, residual = [], []
        for tr, te in folds:
            predicted = model.fit(X.iloc[tr], y.iloc[tr]).predict(X.iloc[te])
            maes.append(mean_absolute_error(y.iloc[te], predicted))
            residual.extend(y.iloc[te].to_numpy() - predicted)
        cv[name] = np.mean(maes)
        residuals[name] = float(np.std(residual, ddof=1))
    best_name  = min(cv, key=cv.get)
    best_model = models[best_name]
    best_model.fit(X, y)
    return {"model": best_model, "features": feats, "best_name": best_name,
            "cv_mae": cv[best_name], "mean_stat": float(y.mean()), "std_stat": float(y.std()),
            "residual_std": residuals[best_name], "date_folds": [
                {"train_through": str(dates.iloc[tr].max().date()), "test_from": str(dates.iloc[te].min().date())}
                for tr, te in folds]}

def train_all(data_path=DATA_PATH):
    print("\n═══ NBA PROPS MODEL — TRAINING ═══\n")
    if not os.path.exists(data_path):
        print(f"[WARN] No player logs found at {data_path}")
        print("       Props are fail-closed: no synthetic placeholder predictions will be created.")
        thresholds = {"pts":2.5,"reb":1.5,"ast":1.0,"threes":0.5,"pra":3.5}
        path = os.path.join(MODEL_DIR, "props_models.pkl")
        candidate_bundle = {
                "status": "insufficient_data",
                "models": {},
                "thresholds": thresholds,
                "data_path": data_path,
            }
        publish_model_bundle(candidate_bundle, path, "props")
        print(f"No valid training data; existing champion is retained: {path}")
        return {}

    df = pd.read_csv(data_path, parse_dates=["game_date"])
    df = prepare_data(df)
    print(f"Players: {df['player'].nunique()} | Rows: {len(df):,}\n")

    thresholds = {"pts":2.5,"reb":1.5,"ast":1.0,"threes":0.5,"pra":3.5}
    bundles    = {}

    print(f"{'Prop':<10} {'Model':<8} {'MAE':>8} {'Mean':>8}")
    print("─"*38)
    for target in PROP_TARGETS:
        b = train_prop_model(df, target)
        if b is None:
            print(f"  {target:<8} skipped (insufficient data)")
            continue
        bundles[target] = b
        print(f"  {target:<8} {b['best_name']:<8} {b['cv_mae']:>8.2f} {b['mean_stat']:>8.1f}")

    path = os.path.join(MODEL_DIR, "props_models.pkl")
    trained_through = df["game_date"].max().date().isoformat() if not df.empty else None
    candidate_bundle = {
            "status": "trained" if bundles else "insufficient_data",
            "models": bundles,
            "thresholds": thresholds,
            "trained_rows": int(len(df)),
            "trained_through": trained_through,
            "validation": "expanding_date_groups_fold_imputation",
            "feature_semantics_version": FEATURE_VERSION,
            "feature_policy": "prior_only_shifted_inputs",
        }
    publish_model_bundle(candidate_bundle, path, "props")
    print(f"\nActive model → {path} (see publication report)")
    print(f"   Status: {'trained' if bundles else 'insufficient_data'} | Through: {trained_through}")
    return bundles

def predict_props(player_games, model_path=os.path.join(MODEL_DIR,"props_models.pkl")):
    with open(model_path, "rb") as f:
        bundle = pickle.load(f)
    if bundle.get("status") not in (None, "trained") or not bundle.get("models"):
        return pd.DataFrame()
    models, thresholds = bundle["models"], bundle["thresholds"]
    rows = []
    for pg in player_games:
        row = {
            "minutes": pg.get("proj_minutes",30), "roll5_minutes": pg.get("roll5_minutes",30),
            "usage": pg.get("usage",0.20), "ts_pct": pg.get("ts_pct",0.58),
            "opp_drtg_pos": pg.get("opp_drtg_pos",114), "avg_pace": pg.get("avg_pace",99),
            "team_pace": pg.get("team_pace",99), "rest_days": pg.get("rest_days",2),
            "is_home": int(pg.get("is_home",0)),
            "roll5_pts": pg.get("roll5_pts",18), "roll5_reb": pg.get("roll5_reb",6),
            "roll5_ast": pg.get("roll5_ast",4), "roll5_threes": pg.get("roll5_threes",2),
            "roll5_pra": pg.get("roll5_pra",28), "season_game_num": pg.get("game_num",10),
            "month": pg.get("month",11), "def_adj": 114-pg.get("opp_drtg_pos",114),
            "usage_pace": pg.get("usage",0.20)*pg.get("avg_pace",99)/99.0,
        }
        result = {"player": pg["player"], "team": pg.get("team",""),
                  "opp": pg.get("opp_team",""), "props": {}}
        if bundle.get("feature_semantics_version") == FEATURE_VERSION:
            row["roll10_minutes"] = pg.get("roll10_minutes", row["roll5_minutes"])
            row = build_features(row, projected_minutes=pg.get("proj_minutes"), projected_usage=pg.get("usage"))
        for target in PROP_TARGETS:
            if target not in models: continue
            m     = models[target]
            feats = [f for f in m["features"] if f in row]
            X     = pd.DataFrame([row])[feats]
            pred  = float(m["model"].predict(X)[0])
            line  = pg.get(f"posted_{target}")
            edge  = round(pred-line,1) if line else None
            thresh= thresholds[target]
            signal= ("OVER" if edge and edge>thresh else
                     "UNDER" if edge and edge<-thresh else None)
            result["props"][target] = {"pred": round(pred,1), "line": line,
                                       "edge": edge, "signal": signal}
        rows.append(result)
    return pd.DataFrame(rows)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train","predict"], default="train")
    args = parser.parse_args()
    if args.mode == "train":
        train_all()
