"""
spread_model.py
---------------
NBA Spread Model — predicts the margin of victory (home - away)
and evaluates against the closing line.

Two models are built and compared:
  1. Ridge Regression (baseline — interpretable)
  2. Gradient Boosting Regressor (primary — handles non-linearities)

The final output is:
  - A trained model saved to models/spread_model.pkl
  - Feature importance rankings
  - Backtesting results (ATS record, CLV, MAE vs closing spread)

Usage:
    # Train on historical data
    python spread_model.py --mode train --data data/processed/master_all.csv

    # Predict today's games
    python spread_model.py --mode predict
"""

import os
import argparse
import pickle
import warnings
import numpy as np
import pandas as pd
from datetime import datetime

# Sklearn
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore")

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)


# ── Feature Definitions ───────────────────────────────────────────────────────

SPREAD_FEATURES = [
    # Rolling performance (form)
    "home_rolling_5g",          # Home team avg margin, last 5 games
    "away_rolling_5g",          # Away team avg margin, last 5 games
    "home_rolling_10g",         # Home team avg margin, last 10 games
    "away_rolling_10g",         # Away team avg margin, last 10 games
    "rolling_diff_5g",          # Differential (home - away), last 5 games
    "rolling_diff_10g",         # Differential (home - away), last 10 games

    # Season-level efficiency
    "home_net_rtg",             # Home adjusted net rating
    "away_net_rtg",             # Away adjusted net rating
    "net_rtg_diff",             # Net rating differential

    "home_pace",                # Home team pace (possessions per 48 min)
    "away_pace",                # Away team pace
    "avg_pace",                 # Average pace (drives total prediction)

    "home_ortg",                # Home offensive rating
    "away_drtg",                # Away defensive rating (matchup efficiency)
    "away_ortg",                # Away offensive rating
    "home_drtg",                # Home defensive rating

    # Rest and fatigue
    "home_rest_days",           # Days of rest for home team
    "away_rest_days",           # Days of rest for away team
    "rest_diff",                # Rest advantage (home - away)
    "home_back_to_back",        # Home on B2B (0/1)
    "away_back_to_back",        # Away on B2B (0/1)
    "home_three_in_four",       # Home 3-in-4 game fatigue (0/1)
    "away_three_in_four",       # Away 3-in-4 game fatigue (0/1)

    # Travel
    "long_travel",              # Away team crossed conference (0/1)
    "east_to_west",             # East team traveling west (0/1)
    "west_to_east",             # West team traveling east (0/1)

    # Season context
    "season_game_num",          # Game number in season (fatigue accumulates over an 82-game slate)
    "month",                    # Month (Oct-Jun regular season + playoffs)
    "is_playoff",               # Playoff game flag (0/1)
]

# Target variable
TARGET = "actual_spread"  # home_pts - away_pts; positive = home won by X


# ── Data Loading & Feature Engineering ───────────────────────────────────────

def load_and_prepare(path: str) -> pd.DataFrame:
    print(f"Loading data from: {path}")
    df = pd.read_csv(path, parse_dates=["game_date"])
    print(f"  Raw rows: {len(df)}")

    df = df[df[TARGET].notna()].copy()
    print(f"  Rows with results: {len(df)}")

    df = engineer_features(df)

    feature_cols = [f for f in SPREAD_FEATURES if f in df.columns]
    df = df.dropna(subset=feature_cols, thresh=len(feature_cols) // 2)
    print(f"  Rows after feature filtering: {len(df)}")

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add all derived features needed by the model."""

    df["rolling_diff_5g"]  = df.get("home_rolling_5g",  0) - df.get("away_rolling_5g",  0)
    df["rolling_diff_10g"] = df.get("home_rolling_10g", 0) - df.get("away_rolling_10g", 0)

    if "home_net_rtg" in df.columns and "away_net_rtg" in df.columns:
        df["net_rtg_diff"] = df["home_net_rtg"] - df["away_net_rtg"]

    if "home_pace" in df.columns and "away_pace" in df.columns:
        df["avg_pace"] = (df["home_pace"] + df["away_pace"]) / 2

    if "home_rest_days" in df.columns and "away_rest_days" in df.columns:
        df["rest_diff"] = df["home_rest_days"].fillna(2) - df["away_rest_days"].fillna(2)

    if "travel_direction" in df.columns:
        df["east_to_west"] = (df["travel_direction"] == "east_to_west").astype(int)
        df["west_to_east"] = (df["travel_direction"] == "west_to_east").astype(int)
    else:
        df["east_to_west"] = 0
        df["west_to_east"] = 0

    for col in ["home_back_to_back", "away_back_to_back",
                "home_three_in_four", "away_three_in_four", "long_travel"]:
        if col in df.columns:
            df[col] = df[col].astype(int)

    if "game_date" in df.columns and "season" in df.columns:
        df = df.sort_values(["season", "game_date"])
        df["season_game_num"] = df.groupby("season").cumcount() + 1

    if "game_date" in df.columns:
        df["month"] = df["game_date"].dt.month

    # NBA playoffs run April - June.
    if "month" in df.columns:
        df["is_playoff"] = df["month"].isin([4, 5, 6]).astype(int)

    return df


def get_feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return X (features) and y (target) with available columns only."""
    available = [f for f in SPREAD_FEATURES if f in df.columns]
    missing   = [f for f in SPREAD_FEATURES if f not in df.columns]

    if missing:
        print(f"  [INFO] Features not in data (will be added when stats pipeline runs): {missing}")

    X = df[available].fillna(0)
    y = df[TARGET]

    return X, y, available


# ── Models ────────────────────────────────────────────────────────────────────

def build_ridge() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model",  Ridge(alpha=10.0))
    ])


def build_gbr() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model",  GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            min_samples_leaf=10,
            subsample=0.8,
            random_state=42
        ))
    ])


def build_rf() -> Pipeline:
    return Pipeline([
        ("model", RandomForestRegressor(
            n_estimators=200,
            max_depth=6,
            min_samples_leaf=10,
            random_state=42
        ))
    ])


# ── Training & Evaluation ─────────────────────────────────────────────────────

def walk_forward_cv(model, X: pd.DataFrame, y: pd.Series,
                    dates: pd.Series, n_splits: int = 4) -> dict:
    """Time-series aware cross-validation. Trains on past seasons, tests on the next."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    maes, ats_records = [], []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        mae = mean_absolute_error(y_test, preds)
        maes.append(mae)

        correct_side = ((preds > 0) == (y_test > 0)).mean()
        ats_records.append(correct_side)

        print(f"    Fold {fold+1}: MAE={mae:.2f} pts | Directional accuracy={correct_side:.1%}")

    return {"mae_mean": np.mean(maes), "mae_std": np.std(maes),
            "ats_mean": np.mean(ats_records), "ats_std": np.std(ats_records)}


def evaluate_vs_closing_line(preds: np.ndarray, df: pd.DataFrame) -> dict:
    """Compare model predictions against the closing spread line."""
    results = {}

    if "avg_spread_home_line" not in df.columns:
        print("  [SKIP] No closing line data available for CLV analysis.")
        return results

    eval_df = df.copy()
    eval_df["model_pred"]    = preds
    eval_df["closing_line"]  = -eval_df["avg_spread_home_line"]  # Convert to home perspective
    eval_df["actual_spread"] = df[TARGET]

    eval_df["model_edge"]    = eval_df["model_pred"] - eval_df["closing_line"]
    eval_df["home_covered"]  = eval_df["actual_spread"] > eval_df["closing_line"]

    results["overall_ats"] = eval_df["home_covered"].mean()

    strong_plays = eval_df[eval_df["model_edge"].abs() >= 3.0].copy()
    if len(strong_plays) > 0:
        strong_plays["bet_home"] = strong_plays["model_edge"] > 0
        strong_plays["bet_won"]  = (
            (strong_plays["bet_home"] & strong_plays["home_covered"]) |
            (~strong_plays["bet_home"] & ~strong_plays["home_covered"])
        )
        results["strong_plays_n"]   = len(strong_plays)
        results["strong_plays_ats"] = strong_plays["bet_won"].mean()
        results["avg_clv"]          = strong_plays["model_edge"].abs().mean()

    results["model_mae"]   = mean_absolute_error(eval_df["actual_spread"], eval_df["model_pred"])
    results["line_mae"]    = mean_absolute_error(eval_df["actual_spread"], eval_df["closing_line"])
    results["improvement"] = results["line_mae"] - results["model_mae"]

    return results


def feature_importance_report(model, feature_names: list) -> pd.DataFrame:
    """Extract feature importances from the trained model."""
    try:
        gbr = model.named_steps["model"]
        importances = gbr.feature_importances_
    except AttributeError:
        return pd.DataFrame()

    fi = pd.DataFrame({
        "feature":    feature_names,
        "importance": importances
    }).sort_values("importance", ascending=False)

    return fi


# ── Prediction for Upcoming Games ────────────────────────────────────────────

def predict_game(model, feature_names: list,
                 home_team: str, away_team: str,
                 home_stats: dict, away_stats: dict,
                 schedule_context: dict) -> dict:
    """
    Generate a spread prediction for a single upcoming game.

    home_stats / away_stats should contain:
        net_rtg, ortg, drtg, pace, rolling_5g, rolling_10g

    schedule_context should contain:
        home_rest_days, away_rest_days, home_b2b, away_b2b,
        long_travel, east_to_west, west_to_east, season_game_num, month
    """
    row = {f: 0 for f in feature_names}

    row["home_rolling_5g"]   = home_stats.get("rolling_5g", 0)
    row["away_rolling_5g"]   = away_stats.get("rolling_5g", 0)
    row["home_rolling_10g"]  = home_stats.get("rolling_10g", 0)
    row["away_rolling_10g"]  = away_stats.get("rolling_10g", 0)
    row["rolling_diff_5g"]   = row["home_rolling_5g"] - row["away_rolling_5g"]
    row["rolling_diff_10g"]  = row["home_rolling_10g"] - row["away_rolling_10g"]

    row["home_net_rtg"]  = home_stats.get("net_rtg", 0)
    row["away_net_rtg"]  = away_stats.get("net_rtg", 0)
    row["net_rtg_diff"]  = row["home_net_rtg"] - row["away_net_rtg"]
    row["home_ortg"]     = home_stats.get("ortg", 114)
    row["away_ortg"]     = away_stats.get("ortg", 114)
    row["home_drtg"]     = home_stats.get("drtg", 114)
    row["away_drtg"]     = away_stats.get("drtg", 114)
    row["home_pace"]     = home_stats.get("pace", 99)
    row["away_pace"]     = away_stats.get("pace", 99)
    row["avg_pace"]      = (row["home_pace"] + row["away_pace"]) / 2

    row["home_rest_days"]     = schedule_context.get("home_rest_days", 2)
    row["away_rest_days"]     = schedule_context.get("away_rest_days", 2)
    row["rest_diff"]          = row["home_rest_days"] - row["away_rest_days"]
    row["home_back_to_back"]  = int(schedule_context.get("home_b2b", False))
    row["away_back_to_back"]  = int(schedule_context.get("away_b2b", False))
    row["home_three_in_four"] = int(schedule_context.get("home_3in4", False))
    row["away_three_in_four"] = int(schedule_context.get("away_3in4", False))
    row["long_travel"]        = int(schedule_context.get("long_travel", False))
    row["east_to_west"]       = int(schedule_context.get("east_to_west", False))
    row["west_to_east"]       = int(schedule_context.get("west_to_east", False))
    row["season_game_num"]    = schedule_context.get("season_game_num", 20)
    row["month"]              = schedule_context.get("month", 11)
    row["is_playoff"]         = int(schedule_context.get("month", 11) in (4, 5, 6))

    X_pred = pd.DataFrame([row])[feature_names]
    pred_spread = model.predict(X_pred)[0]

    return {
        "home_team":    home_team,
        "away_team":    away_team,
        "pred_spread":  round(pred_spread, 1),   # + = home favored by X
        "favored_team": home_team if pred_spread > 0 else away_team,
        "margin":       abs(round(pred_spread, 1)),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def train_mode(data_path: str):
    print("\n═══ NBA SPREAD MODEL — TRAINING ═══\n")

    df = load_and_prepare(data_path)
    df = df.sort_values("game_date").reset_index(drop=True)

    X, y, feature_names = get_feature_matrix(df)
    print(f"\nFeatures used: {len(feature_names)}")
    print(f"Training samples: {len(X)}")
    print(f"Seasons: {sorted(df['season'].unique())}\n")

    print("── Walk-Forward Cross-Validation ──\n")

    models = {
        "Ridge (Baseline)":      build_ridge(),
        "Gradient Boosting":     build_gbr(),
        "Random Forest":         build_rf(),
    }

    cv_results = {}
    for name, model in models.items():
        print(f"  {name}:")
        results = walk_forward_cv(model, X, y, df["game_date"])
        cv_results[name] = results
        print(f"    → Avg MAE: {results['mae_mean']:.2f} ± {results['mae_std']:.2f} pts")
        print(f"    → Directional accuracy: {results['ats_mean']:.1%}\n")

    best_name = min(cv_results, key=lambda k: cv_results[k]["mae_mean"])
    print(f"✅ Best model: {best_name}")

    print(f"\n── Training final {best_name} on full dataset ──")
    final_model = models[best_name]
    final_model.fit(X, y)

    print("\n── Closing Line Value Analysis ──")
    train_preds = final_model.predict(X)
    clv_results = evaluate_vs_closing_line(train_preds, df)

    if clv_results:
        print(f"  Model MAE vs actual:     {clv_results.get('model_mae', 'N/A'):.2f} pts")
        print(f"  Closing line MAE:        {clv_results.get('line_mae', 'N/A'):.2f} pts")
        print(f"  Model improvement:       {clv_results.get('improvement', 'N/A'):.2f} pts")
        if "strong_plays_ats" in clv_results:
            n   = clv_results['strong_plays_n']
            ats = clv_results['strong_plays_ats']
            clv = clv_results['avg_clv']
            print(f"  Strong plays (≥3pt edge): {n} games | {ats:.1%} ATS | Avg CLV: {clv:.1f} pts")

    print("\n── Feature Importance (Top 15) ──")
    fi = feature_importance_report(final_model, feature_names)
    if not fi.empty:
        print(fi.head(15).to_string(index=False))

        fi_path = os.path.join(MODEL_DIR, "feature_importance_spread.csv")
        fi.to_csv(fi_path, index=False)
        print(f"\n  Saved → {fi_path}")

    model_path = os.path.join(MODEL_DIR, "spread_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump({
            "model":         final_model,
            "feature_names": feature_names,
            "model_name":    best_name,
            "trained_on":    datetime.now().isoformat(),
            "cv_results":    cv_results,
            "clv_results":   clv_results,
        }, f)

    print(f"\n✅ Model saved → {model_path}")

    print("\n═══ TRAINING SUMMARY ═══")
    print(f"  Model type:        {best_name}")
    print(f"  Features used:     {len(feature_names)}")
    print(f"  Training games:    {len(X)}")
    best = cv_results[best_name]
    print(f"  CV MAE:            {best['mae_mean']:.2f} ± {best['mae_std']:.2f} pts")
    print(f"  CV Directional:    {best['ats_mean']:.1%}")
    print("\n  Next step: run predict mode or build the totals model.")


def predict_mode(model_path: str = None):
    """Demo predict mode using placeholder games."""

    path = model_path or os.path.join(MODEL_DIR, "spread_model.pkl")

    if not os.path.exists(path):
        print("[ERROR] No trained model found. Run --mode train first.")
        return

    with open(path, "rb") as f:
        bundle = pickle.load(f)

    model         = bundle["model"]
    feature_names = bundle["feature_names"]

    print(f"\n═══ NBA SPREAD PREDICTIONS ═══")
    print(f"Model: {bundle['model_name']} | Trained: {bundle['trained_on'][:10]}\n")

    sample_games = [
        {
            "home": "Boston Celtics",
            "away": "New York Knicks",
            "home_stats": {"net_rtg": 6.1, "ortg": 118, "drtg": 112, "pace": 98, "rolling_5g": 4.2, "rolling_10g": 3.0},
            "away_stats": {"net_rtg": 2.5, "ortg": 115, "drtg": 112, "pace": 97, "rolling_5g": -1.0, "rolling_10g": 1.8},
            "context": {"home_rest_days": 1, "away_rest_days": 1, "home_b2b": False, "away_b2b": False,
                        "long_travel": False, "season_game_num": 10, "month": 11},
        },
        {
            "home": "Utah Jazz",
            "away": "Denver Nuggets",
            "home_stats": {"net_rtg": -5.0, "ortg": 110, "drtg": 115, "pace": 100, "rolling_5g": -3.0, "rolling_10g": -2.0},
            "away_stats": {"net_rtg": 7.5, "ortg": 119, "drtg": 111, "pace": 98, "rolling_5g": 6.0, "rolling_10g": 5.2},
            "context": {"home_rest_days": 2, "away_rest_days": 1, "home_b2b": False, "away_b2b": True,
                        "long_travel": True, "west_to_east": False, "season_game_num": 10, "month": 11},
        },
    ]

    print(f"{'Game':<40} {'Pred Spread':>12} {'Favored':>24} {'By':>6}")
    print("─" * 86)

    for g in sample_games:
        result = predict_game(
            model, feature_names,
            g["home"], g["away"],
            g["home_stats"], g["away_stats"], g["context"]
        )
        game_str = f"{g['away']} @ {g['home']}"
        print(f"{game_str:<40} {result['pred_spread']:>+12.1f} {result['favored_team']:>24} {result['margin']:>5.1f}")

    print("\nNote: Positive spread = home team favored by that margin.")
    print("These predictions require the stats pipeline to be populated with current season data.")


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NBA Spread Model")
    parser.add_argument("--mode", choices=["train", "predict"], default="train")
    parser.add_argument("--data", type=str, default="data/processed/master_all.csv")
    parser.add_argument("--model", type=str, default=None, help="Path to saved model (predict mode)")
    args = parser.parse_args()

    if args.mode == "train":
        train_mode(args.data)
    else:
        predict_mode(args.model)
