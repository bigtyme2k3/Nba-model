"""
game_model_promotion.py
-----------------------
Champion/challenger promotion gate for NBA spread and totals bundles.

The workflow snapshots the currently committed champion bundles before training.
Candidate bundles are then compared on their own stored walk-forward metrics.

A candidate is publishable only when:
- it is a trained/fresh bundle with prior-only team context policy
- MAE does not regress beyond tolerance versus the committed champion
- spread directional accuracy does not materially regress
- absolute sanity ceilings are satisfied

This gate controls artifact publication only. It is not evidence of betting edge.
"""
from __future__ import annotations

import argparse
import json
import pickle
from datetime import datetime, timezone
from pathlib import Path

SPREAD_MAE_TOLERANCE = 0.03
TOTALS_MAE_TOLERANCE = 0.05
SPREAD_DIRECTIONAL_TOLERANCE = 0.005
ABS_SPREAD_MAE_MAX = 12.5
ABS_TOTALS_MAE_MAX = 17.0
EXPECTED_CONTEXT_POLICY = "BOTH_TEAMS_PRIOR_ONLY_REQUIRED"


def load_bundle(path):
    p=Path(path)
    if not p.exists():
        return None
    with p.open("rb") as fh:
        return pickle.load(fh)


def metric(bundle, model_type, name):
    if not bundle:
        return None
    model_name=bundle.get("model_name")
    results=bundle.get("cv_results") or {}
    rec=results.get(model_name) or {}
    if name=="mae":
        val=rec.get("mae")
        if val is None:
            val=rec.get("mae_mean")
    elif name=="directional":
        val=rec.get("ats_mean")
    else:
        val=None
    try:
        return float(val) if val is not None else None
    except Exception:
        return None


def bundle_summary(bundle, model_type):
    if not bundle:
        return {"present":False}
    return {
        "present":True,
        "status":bundle.get("status"),
        "model_name":bundle.get("model_name"),
        "trained_through":bundle.get("trained_through"),
        "trained_rows":bundle.get("trained_rows"),
        "validation":bundle.get("validation"),
        "team_context_policy":bundle.get("team_context_policy"),
        "feature_names_count":bundle.get("feature_names_count") or len(bundle.get("feature_names") or []),
        "mae":metric(bundle,model_type,"mae"),
        "directional_accuracy":metric(bundle,model_type,"directional") if model_type=="spread" else None,
    }


def evaluate_model(model_type, champion, candidate):
    champ=bundle_summary(champion,model_type)
    cand=bundle_summary(candidate,model_type)
    checks={}

    checks["candidate_present"]=bool(candidate)
    checks["candidate_trained"]=cand.get("status")=="trained"
    checks["prior_context_policy"]=cand.get("team_context_policy")==EXPECTED_CONTEXT_POLICY

    c_mae=cand.get("mae")
    h_mae=champ.get("mae")
    abs_max=ABS_SPREAD_MAE_MAX if model_type=="spread" else ABS_TOTALS_MAE_MAX
    checks["absolute_mae_sane"]=c_mae is not None and c_mae <= abs_max

    tol=SPREAD_MAE_TOLERANCE if model_type=="spread" else TOTALS_MAE_TOLERANCE
    checks["mae_nonregression"]=(
        c_mae is not None and
        (h_mae is None or c_mae <= h_mae + tol)
    )

    if model_type=="spread":
        c_dir=cand.get("directional_accuracy")
        h_dir=champ.get("directional_accuracy")
        checks["directional_nonregression"]=(
            c_dir is not None and
            (h_dir is None or c_dir + SPREAD_DIRECTIONAL_TOLERANCE >= h_dir)
        )

    promotable=all(checks.values())
    return {
        "champion":champ,
        "candidate":cand,
        "checks":checks,
        "promotable":promotable,
        "mae_delta_candidate_minus_champion":(
            round(c_mae-h_mae,5)
            if c_mae is not None and h_mae is not None else None
        ),
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--champion-spread",required=True)
    p.add_argument("--champion-totals",required=True)
    p.add_argument("--candidate-spread",required=True)
    p.add_argument("--candidate-totals",required=True)
    p.add_argument("--out",default="data/tracking/game_model_promotion.json")
    args=p.parse_args()

    report={
        "generated_at_utc":datetime.now(timezone.utc).isoformat(),
        "policy":"CHAMPION_CHALLENGER_WALK_FORWARD_NONREGRESSION",
        "spread":evaluate_model(
            "spread",
            load_bundle(args.champion_spread),
            load_bundle(args.candidate_spread),
        ),
        "totals":evaluate_model(
            "totals",
            load_bundle(args.champion_totals),
            load_bundle(args.candidate_totals),
        ),
    }
    report["promotion_approved"]=(
        report["spread"]["promotable"] and report["totals"]["promotable"]
    )

    out=Path(args.out)
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(report,indent=2))

    if not report["promotion_approved"]:
        raise SystemExit("Game-model promotion gate rejected one or more candidates.")


if __name__=="__main__":
    main()
