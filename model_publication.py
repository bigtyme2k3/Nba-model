"""Guard every training entry point before replacing a champion artifact."""
from __future__ import annotations
import hashlib
import json
import math
import pickle
from datetime import date, datetime, timezone
from pathlib import Path


def artifact_id(path):
    path = Path(path)
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def model_family_id(path):
    """Stable across refits; changes with algorithm, inputs, or feature semantics."""
    path = Path(path)
    if not path.exists():
        return None
    try:
        with path.open("rb") as handle:
            bundle = pickle.load(handle)
    except Exception:
        return None
    signature = {key: bundle.get(key) for key in ("feature_semantics_version", "validation", "model_name", "feature_names")}
    if bundle.get("models"):
        signature["props"] = {stat: {"algorithm": model.get("best_name"), "features": model.get("features")}
                              for stat, model in sorted(bundle["models"].items())}
    return hashlib.sha256(json.dumps(signature, sort_keys=True, default=str).encode()).hexdigest()


def evaluate(kind, champion, candidate):
    if kind in {"spread", "totals"}:
        from game_model_promotion import evaluate_model
        report = evaluate_model(kind, champion, candidate)
        checks = report["checks"]
    else:
        from prop_features import VERSION
        checks = {"trained": candidate.get("status") == "trained",
                  "feature_semantics": candidate.get("feature_semantics_version") == VERSION,
                  "date_grouped_validation": candidate.get("validation") == "expanding_date_groups_fold_imputation"}
        limits = {"pts": 6.5, "reb": 3.0, "ast": 2.3, "threes": 1.3, "pra": 8.0}
        for stat, ceiling in limits.items():
            metric = (candidate.get("models", {}).get(stat) or {}).get("cv_mae")
            old = ((champion or {}).get("models", {}).get(stat) or {}).get("cv_mae")
            checks[f"{stat}_sane_nonregression"] = (metric is not None and math.isfinite(metric)
                and metric <= ceiling and (old is None or metric <= old + .03))
        report = {"checks": checks}
    through = candidate.get("trained_through")
    old_through = (champion or {}).get("trained_through")
    try:
        trained = date.fromisoformat(through)
        checks["valid_training_date"] = trained <= date.today() and (not old_through or through >= old_through)
    except (TypeError, ValueError):
        checks["valid_training_date"] = False
    report["promotable"] = all(checks.values())
    return report


def publish_model_bundle(candidate, path, kind):
    path = Path(path)
    champion = None
    if path.exists():
        # An unreadable champion requires repair, never an unreviewed replacement.
        with path.open("rb") as handle:
            champion = pickle.load(handle)
    report = evaluate(kind, champion, candidate)
    report.update(model=kind, generated_at_utc=datetime.now(timezone.utc).isoformat(),
                  prior_artifact_id=artifact_id(path), trained_through=candidate.get("trained_through"))
    if report["promotable"]:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(".pkl.tmp")
        with temp.open("wb") as handle:
            pickle.dump(candidate, handle)
        temp.replace(path)
    report["active_artifact_id"] = artifact_id(path)
    out = Path("data/tracking/model_publication") / f"{kind}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Model publication {kind}: {'APPROVED' if report['promotable'] else 'REJECTED; champion retained'}")
    return report["promotable"]
