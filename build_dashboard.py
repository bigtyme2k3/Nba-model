"""
build_dashboard.py
------------------
Reads the target predictions JSON and bakes it into docs/index.html.

If docs/index.html doesn't exist yet, a self-contained dashboard page is
generated from scratch (see fallback_html below) — nothing else needs to
be committed before the first pipeline run.
"""
from __future__ import annotations
import glob, json, os, re
from dashboard_data import prepare_display
from html import escape
from pathlib import Path
from datetime import date
from active_slate_date import resolve_target_date
from typing import Any, Dict, List
import pandas as pd

PREDICTIONS_DIR = 'predictions'
RAW_DIR = 'data/raw'
DASH_DIR = 'data/dashboard'
TRACKING_JSON = 'data/tracking/model_tracking.json'
READINESS_JSON = 'data/tracking/nba_v1_readiness.json'
FORWARD_PERF_JSON = 'data/tracking/nba_forward_performance.json'
EXECUTION_PERF_JSON = 'data/tracking/nba_execution_performance.json'
CLV_REPORT_JSON = 'data/tracking/nba_prop_clv_report.json'
OUTPUT_HTML = 'docs/index.html'


def target_today() -> str:
    return os.environ.get('TARGET') or os.environ.get('NBA_TARGET_DATE') or resolve_target_date()


def load_json(path: str, default: Any):
    try:
        if os.path.exists(path):
            with open(path, encoding='utf-8') as handle:
                return json.load(handle)
    except Exception as exc:
        print(f'  [WARN] Could not read {path}: {exc}')
    return default


def empty_tracking() -> Dict[str, Any]:
    return {'overall': '0-0-0', 'wins': 0, 'losses': 0, 'pushes': 0, 'win_pct': 0, 'roi': 0,
            'profit_units': 0, 'clv_avg': 0, 'by_type': {}, 'by_conf': {}, 'recent_10': []}


def empty_dashboard_data() -> Dict[str, Any]:
    today = target_today()
    return {'date': today, 'generated': None, 'games': [], 'best_bets': [], 'props': [],
            'player_points': [], 'props_board': [], 'line_shopping': [],
            'tracking': empty_tracking(), 'model_tracking': empty_tracking(),
            'readiness': {}, 'forward_performance': {}, 'execution_performance': {},
            'clv_report': {}, 'evidence': {},
            'data_health': {'odds': 'missing', 'props': 'missing', 'line_shopping': 'missing',
                             'player_points': 'missing', 'spreads_found': 0, 'totals_found': 0,
                             'props_found': 0, 'player_points_found': 0, 'line_shopping_rows': 0,
                             'games': 0, 'actionable_bets': 0, 'high_bets': 0, 'last_updated_utc': None},
            'model_stats': {'spread': {'algo': 'Ridge / GBR', 'n': 0},
                             'totals': {'algo': 'Ridge / RF', 'n': 0},
                             'props': {'algo': 'Ridge / GBR', 'n': 0}}}


def find_predictions() -> Dict[str, Any]:
    target = target_today()
    paths = sorted(glob.glob(os.path.join(PREDICTIONS_DIR, 'predictions_*.json')), reverse=True)
    path = os.path.join(PREDICTIONS_DIR, f'predictions_{target}.json')
    data = load_json(path, {})
    if data.get('date') == target:
        data['slate_status'] = 'CURRENT'
        return data
    empty = empty_dashboard_data()
    empty['slate_status'] = 'AWAITING_PREDICTIONS'
    for previous in paths:
        old = load_json(previous, {})
        if old.get('generated') and str(old.get('date', '')) < target:
            empty['last_successful_prediction_at'] = old['generated']
            empty['last_successful_slate_date'] = old.get('date')
            break
    print(f'  [WARN] Awaiting predictions for {target}')
    return empty


def csv_value(value: Any) -> Any:
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return value


def load_csv_records(paths: List[str], target_date=None) -> List[Dict[str, Any]]:
    for path in paths:
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                if target_date:
                    date_col = next((c for c in ('game_date', 'date', 'target_date') if c in df.columns), None)
                    if not date_col:
                        continue
                    df = df[df[date_col].astype(str).eq(str(target_date))]
                if len(df) > 0:
                    print(f'  Loaded CSV: {path} ({len(df)} rows)')
                    return [{k: csv_value(v) for k, v in row.to_dict().items()} for _, row in df.iterrows()]
                print(f'  Header-only CSV skipped: {path}')
            except Exception as exc:
                print(f'  [WARN] Could not load {path}: {exc}')
    return []


def load_player_points_for_date(target_date: str) -> List[Dict[str, Any]]:
    return load_csv_records([os.path.join(RAW_DIR, f'player_points_{target_date}.csv'),
                              os.path.join(RAW_DIR, 'player_points_today.csv')], target_date)


def load_line_shopping_for_date(target_date: str) -> List[Dict[str, Any]]:
    return load_csv_records([os.path.join(RAW_DIR, f'line_shopping_best_{target_date}.csv'),
                              os.path.join(RAW_DIR, f'line_shopping_{target_date}.csv'),
                              os.path.join(RAW_DIR, 'line_shopping_best_today.csv'),
                              os.path.join(RAW_DIR, 'line_shopping_today.csv')], target_date)


def load_tracking() -> Dict[str, Any]:
    return load_json(TRACKING_JSON, empty_tracking())


def enrich_data(data: Dict[str, Any]) -> Dict[str, Any]:
    target_date = target_today()
    if data.get('date') != target_date:
        data = empty_dashboard_data()
        data['slate_status'] = 'AWAITING_PREDICTIONS'

    line_shopping = data.get('line_shopping') or load_line_shopping_for_date(target_date)
    data['line_shopping'] = line_shopping
    if line_shopping:
        h = data.setdefault('data_health', {})
        h['line_shopping'] = 'loaded'
        h['line_shopping_rows'] = max(int(h.get('line_shopping_rows', 0) or 0), len(line_shopping))

    h = data.setdefault('data_health', {})
    spread_rows = sum(1 for g in data.get('games', []) if g.get('spread', {}).get('posted_line') is not None)
    total_rows = sum(1 for g in data.get('games', []) if g.get('totals', {}).get('line') is not None)
    h['spreads_found'] = max(int(h.get('spreads_found', 0) or 0), spread_rows)
    h['totals_found'] = max(int(h.get('totals_found', 0) or 0), total_rows)
    h['games'] = len(data.get('games', []))
    h['actionable_bets'] = len(data.get('best_bets', []))
    h['high_bets'] = sum(1 for b in data.get('best_bets', []) if b.get('stars', 0) >= 3)
    h['odds'] = 'loaded' if (spread_rows or total_rows) else h.get('odds', 'missing')

    legacy_tracking = data.get('tracking') or data.get('model_tracking') or load_tracking()

    readiness = load_json(READINESS_JSON, {})
    forward_perf = load_json(FORWARD_PERF_JSON, {})
    same_epoch = (
        bool(readiness.get('evidence_epoch'))
        and readiness.get('evidence_epoch') == forward_perf.get('evidence_epoch')
    )
    tracking = forward_perf if same_epoch else legacy_tracking
    data['tracking'] = tracking
    data['model_tracking'] = tracking
    execution_perf = load_json(EXECUTION_PERF_JSON, {})
    clv_report = load_json(CLV_REPORT_JSON, {})
    data['reconciliation'] = load_json('data/tracking/reconciliation_status.json', {})
    h['grading'] = data['reconciliation'].get('status', 'UNKNOWN')
    data['readiness'] = readiness
    data['forward_performance'] = forward_perf
    data['execution_performance'] = execution_perf
    data['clv_report'] = clv_report
    data['evidence'] = {
        'status': readiness.get('status', 'FOUNDATION_NOT_READY'),
        'evidence_epoch': readiness.get('evidence_epoch'),
        'production_ready': bool(readiness.get('production_ready', False)),
        'forward_bets': (readiness.get('forward_evidence') or {}).get('bet_rows', 0),
        'forward_roi': (readiness.get('forward_evidence') or {}).get('roi'),
        'brier': (readiness.get('forward_evidence') or {}).get('brier'),
        'matched_closes': (readiness.get('forward_evidence') or {}).get('clv_matched', 0),
        'beat_close_rate': (readiness.get('forward_evidence') or {}).get('beat_close_rate'),
        'avg_line_clv': (readiness.get('forward_evidence') or {}).get('avg_line_clv'),
        'execution_bets': execution_perf.get('bet_rows', 0),
    }
    return prepare_display(data)


def fallback_html(data_json: str) -> str:
    """Render the maintained template on every run so layout upgrades persist."""
    source = Path(__file__).resolve().parent / "dashboard"
    html = (source / "index.html").read_text(encoding="utf-8")
    return (html.replace("__DASHBOARD_CSS__", (source / "style.css").read_text(encoding="utf-8"))
            .replace("__LINEMATE_REVIEW_JS__", (source / "linemate_review.js").read_text(encoding="utf-8"))
            .replace("__DASHBOARD_JS__", (source / "app.js").read_text(encoding="utf-8"))
            .replace("__DASHBOARD_DATA__", data_json))


def inject_data_script(html: str, data_json: str) -> str:
    html = re.sub(r"<script id=[\"']dashboard-data[\"']>.*?</script>\s*", '', html, flags=re.DOTALL)
    replacement = f"const DATA = {data_json}; window.DATA = DATA;\n"
    for pattern in [r"const\s+DATA\s*=\s*.*?;\s*(?=\n\s*const|\n\s*let|\n\s*function|\n\s*window\.|</script>)",
                     r"window\.DATA\s*=\s*.*?;\s*(?=\n|</script>)"]:
        new_html, n = re.subn(pattern, replacement, html, count=1, flags=re.DOTALL)
        if n:
            return new_html
    block = f"<script id=\"dashboard-data\">const DATA = {data_json}; window.DATA = DATA;</script>\n"
    if '</head>' in html:
        return html.replace('</head>', block + '</head>', 1)
    if '<body' in html:
        return re.sub(r'(<body[^>]*>)', r'\1\n' + block, html, count=1, flags=re.IGNORECASE)
    return block + html


def inject_research_link(html: str) -> str:
    """A separate artifact link; research never participates in NBA V1 bet decisions."""
    if 'id="historical-market-research-link"' in html:
        return html
    link = '<aside id="historical-market-research-link" style="padding:10px 20px;text-align:right"><a href="market-research/index.html" style="color:#93c5fd">Historical market/fade research · No live recommended bets</a></aside>'
    return re.sub(r'(<body[^>]*>)', lambda match: match.group(1) + link, html, count=1, flags=re.IGNORECASE)


def build_html(data: Dict[str, Any]) -> bool:
    data = enrich_data(data)
    data_json = json.dumps(data, separators=(',', ':'), ensure_ascii=False, allow_nan=False)
    # Market/player text must not close the data script element.
    data_json = data_json.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    os.makedirs(os.path.dirname(OUTPUT_HTML), exist_ok=True)
    html = fallback_html(data_json)
    Path(OUTPUT_HTML).write_text(html, encoding='utf-8')
    return True


def main() -> None:
    print('\n═══ Building Dashboard ═══\n')
    data = find_predictions()
    build_html(data)
    h = data.get('data_health', {})
    print(f"  ✅ {OUTPUT_HTML} updated")
    print(f"     Target env: {target_today()}")
    print(f"     Date: {data.get('date')}")
    print(f"     Games: {len(data.get('games', []))}")
    print(f"     Best bets: {len(data.get('best_bets', []))}")
    print(f"     Props: {sum(len(p.get('props') or {}) for g in data.get('games', []) for p in g.get('props', []))}")
    print(f"     Odds: {h.get('odds', 'unknown')}")


if __name__ == '__main__':
    main()
