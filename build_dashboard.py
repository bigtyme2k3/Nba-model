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
from datetime import date
from typing import Any, Dict, List
import pandas as pd

PREDICTIONS_DIR = 'predictions'
RAW_DIR = 'data/raw'
DASH_DIR = 'data/dashboard'
TRACKING_JSON = 'data/tracking/model_tracking.json'
OUTPUT_HTML = 'docs/index.html'


def target_today() -> str:
    return os.environ.get('TARGET') or os.environ.get('NBA_TARGET_DATE') or str(date.today())


def load_json(path: str, default: Any):
    try:
        if os.path.exists(path):
            return json.load(open(path, encoding='utf-8'))
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
            'data_health': {'odds': 'missing', 'props': 'missing', 'line_shopping': 'missing',
                             'player_points': 'missing', 'spreads_found': 0, 'totals_found': 0,
                             'props_found': 0, 'player_points_found': 0, 'line_shopping_rows': 0,
                             'games': 0, 'actionable_bets': 0, 'high_bets': 0, 'last_updated_utc': None},
            'model_stats': {'spread': {'algo': 'Ridge / GBR', 'n': 0},
                             'totals': {'algo': 'Ridge / RF', 'n': 0},
                             'props': {'algo': 'Ridge / GBR', 'n': 0}}}


def find_predictions() -> Dict[str, Any]:
    target = target_today()
    candidates = [os.path.join(PREDICTIONS_DIR, f'predictions_{target}.json')] + \
        sorted(glob.glob(os.path.join(PREDICTIONS_DIR, 'predictions_*.json')), reverse=True)
    seen = set()
    for path in candidates:
        if path in seen:
            continue
        seen.add(path)
        if os.path.exists(path):
            data = json.load(open(path, encoding='utf-8'))
            print(f"  Loaded: {path} ({len(data.get('games', []))} games)")
            return data
    print('  [WARN] No predictions file found — using empty data')
    return empty_dashboard_data()


def csv_value(value: Any) -> Any:
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return value


def load_csv_records(paths: List[str]) -> List[Dict[str, Any]]:
    for path in paths:
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                if len(df) > 0:
                    print(f'  Loaded CSV: {path} ({len(df)} rows)')
                    return [{k: csv_value(v) for k, v in row.to_dict().items()} for _, row in df.iterrows()]
                print(f'  Header-only CSV skipped: {path}')
            except Exception as exc:
                print(f'  [WARN] Could not load {path}: {exc}')
    return []


def load_player_points_for_date(target_date: str) -> List[Dict[str, Any]]:
    return load_csv_records([os.path.join(RAW_DIR, f'player_points_{target_date}.csv'),
                              os.path.join(RAW_DIR, 'player_points_today.csv')])


def load_line_shopping_for_date(target_date: str) -> List[Dict[str, Any]]:
    return load_csv_records([os.path.join(RAW_DIR, f'line_shopping_best_{target_date}.csv'),
                              os.path.join(RAW_DIR, f'line_shopping_{target_date}.csv'),
                              os.path.join(RAW_DIR, 'line_shopping_best_today.csv'),
                              os.path.join(RAW_DIR, 'line_shopping_today.csv')])


def load_tracking() -> Dict[str, Any]:
    return load_json(TRACKING_JSON, empty_tracking())


def enrich_data(data: Dict[str, Any]) -> Dict[str, Any]:
    target_date = target_today()
    data['date'] = target_date

    points = data.get('props') or data.get('player_points') or load_player_points_for_date(target_date)
    if points:
        data['props'] = points
        data['player_points'] = points
        data['props_board'] = data.get('props_board') or points
        h = data.setdefault('data_health', {})
        h['player_points'] = 'loaded'
        h['props'] = 'loaded'
        h['player_points_found'] = len(points)
        h['props_found'] = max(int(h.get('props_found', 0) or 0), len(points))
    else:
        data.setdefault('props', [])
        data.setdefault('player_points', [])
        data.setdefault('props_board', [])

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

    tracking = data.get('tracking') or data.get('model_tracking') or load_tracking()
    data['tracking'] = tracking
    data['model_tracking'] = tracking
    return data


def fallback_html(data_json: str) -> str:
    return f"""<!DOCTYPE html><html lang='en'><head><meta charset='UTF-8'><meta name='viewport' content='width=device-width,initial-scale=1.0'><title>NBA Betting Model</title>
<style>
:root{{--bg:#070a12;--panel:#0d1220;--border:#ffffff14;--text:#e2e8f0;--muted:#94a3b8;--accent:#00e5a0;--bad:#f87171;--warn:#fbbf24}}
*{{box-sizing:border-box}}
body{{background:var(--bg);color:var(--text);font-family:'Courier New',ui-monospace,monospace;margin:0}}
.app{{max-width:1200px;margin:0 auto;padding:20px}}
.title{{font-size:30px;font-weight:900;letter-spacing:-0.5px}}
.title span{{color:var(--accent)}}
.sub{{color:var(--muted);margin-top:4px}}
.tabs{{display:flex;gap:8px;margin:20px 0;flex-wrap:wrap}}
button{{background:var(--panel);color:var(--text);border:1px solid var(--border);border-radius:12px;padding:10px 16px;font-family:inherit;font-weight:800;cursor:pointer}}
button.active{{border-color:var(--accent);color:var(--accent)}}
.grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}}
.card,.panel{{background:var(--panel);border:1px solid var(--border);border-radius:18px;padding:16px}}
.card .label{{color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:0.06em}}
.value{{font-size:26px;font-weight:900;color:var(--accent);margin-top:4px}}
.bad{{color:var(--bad)}}
.row{{background:var(--panel);border:1px solid var(--border);border-radius:14px;padding:14px;margin-bottom:10px}}
.row b{{font-size:15px}}
.pill{{display:inline-block;padding:2px 8px;border-radius:999px;font-size:11px;font-weight:800;border:1px solid var(--border);margin-left:6px}}
.meta{{color:var(--muted);font-size:13px;margin-top:6px}}
@media(max-width:800px){{.grid{{grid-template-columns:1fr 1fr}}}}
</style>
<script id='dashboard-data'>const DATA = {data_json}; window.DATA = DATA;</script>
</head><body>
<div class='app'>
  <div class='title'>🏀 NBA <span>Betting Model</span></div>
  <div class='sub'>Daily Report · <span id='date'></span></div>
  <div class='grid' id='health'></div>
  <div class='tabs' id='tabs'>
    <button onclick="show('games')">Games</button>
    <button onclick="show('props')">Props</button>
    <button onclick="show('bets')">Best Bets</button>
    <button onclick="show('tracking')">Model Tracking</button>
  </div>
  <div id='view' class='panel'></div>
</div>
<script>
const safe=(v,d='—')=>v===null||v===undefined||v===''?d:v;
function health(){{
  const h=DATA.data_health||{{}};
  document.getElementById('date').textContent=DATA.date||'';
  document.getElementById('health').innerHTML=[
    ['Odds',h.odds||'missing'],
    ['Props',h.player_points||h.props||'missing'],
    ['Games',h.games||0],
    ['Actionable',h.actionable_bets||0],
  ].map(x=>`<div class='card'><div class='label'>${{x[0]}}</div><div class='value ${{String(x[1]).includes('missing')?'bad':''}}'>${{x[1]}}</div></div>`).join('');
}}
function show(tab){{
  [...document.querySelectorAll('#tabs button')].forEach(b=>b.classList.remove('active'));
  const idx={{games:0,props:1,bets:2,tracking:3}}[tab];
  document.querySelectorAll('#tabs button')[idx].classList.add('active');
  let out='';
  if(tab==='games') out=(DATA.games||[]).map(g=>`<div class='row'><b>${{safe(g.away?.name||g.away_team)}} @ ${{safe(g.home?.name||g.home_team)}}</b><span class='pill'>${{safe(g.spread?.model_line)}}</span><span class='pill'>${{safe(g.totals?.play)}} ${{safe(g.totals?.line)}}</span><div class='meta'>Tip ${{safe(g.tip)}}</div></div>`).join('')||'No games today.';
  if(tab==='props') out=(DATA.props||DATA.player_points||[]).slice(0,80).map(p=>`<div class='row'><b>${{safe(p.player)}}</b> ${{safe(p.stat)}} ${{safe(p.signal)}}<div class='meta'>Line ${{safe(p.line)}} · Projection ${{safe(p.pred)}}</div></div>`).join('')||'No props loaded yet.';
  if(tab==='bets') out=(DATA.best_bets||[]).map(b=>`<div class='row'><b>${{safe(b.play)}}</b><span class='pill'>${{'★'.repeat(b.stars||1)}}</span><div class='meta'>${{safe(b.game)}} · edge ${{safe(b.edge)}} · ${{safe(b.units)}}u</div></div>`).join('')||'No best bets yet — run the pipeline once odds/props are loaded.';
  if(tab==='tracking') out=`<pre style="white-space:pre-wrap">${{JSON.stringify(DATA.tracking||DATA.model_tracking||{{}},null,2)}}</pre>`;
  document.getElementById('view').innerHTML=out;
}}
health();show('games');
</script>
</body></html>"""


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


def build_html(data: Dict[str, Any]) -> bool:
    data = enrich_data(data)
    data_json = json.dumps(data, separators=(',', ':'), ensure_ascii=False)
    os.makedirs(os.path.dirname(OUTPUT_HTML), exist_ok=True)
    if os.path.exists(OUTPUT_HTML):
        html = open(OUTPUT_HTML, encoding='utf-8').read()
        html = inject_data_script(html, data_json)
    else:
        html = fallback_html(data_json)
    open(OUTPUT_HTML, 'w', encoding='utf-8').write(html)
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
    print(f"     Props: {len(data.get('props', []))}")
    print(f"     Odds: {h.get('odds', 'unknown')}")


if __name__ == '__main__':
    main()
