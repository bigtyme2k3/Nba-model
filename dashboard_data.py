"""Prepare dated display records without changing prediction or betting policy."""
from __future__ import annotations

import json
import math
from copy import deepcopy
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from game_markets import BOOKS, priced_evidence, valid_price
from teams import normalize_team
from prediction_diagnostics import evaluate_day, evaluate_history


def num(value):
    try:
        val = float(value)
        return val if math.isfinite(val) else None
    except (ValueError, TypeError):
        return None


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def norm(value):
    return ' '.join(str(value or '').replace('’', "'").lower().split())


def matchup(home, away):
    return f'{normalize_team(str(away))} @ {normalize_team(str(home))}'


def records(path):
    try:
        return clean(pd.read_csv(path, low_memory=False).to_dict('records'))
    except (OSError, ValueError, pd.errors.EmptyDataError):
        return []


def dated_records(paths, target):
    for path in paths:
        rows = [r for r in records(path) if str(r.get('game_date') or r.get('date')) == str(target)]
        if rows:
            return rows
    return []


def load_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return {}


def quote_index(target, raw_dir):
    raw_dir = Path(raw_dir)
    rows = dated_records([raw_dir / f'sportsbook_props_{target}.csv', raw_dir / 'sportsbook_props_today.csv'], target)
    index = defaultdict(list)
    for row in rows:
        if row.get('book_key') not in BOOKS or row.get('side') not in {'OVER', 'UNDER'}:
            continue
        if num(row.get('line')) is None or not valid_price(row.get('price')):
            continue
        key = (norm(matchup(row.get('home_team'), row.get('away_team'))), norm(row.get('player')), str(row.get('stat')).upper())
        index[key].append({'book': row.get('book'), 'book_key': row['book_key'],
                           'side': row['side'], 'line': num(row['line']), 'odds': int(row['price']),
                           'updated_at': row.get('last_update'), 'captured_at': row.get('scraped_at')})
    return index, rows


def flatten_props(data, quotes=None, profiles=None):
    quotes, profiles = quotes or {}, profiles or {}
    final = {(norm(b.get('game')), norm(b.get('player')), str(b.get('stat')).upper()): b
             for b in data.get('best_bets', []) if b.get('type') == 'PROP'}
    out = []
    for game in data.get('games', []):
        game_name = matchup(game.get('home', {}).get('name'), game.get('away', {}).get('name'))
        for player in game.get('props', []):
            name = player.get('player')
            profile = profiles.get(norm(name), {})
            for stat, prop in (player.get('props') or {}).items():
                stat = stat.upper()
                key = (norm(game_name), norm(name), stat)
                prediction, line = num(prop.get('pred')), num(prop.get('line'))
                if prediction is None:
                    continue
                lean = ('OVER' if prediction > line else 'UNDER' if prediction < line else None) if line is not None else None
                meta = final.get(key)
                eligible = bool(meta and prop.get('signal') and not prop.get('availability_block'))
                side = prop.get('signal') if eligible else lean
                offers = quotes.get(key, [])
                matches = [q for q in offers if q['line'] == line and q['side'] == side]
                # A research row's selected price must belong to the displayed side/line.
                shown = max(matches, key=lambda q: q['odds']) if matches else None
                if eligible:
                    shown = {'book': prop.get('book'), 'book_key': prop.get('book_key'),
                             'line': line, 'side': side, 'odds': prop.get('odds')}
                def recent(window):
                    if stat == 'PRA':
                        parts = [num(profile.get(f'roll{window}_{s}')) for s in ('pts', 'reb', 'ast')]
                        return sum(parts) if all(x is not None for x in parts) else None
                    return num(profile.get(f'roll{window}_{stat.lower()}'))
                out.append({'player': name, 'team': player.get('team'), 'opp': player.get('opp'),
                            'game': game_name, 'tip': game.get('tip'), 'stat': stat,
                            'projection': prediction, 'line': line, 'gap': prediction-line if line is not None else None,
                            'direction': side, 'decision': 'BET' if eligible else 'RESEARCH',
                            'quote': shown, 'offers': offers,
                            'minutes': num(player.get('proj_min')), 'base_minutes': num(player.get('base_proj_min')),
                            'injury_status': player.get('injury_status'), 'availability_block': prop.get('availability_block', False),
                            'block_reason': prop.get('block_reason'),
                            'recent_minutes': num(player.get('recent_minutes')),
                            'minutes_gap': num(player.get('minutes_gap')),
                            'minutes_guard': bool(player.get('minutes_guard', False)),
                            'shadow_stabilized_minutes': num(player.get('shadow_stabilized_minutes')),
                            'rotation_risk': player.get('rotation_risk_band'), 'role': player.get('projected_role'),
                            'minutes_delta': num(player.get('opportunity_minutes_delta')),
                            'vacancy_sources': player.get('vacancy_sources'),
                            'last5': recent(5), 'last10': recent(10),
                            'model_probability': num(prop.get('model_probability')) if eligible else None,
                            'ev': num(prop.get('ev')) if eligible else None,
                            'units': num(meta.get('units')) if eligible else None})
    return sorted(out, key=lambda r: (str(r['tip']), str(r['player']), r['stat']))


def prepare_display(data, raw_dir='data/raw', tracking_dir='data/tracking', processed_dir='data/processed', predictions_dir='predictions'):
    data = deepcopy(data)
    target = data['date']
    index, raw_quotes = quote_index(target, raw_dir)
    profiles = {norm(row.get('player')): row for row in records(Path(processed_dir) / 'player_profiles_latest.csv')
                if row.get('last_game_date') and str(row['last_game_date']) < target}
    props = flatten_props(data, index, profiles)
    data['props'] = props
    data['player_points'] = data['props_board'] = []
    h = data.setdefault('data_health', {})
    players = {row['player'] for row in props}
    h.update(props='loaded' if props else 'missing', player_points='loaded' if props else 'missing',
             props_found=len(props), player_points_found=len(players), modeled_players=len(players),
             quote_rows=sum(map(len, index.values())), priced_players=len({r.get('player') for r in raw_quotes}),
             games=len(data.get('games', [])), actionable_bets=len(data.get('best_bets', [])))
    status = load_json(Path(raw_dir) / 'sportsbook_props_status.json')
    data['props_feed'] = status if status.get('target_date') == target else {}
    calibration = load_json('models/edge_calibration.json')
    data['calibration'] = calibration

    game_odds = dated_records([Path(raw_dir) / f'odds_{target}.csv', Path(raw_dir) / 'odds_today.csv'], target)
    game_lookup = {norm(matchup(r.get('home_team'), r.get('away_team'))): r for r in game_odds}
    for game in data.get('games', []):
        name = matchup(game.get('home', {}).get('name'), game.get('away', {}).get('name'))
        raw = game_lookup.get(norm(name), {})
        try:
            offers = json.loads(raw.get('game_offers') or '[]')
        except (TypeError, ValueError):
            offers = []
        game['market_quotes'] = [q for q in offers if q.get('book_key') in BOOKS and valid_price(q.get('price'))]
        game['display_name'] = name
        game['priced_at'] = raw.get('scraped_at')
        game.pop('props', None)

    track = Path(tracking_dir)
    result_rows = records(track / 'nba_forward_certified.csv')
    epoch = (data.get('forward_performance') or {}).get('evidence_epoch')
    eligible = [r for r in result_rows if epoch and r.get('decision') == 'BET' and priced_evidence(r)
                and r.get('outcome') in {'WIN', 'LOSS', 'PUSH'} and r.get('evidence_epoch') == epoch]
    keep = ('prediction_id', 'date', 'game', 'market', 'player', 'stat', 'side', 'line', 'odds',
            'sportsbook', 'sportsbook_key', 'projection', 'actual', 'outcome', 'decision')
    data['results'] = [{**{k: r.get(k) for k in keep}, 'profit_units': (
        (-1 if r['outcome'] == 'LOSS' else 0 if r['outcome'] == 'PUSH' else
         float(r['odds'])/100 if float(r['odds']) > 0 else 100/abs(float(r['odds']))))}
        for r in sorted(eligible, key=lambda r: (str(r.get('date')), str(r.get('prediction_id'))), reverse=True)]
    data['yesterday'] = (date.fromisoformat(target)-timedelta(days=1)).isoformat()
    data['yesterday_diagnostics'] = evaluate_day(
        data['yesterday'], predictions_dir=predictions_dir, raw_dir=raw_dir)
    data['trailing_diagnostics'] = evaluate_history(
        data['yesterday'], lookback_days=14, predictions_dir=predictions_dir, raw_dir=raw_dir)
    h['injury_feed_healthy'] = bool((data.get('injury_feed') or {}).get('healthy', False))
    data['display_schema'] = 'NBA_DASHBOARD_V2'
    return clean(data)
