import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import pandas as pd

import build_dashboard
from dashboard_data import flatten_props, norm, prepare_display, quote_index


class DashboardDataTests(unittest.TestCase):
    def payload(self):
        return {'date': '2026-11-10', 'games': [{
            'home': {'name': 'New York Knicks'}, 'away': {'name': 'Milwaukee Bucks'},
            'tip': '2026-11-11T00:30Z', 'props': [{
                'player': 'Jalen Brunson', 'team': 'New York Knicks', 'opp': 'Milwaukee Bucks',
                'proj_min': 34, 'props': {
                    'THREES': {'pred': 3.5, 'line': 2.5, 'odds': -137, 'side': 'UNDER',
                               'book_key': 'draftkings', 'signal': None, 'ev': .99},
                    'REB': {'pred': 4, 'line': None},
                }}]}], 'best_bets': [], 'forward_performance': {'evidence_epoch': 'epoch'}}

    def key(self):
        return (norm('Milwaukee Bucks @ New York Knicks'), norm('Jalen Brunson'), 'THREES')

    def offer(self, side='OVER', line=2.5, odds=104):
        return {'book': 'DraftKings', 'book_key': 'draftkings', 'side': side, 'line': line, 'odds': odds}

    def test_nested_projections_use_same_side_same_line_prices(self):
        rows = flatten_props(self.payload(), {self.key(): [self.offer('UNDER', odds=-137), self.offer(), self.offer(line=3.5, odds=180)]})
        threes = next(r for r in rows if r['stat'] == 'THREES')
        self.assertEqual((len(rows), threes['direction'], threes['quote']['odds']), (2, 'OVER', 104))
        self.assertEqual(threes['decision'], 'RESEARCH')
        self.assertIsNone(threes['ev'])
        rebound = next(r for r in rows if r['stat'] == 'REB')
        self.assertIsNone(rebound['direction'])
        self.assertIsNone(rebound['quote'])

    def test_opposite_side_and_other_line_are_not_borrowed(self):
        rows = flatten_props(self.payload(), {self.key(): [self.offer('UNDER'), self.offer(line=3.5)]})
        self.assertTrue(all(r['quote'] is None for r in rows))

    def test_only_final_unblocked_card_is_qualified(self):
        data = self.payload()
        data['best_bets'] = [{'type': 'PROP', 'game': 'Milwaukee Bucks @ New York Knicks', 'player': 'Jalen Brunson', 'stat': 'THREES', 'units': .2}]
        prop = data['games'][0]['props'][0]['props']['THREES']
        prop['signal'] = 'OVER'
        prop['odds'] = 104
        row = next(r for r in flatten_props(data) if r['stat'] == 'THREES')
        self.assertEqual((row['decision'], row['quote']['odds'], row['units']), ('BET', 104, .2))
        prop['availability_block'] = True
        self.assertTrue(all(r['decision'] == 'RESEARCH' for r in flatten_props(data)))

    def test_stale_and_undated_quotes_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'sportsbook_props_today.csv'
            for row in ({'game_date': '2026-11-09', 'player': 'Old'}, {'player': 'Undated'}):
                pd.DataFrame([row]).to_csv(path, index=False)
                self.assertEqual(quote_index('2026-11-10', temp)[1], [])

    def test_display_copy_counts_and_prior_only_profiles(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            data = self.payload()
            before = deepcopy(data)
            profiles = path / 'player_profiles_latest.csv'
            for last_game_date, expected in [('2026-11-09', 2.9), ('2026-11-10', None)]:
                pd.DataFrame([{'player': 'Jalen Brunson', 'last_game_date': last_game_date, 'roll10_threes': 2.9}]).to_csv(profiles, index=False)
                shown = prepare_display(data, temp, temp, temp)
                self.assertEqual(next(r for r in shown['props'] if r['stat'] == 'THREES')['last10'], expected)
                self.assertEqual((shown['data_health']['props'], shown['data_health']['props_found'], shown['data_health']['modeled_players']), ('loaded', 2, 1))
                self.assertNotIn('props', shown['games'][0])
                self.assertEqual(data, before)

    def test_results_require_epoch_and_exact_priced_evidence(self):
        base = {'date': '2026-11-09', 'decision': 'BET', 'market': 'PROP', 'side': 'OVER',
                'sportsbook_key': 'draftkings', 'odds': 120, 'outcome': 'WIN', 'evidence_epoch': 'epoch'}
        rows = [base, {**base, 'odds': -125}, {**base, 'outcome': 'LOSS'},
                {**base, 'evidence_epoch': 'old'}, {**base, 'decision': 'PASS'},
                {**base, 'market': 'SPREAD', 'side': 'HOME', 'line': -3.5}]
        with tempfile.TemporaryDirectory() as temp:
            pd.DataFrame(rows).to_csv(Path(temp) / 'nba_forward_certified.csv', index=False)
            data = self.payload()
            result = prepare_display(data, temp, temp, temp)['results']
            self.assertEqual(sorted(r['profit_units'] for r in result), [-1, .8, 1.2])
            data['forward_performance'] = {}
            self.assertEqual(prepare_display(data, temp, temp, temp)['results'], [])

    def test_template_rebuild_is_idempotent_and_script_safe(self):
        data = {'date': '2026-11-10', 'label': '</script><script>alert(1)</script>', 'missing': float('nan')}
        from dashboard_data import clean
        with tempfile.TemporaryDirectory() as temp, patch.object(build_dashboard, 'OUTPUT_HTML', str(Path(temp) / 'index.html')), patch.object(build_dashboard, 'enrich_data', side_effect=clean):
            output = Path(temp) / 'index.html'
            output.write_text('Old dashboard sentinel')
            build_dashboard.build_html(data)
            once = output.read_text()
            build_dashboard.build_html(data)
            self.assertEqual(once, output.read_text())
            self.assertNotIn('Old dashboard sentinel', once)
            self.assertNotIn('</script><script>alert', once)
            self.assertEqual(once.count('<script'), 2)
            self.assertIn('\\u003c/script\\u003e', once)
            self.assertIn('"missing":null', once)


if __name__ == '__main__':
    unittest.main()
