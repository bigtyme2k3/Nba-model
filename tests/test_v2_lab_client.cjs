'use strict';
const assert = require('node:assert/strict');
const {sortCards, lineProbabilities, histogramSvg, label, esc} = require('../dashboard/v2_lab.js');
const card = (player, stat, tip, v1, v2, mc, lab='AGREES_WITH_MARKET') => ({player, team:'T', opp:'O', stat, tip, v1_projection:v1, v2_projection:v2, research_label:lab, recorded: mc==null ? null : {mc_probability: mc}});
const cards = [card('B', 'PTS', '2026-11-20T01:00Z', 20, 25, .55), card('A', 'REB', '2026-11-20T00:00Z', 8, 8.5, .70, 'MC_FAVORS_SIDE'), card('C', 'AST', '2026-11-20T02:00Z', 5, null, null)];
assert.deepEqual(sortCards(cards, 'time').map(c => c.player), ['A','B','C']);
assert.deepEqual(sortCards(cards, 'disagreement').map(c => c.player), ['B','A','C']);   // no V2 sorts last
assert.deepEqual(sortCards(cards, 'probability').map(c => c.player), ['A','B','C']);
assert.equal(sortCards(cards, 'time', {label:'MC_FAVORS_SIDE'}).length, 1);
assert.equal(sortCards(cards, 'time', {search:'b'}).length, 1);
assert.equal(cards[0].player, 'B');                                                     // input not mutated
// Bins of width 1 from 0: outcomes 0..4 with counts 1,2,3,2,2 (10 total).
const h = {start:0, width:1, counts:[1,2,3,2,2]};
assert.deepEqual(lineProbabilities(h, 2.5), {over:.4, under:.6});
const w = {start:0, width:2, counts:[4,6]};                                              // 0-1: 4, 2-3: 6
assert.deepEqual(lineProbabilities(w, 1.5), {over:.6, under:.4});
assert.deepEqual(lineProbabilities({counts:[]}, 2.5), {over:null, under:null});
assert.match(histogramSvg(h, 2.5, 'x'), /Line 2\.5/);
assert.doesNotMatch(histogramSvg(h, 2.5, '<b>'), /<b>/);
assert.equal(label('MC_AGAINST_SIDE'), 'MC against recorded side');
assert.equal(esc('<x>'), '&lt;x&gt;');
console.log('V2 lab client: 14 assertions passed');
