'use strict';
const assert = require('node:assert/strict');
const {heatClass, filterPlayers, sortPlayers, filterTonight, chartSvg, hitText} = require('../dashboard/consistency.js');
assert.equal(heatClass({season_rate:.93, gp:14}, {}), 'heat-90');
assert.equal(heatClass({season_rate:.85, gp:14}, {}), 'heat-80');
assert.equal(heatClass({season_rate:.75, gp:14}, {}), 'heat-70');
assert.equal(heatClass({season_rate:.5, gp:14}, {}), 'heat-low');
assert.equal(heatClass({season_rate:.95, gp:8}, {}), 'heat-warn');                 // under 10 games
assert.equal(heatClass({season_rate:.95, gp:14}, {injury_flag:true}), 'heat-warn'); // injury
const P = (rank, player, team, score, cats, rate) => ({rank, player, team, score, gp:12, mpg:30, qualifying_categories:cats, best_category:cats[0], categories:{[cats[0]]:{season_rate:rate}}, status:'PASS'});
const ps = [P(2,'Bee','NY',70,['PTS'],.8), P(1,'Ace','BOS',80,['REB','PTS'],.9), P('R1','Cat','NY',60,['AST'],.95)];
assert.deepEqual(sortPlayers(ps).map(p=>p.player), ['Ace','Bee','Cat']);
assert.deepEqual(sortPlayers(ps,'season').map(p=>p.player), ['Cat','Ace','Bee']);
assert.deepEqual(sortPlayers(ps,'season',-1).map(p=>p.player), ['Bee','Ace','Cat']);
assert.equal(filterPlayers(ps,{team:'NY'}).length, 2);
assert.equal(filterPlayers(ps,{category:'PTS'}).length, 2);
assert.equal(filterPlayers(ps,{search:'ace'}).length, 1);
const rows = [{player:'Ace', team:'BOS', opp:'NY', category:'PTS', side:'OVER', books:{fanduel:{over:-110}, draftkings:{over:null}}}];
assert.equal(filterTonight(rows,{book:'fanduel'}).length, 1);
assert.equal(filterTonight(rows,{book:'draftkings'}).length, 0);
assert.equal(hitText({n:10, over:6, under:3, push:1}, 'OVER'), '6/9 (1P)');
assert.equal(hitText({n:5, over:1, under:4, push:0}, 'UNDER'), '4/5');
const svg = chartSvg([{date:'2026-11-01', opp:'NY', home:true, minutes:30, PTS:25}, {date:'2026-11-03', opp:'<x>', home:false, minutes:12, PTS:18, low_minutes:true}], 'PTS', 21.5);
assert.match(svg, /cs-bar hit/); assert.match(svg, /cs-bar miss/); assert.doesNotMatch(svg, /<x>/);
const under = chartSvg([{date:'d', opp:'A', minutes:30, REB:6}, {date:'e', opp:'B', minutes:30, REB:10}, {date:'f', opp:'C', minutes:30, REB:8}], 'REB', 8.5, 'UNDER');
assert.match(under, /cs-bar hit"[^>]*height/); assert.equal((under.match(/cs-bar hit/g)||[]).length, 2);   // 6 and 8 hit the UNDER
const push = chartSvg([{date:'d', opp:'A', minutes:30, PTS:20}], 'PTS', 20, 'OVER');
assert.match(push, /cs-bar push/);
console.log('Consistency client: 24 assertions passed');
