/* Player Consistency tab (consistency.py). Pregame snapshot; BET only with a validated calibration. */
(function () {
  'use strict';
  const esc = v => String(v ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const num = v => v !== null && v !== undefined && v !== '' && Number.isFinite(Number(v));
  const fmt = (v, d=1) => num(v) ? Number(v).toLocaleString('en-US',{minimumFractionDigits:0,maximumFractionDigits:d}) : '—';
  const pct = (v, d=0) => num(v) ? fmt(Number(v)*100, d)+'%' : '—';
  const odds = v => num(v) ? (Number(v) > 0 ? '+' : '')+Number(v) : '—';
  const signedPct = v => num(v) ? (Number(v) > 0 ? '+' : '')+fmt(Number(v)*100,1)+'%' : '—';
  const CATS = ['PTS','AST','REB','3PM','STL','BLK','PRA','PA','PR','RA'];
  const COMBINED = new Set(['PRA','PA','PR','RA']);
  const BOOKS = [['fanduel','FD'],['draftkings','DK'],['fanatics','FAN']];
  const STATUS_ORDER = {BET:0, LEAN:1, WATCH:2, PASS:3};
  const timeET = v => { const d = new Date(v||''); return Number.isFinite(d.getTime()) ? new Intl.DateTimeFormat('en-US',{timeZone:'America/New_York',hour:'numeric',minute:'2-digit'}).format(d)+' ET' : '—'; };
  const dayET = v => { const d = new Date(String(v)+'T12:00:00Z'); return Number.isFinite(d.getTime()) ? new Intl.DateTimeFormat('en-US',{timeZone:'America/New_York',month:'short',day:'numeric'}).format(d) : '—'; };

  // Heatmap class by season hit rate at the threshold; red when the sample or availability is a problem.
  function heatClass(cat, player) {
    if (!cat || !num(cat.season_rate)) return 'heat-warn';
    if ((cat.gp || 0) < 10 || player?.injury_flag) return 'heat-warn';
    const r = Number(cat.season_rate);
    return r >= .9 ? 'heat-90' : r >= .8 ? 'heat-80' : r >= .7 ? 'heat-70' : 'heat-low';
  }

  function filterPlayers(players, f={}) {
    const q = String(f.search||'').trim().toLowerCase();
    return (players||[]).filter(p => (!q || `${p.player} ${p.team}`.toLowerCase().includes(q))
      && (!f.team || p.team === f.team)
      && (!f.category || (p.qualifying_categories||[]).includes(f.category)));
  }

  const SORTS = {
    rank: p => typeof p.rank === 'number' ? p.rank : 100 + Number(String(p.rank).slice(1)),
    player: p => p.player, gp: p => -p.gp, mpg: p => -p.mpg, score: p => -p.score,
    season: p => -(p.categories?.[p.best_category]?.season_rate ?? -1),
    l10: p => -(p.categories?.[p.best_category]?.l10_rate ?? -1),
    l5: p => -(p.categories?.[p.best_category]?.l5_rate ?? -1),
    streak: p => -(p.categories?.[p.best_category]?.streak ?? -1),
    edge: p => -(p.best_line?.edge ?? -9), status: p => STATUS_ORDER[p.status] ?? 9,
    next: p => String(p.next_game?.tip || p.next_game?.date || '9999'),
  };
  function sortPlayers(players, key='rank', dir=1) {
    const f = SORTS[key] || SORTS.rank;
    return [...players].sort((a,b) => { const x=f(a), y=f(b); return (x<y?-1:x>y?1:0)*dir || SORTS.rank(a)-SORTS.rank(b); });
  }

  function filterTonight(rows, f={}) {
    const q = String(f.search||'').trim().toLowerCase();
    return (rows||[]).filter(r => (!q || `${r.player} ${r.team} ${r.opp}`.toLowerCase().includes(q))
      && (!f.team || r.team === f.team) && (!f.category || r.category === f.category)
      && (!f.book || r.books?.[f.book]?.[String(r.side).toLowerCase()] != null));
  }

  // Last-N chart: one bar per game, a dashed line at the threshold or tonight's exact line.
  function chartSvg(games, cat, line, side='OVER') {
    const g = (games||[]).filter(x => num(x[cat]));
    if (!g.length) return '<p class="small muted">No games.</p>';
    const W = 560, H = 190, pad = 26, bw = (W - pad) / g.length;
    const max = Math.max(...g.map(x => x[cat]), num(line) ? line : 0, 1) * 1.15;
    const y = v => H - 34 - (v / max) * (H - 50);
    const bars = g.map((x,i) => {
      const push = num(line) && x[cat] === Number(line), h = H - 34 - y(x[cat]);
      const hit = !num(line) || push ? null : (side === 'UNDER' ? x[cat] < line : x[cat] > line);
      const cls = push ? 'cs-bar push' : hit === null ? 'cs-bar' : hit ? 'cs-bar hit' : 'cs-bar miss';
      const label = `${x.date} ${x.home?'vs':'@'} ${x.opp}: ${x[cat]} ${cat}, ${fmt(x.minutes,0)} min${x.b2b?', back-to-back':''}${x.low_minutes?', low minutes':''}`;
      return `<g><title>${esc(label)}</title><rect class="${cls}" x="${pad + i*bw + 2}" y="${y(x[cat])}" width="${Math.max(bw-4,2)}" height="${Math.max(h,1)}" rx="2"></rect>`
        + `<text class="cs-val" x="${pad + i*bw + bw/2}" y="${y(x[cat]) - 3}" text-anchor="middle">${esc(fmt(x[cat],0))}</text>`
        + `<text class="cs-axis" x="${pad + i*bw + bw/2}" y="${H-20}" text-anchor="middle">${esc((x.home?'':'@')+(x.opp||''))}</text>`
        + `<text class="cs-axis ${x.low_minutes?'warn':''}" x="${pad + i*bw + bw/2}" y="${H-8}" text-anchor="middle">${esc(fmt(x.minutes,0))}${x.b2b?'·B':''}</text></g>`;
    }).join('');
    const ref = num(line) ? `<line class="cs-line" x1="${pad}" x2="${W}" y1="${y(line)}" y2="${y(line)}"></line><text class="cs-axis" x="2" y="${y(line)+3}">${esc(fmt(line))}</text>` : '';
    return `<svg class="cs-chart" viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(`Last ${g.length} games, ${cat}`)}">${bars}${ref}</svg>`;
  }

  function hitText(h, side) {
    if (!h) return '—';
    const k = String(side||'OVER').toLowerCase();
    return `${h[k]}/${h.n - h.push}${h.push ? ` (${h.push}P)` : ''}`;
  }

  const api = {heatClass, filterPlayers, sortPlayers, filterTonight, chartSvg, hitText};
  if (typeof module !== 'undefined' && module.exports) { module.exports = api; return; }

  const state = {f:{search:'',team:'',category:'',book:'',date:''}, sort:'rank', dir:1, mode:'A', player:null, chartCat:null, showReserve:false};
  const badge = (t, k='') => `<span class="badge ${k}">${esc(t)}</span>`;
  const statusBadge = s => badge(s||'PASS', s==='BET'?'bet':s==='LEAN'?'research':s==='WATCH'?'warning':'');
  let view, C, R;

  function kpis() {
    const k = C.kpis || {}, r7 = k.results_7d || {}, t = r7.trends || {}, l = r7.exact_lines || {};
    const results = t.n ? `${t.hits}-${t.misses}` : '—';
    const tiles = [[k.active_players ?? 0,'Active players',`of 20 · ${k.reserve_players ?? 0} reserve`],
      [k.qualified_trends ?? 0,'Qualified trends','80%+ at a meaningful threshold'],
      [k.active_streaks ?? 0,'Active streaks','5+ games in a row'],
      [k.tonight_players ?? 0,'Playing tonight','active list'],
      [k.potential_edges ?? 0,'Potential edges',`${k.bets ?? 0} BET · rest LEAN`],
      [results,'Results · 7 days',l.n ? `lines ${l.hits}-${l.misses}` : 'trends hit-miss']];
    const fr = C.freshness || {};
    const missing = (fr.missing_feeds||[]);
    return `<section class="cs-kpis">${tiles.map(([v,label,sub]) => `<div class="summary-item"><div class="summary-value">${esc(v)}</div><div class="summary-label">${esc(label)}</div><div class="cs-sub">${esc(sub)}</div></div>`).join('')}</section>
      <p class="small muted cs-fresh">Refreshed ${esc(timeET(C.generated_at_utc))} (${esc(C.phase||'')}) · box scores through ${esc(dayET(fr.last_box_score_date))}${fr.stale_box_scores?' <b class="loss">· stale</b>':''} · props ${esc(timeET(fr.props_scraped_at))} · injuries ${fr.injury_feed?.healthy?'ok':'<b class="loss">unavailable</b>'}${missing.length?` · <span class="cs-missing">Missing: ${missing.map(esc).join('; ')}</span>`:''}</p>`;
  }

  function filters() {
    const all = [...(C.active||[]), ...(C.reserve||[])];
    const teams = [...new Set(all.map(p => p.team))].sort();
    const dates = [...new Set((R?.history||[]).map(h => h.date))].sort().reverse();
    const opt = (items, sel) => items.map(([v,l]) => `<option value="${esc(v)}" ${v===sel?'selected':''}>${esc(l)}</option>`).join('');
    return `<section class="filters cs-filters"><input id="cs-search" class="search-input" type="search" placeholder="Search players" value="${esc(state.f.search)}" aria-label="Search players">
      <select id="cs-team" aria-label="Team">${opt([['','All teams'],...teams.map(t=>[t,t])], state.f.team)}</select>
      <select id="cs-category" aria-label="Category">${opt([['','All categories'],...CATS.map(c=>[c,c])], state.f.category)}</select>
      <select id="cs-book" aria-label="Sportsbook">${opt([['','All books'],...BOOKS.map(([k,s])=>[k,{fanduel:'FanDuel',draftkings:'DraftKings',fanatics:'Fanatics'}[k]])], state.f.book)}</select>
      <select id="cs-date" aria-label="Results date">${opt([['','All result dates'],...dates.map(d=>[d,dayET(d)])], state.f.date)}</select></section>`;
  }

  function leaderboard() {
    const pool = state.showReserve ? [...(C.active||[]), ...(C.reserve||[])] : (C.active||[]);
    const rows = sortPlayers(filterPlayers(pool, state.f), state.sort, state.dir);
    const th = (key, label) => key ? `<th><button type="button" class="cs-sort" data-cs-sort="${key}" aria-label="Sort by ${esc(label)}">${esc(label)}${state.sort===key?(state.dir>0?' ▲':' ▼'):''}</button></th>` : `<th>${esc(label)}</th>`;
    const head = `<tr>${th('rank','#')}${th('player','Player')}${th('gp','GP / MPG')}${th('next','Next')}${th('','Best')}${th('','Threshold')}${th('season','Season')}${th('l10','L10')}${th('l5','L5')}${th('streak','Streak')}${th('score','Score')}${th('','Model P')}${th('','Best odds')}${th('edge','Edge')}${th('status','Status')}</tr>`;
    const body = rows.map(p => {
      const c = p.categories?.[p.best_category] || {}, bl = p.best_line || {};
      const flags = [p.override ? badge('Manual','warning') : '', p.injury_flag ? badge(p.injury,'warning') : '', p.minutes_restriction ? badge('Min. limit','warning') : ''].join('');
      const next = p.next_game ? (p.next_game.tip ? `${p.next_game.home?'vs':'@'} ${p.next_game.opp} ${timeET(p.next_game.tip)}` : dayET(p.next_game.date)) : '—';
      return `<tr class="${state.player===p.player_id?'cs-selected':''}"><td>${esc(p.rank)}</td><td class="cs-name"><button type="button" class="cs-link" data-cs-player="${p.player_id}">${esc(p.player)}</button><span class="small muted">${esc(p.team)}</span>${flags}</td>
        <td>${p.gp} / ${fmt(p.mpg)}</td><td class="small">${esc(next)}</td><td>${esc(p.best_category)}</td><td>${esc(c.threshold)}+</td>
        <td>${pct(c.season_rate)}</td><td>${pct(c.l10_rate)}</td><td>${pct(c.l5_rate)}</td><td>${c.streak??'—'}</td><td><b>${fmt(p.score)}</b></td>
        <td>${num(bl.calibrated_probability)?pct(bl.calibrated_probability,1):'<span class="muted" title="Not calibrated">—</span>'}</td>
        <td class="small">${bl.best_odds!=null?`${esc(bl.category)} ${esc(bl.side?.[0])}${esc(bl.line)} ${esc(odds(bl.best_odds))}`:'—'}</td>
        <td>${num(bl.edge)?signedPct(bl.edge):'—'}</td><td>${statusBadge(p.status)}${p.status_reason?`<span class="small muted cs-reason">${esc(p.status_reason)}</span>`:''}</td></tr>`;
    }).join('');
    const empty = (C.active||[]).length < 20 ? `<p class="small muted">${(C.active||[]).length} players qualify today; empty slots stay empty.</p>` : '';
    return `<section class="panel cs-section"><div class="cs-head"><h2 class="inline-title">Leaderboard</h2><label class="small"><input id="cs-reserve" type="checkbox" ${state.showReserve?'checked':''}> Show reserve (${(C.reserve||[]).length})</label></div>
      <p class="subtitle">Ranked by consistency score. Best category = the player's highest-scoring qualifying category. Model P and edge appear only when the prop calibration is validated (${esc(C.calibration?.status||'unknown')} now).</p>
      <div class="table-wrap"><table class="cs-table">${head}<tbody>${body || '<tr><td colspan="15" class="muted">No qualified players match these filters.</td></tr>'}</tbody></table></div>${empty}</section>`;
  }

  function matrixA(players) {
    const cats = state.f.category ? [state.f.category] : CATS;
    const body = players.map(p => `<tr><td class="cs-name"><button type="button" class="cs-link" data-cs-player="${p.player_id}">${esc(p.player)}</button></td>${cats.map(cat => {
      const c = p.categories?.[cat];
      if (!c) return '<td class="heat-warn">—</td>';
      const streak = c.streak >= 5 ? `<span class="cs-streak" title="${c.streak}-game streak">●${c.streak}</span>` : '';
      return `<td class="${heatClass(c,p)}" title="${esc(`${cat} ${c.threshold}+: ${c.season_hits}/${c.gp} season, ${c.l10_hits}/${c.l10_n} last 10`)}"><b>${esc(c.threshold)}+</b>${streak}<span>${pct(c.season_rate)} · L10 ${pct(c.l10_rate)}</span></td>`;
    }).join('')}</tr>`).join('');
    return `<div class="table-wrap"><table class="cs-heat"><tr><th>Player</th>${cats.map(c=>`<th>${c}</th>`).join('')}</tr>${body}</table></div>
      <p class="small muted cs-legend"><span class="heat-90">90%+</span><span class="heat-80">80–89%</span><span class="heat-70">70–79%</span><span class="heat-low">under 70%</span><span class="heat-warn">under 10 games or injury</span> ● active streak of 5+</p>`;
  }

  function matrixB() {
    const rows = filterTonight(C.tonight, state.f);
    if (!rows.length) return '<p class="small muted">No exact sportsbook lines for these filters. Lines are collected for PTS, REB, AST, 3PM and PRA; STL, BLK, PA, PR and RA have none.</p>';
    const body = rows.map(r => {
      const k = String(r.side).toLowerCase();
      const books = BOOKS.map(([b]) => { const q = r.books?.[b] || {}; return `<td>${q[k]!=null?odds(q[k]):q.line!=null&&q.line!==r.line?`<span class="muted" title="Different line">${esc(q.line)}</span>`:'—'}</td>`; }).join('');
      return `<tr><td class="cs-name">${esc(r.player)}<span class="small muted">${esc(r.team)} ${r.home?'vs':'@'} ${esc(r.opp)}</span></td><td>${esc(r.category)} ${esc(r.side)} ${esc(r.line)}</td>
        <td>${esc(hitText(r.hits?.l5,r.side))}</td><td>${esc(hitText(r.hits?.l10,r.side))}</td><td>${esc(hitText(r.hits?.season,r.side))}</td>${books}
        <td>${esc(odds(r.best_odds))} <span class="small muted">${esc((BOOKS.find(([b])=>b===r.best_book)||[])[1]||'')}</span></td><td>${pct(r.break_even,1)}</td>
        <td>${fmt(r.projection)}</td><td>${num(r.calibrated_probability)?pct(r.calibrated_probability,1):`<span class="muted" title="Raw model, not calibrated">${num(r.model_probability)?pct(r.model_probability,1)+'*':'—'}</span>`}</td>
        <td>${num(r.edge)?signedPct(r.edge):'—'}</td></tr>`;
    }).join('');
    return `<div class="table-wrap"><table class="cs-table"><tr><th>Player</th><th>Line</th><th>L5</th><th>L10</th><th>Season</th><th>FD</th><th>DK</th><th>FAN</th><th>Best</th><th>Break-even</th><th>Proj.</th><th>Prob.</th><th>Edge</th></tr>${body}</table></div>
      <p class="small muted">Hit counts are at the exact line (pushes excluded and shown as P). Prices are for the side shown; a grey number means that book is on a different line. * raw model probability, not calibrated, so no edge is shown.</p>`;
  }

  function analysis() {
    const players = filterPlayers([...(C.active||[]), ...(state.showReserve ? C.reserve||[] : [])], state.f);
    return `<section class="panel cs-section"><div class="cs-head"><h2 class="inline-title">Statistical analysis</h2><div class="lab-nav">
      <button type="button" class="lab-nav-btn" data-cs-mode="A" aria-pressed="${state.mode==='A'}">A · Consistency matrix</button>
      <button type="button" class="lab-nav-btn" data-cs-mode="B" aria-pressed="${state.mode==='B'}">B · Exact lines tonight</button></div></div>
      ${state.mode==='A' ? matrixA(players) : matrixB()}</section>`;
  }

  function deepDive() {
    const all = [...(C.active||[]), ...(C.reserve||[])];
    const p = all.find(x => x.player_id === state.player) || all[0];
    if (!p) return '';
    const cat = state.chartCat || p.best_category;
    const line = (C.tonight||[]).find(r => r.player_id === p.player_id && r.category === cat);
    const c = p.categories?.[cat] || {};
    const cards = CATS.map(k => { const x = p.categories?.[k] || {}; const tl = (C.tonight||[]).find(r => r.player_id===p.player_id && r.category===k);
      return `<button type="button" class="cs-card ${k===cat?'active':''} ${heatClass(x,p)}" data-cs-chart="${k}"><div class="cs-card-top"><b>${k}</b>${(p.qualifying_categories||[]).includes(k)?badge('Qualified','bet'):''}</div>
        <div class="cs-card-grid"><span>Avg ${fmt(x.mean)}</span><span>SD ${fmt(x.std)}</span><span>Med ${fmt(x.median)}</span><span>L5 ${fmt(x.l5_avg)}</span><span>L10 ${fmt(x.l10_avg)}</span><span>Streak ${x.streak??'—'}</span></div>
        <div class="small">${esc(x.threshold)}+ · ${x.season_hits}/${x.gp} · L10 ${x.l10_hits}/${x.l10_n}</div><div class="small muted">${tl?`Line ${esc(tl.side)} ${esc(tl.line)} · ${esc(odds(tl.best_odds))}`:'No sportsbook line'}</div></button>`; }).join('');
    const sp = p.splits?.[cat] || {};
    const splitRows = Object.entries(sp).map(([label,[n,avg,rate]]) => `<tr><td>${esc(label)}</td><td>${n}</td><td>${fmt(avg)}</td><td>${pct(rate)}</td></tr>`).join('');
    const cs = Object.entries(p.combined_stability||{}).map(([k,v]) => `<tr><td>${k}</td><td>${fmt(v.mean)}</td><td>${fmt(v.std)}</td><td>${fmt(v.cv,2)}</td><td>${fmt(v.offset_ratio,2)}</td></tr>`).join('');
    const ng = p.next_game;
    return `<section class="panel cs-section" id="cs-deep"><div class="cs-head"><h2 class="inline-title">Player deep dive · ${esc(p.player)}</h2>${statusBadge(p.status)}</div>
      <p class="subtitle">#${esc(p.rank)} · ${esc(p.team_name)} ${esc(p.position||'')} · ${p.gp} GP (${p.dnp} DNP excluded, ${p.low_minute_games} under 15 min) · ${fmt(p.mpg)} MPG (CV ${fmt(p.minutes_cv,2)}) · score ${fmt(p.score)} · last game ${esc(dayET(p.last_game))}
      ${ng?` · next ${ng.tip?`${ng.home?'vs':'@'} ${esc(ng.opp)} ${esc(timeET(ng.tip))}`:esc(dayET(ng.date))}`:''}${num(p.projected_minutes)?` · projected ${fmt(p.projected_minutes)} min`:''}${p.injury?` · <b class="loss">${esc(p.injury)}</b>`:''}${p.override?` · <b>Manual override: ${esc(p.override_reason||'')}</b>`:''}</p>
      ${(p.preferences_missed||[]).length?`<p class="small muted">Preferences missed: ${p.preferences_missed.map(esc).join(', ')}</p>`:''}
      <div class="cs-cards">${cards}</div>
      <h3 class="review-heading">Last ${(p.last20||[]).length} games · ${cat} vs ${line?`tonight's line (${line.side} ${line.line})`:`${c.threshold}+ threshold`} (opponent, minutes; ·B = back-to-back)</h3>
      <div class="table-wrap"><div class="cs-chart-wrap">${chartSvg(p.last20, cat, line ? line.line : (num(c.threshold) ? c.threshold - .5 : null), line ? line.side : 'OVER')}</div></div>
      <div class="cs-two"><div><h3 class="review-heading">${cat} splits at ${esc(c.threshold)}+ (games, average, hit rate)</h3><div class="table-wrap"><table class="cs-table"><tr><th>Split</th><th>GP</th><th>Avg</th><th>Hit</th></tr>${splitRows}</table></div></div>
      <div><h3 class="review-heading">Combined-stat stability (from per-game totals)</h3><div class="table-wrap"><table class="cs-table"><tr><th>Stat</th><th>Avg</th><th>SD</th><th>CV</th><th>SD ÷ parts</th></tr>${cs}</table></div>
      <p class="small muted">SD ÷ parts below 1 means the components offset each other, so the combined line is steadier than its pieces.</p></div></div></section>`;
  }

  function tonight() {
    const rows = filterTonight(C.tonight, state.f).sort((a,b) => (STATUS_ORDER[a.recommendation]-STATUS_ORDER[b.recommendation]) || String(a.tip).localeCompare(String(b.tip)));
    const row = r => `<tr><td class="cs-name">${esc(r.player)}<span class="small muted">${r.home?'vs':'@'} ${esc(r.opp)} · ${esc(timeET(r.tip))}</span></td><td>${esc(r.category)} ${esc(r.side)} ${esc(r.line)}</td>
      <td>${esc(hitText(r.hits?.l5,r.side))} · ${esc(hitText(r.hits?.l10,r.side))} · ${esc(hitText(r.hits?.season,r.side))}</td><td>${fmt(r.projection)}</td><td>${fmt(r.projected_minutes)}</td>
      <td class="small">${esc(r.injury || (r.injury_adjustment && r.injury_adjustment!=='NONE' ? r.injury_adjustment.replaceAll('_',' ').toLowerCase() : '—'))}${num(r.opportunity_minutes_delta)&&r.opportunity_minutes_delta?` (${r.opportunity_minutes_delta>0?'+':''}${fmt(r.opportunity_minutes_delta)} min)`:''}</td>
      <td>${esc((BOOKS.find(([b])=>b===r.best_book)||[])[1]||'—')} ${esc(odds(r.best_odds))}</td><td>${pct(r.break_even,1)}</td><td>${num(r.edge)?signedPct(r.edge):'—'}</td>
      <td>${statusBadge(r.recommendation)}${(r.failed_checks||[]).length?`<span class="small muted cs-reason">${esc(r.failed_checks.map(x=>x.replaceAll('_',' ')).join(', '))}</span>`:''}</td></tr>`;
    const live = rows.filter(r => r.recommendation !== 'PASS'), passed = rows.filter(r => r.recommendation === 'PASS');
    const head = '<tr><th>Player</th><th>Line</th><th>L5 · L10 · Season</th><th>Proj.</th><th>Proj. min</th><th>Injury adj.</th><th>Best</th><th>Break-even</th><th>Edge</th><th>Call</th></tr>';
    const body = live.map(row).join('');
    const passTable = passed.length ? `<details class="details-link"><summary>PASS (${passed.length}) · failed a check</summary><div class="table-wrap"><table class="cs-table">${head}${passed.map(row).join('')}</table></div></details>` : '';
    return `<section class="panel cs-section"><h2 class="inline-title">Tonight's opportunities</h2>
      <p class="subtitle">BET needs a calibrated probability that beats break-even by 3 points with every check passing. LEAN: every check passes and the exact-line record is 70%+ (season and last 10) and beats break-even, but it isn't validated. WATCH: needs more information. PASS: fails a check.</p>
      <div class="table-wrap"><table class="cs-table">${head}${body||'<tr><td colspan="10" class="muted">No BET, LEAN or WATCH lines for these filters.</td></tr>'}</table></div>${passTable}</section>`;
  }

  function results() {
    if (!R || !R.graded) return `<section class="panel cs-section"><h2 class="inline-title">Results</h2><p class="small muted">Nothing graded yet. Every monitored trend and exact line is graded after the games.</p></section>`;
    const line = (label, s, units=false) => `<tr><td>${esc(label)}</td><td>${s?.n?`${s.hits}-${s.misses}${s.pushes?`-${s.pushes}`:''}`:'—'}</td><td>${pct(s?.hit_rate,1)}</td><td>${s?.void_dnp||0}</td><td>${units&&num(s?.units_if_followed)?`${s.units_if_followed>0?'+':''}${fmt(s.units_if_followed,2)}u`:'—'}</td></tr>`;
    const hist = (R.history||[]).filter(h => (!state.f.date || h.date === state.f.date) && (!state.f.category || h.category === state.f.category)
      && (!state.f.team || h.team === state.f.team) && (!state.f.search || String(h.player).toLowerCase().includes(state.f.search.toLowerCase()))).slice(0, 80);
    const rows = hist.map(h => `<tr><td>${esc(dayET(h.date))}</td><td>${esc(h.player)}</td><td>${esc(h.kind==='TREND'?`${h.category} ${h.threshold}+`:`${h.category} ${h.side} ${h.line}`)}</td><td>${esc(h.kind==='TREND'?'Trend':h.recommendation)}</td><td>${fmt(h.actual)}</td><td class="${['HIT','WIN'].includes(h.outcome)?'win':['MISS','LOSS'].includes(h.outcome)?'loss':'push'}">${esc(h.outcome)}</td></tr>`).join('');
    return `<section class="panel cs-section"><h2 class="inline-title">Results</h2><p class="subtitle">${esc(R.note||'')}</p>
      <div class="table-wrap"><table class="cs-table"><tr><th>Group</th><th>Record</th><th>Hit rate</th><th>DNP void</th><th>If followed</th></tr>
      ${line('Individual trends (PTS AST REB 3PM STL BLK)', R.individual_trends)}${line('Combined trends (PRA PA PR RA)', R.combined_trends)}
      ${line('Alternate thresholds (all trends)', R.alternate_thresholds)}${line('Exact sportsbook lines', R.exact_lines, true)}
      ${line('LEAN lines', R.by_recommendation?.LEAN, true)}${line('BET-qualified', R.bet_qualified, true)}${line('Last 7 days · trends', R.rolling_7d?.trends)}${line('Last 7 days · lines', R.rolling_7d?.exact_lines, true)}</table></div>
      <details class="details-link"><summary>Graded selections (${hist.length})</summary><div class="table-wrap"><table class="cs-table"><tr><th>Date</th><th>Player</th><th>Selection</th><th>Type</th><th>Actual</th><th>Result</th></tr>${rows}</table></div></details></section>`;
  }

  function render() {
    if (!C || !(C.active||[]).length && !(C.reserve||[]).length) {
      view.innerHTML = `<div class="page-heading"><div><h1>Player consistency</h1><p class="subtitle">The 20 most consistent players and their trends.</p></div></div><div class="empty-state"><h2>No consistency snapshot for this slate</h2><p>It builds with the daily run and again before tip.</p></div>${results()}`;
      return;
    }
    view.innerHTML = `<div class="page-heading"><div><h1>Player consistency</h1><p class="subtitle">20 most consistent players, 10 categories · pregame data only · ${esc(C.version)} (rules ${esc(C.rules_id)})</p></div>${badge('Research · not bets','research')}</div>
      ${kpis()}${filters()}<div id="cs-body">${body()}</div>`;
  }
  const body = () => leaderboard() + analysis() + deepDive() + tonight() + results();
  const refresh = () => { const b = document.getElementById('cs-body'); if (b) b.innerHTML = body(); };

  function bind() {
    view.addEventListener('click', e => {
      if (!document.getElementById('cs-body')) return;
      const t = e.target.closest('[data-cs-sort],[data-cs-player],[data-cs-mode],[data-cs-chart]');
      if (!t) return;
      if (t.dataset.csSort) { const k = t.dataset.csSort; state.dir = state.sort === k ? -state.dir : 1; state.sort = k; refresh(); }
      else if (t.dataset.csPlayer) { state.player = Number(t.dataset.csPlayer); state.chartCat = null; refresh(); document.getElementById('cs-deep')?.scrollIntoView({block:'start'}); }
      else if (t.dataset.csMode) { state.mode = t.dataset.csMode; refresh(); }
      else if (t.dataset.csChart) { state.chartCat = t.dataset.csChart; refresh(); document.getElementById('cs-deep')?.scrollIntoView({block:'start'}); }
    });
    view.addEventListener('input', e => { if (e.target.id === 'cs-search') { state.f.search = e.target.value; refresh(); } });
    view.addEventListener('change', e => {
      const id = e.target.id;
      if (id === 'cs-reserve') { state.showReserve = e.target.checked; refresh(); }
      else if (['cs-team','cs-category','cs-book','cs-date'].includes(id)) { state.f[id.slice(3)] = e.target.value; refresh(); }
    });
  }

  let bound = false;
  window.Consistency = {
    open(target) {
      view = target; C = (window.DATA||{}).consistency || {}; R = (window.DATA||{}).consistency_report || {};
      if (!bound) { bind(); bound = true; }
      render();
    }
  };
}());
