/* V2 Research Lab: Monte Carlo shadow research. Labels only; BET eligibility belongs to V1. */
(function () {
  'use strict';
  const esc = v => String(v ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const num = v => v !== null && v !== undefined && v !== '' && Number.isFinite(Number(v));
  const fmt = (v, d=1) => num(v) ? Number(v).toLocaleString('en-US',{minimumFractionDigits:0,maximumFractionDigits:d}) : '—';
  const pct = (v, d=1) => num(v) ? fmt(Number(v)*100, d)+'%' : '—';
  const odds = v => num(v) ? (Number(v) > 0 ? '+' : '')+Number(v) : '—';
  const fixed = (v, d=4) => num(v) ? Number(v).toFixed(d) : '—';
  const signed = (v, d=3) => num(v) ? (Number(v) > 0 ? '+' : '')+fmt(v, d) : '—';
  const book = k => ({fanduel:'FanDuel',draftkings:'DraftKings',fanatics:'Fanatics',novig:'Novig'}[k] || k || '—');
  const statName = s => ({PTS:'Points',REB:'Rebounds',AST:'Assists',THREES:'Threes',PRA:'Pts + reb + ast'}[s] || s);
  const timeET = v => { const d = new Date(v||''); return Number.isFinite(d.getTime()) ? new Intl.DateTimeFormat('en-US',{timeZone:'America/New_York',hour:'numeric',minute:'2-digit'}).format(d)+' ET' : 'Time unavailable'; };
  const LABELS = {AGREES_WITH_MARKET:['Agrees with market',''], MC_FAVORS_SIDE:['MC favors recorded side','research'],
                  MC_AGAINST_SIDE:['MC against recorded side','warning'], INSUFFICIENT_EVIDENCE:['Insufficient evidence','']};
  const label = key => (LABELS[key] || [String(key||'—').replaceAll('_',' ').toLowerCase(), ''])[0];
  const disagreement = c => num(c.v1_projection) && num(c.v2_projection) ? Math.abs(c.v2_projection - c.v1_projection) : -1;
  const mcProb = c => num(c.recorded?.mc_probability) ? Number(c.recorded.mc_probability) : -1;

  function sortCards(cards, sort='time', filters={}) {
    const q = String(filters.search||'').trim().toLowerCase();
    const rows = (cards||[]).filter(c => (!q || `${c.player} ${c.team} ${c.opp}`.toLowerCase().includes(q))
      && (!filters.stat || c.stat === filters.stat) && (!filters.label || c.research_label === filters.label));
    const by = {
      time: (a,b) => String(a.tip).localeCompare(String(b.tip)) || String(a.player).localeCompare(String(b.player)) || String(a.stat).localeCompare(String(b.stat)),
      disagreement: (a,b) => disagreement(b) - disagreement(a),
      probability: (a,b) => mcProb(b) - mcProb(a),
      market: (a,b) => String(a.stat).localeCompare(String(b.stat)) || String(a.player).localeCompare(String(b.player)),
    }[sort] || ((a,b) => 0);
    return [...rows].sort(by);
  }

  // Outcome probabilities at a line from the stored histogram (bins of `width` from `start`).
  function lineProbabilities(hist, line) {
    const counts = hist?.counts || [], total = counts.reduce((a,b) => a+b, 0);
    if (!total || !num(line)) return {over:null, under:null};
    let over = 0, under = 0;
    counts.forEach((n, i) => {
      const lo = hist.start + i*hist.width, hi = lo + hist.width - 1;   // whole-number outcomes lo..hi
      if (lo > line) over += n; else if (hi < line) under += n;
      else for (let v = lo; v <= hi; v++) { const share = n/hist.width; if (v > line) over += share; else if (v < line) under += share; }
    });
    return {over: over/total, under: under/total};
  }

  function histogramSvg(hist, line, title) {
    const counts = hist?.counts || [];
    if (!counts.length) return '<p class="small muted">No simulation available.</p>';
    const W = 320, H = 140, pad = 18, max = Math.max(...counts), bw = (W - pad*2) / counts.length;
    const x = v => pad + (v - hist.start) / hist.width * bw;
    const bars = counts.map((n, i) => `<rect x="${(pad + i*bw + .5).toFixed(1)}" y="${(H - pad - n/max*(H - pad*2)).toFixed(1)}" width="${Math.max(1, bw - 1).toFixed(1)}" height="${(n/max*(H - pad*2)).toFixed(1)}" class="lab-bar"/>`).join('');
    const end = hist.start + counts.length*hist.width;
    const marker = num(line) && line >= hist.start && line <= end
      ? `<line x1="${x(line + .5).toFixed(1)}" x2="${x(line + .5).toFixed(1)}" y1="${pad - 6}" y2="${H - pad}" class="lab-line"/><text x="${x(line + .5).toFixed(1)}" y="${pad - 8}" class="lab-axis" text-anchor="middle">Line ${esc(fmt(line))}</text>` : '';
    return `<svg viewBox="0 0 ${W} ${H}" class="lab-chart" role="img" aria-label="${esc(title)}">${bars}${marker}<text x="${pad}" y="${H - 4}" class="lab-axis">${esc(fmt(hist.start,0))}</text><text x="${W - pad}" y="${H - 4}" class="lab-axis" text-anchor="end">${esc(fmt(end - 1,0))}</text></svg>`;
  }

  if (typeof module !== 'undefined' && module.exports) {
    module.exports = {sortCards, lineProbabilities, histogramSvg, label, esc};
    return;
  }

  const state = {view:'today', sort:'time', search:'', stat:'', label:'', card:null};
  const badge = (t, k='') => `<span class="badge ${k}">${esc(t)}</span>`;
  const metric = (l, v, accent=false) => `<div><span class="metric-label">${esc(l)}</span><span class="metric-value ${accent?'accent':''}">${esc(v)}</span></div>`;
  const opts = (items, sel) => items.map(([v,l]) => `<option value="${esc(v)}" ${v===sel?'selected':''}>${esc(l)}</option>`).join('');
  const lab = () => (window.DATA || {}).v2_lab || {};
  const note = '<p class="info-note">Shadow research. Monte Carlo has no bet authority: BET eligibility belongs to V1, and this page only labels where the simulation agrees or disagrees.</p>';

  function nav() {
    const b = (id, text) => `<button type="button" class="lab-nav-btn" data-lab-view="${id}" aria-pressed="${state.view===id}">${text}</button>`;
    return `<div class="page-heading"><div><h1>V2 Research Lab</h1><p class="subtitle">Monte Carlo shadow forecasts · ${esc(lab().today?.mc_version || 'version unavailable')}</p></div>${badge('Shadow only','research')}</div><div class="lab-nav" role="group" aria-label="Research Lab views">${b('today','Today’s simulations')}${b('validation','Validation &amp; results')}</div>`;
  }

  function todayView() {
    const t = lab().today;
    if (!t || !t.cards) return note + '<div class="empty-state"><h2>No simulations for this slate</h2><p>Shadow forecasts publish after the daily predictions. They appear here once that step has run.</p></div>';
    const cards = t.cards, recorded = cards.filter(c => c.recorded);
    const count = k => recorded.filter(c => c.research_label === k).length;
    const feat = t.v2_features || {};
    const gate = {PASS:'Pass', FAIL:'Fail'}[lab().forward?.sportsbooks?.all_arms_common_rows?.strict_gate?.mc_v2?.status] || 'Not graded';
    const tiles = [[t.games,'Games evaluated'],[t.players,'Players simulated'],[`${fmt(t.v2_players,0)}/${fmt(t.players,0)}`,'V2 coverage'],
      [{FRESH:'Fresh', STALE:'Stale'}[feat.status] || 'Missing','V2 features'],[`${fmt(recorded.length,0)}/${fmt(cards.length,0)}`,'Recorded quotes'],
      [`${fmt(count('AGREES_WITH_MARKET'),0)}`,'Agree with market'],[`${fmt(count('MC_FAVORS_SIDE')+count('MC_AGAINST_SIDE'),0)}`,'Research signals'],[gate,'Strict gate']];
    const shown = sortCards(cards, state.sort, state).slice(0, 80);
    return nav() + note + `<section class="lab-tiles">${tiles.map(([v,l]) => `<div class="summary-item"><div class="summary-value lab-tile-value">${esc(typeof v === 'number' ? fmt(v,0) : v)}</div><div class="summary-label">${esc(l)}</div></div>`).join('')}</section>
      <p class="small muted">Qualification restrictions: no Monte Carlo signal can create, change or size a bet. V2 features ${esc(feat.target_date ? 'for '+feat.target_date : 'unavailable')}${feat.injury_feed_healthy === false ? ' · injury feed unverified, V2 skipped' : ''}.</p>
      <div class="filters lab-filters"><input class="search-input" id="lab-search" type="search" placeholder="Search player or team" value="${esc(state.search)}" aria-label="Search player or team">
      <select class="search-input" id="lab-sort" aria-label="Sort">${opts([['time','Tip time'],['disagreement','Largest V1–V2 gap'],['probability','MC probability'],['market','Stat market']], state.sort)}</select>
      <select class="search-input" id="lab-stat" aria-label="Stat">${opts([['','All stats'],['PTS','Points'],['REB','Rebounds'],['AST','Assists'],['THREES','Threes'],['PRA','Pts + reb + ast']], state.stat)}</select>
      <select class="search-input" id="lab-label" aria-label="Research status">${opts([['','All statuses'],...Object.keys(LABELS).map(k => [k, label(k)])], state.label)}</select></div>
      <div class="lab-cards">${shown.map(c => card(c, cards.indexOf(c))).join('') || '<p class="small muted">No simulations match these filters.</p>'}</div>
      ${sortCards(cards, state.sort, state).length > 80 ? '<p class="small muted">Showing the first 80. Narrow the filters to see more.</p>' : ''}`;
  }

  function card(c, i) {
    const r = c.recorded;
    const [text, kind] = LABELS[c.research_label] || [label(c.research_label), ''];
    return `<article class="watch-card lab-card"><div class="prop-header"><span class="player-name">${esc(c.player)}</span>${badge(text, kind)}</div>
      <div class="prop-meta">${esc(statName(c.stat))} · ${esc(c.team)} vs ${esc(c.opp)} · ${esc(timeET(c.tip))}</div>
      <div class="prop-numbers">${metric('V1', fmt(c.v1_projection), true)}${metric('V2', fmt(c.v2_projection))}${metric('Sim median', fmt(c.distribution?.p50))}</div>
      <div class="prop-bottom"><span>${r ? `${esc(book(r.book_key))} ${esc(String(r.side||'').toLowerCase())} ${esc(fmt(r.line))} ${esc(odds(r.odds))}` : 'No recorded quote'}</span><span>${r ? 'MC '+esc(pct(r.mc_probability))+' · Mkt '+esc(pct(r.market_probability)) : ''}</span></div>
      <button type="button" class="button" data-lab-card="${i}">Player analysis</button></article>`;
  }

  function playerView(c) {
    const r = c.recorded, d = c.distribution || {}, line = r?.line;
    const p = lineProbabilities(c.histogram, line);
    const sb = (c.offers || []).filter(o => !o.exchange), ex = (c.offers || []).filter(o => o.exchange);
    const table = rows => `<div class="table-wrap"><table class="quote-table"><thead><tr><th>Book</th><th>Line</th><th>Over</th><th>Under</th><th>No-vig over</th><th>MC over</th></tr></thead><tbody>${rows.map(o => `<tr><td>${esc(book(o.book_key))}</td><td>${esc(fmt(o.line))}</td><td>${esc(odds(o.over))}</td><td>${esc(odds(o.under))}</td><td>${esc(pct(o.no_vig_over))}</td><td>${esc(pct(o.mc_over))}</td></tr>`).join('')}</tbody></table></div>`;
    const scen = Object.entries(c.scenarios || {});
    const names = {reduced_minutes:'Reduced minutes (−20%)', increased_minutes:'More minutes (+10%)', increased_usage:'Higher usage (+10%)'};
    return `<div class="page-heading"><div><h1>${esc(c.player)}</h1><p class="subtitle">${esc(statName(c.stat))} · ${esc(c.team)} vs ${esc(c.opp)} · ${esc(timeET(c.tip))}</p></div><button type="button" class="button" data-lab-view="today">← All simulations</button></div>
      <p class="small muted">Availability: ${esc(String(c.injury_status || 'Unknown').replaceAll('_',' ').toLowerCase())} · Projected minutes ${esc(fmt(c.minutes))} · Simulation ${esc(c.mc_arm === 'mc_v2' ? 'around V2' : 'around V1 (no V2 projection)')}</p>
      <section class="panel"><h2>Projection comparison</h2><div class="detail-metrics">${metric('V1 projection', fmt(c.v1_projection), true)}${metric('V2 projection', fmt(c.v2_projection))}${metric('Simulated median', fmt(d.p50))}</div></section>
      <section class="panel"><h2>Simulated outcomes</h2>${histogramSvg(c.histogram, line, `${c.player} ${statName(c.stat)}: simulated outcomes`)}
        <div class="detail-metrics">${metric('Expected range (10–90%)', `${fmt(d.p10)}–${fmt(d.p90)}`)}${metric('Mean', fmt(d.mean))}${metric(num(line) ? `Over ${fmt(line)}` : 'Over', pct(p.over))}${metric(num(line) ? `Under ${fmt(line)}` : 'Under', pct(p.under))}</div></section>
      <section class="panel"><h2>Scenario sensitivity</h2><div class="table-wrap"><table class="quote-table"><thead><tr><th>Scenario</th><th>Mean</th><th>Median</th><th>10–90%</th></tr></thead><tbody><tr><td>Baseline</td><td>${esc(fmt(d.mean))}</td><td>${esc(fmt(d.p50))}</td><td>${esc(fmt(d.p10))}–${esc(fmt(d.p90))}</td></tr>${scen.map(([k,s]) => `<tr><td>${esc(names[k] || k)}</td><td>${esc(fmt(s.mean))}</td><td>${esc(fmt(s.p50))}</td><td>${esc(fmt(s.p10))}–${esc(fmt(s.p90))}</td></tr>`).join('')}</tbody></table></div><p class="small muted">One input changed at a time on the same random draws. Confirmed absences are already in V2's projection.</p></section>
      <section class="panel"><h2>Sportsbook comparison</h2>${sb.length ? table(sb) : '<p class="small muted">No FanDuel, DraftKings or Fanatics quotes recorded.</p>'}<h3 class="review-heading">Novig (exchange reference, scored separately)</h3>${ex.length ? table(ex) : '<p class="small muted">No Novig quotes recorded.</p>'}</section>
      <section class="panel"><h2>Research verdict</h2><p>${r ? `V1 decision: <strong>${esc(r.v1_decision || 'PASS')}</strong>. Recorded ${esc(book(r.book_key))} ${esc(String(r.side||'').toLowerCase())} ${esc(fmt(r.line))} at ${esc(odds(r.odds))}: market ${esc(pct(r.market_probability))}, V1 ${esc(pct(r.v1_probability))}, Monte Carlo ${esc(pct(r.mc_probability))}.` : 'No recorded ledger quote for this market.'}</p><p>${badge(label(c.research_label), (LABELS[c.research_label]||[])[1]||'')}</p>${note}</section>`;
  }

  function scoreTable(title, rows, cols) {
    return `<h3 class="review-heading">${esc(title)}</h3><div class="table-wrap"><table class="quote-table"><thead><tr>${cols.map(c => `<th>${esc(c)}</th>`).join('')}</tr></thead><tbody>${rows.map(r => `<tr>${r.map(v => `<td>${esc(v)}</td>`).join('')}</tr>`).join('')}</tbody></table></div>`;
  }

  function armsBlock(e, title) {
    if (!e || !e.scores) return `<p class="small muted">${esc(title)}: ${fmt(e?.rows || 0,0)} graded rows so far.</p>`;
    const names = {market:'Market no-vig', v1:'V1 recorded', mc_v1:'MC around V1', mc_v2:'MC around V2'};
    const s = e.scores, d = e.calibration_diagnostics || {}, roi = e.hypothetical_roi || {}, sg = e.strict_gate || {}, lg = e.calibration_gate || {};
    const r5 = a => (roi[a] || []).find(x => x.threshold === 0.05);
    return `<p class="small muted">${esc(title)}: ${fmt(e.rows,0)} graded props over ${fmt(e.dates,0)} dates.</p>`
      + scoreTable('Probability accuracy', Object.keys(s).map(a => [names[a] || a, fixed(s[a].brier), fixed(s[a].log_loss), a==='market' ? '—' : signed(s[a].log_loss_minus_market,4), fixed(d[a]?.slope,2)]), ['Method','Brier','Log loss','vs market','Calibration slope'])
      + scoreTable('Profitability (hypothetical, EV ≥ 5% at recorded odds)', Object.keys(s).filter(a => a !== 'market').map(a => [names[a] || a, fmt(r5(a)?.bets,0), pct(r5(a)?.roi), r5(a)?.roi_ci95 ? `${pct(r5(a).roi_ci95[0])} to ${pct(r5(a).roi_ci95[1])}` : '—']), ['Method','Bets','ROI','95% range'])
      + scoreTable('Promotion gates', Object.keys(sg).map(a => [names[a] || a, lg[a]?.status || '—', sg[a]?.status || '—', Object.entries(sg[a]?.checks || {}).filter(([,v]) => !v).map(([k]) => k.replaceAll('_',' ')).join(', ') || 'none']), ['Method','Live gate','Strict gate','Failed checks']) + '<p class="small muted">Live gate: the production rule (hold-out no worse than the market). Strict gate: fixed 2026-11-15, needs a margin, a confidence interval, walk-forward stability and at least 7 hold-out dates. Neither gives Monte Carlo bet authority.</p>';
  }

  function validationView() {
    const L = lab(), h = L.historical || {}, seasons = Object.keys(h.seasons || {});
    const stats = ['pts','reb','ast','threes','pra'];
    let html = nav() + note + '<p class="small">Prediction accuracy, probability accuracy and profitability are reported separately. A closer mean alone earns a probability model no credit.</p>';
    html += '<section class="panel"><h2>Prediction accuracy</h2>';
    const mae = L.v2_mae || {};
    html += Object.keys(mae).length ? scoreTable('Forward ledger: V1 vs V2 projection MAE', Object.entries(mae).filter(([,v]) => v && v.rows !== undefined).map(([k,v]) => [k, fmt(v.rows,0), fmt(v.v1_mae,2), fmt(v.v2_mae,2)]), ['Stat','Rows','V1 MAE','V2 MAE']) : '<p class="small muted">No graded rows with both projections yet.</p>';
    for (const s of seasons) html += scoreTable(`Historical ${s}-${String(Number(s)+1).slice(2)} MAE`, stats.map(st => { const m = h.seasons[s][st]?.mae || {}; return [st.toUpperCase(), fixed(m.v1,3), fixed(m.v2,3), fixed(m.v2_mc_mean,3)]; }), ['Stat','V1','V2','V2 + MC mean']);
    html += '</section><section class="panel"><h2>Probability accuracy · historical</h2><p class="small muted">Pseudo-lines from each player’s prior 5-game average (no sportsbook prices). Fit on 2023-24, tested on later seasons.</p>';
    for (const s of seasons) html += scoreTable(`${s}-${String(Number(s)+1).slice(2)}`, stats.map(st => { const x = h.seasons[s][st] || {}; const pit = x.pit_deciles_v2_mc || []; return [st.toUpperCase(), fixed(x.brier?.v1_normal), fixed(x.brier?.v2_normal), fixed(x.brier?.v2_mc), fixed(x.log_loss?.v1_normal), fixed(x.log_loss?.v2_mc), pct(x.coverage_80?.v2_mc), pit.length ? `${pct(pit[0])} / ${pct(pit[pit.length-1])}` : '—']; }), ['Stat','Brier V1','Brier V2','Brier V2+MC','Log loss V1','Log loss V2+MC','80% coverage','PIT ends']);
    html += '</section><section class="panel"><h2>Recorded sportsbook prices · retrospective</h2><p class="small muted">Rebuilt pregame from archived data, not forecasts recorded at the time.</p>' + armsBlock(L.phase3, 'FanDuel, DraftKings, Fanatics') + '</section>';
    const f = L.forward || {};
    html += `<section class="panel"><h2>Forward shadow forecasts</h2><p class="small muted">Recorded before tip and graded after. ${f.status === 'GRADED' ? `${fmt(f.graded_rows,0)} graded rows, ${fmt(f.dates,0)} dates (${esc(f.first_date)} to ${esc(f.last_date)}).` : 'No graded shadow forecasts yet.'}</p>` + armsBlock(f.sportsbooks?.all_arms_common_rows, 'Sportsbooks') + armsBlock(f.exchanges_novig?.v1_anchored_all_rows, 'Novig (separate)') + '</section>';
    return html;
  }

  function render(view) {
    const t = lab().today;
    view.innerHTML = state.view === 'player' && state.card !== null && t?.cards?.[state.card] ? playerView(t.cards[state.card])
      : state.view === 'validation' ? validationView() : todayView();
  }

  function open(view) {
    state.view = 'today'; state.card = null;
    document.querySelectorAll('[data-tab]').forEach(b => b.setAttribute('aria-selected', 'false'));
    render(view); window.scrollTo({top:0}); view.focus({preventScroll:true});
  }

  document.addEventListener('click', ev => {
    const view = document.getElementById('view');
    const v = ev.target.closest('[data-lab-view]'), c = ev.target.closest('[data-lab-card]');
    if (v) { state.view = v.dataset.labView; state.card = null; render(view); window.scrollTo({top:0}); }
    else if (c) { state.view = 'player'; state.card = Number(c.dataset.labCard); render(view); window.scrollTo({top:0}); }
  });
  const rerenderCards = () => { const view = document.getElementById('view'); const pos = document.activeElement?.id; render(view); if (pos) { const el = document.getElementById(pos); if (el) { el.focus(); if (el.setSelectionRange && el.type === 'search') el.setSelectionRange(el.value.length, el.value.length); } } };
  document.addEventListener('input', ev => { if (ev.target.id === 'lab-search') { state.search = ev.target.value; rerenderCards(); } });
  document.addEventListener('change', ev => {
    const map = {'lab-sort':'sort','lab-stat':'stat','lab-label':'label'};
    if (map[ev.target.id]) { state[map[ev.target.id]] = ev.target.value; rerenderCards(); }
  });
  window.V2Lab = {open};
}());
