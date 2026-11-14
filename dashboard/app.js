(function () {
  'use strict';
  const esc = value => String(value ?? '').replace(/[&<>"']/g, x => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[x]));
  const numeric = value => value !== null && value !== undefined && value !== '' && Number.isFinite(Number(value));
  const fmt = (value, digits=1) => numeric(value) ? Number(value).toLocaleString('en-US',{minimumFractionDigits:0,maximumFractionDigits:digits}) : '—';
  const signed = (value, digits=1) => numeric(value) ? (Number(value)>0?'+':'')+fmt(value,digits) : '—';
  const odds = value => numeric(value) ? (Number(value)>0?'+':'')+String(Number(value)) : '—';
  const pct = value => numeric(value) ? fmt(Number(value)*100,1)+'%' : '—';
  const timeET = value => {
    const d = new Date(value || '');
    return Number.isFinite(d.getTime()) ? new Intl.DateTimeFormat('en-US',{timeZone:'America/New_York',hour:'numeric',minute:'2-digit'}).format(d)+' ET' : 'Time unavailable';
  };
  const dayLabel = value => {
    const d = new Date(String(value)+'T12:00:00Z');
    return Number.isFinite(d.getTime()) ? new Intl.DateTimeFormat('en-US',{timeZone:'America/New_York',weekday:'short',month:'short',day:'numeric'}).format(d) : 'Date unavailable';
  };
  const started = tip => Boolean(tip && new Date(tip).getTime() <= Date.now());
  const bookName = key => ({fanduel:'FanDuel',draftkings:'DraftKings',fanatics:'Fanatics'}[key] || key || 'Unavailable');
  const shortBook = key => ({fanduel:'FD',draftkings:'DK',fanatics:'FAN'}[key] || '—');
  const statName = stat => ({PTS:'Points',REB:'Rebounds',AST:'Assists',THREES:'Threes',PRA:'Pts + reb + ast'}[stat] || stat);
  // First-quarter record this season (W-L-T); context only, not a model input.
  const q1Record = (team,inline) => { const q=team?.q1; if(!q||!(q.q1_wins+q.q1_losses+q.q1_ties)) return ''; const text=`1Q ${q.q1_wins}-${q.q1_losses}-${q.q1_ties}`; return inline?`${esc(text)} · `:`<span class="small muted">${esc(text)}</span>`; };
  const projectedSpread = game => !numeric(game.spread?.pred) ? 'Unavailable' : Math.abs(Number(game.spread.pred))<.5 ? "Pick’em" : `${Number(game.spread.pred)>0?(game.home.abbr||game.home.name):(game.away.abbr||game.away.name)} -${fmt(Math.abs(Number(game.spread.pred)))}`;
  const quoteFor = (row, book='') => {
    if(row.decision==='BET' && (!book || row.quote?.book_key===book)) return row.quote;
    const offers=(row.offers||[]).filter(q=>(!book||q.book_key===book)&&q.line===row.line&&q.side===row.direction);
    return offers.length ? offers.reduce((a,b)=>Number(a.odds)>Number(b.odds)?a:b) : (!book ? row.quote : null);
  };
  const filterProps = (rows, filters={}) => {
    const search=String(filters.search||'').trim().toLowerCase();
    const result=rows.filter(r=>(!search || `${r.player} ${r.team} ${r.opp}`.toLowerCase().includes(search))
      && (!filters.game||r.game===filters.game) && (!filters.stat||r.stat===filters.stat)
      && (!filters.direction||r.direction===filters.direction) && (!filters.decision||r.decision===filters.decision)
      && (!filters.book||Boolean(quoteFor(r,filters.book))));
    if(filters.sort==='gap'&&filters.stat) return result.sort((a,b)=>Math.abs(b.gap??0)-Math.abs(a.gap??0));
    return result.sort((a,b)=>filters.sort==='time' ? String(a.tip).localeCompare(String(b.tip))||String(a.player).localeCompare(String(b.player)) : String(a.player).localeCompare(String(b.player))||String(a.stat).localeCompare(String(b.stat)));
  };
  const groupedQuotes = offers => {
    const grouped=new Map();
    for(const q of offers||[]) {
      const key=q.book_key+'|'+q.line;
      const row=grouped.get(key)||{book_key:q.book_key,line:q.line};
      if(q.side==='OVER'||q.side==='UNDER') row[q.side]=q.odds;
      grouped.set(key,row);
    }
    return [...grouped.values()].sort((a,b)=>String(a.book_key).localeCompare(String(b.book_key))||a.line-b.line);
  };
  if(typeof module!=='undefined' && module.exports) {
    module.exports={esc,numeric,fmt,signed,odds,pct,timeET,dayLabel,quoteFor,filterProps,groupedQuotes};
    return;
  }

  const D=window.DATA||{}, props=D.props||[], games=D.games||[];
  const state={tab:'today',filters:{search:'',game:'',stat:'',book:'',direction:'',decision:'',sort:'player'},limit:25,game:'',resultDate:'',resultBook:'',resultOutcome:''};
  const view=document.getElementById('view');
  const button=(text,action,extra='',primary=false)=>`<button type="button" class="button ${primary?'primary':''}" data-action="${esc(action)}" ${extra}>${esc(text)}</button>`;
  const badge=(text,kind='')=>`<span class="badge ${kind}">${esc(text)}</span>`;
  const heading=(title,subtitle='',aside='')=>`<div class="page-heading"><div><h1>${esc(title)}</h1>${subtitle?`<p class="subtitle">${esc(subtitle)}</p>`:''}</div>${aside}</div>`;
  const empty=(title,message,action='')=>`<div class="empty-state"><h2>${esc(title)}</h2><p>${esc(message)}</p>${action}</div>`;
  const metric=(label,value,accent=false)=>`<div><span class="metric-label">${esc(label)}</span><span class="metric-value ${accent?'accent':''}">${esc(value)}</span></div>`;
  const gatePending=()=>{
    const market=D.calibration?.markets?.prop_price;
    return !market || market.status==='MARKET_ONLY' || market.validation==='INSUFFICIENT_DATED_EVIDENCE';
  };
  const options=(items,selected)=>items.map(([value,label])=>`<option value="${esc(value)}" ${value===selected?'selected':''}>${esc(label)}</option>`).join('');
  function progress(label,current,target) {
    const amount=Number(current)||0, maximum=Number(target)||1;
    return `<div class="progress-item"><div class="progress-text"><span>${esc(label)}</span><span>${fmt(amount,0)} / ${fmt(maximum,0)}</span></div><div class="progress-track" role="progressbar" aria-label="${esc(label)}" aria-valuenow="${amount}" aria-valuemin="0" aria-valuemax="${maximum}"><div class="progress-fill" style="width:${Math.max(0,Math.min(100,amount/maximum*100))}%"></div></div></div>`;
  }
  function freshness() {
    const waiting=D.slate_status==='AWAITING_PREDICTIONS', stamp=D.generated||D.last_successful_prediction_at;
    document.getElementById('slate-freshness').innerHTML=`<strong class="${waiting?'waiting':''}">${waiting?'Awaiting today’s predictions':esc(dayLabel(D.date))}</strong><span title="${esc(stamp||'No successful refresh')}">${D.generated?'Updated':'Last refresh'} ${esc(stamp?timeET(stamp):'unavailable')}</span>`;
    const h=D.data_health||{};
    document.getElementById('summary-strip').innerHTML=[[games.length,'Games'],[h.modeled_players||0,'Players'],[h.quote_rows||0,'Prop quotes'],[(D.best_bets||[]).length,'Qualified']].map(([value,label])=>`<div class="summary-item"><div class="summary-value ${label==='Qualified'&&value?'accent':''}">${fmt(value,0)}</div><div class="summary-label">${label}</div></div>`).join('');
  }
  function calibrationPanel() {
    if(D.slate_status==='AWAITING_PREDICTIONS') return empty('Awaiting today’s predictions','Current-slate cards will appear after the next successful refresh. Previous-slate cards stay in Results.');
    if((D.best_bets||[]).length) return '';
    if(!D.data_health?.quote_rows) return empty('Awaiting sportsbook quotes','Current prices will appear when the market feed provides dated sportsbook offers.');
    if(!props.length) return empty('Awaiting player projections','Sportsbook quotes are loaded. Player projections will appear after the next successful model refresh.');
    const pending=gatePending();
    const blocks=Object.entries(D.qualification_audit?.rejection_counts||{})
      .filter(([key])=>key!=='QUALIFIED_GAME_MARKET'&&key!=='PRICED_PROP_SIGNAL')
      .sort((a,b)=>b[1]-a[1]).slice(0,3)
      .map(([key,n])=>`${key.replaceAll('_',' ').toLowerCase()}: ${n}`).join(' · ');
    return `<section class="panel calibration-panel"><div>${badge(pending?'Validation in progress':'Research mode','warning')}<h2>${pending?'Calibration in progress':'No qualified bets in this snapshot'}</h2><p>${pending?'Projections and prices are available. BET signals unlock when later-date validation passes.':'The model has not identified a qualifying play at the recorded lines and prices.'}</p>${blocks?`<p class="small muted">Leading blocks: ${esc(blocks)}</p>`:''}</div>${button('Explore player props','view-props','',true)}</section>`;
  }
  function gameQuoteTable(game) {
    const quotes=game.market_quotes||[];
    if(!quotes.length) return '<p class="snapshot-note">Exact sportsbook quotes are unavailable for this snapshot. Model projections remain visible.</p>';
    const keys=[...new Set(quotes.map(q=>q.book_key))].sort();
    const cell=(book,market,side)=>{
      const q=quotes.find(x=>x.book_key===book&&x.market===market&&x.side===side);
      return q?`${esc(market==='SPREAD'?signed(q.line):fmt(q.line))}<small>${esc(odds(q.price))}</small>`:'—';
    };
    return `<p class="review-heading">Spread · exact handicap and price</p><table class="quote-table"><thead><tr><th>Book</th><th>${esc(game.away.abbr||'Away')}</th><th>${esc(game.home.abbr||'Home')}</th></tr></thead><tbody>${keys.map(book=>`<tr><td>${esc(shortBook(book))}</td><td>${cell(book,'SPREAD','AWAY')}</td><td>${cell(book,'SPREAD','HOME')}</td></tr>`).join('')}</tbody></table><p class="review-heading">Total · exact line and price</p><table class="quote-table"><thead><tr><th>Book</th><th>Over</th><th>Under</th></tr></thead><tbody>${keys.map(book=>`<tr><td>${esc(shortBook(book))}</td><td>${cell(book,'TOTAL','OVER')}</td><td>${cell(book,'TOTAL','UNDER')}</td></tr>`).join('')}</tbody></table><p class="snapshot-note">Quotes captured ${esc(timeET(game.priced_at))}. Each price belongs to the line directly above it.</p>`;
  }
  function marketNote(game,kind) {
    const model=kind==='SPREAD'?game.spread:game.totals;
    const q=model?.selected_quote||model?.reference_quote||(game.market_quotes||[]).find(x=>x.market===kind);
    if(!q) return 'Book quote unavailable';
    const label=kind==='SPREAD'?(q.side==='HOME'?game.home.abbr:game.away.abbr)+' '+signed(q.line):q.side+' '+fmt(q.line);
    return `${shortBook(q.book_key)} · ${label} (${odds(q.price)})`;
  }
  function gameCard(game,expanded=false) {
    const index=games.indexOf(game), isStarted=started(game.tip);
    const bet=(D.best_bets||[]).some(b=>b.game===game.display_name);
    return `<article class="game-card"><div class="game-top"><span>${esc(timeET(game.tip))}</span>${badge(isStarted?'Started · pregame snapshot':bet?'Qualified bet':'Pregame',bet&&!isStarted?'bet':'')}</div><div class="team-row"><div class="team-name"><span class="team-code">${esc(game.away.abbr||'AWY')}</span><span>${esc(game.away.name)}</span></div>${q1Record(game.away)}</div><div class="team-row"><div class="team-name"><span class="team-code">${esc(game.home.abbr||'HME')}</span><span>${esc(game.home.name)}</span></div><span class="small muted">${q1Record(game.home,true)}HOME</span></div><div class="game-models"><div>${metric('Model spread',projectedSpread(game),true)}<p class="market-note">${esc(marketNote(game,'SPREAD'))}</p></div><div>${metric('Model total',fmt(game.totals?.pred),true)}<p class="market-note">${esc(marketNote(game,'TOTAL'))}</p></div></div><details class="details-link" ${expanded?'open':''}><summary>Compare books &amp; view game props</summary><div class="details-content">${gameQuoteTable(game)}${numeric(game.spread?.shadow_anchored_margin)?`<p class="snapshot-note">Shadow spread: ${esc(projectedSpread({spread:{pred:game.spread.shadow_anchored_margin},home:game.home,away:game.away}))}. Market-anchored research scenario only, not a bet.</p>`:''}${button('Player props','game-props',`data-game-index="${index}"`)}</div></details>${window.LinemateReview?.render(game,D.date,index)||''}</article>`;
  }
  function qualifiedBets() {
    const bets=(D.best_bets||[]).filter(b=>!started(b.tip));
    if(!bets.length) return '';
    return `<div class="section-heading"><h2>Qualified picks</h2>${badge('Forward test','bet')}</div><div class="game-grid">${bets.map(b=>`<article class="panel"><div class="small muted">${esc(b.game)}</div><h2 class="inline-title">${esc(b.play)}</h2><p class="subtitle">${esc(bookName(b.book_key))} · ${esc(odds(b.market_odds))} · ${esc(fmt(b.units))}u</p></article>`).join('')}</div>`;
  }
  function renderToday() {
    const previews=props.filter(r=>r.direction&&r.line!==null&&!r.availability_block&&!started(r.tip)).filter((r,i,all)=>all.findIndex(x=>x.player===r.player)===i).slice(0,3);
    view.innerHTML=heading('Your daily court view',`${fmt(games.length,0)} games · ${fmt(D.data_health?.modeled_players||0,0)} players projected`,badge('Forward testing','research'))+calibrationPanel()+qualifiedBets()+`<div class="section-heading"><h2>Today’s slate</h2><button data-action="view-games" type="button">All games →</button></div><div class="game-grid">${games.map(g=>gameCard(g)).join('')||empty('No current games','The current slate will appear after a successful refresh.')}</div>`+(previews.length?`<div class="section-heading"><h2>Research previews</h2><span class="small muted">Projections to explore</span></div><div class="watch-grid">${previews.map(row=>`<article class="watch-card">${badge('Research','research')}<p><strong>${esc(row.player)}</strong><br><span class="small muted">${esc(statName(row.stat))}</span></p>${metric('Projection',fmt(row.projection),true)}<p class="subtitle">Model lean ${esc(row.direction.toLowerCase())} ${esc(fmt(row.line))}</p>${button('View player','player-props',`data-player="${esc(row.player)}"`)}</article>`).join('')}</div>`:'');
  }
  function renderGames() {
    view.innerHTML=heading('Games','Model projections, exact book quotes and your private Linemate review.')+`<div class="filters"><label class="small muted" for="game-select">Matchup</label><select class="search-input" id="game-select">${options([['','All games'],...games.map(g=>[g.display_name,g.display_name])],state.game)}</select></div><div class="game-grid" id="games-list">${games.filter(g=>!state.game||g.display_name===state.game).map(g=>gameCard(g,Boolean(state.game))).join('')||empty('No current games','Waiting for a dated slate.')}</div>`;
  }
  function propComparison(row) {
    const grouped=groupedQuotes(row.offers);
    if(!grouped.length) return '<p class="snapshot-note">Exact book comparisons are unavailable for this snapshot.</p>';
    return `<table class="quote-table"><thead><tr><th>Book</th><th>Line</th><th>Over</th><th>Under</th></tr></thead><tbody>${grouped.map(q=>`<tr><td>${esc(shortBook(q.book_key))}</td><td>${esc(fmt(q.line))}</td><td>${esc(odds(q.OVER))}</td><td>${esc(odds(q.UNDER))}</td></tr>`).join('')}</tbody></table><p class="snapshot-note">Prices are recorded quotes. Compare the same line and side. Captured ${esc(timeET(row.offers[0]?.captured_at))}.</p>`;
  }
  function linematePropUrl(row) {
    const game=games.find(g=>g.display_name===row.game);
    return game?window.LinemateReview.matchupUrl(game,D.date):'https://linemate.io/nba';
  }
  function propCard(row) {
    const quote=quoteFor(row,state.filters.book), isStarted=started(row.tip);
    const blocked=row.availability_block;
    const blockText=String(row.block_reason||'availability').replaceAll('_',' ').toLowerCase();
    const stateLabel=isStarted?'Pregame record':blocked?'Availability blocked':row.decision==='BET'?'Qualified BET':'Research';
    return `<details class="prop-card"><summary><div class="prop-header"><span class="player-name">${esc(row.player)}</span>${badge(stateLabel,row.decision==='BET'&&!isStarted?'bet':blocked?'warning':'research')}</div><div class="prop-meta">${esc(statName(row.stat))} · ${esc(row.team)} vs ${esc(row.opp)} · ${esc(timeET(row.tip))}</div><div class="prop-numbers">${metric('Model',fmt(row.projection),true)}${metric('Book line',fmt(row.line))}${metric('Gap',signed(row.gap))}</div><div class="prop-bottom"><span>${row.direction?'Model lean '+esc(row.direction.toLowerCase()):'No directional lean'}</span><span class="quote-price">${quote?esc(shortBook(quote.book_key)+' '+odds(quote.odds)):'Price unavailable'}</span></div><div class="expand-hint">Minutes, recent averages &amp; all sportsbook lines ↓</div></summary><div class="prop-expanded"><div class="detail-metrics">${metric('Expected min',fmt(row.minutes))}${metric('Last 5 avg',fmt(row.last5))}${metric('Last 10 avg',fmt(row.last10))}</div><div class="small muted">${esc((row.role||'Role unavailable').replaceAll('_',' '))} · Rotation ${esc((row.rotation_risk||'unknown').toLowerCase())}<br>Availability: ${esc(row.injury_status==='ACTIVE_OR_UNKNOWN'?'No confirmed restriction recorded':(row.injury_status||'Unknown').replaceAll('_',' '))}</div>${row.minutes_delta?`<p class="small muted">Opportunity adjustment: ${esc(signed(row.minutes_delta))} minutes${row.vacancy_sources?' · '+esc(row.vacancy_sources):''}</p>`:''}${row.minutes_guard?`<p class="small muted">Minutes risk: projected ${esc(fmt(row.minutes))} vs last-5 ${esc(fmt(row.recent_minutes))}. Stabilized minutes ${esc(fmt(row.shadow_stabilized_minutes))} is shadow-only.</p>`:''}<p class="info-note">${blocked?'Signal blocked: '+blockText+'.':row.decision==='BET'?'This play passed the recorded decision rules. Verify the current book line and price.':gatePending()?'Calibration pending. The gap is a projection comparison; this is a research observation.':'This observation did not pass all qualification rules.'}</p>${row.decision==='BET'?`<p class="small">Recorded model probability ${esc(pct(row.model_probability))} · EV ${esc(pct(row.ev))}</p>`:''}${propComparison(row)}<p class="small muted">Compare recent form, home/away and teammate splits in <a href="${esc(window.LinemateReview?linematePropUrl(row):'https://linemate.io/nba')}" target="_blank" rel="noopener noreferrer">Linemate ↗</a>. Save observations under this matchup in Games.</p></div></details>`;
  }
  function renderProps() {
    const f=state.filters;
    const filtersOpen=document.querySelector('.filter-toggle')?.open??window.matchMedia('(min-width:900px)').matches;
    const select=(id,label,items)=>`<label for="${id}">${esc(label)}<select id="${id}">${options(items,f[id.replace('filter-','')])}</select></label>`;
    view.innerHTML=heading('Player props','Explore projections, minutes and exact book prices.',badge(`${props.length} projections`,'research'))+`<section class="filters" aria-label="Prop filters"><div class="search-wrap"><input class="search-input" id="player-search" type="search" placeholder="Search a player or team" aria-label="Search a player or team" value="${esc(f.search)}">${button('Reset','reset-filters','',false)}</div><details class="filter-toggle" ${filtersOpen?'open':''}><summary>Filters · game, stat, book and model lean</summary><div class="filter-grid">${select('filter-game','Game',[['','All games'],...games.map(g=>[g.display_name,g.display_name])])}${select('filter-stat','Stat',[['','All stats'],...['PTS','REB','AST','THREES','PRA'].map(s=>[s,statName(s)])])}${select('filter-book','Sportsbook',[['','All three books'],...['fanduel','draftkings','fanatics'].map(b=>[b,bookName(b)])])}${select('filter-direction','Model lean',[['','Over & under'],['OVER','Over'],['UNDER','Under']])}${select('filter-decision','Status',[['','All projections'],['BET','Qualified only'],['RESEARCH','Research only']])}${select('filter-sort','Sort',[['player','Player A–Z'],['time','Game time'],...(f.stat?[['gap','Largest gap in this stat']]:[])])}</div></details></section><div class="filter-footer"><span id="props-count" role="status"></span><span>Tap a player for detail</span></div><div id="props-list" class="prop-grid"></div><div id="props-more"></div>`;
    renderPropRows();
  }
  function renderPropRows() {
    const rows=filterProps(props,state.filters), visible=rows.slice(0,state.limit);
    const players=new Set(rows.map(r=>r.player)).size;
    document.getElementById('props-count').textContent=`${rows.length} projection${rows.length===1?'':'s'} · ${players} player${players===1?'':'s'}`;
    document.getElementById('props-list').innerHTML=visible.map(propCard).join('')||empty(props.length?'No matching projections':'Player projections unavailable',state.filters.decision==='BET'&&gatePending()?'Calibration is pending. Choose All projections to explore research observations.':props.length?'Try another player, game, stat or book.':'Awaiting the next current-slate prediction refresh.');
    document.getElementById('props-more').innerHTML=rows.length>state.limit?`<button type="button" class="button load-more" data-action="more-props">Show 25 more (${visible.length} of ${rows.length})</button>`:'';
  }
  function chart(rows) {
    const byDay=new Map();for(const r of rows) byDay.set(r.date,(byDay.get(r.date)||0)+Number(r.profit_units||0));
    const daily=[...byDay.entries()].sort((a,b)=>a[0].localeCompare(b[0]));
    if(!daily.length)return '<p class="small muted">The chart appears when priced BET results are available.</p>';
    let cumulative=0;const values=[0,...daily.map(([,profit])=>cumulative+=profit)];
    const low=Math.min(0,...values),high=Math.max(1,...values),x=i=>30+i/(values.length-1)*560,y=v=>115-(v-low)/(high-low)*90;
    const path=values.map((v,i)=>`${i?'L':'M'}${x(i).toFixed(1)},${y(v).toFixed(1)}`).join(' ');
    return `<svg class="chart" viewBox="0 0 620 150" role="img" aria-label="Cumulative flat one-unit forward-test profit ${esc(signed(cumulative,2))} units"><line class="chart-zero" x1="30" x2="590" y1="${y(0)}" y2="${y(0)}"/><path class="chart-line" d="${path}"/><text class="chart-label" x="30" y="142">${esc(daily[0][0].slice(5))}</text><text class="chart-label" x="590" y="142" text-anchor="end">${esc(daily[daily.length-1][0].slice(5))}</text><text class="chart-label" x="590" y="16" text-anchor="end">${esc(signed(cumulative,2))}u</text></svg>`;
  }
  function modelStatus() {
    const evidence=D.readiness?.forward_evidence||{}, policy=D.readiness?.promotion_policy||{}, calibration=D.calibration?.markets?.prop_price||{};
    return `<details id="model-status" class="model-status"><summary>Model status &amp; validation</summary><p>Current model evidence is tracked separately from the historical forward-test record.</p>${progress('Current priced BETs',evidence.bet_rows,policy.min_forward_bets||200)}${progress('Current prop BETs',evidence.prop_bets,policy.min_prop_forward_bets||100)}${progress('Matched closing lines',evidence.clv_matched,policy.min_matched_closes||100)}<p>Calibration observations: ${fmt(calibration.graded||0,0)}. Validation uses earlier dates for fitting and later dates for testing.</p>${progress('Calibration training observations',calibration.training_rows,D.calibration?.min_graded||200)}${progress('Calibration holdout observations',calibration.holdout_rows,50)}<p>Data: ${esc(D.readiness?.data_model_healthy?'Ready':'Awaiting healthy inputs')} · Grading: ${esc((D.reconciliation?.status||'Unknown').toLowerCase())}</p></details>`;
  }

  function renderDiagnostics() {
    const report=D.yesterday_diagnostics||{}, sums=report.summary||{}, trailing=D.trailing_diagnostics||{};
    if(report.status!=='COMPLETE'&&report.status!=='PARTIAL_RESULTS')
      return '<section class="panel"><h2>Yesterday · model vs market</h2><p class="small muted">Waiting for dated, final box scores and pregame forecasts.</p></section>';
    const statRows=['PTS','REB','AST','THREES','PRA','MINUTES'].map(key=>{
      const r=report.stats?.[key]||{};
      return `<tr><td>${esc(key)}</td><td>${esc(fmt(r.sample,0))}</td><td>${esc(fmt(r.mae,2))}</td><td>${r.priced_directions?`${esc(fmt(r.directions_correct,0))}/${esc(fmt(r.priced_directions,0))}`:'—'}</td></tr>`;
    }).join('');
    const gameRows=(report.games||[]).map(g=>`<tr><td>${esc(g.game)}</td><td>${esc(fmt(g.model_margin))} / ${esc(fmt(g.actual_margin))}</td><td>${esc(fmt(g.model_total))} / ${esc(fmt(g.actual_total))}</td></tr>`).join('');
    const beatsMarket=numeric(sums.margin_mae_model)&&numeric(sums.margin_mae_market)
      ?Number(sums.margin_mae_model)<Number(sums.margin_mae_market):false;
    return `<section class="panel"><h2>Yesterday · model vs market ${badge('Research only','research')}</h2><p class="subtitle">${esc(dayLabel(report.date))} · ${esc(fmt(sums.completed_games,0))} final games of ${esc(fmt(sums.scheduled_games,0))}. No research lean is counted as a bet.</p><div class="detail-metrics" style="margin-top:16px">${metric('Winner calls',fmt(sums.winner_correct,0)+'/'+fmt(sums.winner_sample,0))}${metric('Spread MAE · model',fmt(sums.margin_mae_model,2))}${metric('Spread MAE · book',fmt(sums.margin_mae_market,2))}${metric('Total MAE · model',fmt(sums.total_mae_model,2))}${metric('Total MAE · book',fmt(sums.total_mae_market,2))}</div><p class="snapshot-note">${beatsMarket?'Model beat the posted spread benchmark this day.':'Book spread benchmark matched or beat the model this day.'} Shadow spread (35% model + 65% book): ${esc(fmt(sums.margin_mae_shadow,2))} MAE. Shadow forecasts are not approved betting lines.</p>${trailing.completed_games?`<p class="snapshot-note">Trailing ${esc(fmt(trailing.lookback_days,0))}-day window · ${esc(fmt(trailing.completed_games,0))} completed games: spread MAE model ${esc(fmt(trailing.margin_mae_model,2))}, market ${esc(fmt(trailing.margin_mae_market,2))}, shadow ${esc(fmt(trailing.margin_mae_shadow,2))}. Total MAE model ${esc(fmt(trailing.total_mae_model,2))}, market ${esc(fmt(trailing.total_mae_market,2))}.</p>`:''}<details class="details-link"><summary>Game-by-game actuals and prop accuracy</summary><div class="details-content"><table class="quote-table"><thead><tr><th>Game</th><th>Margin model / final</th><th>Total model / final</th></tr></thead><tbody>${gameRows}</tbody></table><p class="review-heading">Player stat forecast error</p><table class="quote-table"><thead><tr><th>Market</th><th>Players</th><th>MAE</th><th>Direction</th></tr></thead><tbody>${statRows}</tbody></table></div></details></section>`;
  }
  function renderResults() {
    const perf=D.forward_performance||{}, clv=D.clv_report||{}, rows=D.results||[];
    const record=numeric(perf.hit_rate)?`${perf.wins||0}–${perf.losses||0}`:'—';
    const stats=[['Record',record],['Hit rate',pct(perf.hit_rate)],['Flat 1u profit',signed(perf.profit_units,2)+'u'],['ROI',pct(perf.roi)]];
    const days=[...new Set(rows.map(r=>r.date))].sort().reverse();
    view.innerHTML=heading('Results','Historical forward test · priced BET decisions only.',badge('Recorded prices','research'))+`<div class="results-grid">${stats.map(([label,value])=>`<div class="result-stat">${metric(label,value)}</div>`).join('')}</div>${renderDiagnostics()}<div class="results-layout"><section class="panel"><h2 class="inline-title">Flat 1u test performance</h2>${chart(rows)}<p class="snapshot-note">Each graded bet risks 1u at its recorded sportsbook price. Consensus-priced game entries are excluded.</p></section><section class="panel"><h2 class="inline-title">Closing-line evidence</h2><p class="subtitle">Did the entry beat the same sportsbook’s closing quote?</p><div class="detail-metrics" style="margin-top:16px">${metric('Matched',fmt(clv.matched_closes||0,0))}${metric('Beat close',pct(clv.beat_close_rate))}${metric('Line CLV',fmt(clv.avg_line_clv,2))}</div><p class="small muted">Pending observations: ${fmt(perf.pending_rows||0,0)}<br>Excluded unpriced BET entries: ${fmt(perf.excluded_unpriced_bets||0,0)}</p></section></div><section class="panel">${modelStatus()}</section><div class="section-heading"><h2>Graded bet history</h2><span class="small muted">${rows.length} results</span></div><section class="filters"><div class="filter-grid"><label for="result-date">Date<select id="result-date">${options([['','All graded dates'],[D.yesterday,'Yesterday · '+dayLabel(D.yesterday)],...days.filter(d=>d!==D.yesterday).map(d=>[d,dayLabel(d)])],state.resultDate)}</select></label><label for="result-book">Sportsbook<select id="result-book">${options([['','All books'],...['fanduel','draftkings','fanatics'].map(b=>[b,bookName(b)])],state.resultBook)}</select></label><label for="result-outcome">Outcome<select id="result-outcome">${options([['','All outcomes'],['WIN','Win'],['LOSS','Loss'],['PUSH','Push']],state.resultOutcome)}</select></label></div></section><section class="panel" id="result-history"></section>`;
    renderResultRows();
  }
  function renderResultRows() {
    const rows=(D.results||[]).filter(r=>(!state.resultDate||r.date===state.resultDate)&&(!state.resultBook||r.sportsbook_key===state.resultBook)&&(!state.resultOutcome||r.outcome===state.resultOutcome));
    document.getElementById('result-history').innerHTML=rows.map(r=>`<article class="result-row"><div><strong>${esc(r.player?`${r.player} · ${r.stat} ${r.side} ${fmt(r.line)}`:`${r.game} · ${r.market} ${r.side} ${fmt(r.line)}`)}</strong><p class="subtitle">${esc(dayLabel(r.date))} · ${esc(bookName(r.sportsbook_key))} ${esc(odds(r.odds))} · Actual ${esc(fmt(r.actual))}</p></div><div><span class="badge ${r.outcome==='WIN'?'bet':''}">${esc(r.outcome)}</span><div class="small ${r.outcome.toLowerCase()}">${esc(signed(r.profit_units,2))}u</div></div></article>`).join('')||empty('No priced BET results for this filter','Research observations and excluded consensus entries do not count as graded bets.');
  }
  function renderTab(tab,focus=false) {
    state.tab=tab;
    for(const button of document.querySelectorAll('[data-tab]')){const selected=button.dataset.tab===tab;button.setAttribute('aria-selected',String(selected));button.tabIndex=selected?0:-1;}
    view.setAttribute('aria-labelledby','tab-'+tab);
    ({today:renderToday,games:renderGames,props:renderProps,results:renderResults}[tab]||renderToday)();
    window.scrollTo({top:0,behavior:'instant'});
    if(focus)view.focus({preventScroll:true});
  }
  document.addEventListener('click',event=>{
    const control=event.target.closest('[data-tab],[data-action]');if(!control)return;
    if(control.dataset.tab){renderTab(control.dataset.tab);return;}
    switch(control.dataset.action){
      case 'view-props':renderTab('props',true);break;
      case 'view-games':state.game='';renderTab('games',true);break;
      case 'linemate-review':state.game='';renderTab('games',true);document.querySelector('.more-menu').open=false;const review=document.querySelector('.linemate-review');if(review){review.open=true;review.scrollIntoView({block:'start'});}break;
      case 'game-props':state.filters.game=games[Number(control.dataset.gameIndex)]?.display_name||'';state.filters.search='';state.limit=25;renderTab('props',true);break;
      case 'player-props':state.filters={search:control.dataset.player||'',game:'',stat:'',book:'',direction:'',decision:'',sort:'player'};state.limit=25;renderTab('props',true);break;
      case 'reset-filters':state.filters={search:'',game:'',stat:'',book:'',direction:'',decision:'',sort:'player'};state.limit=25;renderProps();document.getElementById('player-search').focus();break;
      case 'more-props':state.limit+=25;renderPropRows();break;
      case 'model-status':renderTab('results',true);document.querySelector('.more-menu').open=false;document.getElementById('model-status').open=true;document.getElementById('model-status').scrollIntoView({block:'center'});break;
    }
  });
  document.addEventListener('input',event=>{if(event.target.id==='player-search'){state.filters.search=event.target.value;state.limit=25;renderPropRows();}});
  document.addEventListener('change',event=>{
    const id=event.target.id;
    if(id.startsWith('filter-')){state.filters[id.slice(7)]=event.target.value;state.limit=25;if(id==='filter-stat'){if(!state.filters.stat&&state.filters.sort==='gap')state.filters.sort='player';renderProps();}else renderPropRows();}
    if(id==='game-select'){state.game=event.target.value;renderGames();}
    if(id==='result-date'){state.resultDate=event.target.value;renderResultRows();}
    if(id==='result-book'){state.resultBook=event.target.value;renderResultRows();}
    if(id==='result-outcome'){state.resultOutcome=event.target.value;renderResultRows();}
  });
  document.querySelector('.tabs').addEventListener('keydown',event=>{
    const keys=['ArrowLeft','ArrowRight','Home','End'];if(!keys.includes(event.key))return;
    const tabs=[...document.querySelectorAll('[data-tab]')], index=tabs.indexOf(document.activeElement);if(index<0)return;
    event.preventDefault();const next=event.key==='Home'?0:event.key==='End'?tabs.length-1:(index+(event.key==='ArrowRight'?1:-1)+tabs.length)%tabs.length;
    renderTab(tabs[next].dataset.tab);tabs[next].focus();
  });
  freshness();renderTab('today');
}());
