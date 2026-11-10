/* Linemate is a manual research workspace. No account data or site scraping. */
(function () {
  'use strict';
  const KEY='nba-linemate-reviews-v1';
  const esc=v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const text=(v,max=800)=>String(v??'').trim().slice(0,max);
  const kinds={ATS:'ATS record · each game’s spread',LINE:'Hit rate · one selected line',CONTEXT:'Injury, lineup or rotation'};
  const splits={SEASON:'Season',LAST5:'Last 5',LAST10:'Last 10',HOME:'Home',AWAY:'Away',H2H:'Head to head',B2B:'Back-to-back',WITH_WITHOUT:'With / without teammate',OTHER:'Other'};
  function matchupUrl(game,date) {
    const aliases={BKN:'BRO',OKC:'OKL'};
    const code=t=>aliases[t?.abbr]||t?.abbr;
    const away=code(game.away),home=code(game.home);
    if(!/^\d{4}-\d{2}-\d{2}$/.test(date)||![away,home].every(v=>/^[A-Z]{2,3}$/.test(v||'')))return 'https://linemate.io/nba';
    return `https://linemate.io/nba/summary/${date.replaceAll('-','')}-${away}-${home}`;
  }
  function normalize(input) {
    if(!kinds[input.kind])throw Error('Choose an observation type.');
    const row={kind:input.kind,subject:text(input.subject,100),split:input.split,basis:text(input.basis,160),note:text(input.note),verdict:input.verdict};
    if(!row.subject)throw Error('Enter the team or player.');
    if(!splits[row.split])throw Error('Choose a split.');
    if(!['SUPPORTS','CONFLICTS','UNCLEAR'].includes(row.verdict))throw Error('Choose how it compares with the model.');
    const count=(name,optional=false)=>{
      const value=input[name];
      if(optional&&(value===''||value==null))return 0;
      if(value===''||value==null||!Number.isInteger(Number(value))||Number(value)<0||Number(value)>5000)throw Error('Enter whole-number counts between 0 and 5,000.');
      return Number(value);
    };
    if(row.kind==='ATS') {
      row.wins=count('wins');row.losses=count('losses');row.pushes=count('pushes',true);
      if(!row.wins&&!row.losses&&!row.pushes)throw Error('Enter at least one graded ATS game.');
      if(!row.basis)throw Error('Identify the sportsbook and spread timing for this ATS record.');
    } else if(row.kind==='LINE') {
      if(input.line===''||input.line==null||!Number.isFinite(Number(input.line))||Math.abs(Number(input.line))>500)throw Error('Enter the exact selected line.');
      if(!['OVER','UNDER','COVER'].includes(input.side))throw Error('Choose Over, Under or Cover.');
      row.line=Number(input.line);row.side=input.side;row.hits=count('hits');row.sample=count('sample');
      if(!row.sample||row.hits>row.sample)throw Error('Hits must not exceed a positive sample size.');
      if(!row.basis)throw Error('Name the market or stat used for this hit rate.');
    } else if(!row.note)throw Error('Describe the injury, lineup or rotation observation.');
    return row;
  }
  function load(storage) {
    const rows=JSON.parse(storage.getItem(KEY)||'[]');
    if(!Array.isArray(rows))throw Error('Saved review notes could not be read. Your saved data has not been changed.');
    return rows.filter(r=>r&&typeof r==='object');
  }
  function append(storage,input,slate,game,source,now=new Date().toISOString()) {
    const row={...normalize(input),slate,game,source,reviewed_at:now};
    const rows=load(storage);rows.push(row);
    if(rows.length>500)throw Error('The review notebook has 500 entries. Download a backup and remove old entries before adding more.');
    storage.setItem(KEY,JSON.stringify(rows));return row;
  }
  const recordLabel=r=>r.kind==='ATS'?`ATS ${r.wins}–${r.losses}–${r.pushes}`:r.kind==='LINE'?`${r.side} ${r.line}: ${r.hits}/${r.sample} at this line`:'Availability / rotation';
  const opts=obj=>Object.entries(obj).map(([v,t])=>`<option value="${v}">${esc(t)}</option>`).join('');
  const select=(name,label,items)=>`<label>${label}<select name="${name}">${opts(items)}</select></label>`;
  const number=(name,label)=>`<label>${label}<input type="number" name="${name}" min="0" max="5000" step="1" inputmode="numeric"></label>`;
  function list(rows) {
    return rows.length?rows.map((r,index)=>`<article class="linemate-entry"><div class="linemate-entry-top"><strong>${esc(r.subject)}</strong><span class="badge research">${esc(r.verdict==='SUPPORTS'?'Supports model':r.verdict==='CONFLICTS'?'Conflicts with model':'Unclear')}</span></div><p>${esc(recordLabel(r))} · ${esc(splits[r.split]||r.split)}</p>${r.basis?`<p class="small muted">Basis: ${esc(r.basis)}</p>`:''}${r.note?`<p class="linemate-note">${esc(r.note)}</p>`:''}<p class="snapshot-note">Manual observation · ${esc(new Date(r.reviewed_at).toLocaleString('en-US',{timeZone:'America/New_York',month:'short',day:'numeric',hour:'numeric',minute:'2-digit'}))} ET</p><button type="button" class="small" data-review-remove="${index}">Remove this note</button></article>`).join(''):'<p class="small muted">No Linemate observations saved for this matchup and date.</p>';
  }
  function render(game,date,index) {
    const name=game.display_name||`${game.away.name} @ ${game.home.name}`;
    let rows=[],storageError=false;
    try {rows=load(window.localStorage).filter(r=>r.slate===date&&r.game===name);}catch(_){storageError=true;}
    return `<details class="linemate-review" data-review-game="${index}"><summary>Linemate review · ATS, splits &amp; injuries${rows.length?` (${rows.length})`:''}</summary><div class="details-content"><div class="linemate-links"><a class="button" href="${esc(matchupUrl(game,date))}" target="_blank" rel="noopener noreferrer">Open Linemate matchup ↗</a><a href="https://linemate.io/nba/trends" target="_blank" rel="noopener noreferrer">NBA trends ↗</a></div><p class="info-note"><strong>Manual research · no live data connection</strong><br>A hit rate at today’s selected spread is different from ATS against each game’s original spread. Record the exact line, split and sample. Historical hit rates are not model win probabilities.</p><p class="snapshot-note">Notes stay in this browser on this device. They do not sync to GitHub, change projections or qualify bets. Linemate+ views may require your own login.</p><div class="linemate-saved">${list(rows)}</div><form class="linemate-form" data-game-index="${index}"><h3>Add a research observation</h3><div class="linemate-fields">${select('kind','Observation type',kinds)}<label>Team or player<input name="subject" maxlength="100" placeholder="Team or player name" required></label>${select('split','Split / timeframe',splits)}${select('verdict','Comparison with model',{UNCLEAR:'Unclear / needs review',SUPPORTS:'Supports model direction',CONFLICTS:'Conflicts with model direction'})}</div><div class="linemate-counts" data-review-kind="ATS">${number('wins','ATS wins')}${number('losses','ATS losses')}${number('pushes','ATS pushes')}</div><div class="linemate-counts" data-review-kind="LINE" hidden><label>Exact line<input name="line" type="number" min="-500" max="500" step="any"></label>${select('side','Side',{COVER:'Cover',OVER:'Over',UNDER:'Under'})}${number('hits','Hits')}${number('sample','Games in sample')}</div><label>Market / source basis<input name="basis" maxlength="160" placeholder="ATS: sportsbook + line timing; hit rate: stat / market"></label><label>Observation<textarea name="note" maxlength="800" rows="3" placeholder="Availability, teammate absence, minutes, or why the source agrees or conflicts"></textarea></label><button class="button primary" type="submit">Save private observation</button><p class="linemate-feedback small" role="status">${storageError?'Browser storage is unavailable. Notes cannot be saved.':''}</p></form><button type="button" class="button linemate-backup" data-review-export>Download my review notes</button></div></details>`;
  }
  if(typeof module!=='undefined'&&module.exports){module.exports={normalize,load,append,matchupUrl,recordLabel};return;}
  function rowsFor(game,date){return load(window.localStorage).filter(r=>r.game===game&&r.slate===date);}
  document.addEventListener('change',event=>{
    if(!event.target.matches('.linemate-form select[name="kind"]'))return;
    for(const box of event.target.closest('form').querySelectorAll('[data-review-kind]')) {
      box.hidden=box.dataset.reviewKind!==event.target.value;
      for(const control of box.querySelectorAll('input,select'))control.disabled=box.hidden;
    }
  });
  document.addEventListener('submit',event=>{
    const form=event.target.closest('.linemate-form');if(!form)return;event.preventDefault();
    const feedback=form.querySelector('.linemate-feedback');
    try {
      const game=window.DATA.games[Number(form.dataset.gameIndex)],date=window.DATA.date;
      const name=game.display_name||`${game.away.name} @ ${game.home.name}`;
      append(window.localStorage,Object.fromEntries(new FormData(form)),date,name,matchupUrl(game,date));
      const panel=form.closest('.linemate-review');panel.querySelector('.linemate-saved').innerHTML=list(rowsFor(name,date));
      panel.querySelector('summary').textContent=`Linemate review · ATS, splits & injuries (${rowsFor(name,date).length})`;
      feedback.textContent='Saved in this browser. Model projections and BET status are unchanged.';form.querySelector('[name="note"]').value='';
    } catch(error){feedback.textContent=error.name==='QuotaExceededError'?'Browser storage is full. Download your notes before clearing space.':error.message;}
  });
  document.addEventListener('click',event=>{
    const remove=event.target.closest('[data-review-remove]');
    if(remove) {
      const panel=remove.closest('.linemate-review'),game=window.DATA.games[Number(panel.dataset.reviewGame)],date=window.DATA.date,name=game.display_name||`${game.away.name} @ ${game.home.name}`;
      try {
        const rows=load(window.localStorage),matching=rows.map((r,i)=>({r,i})).filter(x=>x.r.game===name&&x.r.slate===date),target=matching[Number(remove.dataset.reviewRemove)];
        if(!target)return;rows.splice(target.i,1);window.localStorage.setItem(KEY,JSON.stringify(rows));
        panel.querySelector('.linemate-saved').innerHTML=list(rowsFor(name,date));panel.querySelector('summary').textContent=`Linemate review · ATS, splits & injuries (${rowsFor(name,date).length})`;
      } catch(_){panel.querySelector('.linemate-feedback').textContent='Could not remove the note. Browser storage is unavailable.';}
    }
    if(event.target.closest('[data-review-export]')) {
      try {
        const url=URL.createObjectURL(new Blob([JSON.stringify(load(window.localStorage),null,2)],{type:'application/json'}));
        const link=document.createElement('a');link.href=url;link.download='nba-linemate-review-notes.json';link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
      } catch(_){event.target.closest('.linemate-review').querySelector('.linemate-feedback').textContent='Could not download the review notes.';}
    }
  });
  window.LinemateReview={render,matchupUrl};
}());
