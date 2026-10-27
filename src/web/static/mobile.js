// Phone presentation for the existing packet store and renderers in index.html.
// No extra receiver polling: stats, health and connection events update this view.
const PHONE=window.matchMedia('(max-width: 900px)');
let MOBILE_LIMIT=60, MOBILE_HEALTH={}, MOBILE_FREQUENCY=null, MOBILE_BANDWIDTH=null;
let MOBILE_CONNECTION='connecting', MOBILE_FEED_COUNT=0, MOBILE_LAST=0;

function setMobileConnection(state){
  MOBILE_CONNECTION=state;
  updateMissionStatus();
}
function updateMissionStatus(){
  const state=document.getElementById('mission-state');
  if(!state)return;
  const connected=MOBILE_CONNECTION==='live'||MOBILE_CONNECTION==='snapshot';
  const warnings=MOBILE_HEALTH.warn||MOBILE_HEALTH.preflight_note;
  const states={connecting:'Connecting…',reconnecting:'Reconnecting…',unavailable:'Receiver unavailable',syncing:'Syncing intercepts…',snapshot:'Session preview'};
  state.textContent=connected?(MOBILE_CONNECTION==='snapshot'?'Session preview':PIPE_RUNNING?'Listening':'Receiver stopped'):(states[MOBILE_CONNECTION]||'Connection unavailable');
  state.dataset.state=connected?(PIPE_RUNNING?'on':'off'):'warning';
  document.getElementById('mission-note').textContent=!connected?'Status may be out of date. Check the connection to your receiver.':warnings?'Receiver needs attention — see warning below.':PIPE_RUNNING?'Receiving across the band.':'Start the receiver or open a saved session.';
  if(MOBILE_CONNECTION==='snapshot')document.getElementById('mission-note').textContent='Saved view · live updates are off.';
  for(const [target,source] of [['mission-received','s_pkts'],['mission-decoded','s_dec'],['mission-nodes','s_nodes']])document.getElementById(target).textContent=document.getElementById(source).textContent;
  document.getElementById('mission-frequency').textContent=MOBILE_FREQUENCY==null?'Frequency unavailable':Number(MOBILE_FREQUENCY).toFixed(3)+' MHz';
  document.getElementById('mission-bandwidth').textContent=MOBILE_BANDWIDTH>0?Number((MOBILE_BANDWIDTH/1e6).toFixed(3))+' MHz BW':'Bandwidth unavailable';
  MOBILE_LAST=PK.reduce((last,p)=>Math.max(last,Number(p.ts)||0),0);
  updateMobileLast();
}
function updateMobileLast(){
  const el=document.getElementById('mission-last');
  if(!MOBILE_LAST){el.textContent='No intercepts yet';return;}
  const age=Math.max(0,Math.floor(Date.now()/1000-MOBILE_LAST));
  el.textContent='Last received '+(age<5?'just now':age<60?age+'s ago':age<3600?Math.floor(age/60)+'m ago':fmtDateTime(MOBILE_LAST));
  el.title=fmtDateTime(MOBILE_LAST);
}
function updateMobileTab(){
  document.body.dataset.tab=TAB;
  if(PHONE.matches&&TAB==='pkts'){
    document.getElementById('view-title').textContent='Intercepts';
    document.getElementById('view-description').textContent='Received messages and activity.';
  }
  const section=TAB==='net'?'nodes':TAB;
  document.querySelectorAll('[data-mobile-tab]').forEach(b=>{
    if(b.dataset.mobileTab===section)b.setAttribute('aria-current','page');else b.removeAttribute('aria-current');
  });
  const more=document.getElementById('mobile-more');
  if(!['pkts','nodes','map'].includes(section))more.setAttribute('aria-current','page');else more.removeAttribute('aria-current');
}
function holdMobileFeed(force){
  if(force||!PHONE.matches||!PK.length||!document.querySelector('.intercept-card'))return false;
  // Keep the item under the user's finger stable while capture continues.
  if(window.scrollY>180||document.querySelector('.intercept-card[open]')){
    const button=document.getElementById('feed-refresh');
    const added=PK.length-MOBILE_FEED_COUNT;
    button.textContent=added>0?added+' new receptions · Show latest':'Updates available · Show latest';
    button.hidden=false;
    return true;
  }
  return false;
}
function mobilePacketSummary(p){
  if(p.decrypted&&p.text)return esc(p.text);
  if(proto(p)==='meshtastic')return p.decrypted?meshSummary(p):'Encrypted message. Contents unavailable.';
  if(p.summary)return esc(p.summary);
  if(p.lat!=null&&p.lon!=null)return 'Position: '+esc(p.lat)+', '+esc(p.lon);
  return 'Payload captured. Open details to inspect.';
}
function renderMobileIntercepts(reps){
  const el=document.getElementById('pktbody');
  MOBILE_FEED_COUNT=PK.length;
  document.getElementById('feed-refresh').hidden=true;
  document.getElementById('shown').textContent=reps.length+' matching / '+PK.length+' receptions';
  const filtered=FILTER||PFILT!=='all';
  document.getElementById('mobile-filters').textContent=filtered?'Filter · active':'Filter';
  const empty=PK.length?'No matching intercepts':PIPE_RUNNING?'Listening for intercepts':'Ready to receive';
  const hint=PK.length?'Try another search or filter.':PIPE_RUNNING?'Messages and activity will appear here.':'Start your receiver, or open a saved session from More.';
  el.innerHTML=reps.length?reps.slice(0,MOBILE_LIMIT).map((p,i)=>{
    const protocol=PROTO_LABEL[proto(p)]||'Unknown protocol';
    const status=p.ambiguous?'Ambiguous':p.confidence==='candidate'?'Unverified':p.decrypted?'Decoded':proto(p)==='meshtastic'?'Encrypted':'Captured';
    const sender=disp(p.from)||p.node_long||p.node_short||'Unidentified sender';
    const frequency=p.freq_mhz!=null&&Number.isFinite(Number(p.freq_mhz))?Number(p.freq_mhz).toFixed(3).replace(/0+$/, '').replace(/\.$/, '.0')+' MHz':null;
    const meta=[protocol,frequency].filter(Boolean).map(esc).join(' · ');
    return `<details class="intercept-card" data-intercept="${i}">
      <summary><span class="intercept-top"><strong>${esc(sender)}</strong><time title="${esc(fmtDateTime(p.ts))}">${p.ts?fmtClock(p.ts):'Time unknown'}</time></span>
      <span class="intercept-message">${mobilePacketSummary(p)}</span>
      <span class="intercept-bottom"><span>${meta}</span><span class="intercept-state">${status}</span></span>
      <span class="intercept-expand">Details <span aria-hidden="true">⌄</span></span></summary>
      <div class="intercept-detail"></div></details>`;
  }).join(''):`<div class="empty"><strong>${empty}</strong><span>${hint}</span></div>`;
  // Populate the detailed renderer only on demand, avoiding hundreds of hidden trees.
  el.querySelectorAll('.intercept-card').forEach(card=>card.addEventListener('toggle',()=>{
    if(!card.open)return;
    const p=reps[Number(card.dataset.intercept)];
    const detail=card.querySelector('.intercept-detail');
    if(detail.dataset.loaded)return;
    detail.dataset.loaded='true';
    detail.innerHTML=pktState(p)+pktDetail(p);
    if(!p.decrypted&&p.enc_hex){
      const retry=document.createElement('button');retry.textContent='Try a decryption key';
      retry.onclick=e=>{e.stopPropagation();const r=retry.getBoundingClientRect();showDecodeMenu(p.pktid,p.hops||0,Math.min(r.left,window.innerWidth-190),Math.min(r.bottom,window.innerHeight-220));};
      detail.appendChild(retry);
    }
  }));
  if(reps.length>MOBILE_LIMIT){
    const more=document.createElement('button');more.id='intercepts-more';
    more.textContent='Show earlier intercepts ('+(reps.length-MOBILE_LIMIT)+' more)';
    more.onclick=()=>{MOBILE_LIMIT+=60;renderPkts(true);};el.appendChild(more);
  }
}
function initMobileWorkbench(){
  const menu=document.getElementById('mobile-menu');
  document.getElementById('mobile-more').onclick=()=>{
    document.getElementById('mobile-theme').textContent=document.getElementById('btnTheme').textContent;
    menu.showModal();
  };
  document.getElementById('mobile-menu-close').onclick=()=>menu.close();
  menu.addEventListener('click',e=>{if(e.target===menu){const r=menu.getBoundingClientRect();if(e.clientX<r.left||e.clientX>r.right||e.clientY<r.top||e.clientY>r.bottom)menu.close();}});
  document.querySelectorAll('[data-mobile-tab]').forEach(button=>button.onclick=()=>{
    menu.close();activateTab(button.dataset.mobileTab==='nodes'?NODE_VIEW:button.dataset.mobileTab);renderPktsIfVisible();window.scrollTo(0,0);
  });
  document.querySelectorAll('[data-mobile-action]').forEach(button=>button.onclick=()=>{
    const target=button.dataset.mobileAction;menu.close();document.getElementById(target).click();
  });
  const filterButton=document.getElementById('mobile-filters');
  filterButton.onclick=()=>{
    const open=filterButton.getAttribute('aria-expanded')!=='true';
    filterButton.setAttribute('aria-expanded',String(open));document.body.classList.toggle('filters-open',open);
    if(open)document.getElementById('filter').focus({preventScroll:true});
  };
  document.getElementById('feed-refresh').onclick=()=>{renderPkts(true);document.getElementById('view-title').scrollIntoView({block:'start'});};
  PHONE.addEventListener('change',()=>{menu.close();activateTab(TAB);renderPktsIfVisible();});
  if(STATIC)setMobileConnection('snapshot');
  setInterval(()=>{if(PHONE.matches&&!document.hidden)updateMobileLast();},10000);
}
function renderPktsIfVisible(){if(TAB==='pkts')renderPkts(true);}
