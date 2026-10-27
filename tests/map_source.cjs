// Real Leaflet/UI; only external tile traffic and installed-pack metadata are fixtures.
const {test}=require('node:test');
const assert=require('node:assert/strict');
const puppeteer=require(process.env.PUPPETEER_MODULE||'puppeteer-core');
const url=process.env.UI_URL||'http://127.0.0.1:5057/';
const pixel=Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aCq0AAAAASUVORK5CYII=','base64');
const pack={available:true,name:'Field region',minzoom:0,maxzoom:14,bounds:[-73,42,-71,44],attribution:'Test region',revision:'field'};
async function fixture(t,{initialMode,onlineHint=true,available=true,network='up'}={}){
  const browser=await puppeteer.launch({executablePath:process.env.CHROME_BIN,headless:true,args:['--no-sandbox']});
  t.after(()=>browser.close());
  const page=await browser.newPage(),state={network,external:[],errors:[],pending:[]};
  page.on('pageerror',e=>state.errors.push(e.message));
  await page.setCacheEnabled(false);
  await page.setViewport({width:390,height:844});
  await page.evaluateOnNewDocument((mode,hint)=>{
    if(mode)localStorage.setItem('lora-map-source',mode);
    Object.defineProperty(navigator,'onLine',{configurable:true,get:()=>hint});
  },initialMode,onlineHint);
  await page.setRequestInterception(true);
  page.on('request',r=>{
    const u=new URL(r.url());
    if(u.hostname==='tile.openstreetmap.org'){
      state.external.push(r.url());
      if(state.network==='hang'){state.pending.push(r);return;}
      return state.network==='up'?r.respond({contentType:'image/png',body:pixel}):r.abort();
    }
    if(/^https?:$/.test(u.protocol)&&u.origin!==new URL(url).origin)return r.abort();
    if(u.pathname==='/api/map')return r.respond({contentType:'application/json',body:JSON.stringify(available?pack:{available:false})});
    if(u.pathname.startsWith('/api/map/tiles/'))return r.respond({contentType:'image/png',body:pixel});
    r.continue();
  });
  await page.goto(url+'?static#map',{waitUntil:'domcontentloaded'});
  await page.waitForFunction(()=>window.OfflineMap&&OfflineMap.overviewLoaded);
  await page.evaluate(()=>MAP.setView([42.86,-72.11],12,{animate:false}));
  return {page,state};
}
async function status(page,pattern,timeout=5000){
  await page.waitForFunction(text=>document.getElementById('offline-map-status').textContent.includes(text),{timeout},pattern);
}
async function choose(page,mode){
  await page.$eval('#offline-map-options',e=>e.open=true);
  await page.select('#map-source',mode);
}
async function onlineTiles(page){
  return page.$$eval('.leaflet-tile-loaded',imgs=>imgs.filter(i=>i.src.startsWith('https://tile.openstreetmap.org/')&&i.naturalWidth>0).length);
}

// Catches defaulting to local maps or failing to switch after the actual tile server fails.
test('Auto prefers internet tiles, falls back on tile errors, and recovers without losing positions',async t=>{
  const {page,state}=await fixture(t);
  await status(page,'Online');
  assert.equal(await page.$eval('#map-source',e=>e.value),'auto');
  assert.ok(await onlineTiles(page));
  assert.match(await page.$eval('.leaflet-control-attribution',e=>e.textContent),/OpenStreetMap/);
  await page.evaluate(()=>{
    PK.push({from:'!field',lat:42.86,lon:-72.11,ts:1});drawMap();
    MAP.setView([42.86,-72.11],12,{animate:false});
  });
  await status(page,'Online');
  state.network='down';
  await page.evaluate(()=>MAP.panTo([42.9,-72.2],{animate:false}));
  await status(page,'Field region');
  assert.equal(await onlineTiles(page),0);
  assert.match(await page.$eval('#mapinfo',e=>e.textContent),/1 located node/);
  const view=await page.evaluate(()=>({center:MAP.getCenter(),zoom:MAP.getZoom()}));
  state.network='up';
  await page.evaluate(()=>window.dispatchEvent(new Event('online')));
  await status(page,'Online');
  assert.ok(await onlineTiles(page));
  assert.deepEqual(await page.evaluate(()=>({center:MAP.getCenter(),zoom:MAP.getZoom()})),view);
  assert.deepEqual(state.errors,[]);
});

// Catches connectivity detection that trusts the Wi-Fi hint, and absence of periodic recovery.
test('Auto starts offline when tiles fail and retries without a browser network event',async t=>{
  const {page,state}=await fixture(t,{network:'down',available:false});
  await status(page,'overview');
  assert.ok(state.external.length>0,'Auto must try the internet map when Wi-Fi reports online');
  state.network='up';
  await status(page,'Online',35000);
  assert.ok(await onlineTiles(page));
  assert.deepEqual(state.errors,[]);
});

// Catches hung requests leaving Auto indefinitely checking instead of using the local pack.
test('Auto falls back after a stalled tile request and ignores its late response',async t=>{
  const {page,state}=await fixture(t,{network:'hang'});
  await status(page,'Checking');
  await page.waitForFunction(()=>document.getElementById('offline-map-status').textContent.startsWith('Field region'),{timeout:11000});
  assert.ok(state.pending.length>0);
  await Promise.all(state.pending.map(r=>r.respond({contentType:'image/png',body:pixel}).catch(()=>{})));
  assert.equal(await onlineTiles(page),0);
  assert.match(await page.$eval('#offline-map-status',e=>e.textContent),/Field region/);
  assert.deepEqual(state.errors,[]);
});

// Catches lost preferences or external requests after explicitly choosing Offline.
test('Offline is saved across reloads and suppresses external map requests',async t=>{
  const {page,state}=await fixture(t);
  await status(page,'Online');
  await choose(page,'offline');
  await status(page,'Field region');
  const before=state.external.length;
  await page.reload({waitUntil:'networkidle0'});
  assert.equal(await page.$eval('#map-source',e=>e.value),'offline');
  await page.evaluate(()=>{MAP.setView([42.86,-72.11],12,{animate:false});window.dispatchEvent(new Event('online'));});
  await status(page,'Field region');
  await page.waitForFunction(()=>!!document.querySelector('.leaflet-tile-loaded'));
  assert.equal(state.external.length,before);
  assert.equal(await onlineTiles(page),0);
  assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));
  assert.deepEqual(state.errors,[]);
});

// Catches stale responses overriding a manual switch, and misleading forced-Online status.
test('manual Online reports failure and switching to Offline cancels pending online activation',async t=>{
  const {page,state}=await fixture(t,{initialMode:'offline',network:'down'});
  await status(page,'Field region');
  assert.equal(state.external.length,0);
  await choose(page,'online');
  await status(page,'Online map unavailable');
  assert.equal(await page.$eval('#map-source',e=>e.value),'online');
  assert.equal(await page.evaluate(()=>!!OfflineMap.tiles),false);
  state.network='hang';
  await choose(page,'auto');
  await status(page,'Checking');
  await page.waitForFunction(()=>document.querySelector('img[src^="https://tile.openstreetmap.org/"]'));
  await choose(page,'offline');
  await status(page,'Field region');
  await Promise.all(state.pending.map(r=>r.respond({contentType:'image/png',body:pixel}).catch(()=>{})));
  assert.equal(await onlineTiles(page),0);
  assert.match(await page.$eval('#offline-map-status',e=>e.textContent),/Field region/);
  assert.deepEqual(state.errors,[]);
});

test('Auto respects a known offline connection without attempting internet tiles',async t=>{
  const {page,state}=await fixture(t,{onlineHint:false});
  await status(page,'Field region');
  assert.equal(state.external.length,0);
  assert.deepEqual(state.errors,[]);
});
