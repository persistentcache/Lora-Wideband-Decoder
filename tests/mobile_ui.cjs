// Browser acceptance check; run against a local Flask preview (see mobile plan).
const {test}=require('node:test');
const assert=require('node:assert/strict');
const puppeteer=require(process.env.PUPPETEER_MODULE||'puppeteer-core');
test('phone prioritizes intercepts, retains tools, and preserves desktop',async()=>{
  const browser=await puppeteer.launch({executablePath:process.env.CHROME_BIN,headless:true,args:['--no-sandbox']});
  try {
    const page=await browser.newPage(), errors=[];
    page.on('pageerror',e=>errors.push(e.message));
    await page.setViewport({width:390,height:844});
    await page.goto((process.env.UI_URL||'http://127.0.0.1:5057/')+'?static',{waitUntil:'networkidle0'});
    await page.evaluate(()=>{
      PK.length=0;
      for(let i=0;i<85;i++)PK.push({ts:Date.now()/1000-i*40,pktid:i,proto:'meshtastic',from:'!abc123',to:'!ffffffff',node_short:'RIDGE',freq_mhz:906.875,chan:'LongFast',port_name:'TEXT_MESSAGE_APP',decrypted:i%3!==0,text:i%3?'Team arrived at the north ridge. Signal is clear.':'',rssi:-92,hops:0});
      PK.forEach(noteName);renderStats({packets:85,decoded:56,encrypted:29,nodes:1,pipeline_running:true});renderCurrent();
    });
    assert.equal(await page.$$eval('.intercept-card',e=>e.length),60);
    assert.equal(await page.$eval('#view-title',e=>e.textContent),'Intercepts');
    assert.deepEqual(await page.$$eval('.mobile-tabs button',buttons=>buttons.map(b=>b.textContent)),['Intercepts','Nodes','Map','More']);
    assert.match(await page.$eval('.intercept-bottom',e=>e.textContent),/Meshtastic · 906.875 MHz/);
    assert.doesNotMatch(await page.$eval('.intercept-bottom',e=>e.textContent),/LongFast/);
    assert.ok(await page.$eval('.intercept-card',e=>e.getBoundingClientRect().top<600));
    assert.equal(await page.$eval('#packet-tools',e=>getComputedStyle(e).display),'none');
    await page.click('.intercept-card summary');
    await page.waitForSelector('.intercept-card[open] .intercept-detail[data-loaded]');
    assert.match(await page.$eval('.intercept-card[open]',e=>e.textContent),/906.875/);
    await page.evaluate(()=>{PK.push({...PK[1],pktid:200,text:'New arrival',ts:Date.now()/1000});renderCurrent();});
    assert.ok(await page.$('.intercept-card[open]'));
    assert.equal(await page.$eval('#feed-refresh',e=>e.hidden),false);
    await page.click('#feed-refresh');
    assert.match(await page.$eval('.intercept-card',e=>e.textContent),/New arrival/);
    await page.click('#intercepts-more');
    assert.equal(await page.$$eval('.intercept-card',e=>e.length),86);
    await page.evaluate(()=>window.scrollTo(0,0));
    await page.click('#mobile-filters');await page.select('#packet-kind','enc');
    assert.equal(await page.$$eval('.intercept-card',e=>e.length),29);
    await page.select('#packet-kind','all');await page.type('#filter','not-in-any-packet');
    assert.match(await page.$eval('#pktbody',e=>e.textContent),/No matching/);
    await page.$eval('#filter',e=>{e.value='';e.dispatchEvent(new Event('input'));});
    await page.click('#mobile-filters');
    await page.evaluate(()=>setMobileConnection('reconnecting'));
    assert.match(await page.$eval('#mission-state',e=>e.textContent),/Reconnecting/);
    await page.evaluate(()=>setMobileConnection('live'));
    assert.match(await page.$eval('#mission-state',e=>e.textContent),/Listening/);
    await page.click('#mobile-more');await page.click('[data-mobile-tab=config]');await page.waitForNetworkIdle({idleTime:100});await page.waitForSelector('#sdrsave',{visible:true});
    await page.evaluate(()=>document.querySelectorAll('#tab-config details').forEach(e=>e.open=true));
    assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));
    await page.click('#btnSessions');await page.waitForSelector('#sesssave');
    await page.keyboard.press('Escape');
    await page.click('[data-mobile-tab=pkts]');
    await page.setViewport({width:1440,height:960});
    await page.waitForSelector('#pktbody table');
    assert.equal(await page.$$eval('.pktrow',e=>e.length),86);
    assert.deepEqual(errors,[]);
  } finally {await browser.close();}
});

test('live connection recovery, mixed traffic and phone layouts',async()=>{
  const browser=await puppeteer.launch({executablePath:process.env.CHROME_BIN,headless:true,args:['--no-sandbox']});
  try {
    const page=await browser.newPage(),errors=[];let stateReads=0,settingsSaved=false,delayState=false,delayedRequest=null;
    const now=Date.now()/1000;
    const packets=[
      {ts:now,pktid:501,proto:'meshtastic',from:'!ridge',node_short:'RIDGE',decrypted:true,port_name:'TEXT_MESSAGE_APP',text:'At the north ridge. Clear signal from base. <img src=x onerror=alert(1)>',chan:'LongFast',freq_mhz:906.875},
      {ts:now-20,pktid:502,proto:'meshtastic',from:'!base',decrypted:false,enc_hex:'aabb',freq_mhz:906.875},
      {ts:now-40,proto:'unknown',raw_hex:'0011',confidence:'candidate',summary:'Unidentified traffic',freq_mhz:915.1},
      {ts:now-60,pktid:503,proto:'meshtastic',from:'!tracker',decrypted:true,port_name:'POSITION_APP',lat:42.1,lon:-71.2},
      {ts:now-80,proto:'meshcore',from:'!long-name'.repeat(15),decrypted:true,text:'Long message '.repeat(100)}
    ];
    page.on('pageerror',e=>errors.push(e.message));
    await page.evaluateOnNewDocument(()=>{
      window.EventSource=class extends EventTarget {constructor(){super();window.testStream=this;}close(){}};
    });
    await page.setRequestInterception(true);
    page.on('request',req=>{
      const path=new URL(req.url()).pathname;
      if(path==='/api/state'){
        stateReads++;
        if(delayState){delayedRequest=req;return;}
        return req.respond({contentType:'application/json',body:JSON.stringify({packets,stats:{packets:packets.length,decoded:3,encrypted:2,nodes:4,pipeline_running:true},config:{center_mhz:915,bandwidth_hz:12000000}})});
      }
      if(req.method()==='POST'){
        if(path==='/api/settings')settingsSaved=true;
        return req.respond({contentType:'application/json',body:'{"ok":true}'});
      }
      req.continue();
    });
    await page.setViewport({width:390,height:844});
    await page.goto(process.env.UI_URL||'http://127.0.0.1:5057/',{waitUntil:'networkidle0'});
    const emit=async(type,data)=>page.evaluate(({type,data})=>window.testStream.dispatchEvent(new MessageEvent(type,{data:JSON.stringify(data)})),{type,data});
    await emit('open');
    assert.equal(await page.$eval('#mission-bandwidth',e=>e.textContent),'12 MHz BW');
    await emit('psd',{center_mhz:914,bandwidth_hz:5000000});
    assert.equal(await page.$eval('#mission-bandwidth',e=>e.textContent),'5 MHz BW');
    assert.match(await page.$eval('#mission-state',e=>e.textContent),/Listening/);
    assert.equal(await page.$('.intercept-message img'),null);
    assert.match(await page.$eval('#pktbody',e=>e.textContent),/Unverified/);
    await page.screenshot({path:'/tmp/lora-mobile-mission-dark.png'});
    await page.click('.intercept-card:nth-child(2) summary');
    await page.waitForSelector('.intercept-card[open] .intercept-detail button');
    await page.$eval('.intercept-card[open] .intercept-detail button',e=>e.scrollIntoView({block:'center'}));
    await page.evaluate(()=>new Promise(r=>requestAnimationFrame(()=>requestAnimationFrame(r))));
    await page.click('.intercept-card[open] .intercept-detail button');
    assert.ok(await page.$('.ctxmenu'),'Decryption menu must remain open after tap');
    await page.click('.intercept-card:nth-child(2) summary');
    for(const width of [360,390,760,844]){
      await page.setViewport({width,height:width===844?390:844});
      assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),`feed overflow at ${width}`);
      await page.click('.intercept-card summary');
      assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));
      await page.click('.intercept-card summary');
      await page.click('#mobile-more');await page.click('[data-mobile-action=btnTheme]');
      await page.click('#mobile-more');await page.click('[data-mobile-tab=config]');await page.waitForNetworkIdle({idleTime:100});await page.waitForSelector('#sdrsave',{visible:true});
      await page.evaluate(()=>document.querySelectorAll('#tab-config details').forEach(e=>e.open=true));
      assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),`settings overflow at ${width}`);
      await page.click('#sdrsave');
      await page.click('[data-mobile-tab=pkts]');
    }
    assert.ok(settingsSaved);
    await page.setViewport({width:390,height:844});
    await page.screenshot({path:'/tmp/lora-mobile-mission-light.png'});
    await emit('error');
    assert.match(await page.$eval('#mission-state',e=>e.textContent),/Reconnecting/);
    assert.match(await page.$eval('#mission-note',e=>e.textContent),/out of date/);
    packets.unshift({...packets[0],pktid:999,text:'Received while phone disconnected'});
    await emit('open');
    await page.waitForFunction(()=>document.getElementById('mission-state').textContent==='Listening');
    assert.equal(stateReads,2);
    assert.match(await page.$eval('#pktbody',e=>e.textContent),/Received while phone disconnected/);
    // An explicit session replacement must not leave a previously expanded card behind.
    await page.click('.intercept-card summary');
    packets.splice(0,packets.length,{...packets[0],pktid:1000,text:'Different session'});
    await page.evaluate(()=>reloadState());
    assert.equal(await page.$$eval('.intercept-card',e=>e.length),1);
    assert.match(await page.$eval('#pktbody',e=>e.textContent),/Different session/);
    // A late snapshot cannot override a subsequent stream disconnect.
    delayState=true;
    await emit('error');await emit('open');
    for(let n=0;!delayedRequest&&n<100;n++)await new Promise(r=>setTimeout(r,10));
    assert.ok(delayedRequest,'reconnect must request a snapshot');
    await emit('packet',{...packets[0],pktid:1001,text:'Arrived during snapshot',ts:Date.now()/1000});
    await emit('stats',{packets:2,decoded:2,encrypted:0,nodes:1,pipeline_running:false});
    await emit('error');
    await delayedRequest.respond({contentType:'application/json',body:JSON.stringify({packets,stats:{packets:1,decoded:1,encrypted:0,nodes:1,pipeline_running:true}})});
    await page.waitForFunction(()=>PK.some(p=>p.pktid===1000)&&PK.some(p=>p.pktid===1001));
    await new Promise(r=>setTimeout(r,100));
    assert.match(await page.$eval('#mission-state',e=>e.textContent),/Reconnecting/);
    assert.equal(await page.evaluate(()=>PIPE_RUNNING),false,'late snapshot must not restore stale running state');
    await page.click('#mobile-more');await page.click('[data-mobile-tab=config]');await page.waitForNetworkIdle({idleTime:100});await page.waitForSelector('#sdrcen',{visible:true});
    await page.$eval('#sdrcen',e=>e.value='920.5');
    delayState=false;await emit('open');
    await page.waitForFunction(()=>document.getElementById('mission-state').textContent==='Listening');
    assert.equal(await page.$eval('#sdrcen',e=>e.value),'920.5','reconnect preserves unsaved settings');
    await emit('health',{warn:'Receiver cannot keep up'});
    assert.match(await page.$eval('#mission-note',e=>e.textContent),/attention/);
    await emit('stats',{packets:6,decoded:4,encrypted:2,nodes:4,pipeline_running:false});
    assert.match(await page.$eval('#mission-state',e=>e.textContent),/stopped/);
    delayState=true;delayedRequest=null;await emit('error');await emit('open');
    for(let n=0;!delayedRequest&&n<100;n++)await new Promise(r=>setTimeout(r,10));
    assert.ok(delayedRequest);
    page.once('dialog',dialog=>dialog.accept());
    await page.evaluate(()=>document.getElementById('btnClear').click());
    await delayedRequest.respond({contentType:'application/json',body:JSON.stringify({packets,stats:{packets:1,decoded:1,encrypted:0,nodes:1,pipeline_running:true}})});
    await new Promise(r=>setTimeout(r,100));
    assert.equal(await page.evaluate(()=>PK.length),0,'late reconnect must not undo Clear');
    assert.deepEqual(errors,[]);
  }finally{await browser.close();}
});


test('Sessions is beside receiver controls and Nodes contains both views',async()=>{
  const browser=await puppeteer.launch({executablePath:process.env.CHROME_BIN,headless:true,args:['--no-sandbox']});
  try {
    const page=await browser.newPage();
    await page.setViewport({width:390,height:844});
    await page.goto((process.env.UI_URL||'http://127.0.0.1:5057/')+'?static',{waitUntil:'networkidle0'});
    for(const width of [320,390,1440]){
      await page.setViewport({width,height:844});
      assert.ok(await page.$eval('#btnSessions',e=>e.getBoundingClientRect().width>0));
      const boxes=await page.evaluate(()=>['btnStart','btnSessions'].map(id=>{const r=document.getElementById(id).getBoundingClientRect();return {top:r.top,left:r.left,right:r.right};}));
      assert.equal(boxes[0].top,boxes[1].top);
      assert.ok(boxes[1].left>boxes[0].right);
      assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));
      await page.click('#btnSessions');await page.waitForSelector('#sesssave');await page.click('#sessclose');
      await page.evaluate(()=>activateTab('nodes'));
      await page.click('[data-node-view=net]');
      assert.equal(await page.$eval('#view-title',e=>e.textContent),'Nodes');
      assert.equal(await page.$eval('[data-node-view=net]',e=>e.getAttribute('aria-pressed')),'true');
      assert.ok(await page.$eval('#net',e=>e.getBoundingClientRect().width>0));
      assert.equal(await page.$eval('nav a[aria-current]',e=>e.dataset.tab),'nodes');
      assert.equal(await page.$eval('.mobile-tabs [aria-current]',e=>e.dataset.mobileTab),'nodes');
      await page.click('[data-node-view=nodes]');
      assert.equal(await page.$eval('#tab-net',e=>getComputedStyle(e).display),'none');
      assert.equal(await page.$eval('#node-view-switch',e=>e.hidden),false);
    }
    await page.goto((process.env.UI_URL||'http://127.0.0.1:5057/')+'?static#net',{waitUntil:'networkidle0'});
    assert.equal(await page.$eval('[data-node-view=net]',e=>e.getAttribute('aria-pressed')),'true');
    await page.click('nav a[data-tab=pkts]');await page.click('nav a[data-tab=nodes]');
    assert.equal(await page.$eval('[data-node-view=net]',e=>e.getAttribute('aria-pressed')),'true');
  }finally{await browser.close();}
});

test('empty intercept view follows receiver start and stop without a packet',async()=>{
  const browser=await puppeteer.launch({executablePath:process.env.CHROME_BIN,headless:true,args:['--no-sandbox']});
  try {
    const page=await browser.newPage();
    await page.evaluateOnNewDocument(()=>{
      window.EventSource=class extends EventTarget {constructor(){super();window.testStream=this;}close(){}};
    });
    await page.goto(process.env.UI_URL||'http://127.0.0.1:5057/',{waitUntil:'networkidle0'});
    const stats=running=>page.evaluate(running=>window.testStream.dispatchEvent(new MessageEvent('stats',{data:JSON.stringify({packets:0,decoded:0,encrypted:0,nodes:0,pipeline_running:running})})),running);
    for(const width of [390,1440]){
      await page.setViewport({width,height:844});
      await page.evaluate(()=>{PK.length=0;renderStats({packets:0,decoded:0,encrypted:0,nodes:0,pipeline_running:false});activateTab('pkts');});
      assert.match(await page.$eval('#pktbody',e=>e.textContent),/Ready to receive/);
      await stats(true);
      assert.match(await page.$eval('#pktbody',e=>e.textContent),/Listening for (intercepts|packets)/);
      assert.doesNotMatch(await page.$eval('#pktbody',e=>e.textContent),/[Ss]tart.*receiver/);
      await stats(false);
      assert.match(await page.$eval('#pktbody',e=>e.textContent),/Ready to receive/);
      await page.evaluate(()=>{PK.push({pktid:1,from:'!abc',text:'hello',decrypted:true});FILTER='no match';renderPkts(true);});
      await stats(true);
      assert.match(await page.$eval('#pktbody',e=>e.textContent),/No matching/);
      await page.evaluate(()=>FILTER='');
    }
  }finally{await browser.close();}
});
