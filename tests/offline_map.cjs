// Fresh browser, no internet/CDN/cache available. Flask must be running at UI_URL.
const {test}=require('node:test');
const assert=require('node:assert/strict');
const puppeteer=require(process.env.PUPPETEER_MODULE||'puppeteer-core');
const url=process.env.UI_URL||'http://127.0.0.1:5057/';
test('map and received positions work with every external request blocked',async()=>{
  const browser=await puppeteer.launch({executablePath:process.env.CHROME_BIN,headless:true,args:['--no-sandbox']});
  try{
    const page=await browser.newPage(),external=[],errors=[];
    await page.evaluateOnNewDocument(()=>localStorage.setItem('lora-map-source','offline'));
    page.on('pageerror',e=>errors.push(e.message));
    await page.setCacheEnabled(false);
    await page.setRequestInterception(true);
    page.on('request',r=>{
      if(/^https?:/.test(r.url())&&new URL(r.url()).origin!==new URL(url).origin){external.push(r.url());return r.abort();}
      if(new URL(r.url()).pathname==='/api/map')return r.respond({contentType:'application/json',body:'{"available":false}'});
      r.continue();
    });
    await page.setViewport({width:390,height:844});
    await page.goto(url+'?static#map',{waitUntil:'networkidle0'});
    assert.equal(await page.evaluate(()=>typeof window.L),'object','Leaflet must be served locally');
    await page.waitForFunction(()=>window.OfflineMap&&OfflineMap.overviewLoaded);
    await page.evaluate(()=>{
      PK.push({from:'!base',lat:42.860,lon:-72.110,ts:1},
              {from:'!base',lat:42.861,lon:-72.108,ts:2,pos_uncertainty_m:100,pos_precision_bits:18},
              {from:'!ridge',lat:42.868,lon:-72.108,ts:2},
              {from:'!base',waypoint:{lat:42.855,lon:-72.102,name:'Trailhead <script>bad</script>'},ts:2});
      drawMap();
    });
    assert.match(await page.$eval('#mapinfo',e=>e.textContent),/2 located nodes.*1 waypoint/);
    assert.match(await page.$eval('#offline-map-status',e=>e.textContent),/overview/i);
    await page.evaluate(()=>MAP_LAYER.getLayers().find(l=>l.getPopup()).openPopup());
    assert.match(await page.$eval('.leaflet-popup',e=>e.textContent),/42.86100.*72.10800/);
    await page.evaluate(()=>MAP.closePopup());
    await page.evaluate(()=>{activateTab('nodes');document.querySelector('tr.noderow').click();});
    await page.click('[data-map-lat]');
    assert.match(await page.$eval('#view-title',e=>e.textContent),/Map/);
    await page.click('#offline-map-options summary');
    assert.ok(await page.$('#offline-map-file'));
    assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));
    await page.screenshot({path:'/tmp/lora-map-offline.png'});
    assert.deepEqual(errors,[]);assert.deepEqual(external,[],'No external requests, even on a fresh visit');
  }finally{await browser.close();}
});

test('regional pack loads local tiles and upload errors preserve the current layer',async()=>{
  const {mkdtempSync,writeFileSync,rmSync}=require('node:fs');
  const {tmpdir}=require('node:os');const {join}=require('node:path');
  const dir=mkdtempSync(join(tmpdir(),'lora-map-test-')),fixture=join(dir,'sample.mbtiles');
  writeFileSync(fixture,'browser upload fixture');
  const browser=await puppeteer.launch({executablePath:process.env.CHROME_BIN,headless:true,args:['--no-sandbox']});
  try{
    const page=await browser.newPage(),external=[],tiles=[];
    await page.evaluateOnNewDocument(()=>localStorage.setItem('lora-map-source','offline'));
    let available=false,uploads=0,missingTiles=false;
    const pack={available:true,name:'Sample region',minzoom:0,maxzoom:14,bounds:[-72.2,42.8,-72.0,42.9],attribution:'Example <img src=x onerror=alert(1)>',revision:'sample'};
    const pixel=Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aCq0AAAAASUVORK5CYII=','base64');
    await page.setCacheEnabled(false);
    await page.setRequestInterception(true);
    page.on('request',r=>{
      const u=new URL(r.url());
      if(/^https?:$/.test(u.protocol)&&u.origin!==new URL(url).origin){external.push(r.url());return r.abort();}
      if(u.pathname==='/api/map')return r.respond({contentType:'application/json',body:JSON.stringify(available?pack:{available:false})});
      if(u.pathname==='/api/map/upload'){
        uploads++;available=true;
        return r.respond({status:uploads===1?200:400,contentType:'application/json',body:JSON.stringify(uploads===1?{ok:true,map:pack}:{ok:false,error:'Vector MBTiles are not supported.'})});
      }
      if(u.pathname.startsWith('/api/map/tiles/')){tiles.push(u.pathname);return missingTiles?r.respond({status:404,body:''}):r.respond({contentType:'image/png',body:pixel});}
      r.continue();
    });
    await page.setViewport({width:390,height:844});
    await page.goto(url+'?static#map',{waitUntil:'networkidle0'});
    await page.click('#offline-map-options summary');
    await (await page.$('#offline-map-file')).uploadFile(fixture);
    await page.click('#offline-map-upload');
    await page.waitForFunction(()=>document.getElementById('offline-map-feedback').textContent.includes('Map ready'));
    await page.click('#offline-map-fit');
    await page.waitForFunction(()=>document.querySelector('.leaflet-tile-loaded'));
    assert.ok(tiles.length>0);
    assert.ok(await page.$$eval('.leaflet-tile-loaded',imgs=>imgs.some(i=>i.src.includes('/api/map/tiles/')&&i.naturalWidth>0)));
    assert.equal(await page.$('.leaflet-control-attribution img'),null);
    await (await page.$('#offline-map-file')).uploadFile(fixture);await page.click('#offline-map-upload');
    await page.waitForFunction(()=>document.getElementById('offline-map-feedback').textContent.includes('Vector MBTiles'));
    assert.equal(await page.evaluate(()=>OfflineMap.pack.name),'Sample region');
    missingTiles=true;
    await page.evaluate(()=>OfflineMap.tiles.redraw());
    await page.waitForFunction(()=>document.getElementById('offline-map-status').textContent.includes('No saved tiles'));
    await page.click('#offline-map-use');
    assert.match(await page.$eval('#offline-map-status',e=>e.textContent),/overview/);
    assert.deepEqual(external,[]);
  }finally{await browser.close();rmSync(dir,{recursive:true,force:true});}
});
