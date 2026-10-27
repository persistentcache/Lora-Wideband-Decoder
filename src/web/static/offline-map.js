// Local map assets stay available underneath an optional internet tile layer.
window.OfflineMap={
  map:null,overview:null,overviewLoaded:false,overviewError:false,pack:null,tiles:null,missing:false,
  mode:'auto',online:null,onlineState:'idle',onlineTimer:null,
  init(map){
    this.map=map;
    try{const saved=localStorage.getItem('lora-map-source');if(['auto','online','offline'].includes(saved))this.mode=saved;}catch(_){}
    const source=document.getElementById('map-source');source.value=this.mode;
    source.onchange=()=>{
      this.mode=source.value;
      try{localStorage.setItem('lora-map-source',this.mode);}catch(_){}
      this.selectSource();
    };
    L.control.scale({imperial:false}).addTo(map);
    map.createPane('overview');map.getPane('overview').style.zIndex=190;
    fetch('/static/maps/world.geojson').then(r=>{if(!r.ok)throw Error('Overview unavailable');return r.json();}).then(data=>{
      this.overview=L.geoJSON(data,{pane:'overview',interactive:false,style:this.overviewStyle(),
        attribution:'Natural Earth · offline overview'}).addTo(map);
      this.overviewLoaded=true;
      this.describe();
    }).catch(()=>{this.overviewError=true;this.describe();});
    document.getElementById('offline-map-use').onchange=()=>this.applyPack();
    document.getElementById('offline-map-fit').onclick=()=>{
      if(this.pack&&this.pack.bounds){const [w,s,e,n]=this.pack.bounds;map.fitBounds([[s,w],[n,e]],{padding:[12,12],maxZoom:this.pack.maxzoom});}
    };
    const file=document.getElementById('offline-map-file'),button=document.getElementById('offline-map-upload');
    file.onchange=()=>{button.disabled=!file.files.length;document.getElementById('offline-map-feedback').textContent='';};
    button.onclick=()=>this.upload(file.files[0]);
    map.on('moveend zoomend',()=>this.describe());
    new MutationObserver(()=>{if(this.overview)this.overview.setStyle(this.overviewStyle());}).observe(document.documentElement,{attributes:true,attributeFilter:['data-theme']});
    window.addEventListener('offline',()=>{if(this.mode!=='offline')this.onlineFailed();});
    window.addEventListener('online',()=>this.retryOnline());
    document.addEventListener('visibilitychange',()=>this.retryOnline());
    // Retry only a visible map, and only after losing the internet layer.
    setInterval(()=>this.retryOnline(),30000);
    this.refresh();
    this.selectSource();
  },
  selectSource(){
    this.removeOnline();
    this.onlineState='idle';
    this.applyPack();
    if(this.mode!=='offline')this.startOnline();
  },
  removeOnline(){
    clearTimeout(this.onlineTimer);this.onlineTimer=null;
    const layer=this.online;this.online=null;
    if(layer)this.map.removeLayer(layer);
  },
  retryOnline(){
    if(this.mode==='offline'||this.online||document.hidden||!this.map.getContainer().getClientRects().length)return;
    this.startOnline();
  },
  startOnline(){
    if(navigator.onLine===false){this.onlineFailed();return;}
    this.onlineState='checking';
    const layer=L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png',{
      maxNativeZoom:19,maxZoom:Math.max(22,this.pack?this.pack.maxzoom:0),
      attribution:'&copy; <a href="https://www.openstreetmap.org/copyright" target="_blank" rel="noopener">OpenStreetMap</a> contributors',
      keepBuffer:1,zIndex:250,opacity:0
    });
    this.online=layer;
    layer.on('loading',()=>{
      if(this.online!==layer)return;
      clearTimeout(this.onlineTimer);
      this.onlineTimer=setTimeout(()=>{if(this.online===layer)this.onlineFailed();},8000);
    });
    layer.on('tileload',()=>{
      if(this.online!==layer||this.onlineState==='ready')return;
      this.onlineState='ready';layer.setOpacity(1);this.applyPack();
    });
    layer.on('tileerror',()=>{if(this.online===layer)this.onlineFailed();});
    layer.on('load',()=>{if(this.online===layer){clearTimeout(this.onlineTimer);this.onlineTimer=null;}});
    layer.addTo(this.map);
    this.describe();
  },
  onlineFailed(){
    this.removeOnline();this.onlineState='unavailable';this.applyPack();
  },
  overviewStyle(){const style=getComputedStyle(document.documentElement);return {color:style.getPropertyValue('--mut').trim(),fillColor:style.getPropertyValue('--panel2').trim(),weight:1,fillOpacity:1};},
  status(text){document.getElementById('offline-map-status').textContent=text;},
  refresh(){
    return fetch('/api/map',{cache:'no-store'}).then(r=>{if(!r.ok)throw Error('Could not read offline map status');return r.json();}).then(pack=>this.install(pack)).catch(error=>{
      this.describe();
      document.getElementById('offline-map-feedback').textContent=error.message;
    });
  },
  install(pack){
    this.pack=pack.available?pack:null;
    this.map.setMaxZoom(Math.max(22,this.pack?this.pack.maxzoom:0));
    if(this.online)this.online.options.maxZoom=this.map.getMaxZoom();
    const toggle=document.getElementById('offline-map-use');toggle.disabled=!this.pack;toggle.checked=!!this.pack;
    document.getElementById('offline-map-fit').disabled=!(this.pack&&this.pack.bounds);
    if(pack.error)document.getElementById('offline-map-feedback').textContent=pack.error;
    this.applyPack();
  },
  applyPack(){
    if(this.tiles){this.map.removeLayer(this.tiles);this.tiles=null;}
    this.missing=false;
    if(this.mode!=='online'&&this.onlineState!=='ready'&&this.pack&&document.getElementById('offline-map-use').checked){
      const p=this.pack;
      // Attribution comes from an uploaded file; render it as text, never as HTML.
      const el=document.createElement('span');el.textContent=p.attribution||p.name;
      this.tiles=L.tileLayer('/api/map/tiles/{z}/{x}/{y}?v='+encodeURIComponent(p.revision),{
        minZoom:p.minzoom,maxZoom:Math.max(22,p.maxzoom),maxNativeZoom:p.maxzoom,noWrap:true,
        attribution:el.innerHTML,keepBuffer:1
      });
      this.tiles.on('loading',()=>{this.missing=false;});
      this.tiles.on('tileerror',()=>{this.missing=true;this.describe();});
      this.tiles.on('load',()=>this.describe());
      this.tiles.addTo(this.map);
    }
    this.describe();
  },
  describe(){
    if(this.onlineState==='ready'){this.status('Online · OpenStreetMap');return;}
    if(this.mode==='online'){
      this.status(this.onlineState==='unavailable'?'Online map unavailable · choose Auto or Offline to use saved maps':'Checking online map…');return;
    }
    const prefix=this.mode==='auto'&&this.onlineState==='checking'?'Checking online map · ':'';
    if(!this.tiles){this.status(prefix+(this.overviewError?'Overview unavailable · positions still shown':'Offline overview · no street detail'));return;}
    const p=this.pack,z=this.map.getZoom();
    let outside=false;
    if(p.bounds){const [w,s,e,n]=p.bounds;outside=!this.map.getBounds().intersects(L.latLngBounds([s,w],[n,e]));}
    if(outside||z<p.minzoom||this.missing){this.status(prefix+'No saved tiles here · showing overview');return;}
    this.status(prefix+p.name+' · offline'+(z>p.maxzoom?' · enlarged past saved detail':''));
  },
  upload(file){
    if(!file)return;
    const feedback=document.getElementById('offline-map-feedback');
    if(file.size>1024**3){feedback.textContent='This map is larger than 1 GB. Choose a smaller region or fewer zoom levels.';return;}
    const button=document.getElementById('offline-map-upload'),input=document.getElementById('offline-map-file');
    button.disabled=true;input.disabled=true;feedback.textContent='Uploading map…';
    const xhr=new XMLHttpRequest();xhr.open('POST','/api/map/upload');xhr.setRequestHeader('Content-Type','application/octet-stream');
    xhr.upload.onprogress=e=>{if(e.lengthComputable)feedback.textContent='Uploading map… '+Math.round(e.loaded/e.total*100)+'%';};
    xhr.upload.onload=()=>{feedback.textContent='Checking and installing map…';};
    xhr.onload=()=>{
      let result;try{result=JSON.parse(xhr.responseText);}catch(_){result={error:'The receiver returned an invalid response.'};}
      if(xhr.status>=200&&xhr.status<300&&result.ok){
        this.install(result.map);input.value='';feedback.textContent='Map ready. It will stay available without internet.';
      }else feedback.textContent=result.error||'Map could not be installed. The previous map is unchanged.';
    };
    xhr.onerror=()=>{feedback.textContent='Connection lost. Reconnect to the receiver and check the map status before retrying.';};
    xhr.onloadend=()=>{input.disabled=false;button.disabled=!input.files.length;};
    xhr.send(file);
  }
};
