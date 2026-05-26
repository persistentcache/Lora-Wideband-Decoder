#!/usr/bin/env python3
"""LoRa wideband intercept — web UI backend (functionality-first rework).

Data model: the pipeline (lora_detect.py) writes one structured JSON record per
packet (decoded OR encrypted-header-only) to a JSONL log.  This backend TAILS
that log, dedups by (pktid,hop), and aggregates:
  - packet feed       (every intercepted packet, sortable/filterable)
  - per-node stats    (avg RSSI of a node's own transmissions, counts, channels)
  - conversations     (who talks to whom — node->node edges, the big picture)
  - encrypted traffic (headers visible, contents hidden — other-key packets)
Served over a small JSON API + Server-Sent Events; pipeline launched from
lora.toml so there are no flags to pass.

Usage:  python3 lora_web.py            # reads lora.toml, serves on [web] host:port
        python3 lora_web.py --config /path/lora.toml --port 5000
"""
import os, sys, json, time, threading, queue, subprocess, argparse, shutil, signal
from collections import deque, defaultdict
from flask import Flask, jsonify, request, Response, render_template

# --- locate scripts/ for the shared config loader ---
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.dirname(_HERE)
for _p in (os.path.join(_PROJECT, 'scripts'), os.path.join(_HERE, 'scripts')):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)
try:
    from lora_config import load_config
except Exception:
    load_config = None
try:
    import decode_header_v3 as dhv   # for web-side retry-decryption of stored packets
except Exception:
    dhv = None
try:
    import sdr_profiles                # multi-SDR capture/probe registry
except Exception:
    sdr_profiles = None
if load_config is None:
    def load_config(path=None):
        return {'radio': {'rate_hz': 28000000, 'bandwidth_hz': 28000000,
                          'center_mhz': 915.0, 'format': 'sc16'},
                'detect': {'threshold': 0.55, 'energy_threshold': 5.0, 'overlap': 0.5,
                           'detect_workers': -1, 'buf_seconds': 16, 'commit_lag': 4},
                'decode': {'budget_s': 10.0, 'workers': 10,
                           'export_dir': '/dev/shm/live_caps',
                           'packet_log': '/tmp/lora_packets.jsonl', 'key': 'default'},
                'web': {'host': '127.0.0.1', 'port': 5000}, '_path': None}

app = Flask(__name__)
CFG = load_config()

# ---------------------------------------------------------------- settings
SETTINGS_PATH = os.path.join(_HERE, 'web_settings.json')
def load_settings():
    _defaults = {'autosave': False, 'waterfall': True,
                 'wide_scan': True,
                 # Advanced Options (Config) — persist across sessions:
                 'protocols': {'meshtastic': True, 'meshcore': True, 'lorawan': True},
                 'unknown': False,          # master: surface unidentified protocols (default OFF)
                 'sdr': 'bladerf',          # selected SDR profile (persists across sessions)
                 'radio': {}}               # web overrides of lora.toml [radio] (rate/bw/center/gain)
    try:
        with open(SETTINGS_PATH) as f:
            return {**_defaults, **json.load(f)}
    except Exception:
        return dict(_defaults)
def save_settings(s):
    try:
        with open(SETTINGS_PATH, 'w') as f:
            json.dump(s, f)
    except Exception:
        pass
SETTINGS = load_settings()

# ---------------------------------------------------------------- channel keys
# Persistent key list (survives all sessions) the decoder reads via LORA_KEYS.
# Format: {"try_default": bool, "keys": [{"id","label","key","scope":"all"|"node","node"}]}
KEYS_PATH = os.path.join(_HERE, 'lora_keys.json')
import base64 as _b64


def _key_valid(s):
    """True if a key string decodes to 16/24/32 raw bytes (AES-128/192/256)."""
    s = (s or '').strip()
    try:
        h = s.lower()
        if all(c in '0123456789abcdef' for c in h) and len(h) in (32, 48, 64):
            return True
    except Exception:
        pass
    try:
        return len(_b64.b64decode(s + '=' * (-len(s) % 4))) in (16, 24, 32)
    except Exception:
        return False


# Hardcoded public/default channel keys — shown in the UI but VALUE-LOCKED and
# non-deletable; the user can only enable/disable, node-scope, and annotate them.
MESH_DEFAULT_KEY     = 'AQ=='                                # Meshtastic public/default (shorthand)
MESHCORE_DEFAULT_KEY = '8b3387e9c5cdea6ac9e5edbaa115cd72'    # MeshCore public/default


def load_keys():
    try:
        with open(KEYS_PATH) as f:
            d = json.load(f)
    except Exception:
        d = {}
    if not isinstance(d, dict):
        d = {}
    # Meshtastic public/default (built-in, value-locked)
    d.setdefault('try_default', True)        # = enabled toggle for the Meshtastic default
    d.setdefault('default_nodes', [])        # [] = applies to all nodes
    d.setdefault('default_priority', 50)     # lower = tried first
    d.setdefault('default_notes', '')
    # MeshCore public/default (built-in, value-locked)
    md = d.setdefault('meshcore_default', {})
    md.setdefault('enabled', True); md.setdefault('nodes', [])
    md.setdefault('priority', 50); md.setdefault('notes', '')
    # Custom keys
    d.setdefault('keys', [])
    for k in d['keys']:
        k.setdefault('protocol', 'meshtastic')
        k.setdefault('enabled', True)
        k.setdefault('notes', '')
    return d


def save_keys(d):
    try:
        with open(KEYS_PATH, 'w') as f:
            json.dump(d, f, indent=2)
    except Exception:
        pass


# Alert/trigger rules (persist across sessions).  Evaluated client-side; stored here.
ALERTS_PATH = os.path.join(_HERE, 'lora_alerts.json')


def load_alerts():
    try:
        with open(ALERTS_PATH) as f:
            d = json.load(f)
            d.setdefault('rules', [])
            return d
    except Exception:
        return {'rules': []}


def save_alerts(d):
    try:
        with open(ALERTS_PATH, 'w') as f:
            json.dump(d, f, indent=2)
    except Exception:
        pass


# Per-node user annotations (custom label, notes, watchlist) — persist across sessions.
NODES_META_PATH = os.path.join(_HERE, 'lora_nodes.json')


def load_node_meta():
    try:
        with open(NODES_META_PATH) as f:
            return json.load(f) or {}
    except Exception:
        return {}


def save_node_meta(d):
    try:
        with open(NODES_META_PATH, 'w') as f:
            json.dump(d, f, indent=2)
    except Exception:
        pass


def _parse_nodes(v):
    """Normalize a node list from a JSON list or a comma/space-separated string."""
    if isinstance(v, list):
        items = v
    elif isinstance(v, str):
        items = v.replace(',', ' ').split()
    else:
        items = []
    out = []
    for it in items:
        s = str(it).strip()
        if s and s not in out:
            out.append(s)
    return out

# CSV export/import columns.  pktid is included so re-imports dedup correctly.
_CSV_COLS = ['ts', 'pktid', 'from', 'to', 'hops', 'hop_start', 'hop_limit',
             'chan', 'decrypted', 'portnum', 'port_name', 'rssi', 'sf', 'bw',
             'freq_mhz', 'text', 'lat', 'lon', 'battery', 'voltage',
             'node_short', 'node_long', 'key']
_CSV_INT = {'hops', 'hop_start', 'hop_limit', 'portnum', 'rssi', 'sf', 'bw', 'battery'}
_CSV_FLOAT = {'ts', 'freq_mhz', 'lat', 'lon', 'voltage'}


def _csv_row_to_rec(row):
    """Coerce a CSV row (all strings) back to a typed packet record."""
    rec = {}
    for k, v in row.items():
        if v is None or v == '':
            continue
        if k in _CSV_INT:
            try: rec[k] = int(float(v))
            except ValueError: pass
        elif k in _CSV_FLOAT:
            try: rec[k] = float(v)
            except ValueError: pass
        elif k == 'decrypted':
            rec[k] = str(v).strip().lower() in ('true', '1', 'yes')
        else:
            rec[k] = v
    return rec


# ---------------------------------------------------------------- state
STATE_LOCK = threading.Lock()
PACKETS = deque(maxlen=20000)     # all intercepted packets (oldest..newest)
SEEN = set()                      # (pktid, hop) dedup keys
NODES = {}                        # node_id -> aggregate dict
EDGES = defaultdict(lambda: {'count': 0, 'decoded': 0, 'encrypted': 0,
                             'last_ts': None})  # (from,to) -> stats
SUBS = []                         # SSE subscriber queues
SUBS_LOCK = threading.Lock()
PIPELINE = {'proc': None, 'running': False, 'started_at': None, 'err': None}
# Confirmation state (no-key zero-FP layer): a sender becomes trustworthy via
# cross-frame evidence even when we can't decrypt or check a MIC.
LW_DEVICES = {}                   # 'LW:<devaddr>' -> {fcnts:[..], count:n, confirmed:bool}
MC_NODES = {}                     # pubkey[0] (int) -> {pubkey,name,role,from,count,last,lat,lon}
                                  #                    crypto-verified MeshCore nodes (from ADVERTs)
KNOWN_MESH = set()                # Meshtastic node ids decoded as a SENDER (proven real)
PENDING_ENC = []                  # encrypted unicast DMs held until a node corroborates them


def _broadcast(evt):
    with SUBS_LOCK:
        dead = []
        for q in SUBS:
            try:
                q.put_nowait(evt)
            except queue.Full:
                dead.append(q)
        for q in dead:
            SUBS.remove(q)


def _node(nid):
    n = NODES.get(nid)
    if n is None:
        n = {'id': nid, 'count': 0, 'decoded': 0, 'encrypted': 0,
             'first_ts': None, 'last_ts': None, 'rssi_sum': 0.0, 'rssi_n': 0,
             'channels': set(), 'last_text': None, 'last_to': None}
        NODES[nid] = n
    return n


def _promote_lorawan(rec):
    """Promote a 'looks like LoRaWAN' record to a NAMED, confirmed LoRaWAN frame."""
    rec['proto'] = 'lorawan'
    rec['confidence'] = 'confirmed'
    rec['from'] = 'LW:' + rec['devaddr']
    rec.pop('hint', None)


def _confirm(rec):
    """Raise a record's confidence using cross-frame evidence (the no-key path to
    positive identification).  Mutates rec in place.

    - 'verified' (cryptographic, e.g. a MeshCore ADVERT Ed25519 signature or a
      decrypted channel message) is left as-is; its node identity is registered.
    - LoRaWAN is NOT named on structure alone: a frame that only *looks like*
      LoRaWAN arrives as proto='unknown', hint='lorawan'.  Once the same DevAddr
      shows a coherent FCnt (non-decreasing, bounded step) across >=2 frames it is
      promoted to a named, confirmed LoRaWAN device — garbage cannot forge a
      persistent identity with an incrementing counter (P of two frames sharing a
      DevAddr ~ 1/2^32).

    Returns the DevAddr that JUST crossed into 'confirmed' (for a one-time
    broadcast so already-displayed rows can be upgraded), else None.  Must be
    called while holding STATE_LOCK."""
    if rec.get('confidence') == 'verified':
        # Register MeshCore node identities from crypto-verified ADVERTs (keyed by
        # the routing path-hash = pubkey[0]).  This registry is what corroborates
        # non-advert MeshCore traffic below.
        if rec.get('proto') == 'meshcore' and rec.get('mc_pubkey') and rec.get('mc_hash') is not None:
            n = MC_NODES.setdefault(int(rec['mc_hash']), {'count': 0})
            n['count'] += 1
            n['pubkey'] = rec['mc_pubkey']
            n['from'] = rec.get('from'); n['last'] = rec.get('ts')
            if rec.get('mc_name'): n['name'] = rec['mc_name']
            if rec.get('mc_role'): n['role'] = rec['mc_role']
            if rec.get('lat') is not None:
                n['lat'] = rec['lat']; n['lon'] = rec.get('lon')
        return None
    # MeshCore corroboration: a 'looks like MeshCore' frame (proto='unknown',
    # hint='meshcore') routed through >=2 crypto-verified nodes is behaviorally
    # confirmed — it can't be coincidence that two of its 1-byte path hashes both
    # match nodes whose ADVERT signatures we verified.  A single match is too weak
    # (1-byte hash) so it only annotates; the frame stays 'unknown' until >=2.
    if rec.get('hint') == 'meshcore':
        matched = set(int(h) for h in (rec.get('mc_path') or []) if int(h) in MC_NODES)
        if matched:
            rec['mc_known_hops'] = len(matched)
        if len(matched) >= 2:
            rec['proto'] = 'meshcore'
            rec['confidence'] = 'confirmed'
            rec.pop('hint', None)
        return None
    if rec.get('hint') != 'lorawan':
        return None
    devaddr, fcnt = rec.get('devaddr'), rec.get('fcnt')
    if not devaddr or fcnt is None:
        return None
    d = LW_DEVICES.setdefault(devaddr, {'fcnts': [], 'count': 0, 'confirmed': False})
    d['count'] += 1
    d['fcnts'].append(int(fcnt))
    if len(d['fcnts']) > 64:
        d['fcnts'] = d['fcnts'][-64:]
    just = None
    if not d['confirmed'] and len(d['fcnts']) >= 2:
        if any(0 <= b - a <= 2000 for a, b in zip(d['fcnts'], d['fcnts'][1:])):
            d['confirmed'] = True
            just = devaddr
    if d['confirmed']:
        _promote_lorawan(rec)
    return just


def _is_enc_unicast_mesh(rec):
    """An encrypted (undecryptable) Meshtastic DIRECT message: the cleartext header
    gives from→to a SPECIFIC node, but the payload didn't decrypt with our key."""
    return (rec.get('proto') == 'meshtastic' and not rec.get('decrypted')
            and rec.get('from') and rec.get('to') and rec.get('to') != 'broadcast')


def _mesh_corroborated(rec):
    """True iff an endpoint of this DM is a node we've actually DECODED elsewhere."""
    return rec.get('from') in KNOWN_MESH or rec.get('to') in KNOWN_MESH


def _store(rec):
    """Commit a record to state (caller MUST hold STATE_LOCK).  Runs positive-ID
    confirmation + node/edge aggregation, registers proven-real Meshtastic senders,
    and releases any buffered DMs they corroborate."""
    PACKETS.append(rec)
    ts = rec.get('ts') or time.time()
    # Positive-ID confirmation runs BEFORE aggregation so a promoted LoRaWAN frame
    # has its 'from' set in time to feed the Nodes/Network views.
    _just = _confirm(rec)
    if _just:                       # a DevAddr just crossed into 'confirmed'
        for _p in PACKETS:
            if _p.get('devaddr') == _just and _p.get('proto') == 'unknown':
                _promote_lorawan(_p)
        _broadcast({'type': 'confirm',
                    'data': {'devaddr': _just, 'proto': 'lorawan', 'confidence': 'confirmed'}})
    frm = rec.get('from')
    if frm:   # only nodes with a parsed sender feed the Nodes/Network views
        to = rec.get('to') or '?'
        n = _node(frm)
        n['count'] += 1
        if rec.get('decrypted'):
            n['decoded'] += 1
        else:
            n['encrypted'] += 1
        if n['first_ts'] is None:
            n['first_ts'] = ts
        n['last_ts'] = ts
        n['last_to'] = to
        if rec.get('chan'):
            n['channels'].add(rec['chan'])
        if rec.get('text') or rec.get('summary'):
            n['last_text'] = rec.get('text') or rec.get('summary')
        # "Avg RSSI by node" = average over the node's OWN transmissions only
        # (hops==0 → transmitted directly by this node, not relayed).
        if rec.get('hops') == 0 and rec.get('rssi') is not None:
            n['rssi_sum'] += float(rec['rssi'])
            n['rssi_n'] += 1
        e = EDGES[(frm, to)]
        e['count'] += 1
        e['decoded' if rec.get('decrypted') else 'encrypted'] += 1
        e['last_ts'] = ts
    # A newly-DECODED Meshtastic sender is a proven-real node → it corroborates any
    # buffered encrypted DM to/from it.
    if rec.get('proto') == 'meshtastic' and rec.get('decrypted') and frm and frm not in KNOWN_MESH:
        KNOWN_MESH.add(frm)
        _release_pending(frm)


def _release_pending(node):
    """Surface buffered encrypted DMs whose endpoint just became a known node
    (caller holds STATE_LOCK).  Does NOT call _stats() (which re-locks)."""
    for k, r in [(k, r) for (k, r) in PENDING_ENC if r.get('from') == node or r.get('to') == node]:
        try:
            PENDING_ENC.remove((k, r))
        except ValueError:
            continue
        if k in SEEN:
            continue
        SEEN.add(k)
        r['enc_confirmed'] = True        # an endpoint is a known node → not a CRC fluke
        _store(r)
        _broadcast({'type': 'packet', 'data': r})   # _broadcast uses SUBS_LOCK — safe


def ingest(rec):
    """Add one packet record to state.  Returns the record if newly displayed, else
    None — a duplicate, OR an encrypted unicast Meshtastic DM held pending node
    corroboration (a clean CRC-16 coincidence would point at a random node id that
    no real, decoded node matches → no fabricated link)."""
    if rec.get('pktid') is not None:
        key = (rec.get('pktid'), rec.get('hops'))
    else:
        key = (rec.get('proto'), rec.get('raw_hex'))
    with STATE_LOCK:
        if key in SEEN:
            return None
        if _is_enc_unicast_mesh(rec) and not _mesh_corroborated(rec):
            PENDING_ENC.append((key, rec))     # hold — NOT marked seen, may surface later
            if len(PENDING_ENC) > 200:
                PENDING_ENC.pop(0)
            return None
        if _is_enc_unicast_mesh(rec):
            rec['enc_confirmed'] = True         # endpoint already known
        SEEN.add(key)
        _store(rec)
    return rec


def _node_view(n):
    return {
        'id': n['id'], 'count': n['count'], 'decoded': n['decoded'],
        'encrypted': n['encrypted'], 'first_ts': n['first_ts'],
        'last_ts': n['last_ts'], 'last_to': n['last_to'],
        'avg_rssi': round(n['rssi_sum'] / n['rssi_n'], 1) if n['rssi_n'] else None,
        'rssi_samples': n['rssi_n'], 'channels': sorted(n['channels']),
        'last_text': n['last_text'],
    }


def _edges_view():
    with STATE_LOCK:
        return [{'from': k[0], 'to': k[1], 'count': v['count'],
                 'decoded': v['decoded'], 'encrypted': v['encrypted'],
                 'last_ts': v['last_ts']} for k, v in EDGES.items()]


def _stats():
    with STATE_LOCK:
        total = len(PACKETS)
        dec = sum(1 for p in PACKETS if p.get('decrypted'))
        return {'packets': total, 'decoded': dec, 'encrypted': total - dec,
                'nodes': len(NODES), 'edges': len(EDGES),
                'pipeline_running': PIPELINE['running']}


# ---------------------------------------------------------------- log tailer
def tail_packet_log(path):
    """Follow the JSONL, ingesting + broadcasting.  On startup, if auto-save is
    ON we replay the existing log (session persists across restarts); if OFF we
    seek to the end (fresh session).  Detects truncation (Clear / rotation)."""
    pos = 0
    started = False
    while True:
        try:
            if not os.path.exists(path):
                time.sleep(0.5)
                continue
            if not started:
                started = True
                pos = os.path.getsize(path)       # ALWAYS start fresh — data does
                                                  # not carry over; use Import for that
            if os.path.getsize(path) < pos:       # file truncated/rotated
                pos = 0
            with open(path, 'r') as f:
                f.seek(pos)
                new = f.readlines()
                pos = f.tell()
            for line in new:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                if ingest(rec) is not None:
                    _broadcast({'type': 'packet', 'data': rec})
                    _broadcast({'type': 'stats', 'data': _stats()})
            time.sleep(0.3)
        except Exception:
            time.sleep(0.5)


PIPELINE_LOG = '/tmp/lora_web_pipeline.log'
HEALTH = {'msps': None, 'rate_msps': None, 'drops_m': 0.0, 'save_q': None,
          'dec_q': None, 'det': None, 'pipe_ms': None, 'elapsed': None,
          'warn': None, 'detect_workers': None}
import re as _re
_RE_STAT = _re.compile(r'\[STAT\]\s+([\d.]+)s\s+\|\s+([\d.]+)Msps\s+\|\s+win=(\d+)\s+'
                       r'det=(\d+)\s+save_q=(\d+)\s+dec_q=(\d+)\s+active=(\d+)\s+'
                       r'pipe=(\d+)ms(?:\s+drops=([\d.]+)M)?')
_RE_AUTO = _re.compile(r'Detect workers: AUTO = (\d+)')


def _tail_health():
    """Follow the pipeline stderr log, parse [STAT]/warnings → HEALTH + SSE."""
    pos = 0
    rate_msps = float(CFG['radio']['rate_hz']) / 1e6
    HEALTH['rate_msps'] = rate_msps
    while True:
        try:
            if not os.path.exists(PIPELINE_LOG):
                HEALTH.update(msps=None); time.sleep(0.5); continue
            if os.path.getsize(PIPELINE_LOG) < pos:
                pos = 0
            with open(PIPELINE_LOG, 'r', errors='replace') as f:
                f.seek(pos); lines = f.readlines(); pos = f.tell()
            changed = False
            for ln in lines:
                m = _RE_STAT.search(ln)
                if m:
                    HEALTH.update(elapsed=float(m.group(1)), msps=float(m.group(2)),
                                  det=int(m.group(4)), save_q=int(m.group(5)),
                                  dec_q=int(m.group(6)), pipe_ms=int(m.group(8)),
                                  drops_m=float(m.group(9)) if m.group(9) else HEALTH['drops_m'])
                    changed = True
                a = _RE_AUTO.search(ln)
                if a:
                    HEALTH['detect_workers'] = int(a.group(1))
                if 'KEEP-UP WARNING' in ln:
                    HEALTH['warn'] = ln.strip()[:200]; changed = True
            if changed:
                _broadcast({'type': 'health', 'data': HEALTH})
            time.sleep(0.6)
        except Exception:
            time.sleep(0.6)


# Live spectrum waterfall: the gate writes its latest downsampled PSD frame here
# (tmpfs, RAM-backed).  We relay it to browsers as SSE only while someone watches.
PSD_FILE = '/dev/shm/lora_psd.bin'
PSD_OFF = PSD_FILE + '.off'   # marker: when present the gate skips emitting frames


def _apply_waterfall_flag():
    """Reflect the persistent waterfall setting to the gate via a marker file the
    gate polls — lets the user enable/disable live (no pipeline restart) AND keeps
    the gate from spending any cycles on PSD frames when it's off."""
    try:
        if SETTINGS.get('waterfall', True):
            if os.path.exists(PSD_OFF):
                os.remove(PSD_OFF)
        else:
            open(PSD_OFF, 'w').close()
            if os.path.exists(PSD_FILE):
                os.remove(PSD_FILE)   # drop the stale frame so the view goes blank
    except Exception:
        pass


def _tail_psd():
    last_m = None
    rate_mhz = float(CFG['radio']['rate_hz']) / 1e6
    center_mhz = float(CFG['radio'].get('center_mhz', 915.0))
    while True:
        try:
            if not SUBS or not SETTINGS.get('waterfall', True) or not os.path.exists(PSD_FILE):
                time.sleep(0.3); continue
            m = os.path.getmtime(PSD_FILE)
            if m == last_m:
                time.sleep(0.08); continue
            last_m = m
            with open(PSD_FILE, 'rb') as f:
                raw = f.read()
            if raw:
                _broadcast({'type': 'psd', 'data': {
                    'b64': _b64.b64encode(raw).decode('ascii'),
                    'center_mhz': center_mhz, 'rate_mhz': rate_mhz}})
            time.sleep(0.08)
        except Exception:
            time.sleep(0.3)


AUTOSAVE_PATH = os.path.join(_HERE, 'lora_autosave.jsonl')


def _autosave_loop():
    """When auto-save is ON, continuously mirror the FULL current session
    (imported + captured) to an importable file, so it survives a crash/restart.
    It is NOT auto-loaded on startup — the user imports it to resume (data does
    not carry over automatically; only the toggle STATE persists)."""
    last_n = -1
    while True:
        time.sleep(8)
        if not SETTINGS.get('autosave'):
            continue
        with STATE_LOCK:
            n = len(PACKETS)
            pkts = list(PACKETS) if n != last_n else None
        if pkts is None:
            continue
        try:
            tmp = AUTOSAVE_PATH + '.tmp'
            with open(tmp, 'w') as f:
                for p in pkts:
                    f.write(json.dumps(p, separators=(',', ':')) + '\n')
            os.replace(tmp, AUTOSAVE_PATH)
            last_n = n
        except Exception:
            pass


# ---------------------------------------------------------------- pipeline
def _radio_cfg():
    """Effective radio settings: lora.toml defaults overlaid with web overrides
    (SETTINGS['radio']), then CLAMPED to the selected SDR's capabilities so a stale
    or out-of-range value (e.g. a bladeRF's 28 Msps after switching to a HackRF)
    can never reach the device."""
    r = {**CFG['radio'], **(SETTINGS.get('radio') or {})}
    sdr = SETTINGS.get('sdr', 'bladerf')
    if sdr_profiles is not None:
        # No gain set anywhere → use the SELECTED device's own profile default
        # (bladeRF 48, HackRF 40, …) rather than one global value that leaks
        # across devices.
        if r.get('gain') in (None, ''):
            r['gain'] = sdr_profiles.SDR_PROFILES.get(sdr, {}).get('default_gain', 'auto')
        r = sdr_profiles.clamp_radio(sdr, r)
    return r


def _build_pipeline_cmd():
    """Construct the <SDR capture> | lora_detect shell pipeline for the SELECTED SDR.
    The capture half comes from the SDR profile registry (bladeRF's command is
    byte-identical to the long-validated one); the gate half is unchanged.  The
    gate path is shlex-quoted — the project dir can contain spaces (e.g.
    'LORA Project') which would otherwise split the shell word and the pipe would
    collapse a second after starting."""
    import shlex
    r = _radio_cfg(); d = CFG['detect']; dec = CFG['decode']
    rate = int(r['rate_hz']); bw = int(r['bandwidth_hz'])
    cen_hz = int(float(r['center_mhz']) * 1e6)
    sdr = SETTINGS.get('sdr', 'bladerf')
    if sdr_profiles is not None:
        cap = sdr_profiles.build_capture(sdr, cen_hz, rate, bw,
                                         gain=r.get('gain', 'auto'),
                                         soapy_driver=r.get('soapy_driver', 'hackrf'))
        fmt = sdr_profiles.fmt_of(sdr)
    else:   # registry unavailable → original bladeRF command + lora.toml format
        cli = shutil.which('bladeRF-cli') or 'bladeRF-cli'
        cap = (f'{shlex.quote(cli)} -e "set frequency rx {cen_hz}; set samplerate rx {rate}; '
               f'set bandwidth rx {bw}; set agc rx on; '
               f'rx config file=/dev/stdout format=bin n=0 buffers=512 samples=32768 '
               f'xfers=64; rx start; rx wait"')
        fmt = r.get('format', 'sc16')
    scripts = os.path.join(_PROJECT, 'scripts', 'lora_detect.py')
    gate = (f'python3 {shlex.quote(scripts)} -r {rate} -b {bw} -c {r["center_mhz"]} '
            f'-t {fmt} --threshold {d["threshold"]} '
            f'--overlap {d["overlap"]} --energy-threshold {d["energy_threshold"]} '
            f'--detect-workers {d["detect_workers"]} --buf-seconds {d["buf_seconds"]} '
            f'--decode --export-iq {shlex.quote(dec["export_dir"])} -d 1')
    return cap + ' | ' + gate


def _kill_stale_pipeline():
    """Kill any orphaned gate / bladeRF from a previous run so the SDR is free.
    The pipeline runs in its own session (setsid) and is detached, so quitting
    the web does NOT auto-kill it — without this, an orphan keeps the bladeRF
    busy and the next Start launches a second pipeline that can't open the
    device and dies in ~1 s ('running' then 'stopped').  These pkill patterns
    only match the gate/bladeRF, never this web process (python3 lora_web.py)."""
    pats = ['lora_detect.py -r']
    if sdr_profiles is not None:   # kill ANY known SDR capture tool, not just bladeRF
        # Use each profile's kill_pat (script name for SoapySDR) — NEVER the bare
        # 'python3' interpreter, which would nuke the web server and the gate.
        pats += sorted({p.get('kill_pat', p['bin']) for p in sdr_profiles.SDR_PROFILES.values()})
    else:
        pats.append('bladeRF-cli')
    for pat in pats:
        try:
            subprocess.run(['pkill', '-9', '-f', pat], timeout=5,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass


def start_pipeline():
    if PIPELINE['running'] and PIPELINE.get('proc') and PIPELINE['proc'].poll() is None:
        return {'ok': False, 'error': 'already running'}
    _kill_stale_pipeline()          # free the SDR from any orphan first
    time.sleep(1.0)                 # let the USB device release
    dec = CFG['decode']
    # Fresh capture log each run: truncate so the tailer only ingests THIS run's
    # packets.  Already-imported data lives in memory and is unaffected → Export
    # still yields imported-old + newly-captured.
    try:
        open(dec.get('packet_log', '/tmp/lora_packets.jsonl'), 'w').close()
    except Exception:
        pass
    env = dict(os.environ)
    env['LORA_PKT_LOG'] = dec['packet_log']
    env['LORA_DECODE_BUDGET_S'] = str(dec['budget_s'])
    env['LORA_DECODE_WORKERS'] = str(dec['workers'])
    env['LORA_KEYS'] = KEYS_PATH    # decoder reads the multi-key list (live mtime reload)
    env['LORA_PSD_FILE'] = PSD_FILE  # gate writes waterfall frames here (reuses its PSD)
    env['LORA_UNKNOWN_REPORT'] = UNKNOWN_PATH  # gate logs unknown-protocol frames here
    # Advanced Options → decoder: which protocols to attempt + whether to surface
    # unknowns (off also lets the decoder early-bail on irrelevant sync words).
    env['LORA_UNKNOWN'] = '1' if SETTINGS.get('unknown') else '0'
    _pr = SETTINGS.get('protocols') or {}
    env['LORA_PROTOCOLS'] = ','.join(k for k in ('meshtastic', 'meshcore', 'lorawan')
                                     if _pr.get(k, True))
    if SETTINGS.get('wide_scan'):
        env['LORA_SCAN_FULL'] = '1'  # sweep all SF×BW (any LoRa), not just presets
    # Max-hold sensitivity is baked in (gate defaults LORA_MAXHOLD on); set
    # LORA_MAXHOLD=0 in the environment only to disable on a very weak CPU.
    _apply_waterfall_flag()          # ensure the on/off marker matches the saved setting
    _apply_unknown_flag()
    if CFG['detect'].get('commit_lag') is not None:
        env['LORA_COMMIT_LAG'] = str(CFG['detect']['commit_lag'])
    os.makedirs(dec['export_dir'], exist_ok=True)
    cmd = _build_pipeline_cmd()
    try:
        proc = subprocess.Popen(cmd, shell=True, env=env, preexec_fn=os.setsid,
                                stdout=subprocess.DEVNULL,
                                stderr=open(PIPELINE_LOG, 'wb'))   # fresh log → health reflects this run only
    except Exception as e:
        PIPELINE['err'] = str(e)
        return {'ok': False, 'error': str(e)}
    PIPELINE.update(proc=proc, running=True, started_at=time.time(), err=None)
    _broadcast({'type': 'stats', 'data': _stats()})
    return {'ok': True}


def stop_pipeline():
    proc = PIPELINE.get('proc')
    if proc and proc.poll() is None:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except Exception:
            try:
                proc.terminate()
            except Exception:
                pass
    _kill_stale_pipeline()          # also reap detached gate/bladeRF children
    PIPELINE.update(proc=None, running=False)
    HEALTH.update(msps=None, warn=None)   # gate gone → clear live readout
    _broadcast({'type': 'stats', 'data': _stats()})
    _broadcast({'type': 'health', 'data': HEALTH})
    return {'ok': True}


def _watch_pipeline():
    while True:
        proc = PIPELINE.get('proc')
        if proc is not None and proc.poll() is not None:
            PIPELINE.update(running=False, proc=None)
            _broadcast({'type': 'stats', 'data': _stats()})
        time.sleep(1.0)


# ---------------------------------------------------------------- routes
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/state')
def api_state():
    with STATE_LOCK:
        pkts = list(PACKETS)[-2000:]
        nodes = [_node_view(n) for n in NODES.values()]
    return jsonify({'packets': pkts, 'nodes': nodes, 'edges': _edges_view(),
                    'stats': _stats(), 'config': {'center_mhz': CFG['radio']['center_mhz'],
                                                  'key': CFG['decode'].get('key')}})


@app.route('/api/packets')
def api_packets():
    limit = int(request.args.get('limit', 2000))
    with STATE_LOCK:
        return jsonify(list(PACKETS)[-limit:])


@app.route('/api/nodes')
def api_nodes():
    with STATE_LOCK:
        return jsonify([_node_view(n) for n in NODES.values()])


@app.route('/api/edges')
def api_edges():
    return jsonify(_edges_view())


@app.route('/api/stats')
def api_stats():
    return jsonify(_stats())


@app.route('/api/mc_nodes')
def api_mc_nodes():
    """Crypto-verified MeshCore node registry (from ADVERT Ed25519 signatures)."""
    with STATE_LOCK:
        out = []
        for h, n in MC_NODES.items():
            out.append({'hash': '0x%02x' % h, 'pubkey': (n.get('pubkey') or '')[:16],
                        'name': n.get('name'), 'role': n.get('role'),
                        'adverts': n.get('count'), 'last': n.get('last'),
                        'lat': n.get('lat'), 'lon': n.get('lon')})
        out.sort(key=lambda x: (x['name'] or x['hash']))
    return jsonify({'nodes': out})


@app.route('/api/health')
def api_health():
    return jsonify(HEALTH)


@app.route('/api/psd')
def api_psd():
    try:
        with open(PSD_FILE, 'rb') as f:
            raw = f.read()
        return jsonify({'b64': _b64.b64encode(raw).decode('ascii'),
                        'center_mhz': float(CFG['radio'].get('center_mhz', 915.0)),
                        'rate_mhz': float(CFG['radio']['rate_hz']) / 1e6})
    except Exception:
        return jsonify({'b64': None})


@app.route('/api/settings', methods=['GET', 'POST'])
def api_settings():
    global SETTINGS
    if request.method == 'POST':
        d = request.get_json(force=True, silent=True) or {}
        if 'autosave' in d:
            SETTINGS['autosave'] = bool(d['autosave'])
        if 'waterfall' in d:
            SETTINGS['waterfall'] = bool(d['waterfall'])
        if 'unknown' in d:
            SETTINGS['unknown'] = bool(d['unknown'])
        if 'protocols' in d and isinstance(d['protocols'], dict):
            pr = dict(SETTINGS.get('protocols') or {'meshtastic': True, 'meshcore': True, 'lorawan': True})
            for k in ('meshtastic', 'meshcore', 'lorawan'):
                if k in d['protocols']:
                    pr[k] = bool(d['protocols'][k])
            SETTINGS['protocols'] = pr
        if 'wide_scan' in d:
            SETTINGS['wide_scan'] = bool(d['wide_scan'])
        if 'sdr' in d and sdr_profiles is not None and d['sdr'] in sdr_profiles.SDR_PROFILES:
            SETTINGS['sdr'] = d['sdr']
        if 'radio' in d and isinstance(d['radio'], dict):
            rad = dict(SETTINGS.get('radio') or {})
            for k in ('rate_hz', 'bandwidth_hz', 'center_mhz'):
                if k in d['radio']:
                    try: rad[k] = float(d['radio'][k])
                    except (TypeError, ValueError): pass
            for k in ('gain', 'soapy_driver'):
                if k in d['radio']:
                    v = str(d['radio'][k]).strip()
                    # soapy_driver reaches a shell=True command — whitelist it.
                    if k == 'soapy_driver' and sdr_profiles is not None:
                        v = sdr_profiles.safe_soapy_driver(v)
                    rad[k] = v
            # Clamp to the (possibly just-changed) SDR's capabilities before saving.
            if sdr_profiles is not None:
                rad = sdr_profiles.clamp_radio(SETTINGS.get('sdr', 'bladerf'), rad)
            SETTINGS['radio'] = rad
        if any(k in d for k in ('autosave', 'waterfall', 'unknown',
                                'protocols', 'wide_scan', 'sdr', 'radio')):
            save_settings(SETTINGS)
            _apply_waterfall_flag()
            _apply_unknown_flag()
    return jsonify(SETTINGS)


@app.route('/api/sdr')
def api_sdr():
    """List supported SDR profiles + the current selection and effective radio cfg."""
    profs = []
    soapy_no_agc = {}
    if sdr_profiles is not None:
        for k, p in sdr_profiles.SDR_PROFILES.items():
            profs.append({'id': k, 'label': p['label'], 'format': p['format'],
                          'tested': p.get('tested', False), 'note': p.get('note', ''),
                          'limits': p.get('limits', {}),
                          'agc_ok': p.get('agc_ok', True),
                          'default_gain': p.get('default_gain', 'auto')})
        # Per-driver AGC overrides for the generic SoapySDR profile, so the UI can
        # react when the driver field changes (e.g. driver=hackrf → no AGC).
        soapy_no_agc = dict(sdr_profiles._SOAPY_NO_AGC)
    return jsonify({'profiles': profs, 'current': SETTINGS.get('sdr', 'bladerf'),
                    'radio': _radio_cfg(), 'soapy_no_agc': soapy_no_agc})


@app.route('/api/sdr/detect', methods=['POST'])
def api_sdr_detect():
    """Run the selected (or posted) SDR's probe command to confirm it's present."""
    if sdr_profiles is None:
        return jsonify({'ok': False, 'error': 'SDR registry unavailable'})
    d = request.get_json(force=True, silent=True) or {}
    sdr = d.get('sdr') or SETTINGS.get('sdr', 'bladerf')
    if sdr not in sdr_profiles.SDR_PROFILES:
        return jsonify({'ok': False, 'error': 'unknown SDR'})
    cmd = sdr_profiles.build_probe(sdr)
    try:
        p = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=12)
        out = (p.stdout or '') + (p.stderr or '')
    except FileNotFoundError:
        return jsonify({'ok': False, 'found': False,
                        'error': 'probe tool not installed (%s)' % sdr_profiles.SDR_PROFILES[sdr]['bin']})
    except Exception as e:
        return jsonify({'ok': False, 'found': False, 'error': str(e)})
    serial = sdr_profiles.parse_serial(sdr, out)
    found = bool(serial) or (p.returncode == 0 and out.strip() != '' and 'not found' not in out.lower()
                             and 'no devices' not in out.lower())
    return jsonify({'ok': True, 'found': found, 'serial': serial,
                    'output': out.strip()[:600]})


@app.route('/api/sdr/limits', methods=['POST'])
def api_sdr_limits():
    """Query the LIVE device for its actual capability ranges so the config tab can
    clamp inputs to what THIS unit supports (e.g. a bladeRF1 = 40 Msps / 28 MHz, not
    the generic profile maxima).  Returns {} if the device has no live-limits probe,
    isn't present, or is held by a running pipeline (the device is single-access) —
    the UI then falls back to the static profile limits."""
    if sdr_profiles is None:
        return jsonify({'ok': False, 'limits': {}})
    d = request.get_json(force=True, silent=True) or {}
    sdr = d.get('sdr') or SETTINGS.get('sdr', 'bladerf')
    if sdr not in sdr_profiles.SDR_PROFILES:
        return jsonify({'ok': False, 'limits': {}})
    # Don't fight a running pipeline for the (single-access) device.
    if PIPELINE.get('running') and PIPELINE.get('proc') and PIPELINE['proc'].poll() is None:
        return jsonify({'ok': False, 'busy': True, 'limits': {}})
    drv = d.get('soapy_driver') or (SETTINGS.get('radio') or {}).get('soapy_driver') or 'hackrf'
    lim = sdr_profiles.query_limits(sdr, soapy_driver=drv)
    return jsonify({'ok': bool(lim), 'limits': lim})


@app.route('/api/keys', methods=['GET'])
def api_keys_get():
    d = load_keys()
    # value-locked built-in key strings, for display in the (greyed) built-in rows
    d['_builtin'] = {'meshtastic': MESH_DEFAULT_KEY, 'meshcore': MESHCORE_DEFAULT_KEY}
    return jsonify(d)


def _mc_bytes16(key):
    """True if `key` decodes to exactly 16 bytes (MeshCore channel crypto is AES-128)."""
    s = (key or '').strip().lower()
    if all(c in '0123456789abcdef' for c in s) and len(s) == 32:
        return True
    try:
        return len(_b64.b64decode(key.strip() + '=' * (-len(key.strip()) % 4))) == 16
    except Exception:
        return False


@app.route('/api/keys', methods=['POST'])
def api_keys_add():
    d = request.get_json(force=True, silent=True) or {}
    key = (d.get('key') or '').strip()
    proto = 'meshcore' if d.get('protocol') == 'meshcore' else 'meshtastic'
    if proto == 'meshcore':
        if not _mc_bytes16(key):
            return jsonify({'ok': False, 'error': 'MeshCore channel key must be 16 bytes (AES-128), hex or base64'})
    elif not _key_valid(key):
        return jsonify({'ok': False, 'error': 'key must be base64/hex for 16, 24 or 32 bytes'})
    nodes = _parse_nodes(d.get('nodes') if d.get('nodes') is not None else d.get('node'))
    scope = 'nodes' if (d.get('scope') in ('node', 'nodes') and nodes) else 'all'
    if d.get('scope') in ('node', 'nodes') and not nodes:
        return jsonify({'ok': False, 'error': 'list at least one node id for a node-scoped key'})
    kc = load_keys()
    try:
        prio = int(d['priority'])
    except (KeyError, TypeError, ValueError):
        prio = 100 if scope == 'all' else 10   # all-scope after default; node-scope before
    kc['keys'].append({
        'id': 'k%d' % int(time.time() * 1000),
        'protocol': proto,
        'label': (d.get('label') or 'key').strip(),
        'key': key,
        'enabled': d.get('enabled') is not False,
        'scope': scope,
        'nodes': nodes,
        'notes': (d.get('notes') or '').strip(),
        'priority': prio,
    })
    save_keys(kc)
    return jsonify({'ok': True, 'keys': kc})


@app.route('/api/keys/<kid>/update', methods=['POST'])
def api_keys_update(kid):
    """Update a CUSTOM key's mutable fields (enabled / notes / scope / nodes /
    priority / label).  The key VALUE is never editable."""
    d = request.get_json(force=True, silent=True) or {}
    kc = load_keys()
    for k in kc['keys']:
        if k.get('id') != kid:
            continue
        if 'enabled' in d:  k['enabled'] = bool(d['enabled'])
        if 'notes' in d:    k['notes'] = (d.get('notes') or '').strip()
        if 'label' in d:    k['label'] = (d.get('label') or k.get('label') or 'key').strip()
        if 'nodes' in d or 'scope' in d:
            nodes = _parse_nodes(d.get('nodes'))
            k['scope'] = 'nodes' if (d.get('scope') in ('node', 'nodes') and nodes) else 'all'
            k['nodes'] = nodes
        if 'priority' in d:
            try: k['priority'] = int(d['priority'])
            except (TypeError, ValueError): pass
        save_keys(kc)
        return jsonify({'ok': True, 'keys': kc})
    return jsonify({'ok': False, 'error': 'unknown key id'})


@app.route('/api/keys/meshcore-default', methods=['POST'])
def api_keys_meshcore_default():
    """Update the value-locked MeshCore public/default built-in (enable / nodes /
    priority / notes only)."""
    d = request.get_json(force=True, silent=True) or {}
    kc = load_keys()
    md = kc['meshcore_default']
    if 'enabled' in d:  md['enabled'] = bool(d['enabled'])
    if 'notes' in d:    md['notes'] = (d.get('notes') or '').strip()
    if 'nodes' in d:    md['nodes'] = _parse_nodes(d['nodes'])
    if 'priority' in d:
        try: md['priority'] = int(d['priority'])
        except (TypeError, ValueError): pass
    save_keys(kc)
    return jsonify({'ok': True, 'keys': kc})


@app.route('/api/keys/<kid>/priority', methods=['POST'])
def api_keys_priority(kid):
    d = request.get_json(force=True, silent=True) or {}
    try:
        prio = int(d.get('priority'))
    except (TypeError, ValueError):
        return jsonify({'ok': False, 'error': 'priority must be an integer'})
    kc = load_keys()
    for k in kc['keys']:
        if k.get('id') == kid:
            k['priority'] = prio
    save_keys(kc)
    return jsonify({'ok': True, 'keys': kc})


@app.route('/api/keys/<kid>', methods=['DELETE'])
def api_keys_del(kid):
    kc = load_keys()
    kc['keys'] = [k for k in kc['keys'] if k.get('id') != kid]
    save_keys(kc)
    return jsonify({'ok': True, 'keys': kc})


@app.route('/api/keys/default', methods=['POST'])
def api_keys_default():
    d = request.get_json(force=True, silent=True) or {}
    kc = load_keys()
    if 'try_default' in d:
        kc['try_default'] = bool(d['try_default'])
    if 'default_nodes' in d:    # [] = all nodes; a list = default only for those
        kc['default_nodes'] = _parse_nodes(d['default_nodes'])
    if 'default_priority' in d:
        try:
            kc['default_priority'] = int(d['default_priority'])
        except (TypeError, ValueError):
            pass
    if 'default_notes' in d:
        kc['default_notes'] = (d.get('default_notes') or '').strip()
    save_keys(kc)
    return jsonify({'ok': True, 'keys': kc})


@app.route('/api/keys/export')
def api_keys_export():
    return Response(json.dumps(load_keys(), indent=2), mimetype='application/json',
                    headers={'Content-Disposition': 'attachment; filename=lora_keys.json'})


@app.route('/api/keys/import', methods=['POST'])
def api_keys_import():
    try:
        d = json.loads(request.get_data(as_text=True) or '{}')
    except Exception:
        return jsonify({'ok': False, 'error': 'invalid JSON'})
    merge = request.args.get('merge') == '1'
    kc = load_keys() if merge else {'try_default': True, 'default_nodes': [],
                                    'default_priority': 50, 'keys': []}
    if 'try_default' in d:
        kc['try_default'] = bool(d['try_default'])
    if 'default_nodes' in d:
        kc['default_nodes'] = _parse_nodes(d['default_nodes'])
    if 'default_priority' in d:
        try:
            kc['default_priority'] = int(d['default_priority'])
        except (TypeError, ValueError):
            pass
    seen = {k.get('key') for k in kc['keys']}
    for k in d.get('keys', []):
        if _key_valid(k.get('key', '')) and k.get('key') not in seen:
            k.setdefault('id', 'k%d' % int(time.time() * 1000 + len(kc['keys'])))
            k['nodes'] = _parse_nodes(k.get('nodes') if k.get('nodes') is not None else k.get('node'))
            k.pop('node', None)
            k['scope'] = 'nodes' if k['nodes'] else 'all'
            try:
                k['priority'] = int(k.get('priority'))
            except (TypeError, ValueError):
                k['priority'] = 100 if k['scope'] == 'all' else 10
            kc['keys'].append(k)
            seen.add(k.get('key'))
    save_keys(kc)
    return jsonify({'ok': True, 'keys': kc})


_ALERT_TYPES = {'new_node', 'node', 'content', 'type', 'signal', 'channel', 'geofence'}


@app.route('/api/alerts', methods=['GET'])
def api_alerts_get():
    return jsonify(load_alerts())


@app.route('/api/alerts', methods=['POST'])
def api_alerts_add():
    d = request.get_json(force=True, silent=True) or {}
    if d.get('type') not in _ALERT_TYPES:
        return jsonify({'ok': False, 'error': 'unknown alert type'})
    ac = load_alerts()
    rule = {
        'id': 'a%d' % int(time.time() * 1000),
        'type': d['type'],
        'label': (d.get('label') or '').strip(),
        'enabled': d.get('enabled', True) is not False,
        'notify': bool(d.get('notify', True)),
        'sound': bool(d.get('sound', False)),
    }
    for k in ('node', 'dir', 'keyword', 'port', 'op', 'value', 'chan',
              'lat', 'lon', 'radius', 'event'):
        if d.get(k) is not None and d.get(k) != '':
            rule[k] = d[k]
    ac['rules'].append(rule)
    save_alerts(ac)
    return jsonify({'ok': True, 'alerts': ac})


@app.route('/api/alerts/<rid>', methods=['POST'])
def api_alerts_update(rid):
    d = request.get_json(force=True, silent=True) or {}
    ac = load_alerts()
    for r in ac['rules']:
        if r.get('id') == rid:
            for k in ('enabled', 'notify', 'sound'):
                if k in d:
                    r[k] = bool(d[k])
    save_alerts(ac)
    return jsonify({'ok': True, 'alerts': ac})


@app.route('/api/alerts/<rid>', methods=['DELETE'])
def api_alerts_del(rid):
    ac = load_alerts()
    ac['rules'] = [r for r in ac['rules'] if r.get('id') != rid]
    save_alerts(ac)
    return jsonify({'ok': True, 'alerts': ac})


@app.route('/api/nodes/meta', methods=['GET'])
def api_nodes_meta():
    return jsonify(load_node_meta())


@app.route('/api/nodes/<nid>/meta', methods=['POST'])
def api_node_meta_set(nid):
    d = request.get_json(force=True, silent=True) or {}
    meta = load_node_meta()
    m = meta.get(nid, {})
    for k in ('label', 'notes', 'watch', 'watch_mode'):
        if k in d:
            if k == 'watch':
                m['watch'] = bool(d['watch'])
            elif k == 'watch_mode':
                m['watch_mode'] = d['watch_mode'] if d['watch_mode'] in ('highlight', 'alert', 'both') else 'highlight'
            else:
                m[k] = (d[k] or '').strip()
    # drop empties so the file stays tidy
    m = {k: v for k, v in m.items() if v not in ('', None, False)}
    if m:
        meta[nid] = m
    else:
        meta.pop(nid, None)
    save_node_meta(meta)
    _broadcast({'type': 'nodemeta', 'data': {'id': nid, 'meta': meta.get(nid, {})}})
    return jsonify({'ok': True, 'meta': meta})


# ---- Unknown-protocol developer report ----
UNKNOWN_PATH = os.path.join(_HERE, 'lora_unknown.jsonl')
UNKNOWN_OFF = UNKNOWN_PATH + '.off'


def _apply_unknown_flag():
    try:
        if SETTINGS.get('unknown', False):
            if os.path.exists(UNKNOWN_OFF):
                os.remove(UNKNOWN_OFF)
        else:
            open(UNKNOWN_OFF, 'w').close()
    except Exception:
        pass


@app.route('/api/unknown')
def api_unknown():
    rows = []
    try:
        with open(UNKNOWN_PATH) as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    except Exception:
        pass
    if request.args.get('download'):
        return Response(''.join(json.dumps(r) + '\n' for r in rows),
                        mimetype='application/json',
                        headers={'Content-Disposition': 'attachment; filename=lora_unknown_report.jsonl'})
    return jsonify({'count': len(rows), 'rows': rows[-500:]})


@app.route('/api/unknown', methods=['DELETE'])
def api_unknown_clear():
    try:
        open(UNKNOWN_PATH, 'w').close()
    except Exception:
        pass
    return jsonify({'ok': True})


@app.route('/api/decode_retry', methods=['POST'])
def api_decode_retry():
    """Re-attempt decryption of a stored ENCRYPTED packet (right-click → Decode).
    Body: {pktid, hops, key_id}.  key_id is a key id, '__default__', or 'all'
    (or omitted = all keys)."""
    if dhv is None:
        return jsonify({'ok': False, 'error': 'decoder module unavailable'})
    d = request.get_json(force=True, silent=True) or {}
    pktid = d.get('pktid')
    try:
        hops = int(d.get('hops'))
    except (TypeError, ValueError):
        hops = None
    with STATE_LOCK:
        rec = next((r for r in PACKETS if r.get('pktid') == pktid and r.get('hops') == hops), None)
    if rec is None:
        return jsonify({'ok': False, 'error': 'packet not found'})
    if rec.get('decrypted'):
        return jsonify({'ok': True, 'already': True, 'packet': rec})
    enc_hex = rec.get('enc_hex')
    if not enc_hex:
        return jsonify({'ok': False, 'error': 'no stored ciphertext for this packet '
                        '(captured before retry support was added)'})
    kc = load_keys()
    keymap = {k.get('id'): k for k in kc.get('keys', [])}
    sel = d.get('key_id') or 'all'
    cands = []                      # (key_str_or_None, label)
    if sel == '__default__':
        cands.append((None, 'default'))
    elif sel != 'all':
        if sel not in keymap:
            return jsonify({'ok': False, 'error': 'unknown key'})
        cands.append((keymap[sel].get('key'), keymap[sel].get('label', 'custom')))
    else:                           # all: default (if enabled) + every custom, by priority
        if kc.get('try_default', True):
            cands.append((None, 'default'))
        for k in sorted(kc.get('keys', []), key=lambda x: x.get('priority', 100)):
            cands.append((k.get('key'), k.get('label', 'custom')))
    result = used = None
    for kstr, label in cands:
        result = dhv.retry_decrypt(enc_hex, rec.get('pktid'), rec.get('from'), kstr)
        if result:
            used = label
            break
    if not result:
        return jsonify({'ok': False, 'error': 'none of the tried keys decoded this packet'})
    with STATE_LOCK:
        was_enc = not rec.get('decrypted')
        rec.update(result)
        rec.pop('enc_hex', None)
        if used and used != 'default':
            rec['key'] = used
        frm = rec.get('from') or '?'
        n = NODES.get(frm)
        if n and was_enc:
            n['encrypted'] = max(0, n['encrypted'] - 1)
            n['decoded'] += 1
            if rec.get('text'):
                n['last_text'] = rec['text']
            e = EDGES.get((frm, rec.get('to') or '?'))
            if e:
                e['encrypted'] = max(0, e['encrypted'] - 1)
                e['decoded'] += 1
    _broadcast({'type': 'update', 'data': rec})
    _broadcast({'type': 'stats', 'data': _stats()})
    return jsonify({'ok': True, 'packet': rec, 'key': used})


@app.route('/api/export')
def api_export():
    """Download the current session as CSV (default) or JSONL."""
    fmt = request.args.get('fmt', 'csv')
    with STATE_LOCK:
        pkts = list(PACKETS)
    stamp = time.strftime('%Y%m%d_%H%M%S')
    if fmt == 'json':
        body = '\n'.join(json.dumps(p, separators=(',', ':')) for p in pkts)
        return Response(body, mimetype='application/x-ndjson',
                        headers={'Content-Disposition':
                                 f'attachment; filename=lora_session_{stamp}.jsonl'})
    import io, csv
    buf = io.StringIO(); w = csv.writer(buf); w.writerow(_CSV_COLS)
    for p in pkts:
        w.writerow([p.get(c, '') for c in _CSV_COLS])
    return Response(buf.getvalue(), mimetype='text/csv',
                    headers={'Content-Disposition':
                             f'attachment; filename=lora_session_{stamp}.csv'})


@app.route('/api/import', methods=['POST'])
def api_import():
    """Load a previously-exported session (JSONL or CSV) into the current state,
    so a user can resume/append to it (then Start the pipeline → new data appends,
    Export → old + new).  Deduped by (pktid,hops) like live packets."""
    data = request.get_data(as_text=True) or ''
    lines = [l for l in data.splitlines() if l.strip()]
    n = 0
    if not lines:
        return jsonify({'imported': 0})
    if lines[0].lstrip().startswith('{'):                      # JSONL
        for l in lines:
            try:
                if ingest(json.loads(l)) is not None:
                    n += 1
            except Exception:
                pass
    else:                                                      # CSV
        import io, csv
        for row in csv.DictReader(io.StringIO(data)):
            try:
                if ingest(_csv_row_to_rec(row)) is not None:
                    n += 1
            except Exception:
                pass
    _broadcast({'type': 'stats', 'data': _stats()})
    return jsonify({'imported': n})


@app.route('/api/pipeline/start', methods=['POST'])
def api_start():
    return jsonify(start_pipeline())


@app.route('/api/pipeline/stop', methods=['POST'])
def api_stop():
    return jsonify(stop_pipeline())


@app.route('/api/clear', methods=['POST'])
def api_clear():
    with STATE_LOCK:
        PACKETS.clear(); SEEN.clear(); NODES.clear(); EDGES.clear()
    # also truncate the persistent log so cleared data doesn't reload on restart
    # (the tailer detects the shrink and resets its read position)
    for _p in (CFG['decode'].get('packet_log', '/tmp/lora_packets.jsonl'), AUTOSAVE_PATH):
        try:
            open(_p, 'w').close()
        except Exception:
            pass
    _broadcast({'type': 'stats', 'data': _stats()})
    return jsonify({'ok': True})


@app.route('/api/stream')
def api_stream():
    def gen():
        q = queue.Queue(maxsize=1000)
        with SUBS_LOCK:
            SUBS.append(q)
        try:
            yield 'event: stats\ndata: %s\n\n' % json.dumps(_stats())
            while True:
                try:
                    evt = q.get(timeout=15)
                    yield 'event: %s\ndata: %s\n\n' % (evt['type'], json.dumps(evt['data']))
                except queue.Empty:
                    yield ': keepalive\n\n'
        finally:
            with SUBS_LOCK:
                if q in SUBS:
                    SUBS.remove(q)
    return Response(gen(), mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default=None)
    ap.add_argument('--host', default=None)
    ap.add_argument('--port', type=int, default=None)
    a = ap.parse_args()
    global CFG
    CFG = load_config(a.config)
    host = a.host or CFG['web'].get('host', '127.0.0.1')
    port = a.port or int(CFG['web'].get('port', 5000))
    log_path = CFG['decode'].get('packet_log', '/tmp/lora_packets.jsonl')
    # Kill the pipeline when the web exits (Ctrl-C / SIGTERM) so it never orphans
    # and holds the SDR — the cause of the "Start → running → stopped" bug.
    import atexit
    atexit.register(_kill_stale_pipeline)
    def _on_signal(_s, _f):
        _kill_stale_pipeline(); os._exit(0)
    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)
    threading.Thread(target=tail_packet_log, args=(log_path,), daemon=True).start()
    try:
        open(PIPELINE_LOG, 'w').close()   # fresh session: drop any prior run's health log
    except Exception:
        pass
    threading.Thread(target=_watch_pipeline, daemon=True).start()
    threading.Thread(target=_autosave_loop, daemon=True).start()
    _apply_waterfall_flag()   # reflect the saved waterfall on/off to the gate marker
    _apply_unknown_flag()
    threading.Thread(target=_tail_health, daemon=True).start()
    threading.Thread(target=_tail_psd, daemon=True).start()
    print(f"lora_web: config={CFG.get('_path')}  tailing={log_path}  "
          f"serving http://{host}:{port}", flush=True)
    if host not in ('127.0.0.1', 'localhost', '::1'):
        print("lora_web: *** WARNING: bound to %s (non-localhost) and the UI has NO "
              "authentication. Anyone who can reach this port can control the SDR and "
              "read intercepted traffic. Use 127.0.0.1, or put a reverse proxy + auth "
              "in front on trusted networks only. ***" % host, file=sys.stderr, flush=True)
    app.run(host=host, port=port, threaded=True, debug=False)


if __name__ == '__main__':
    main()
