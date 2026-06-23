#!/usr/bin/env python3
"""LORA Wideband Decoder — web UI backend.

Data model: the pipeline (src/detector.py) writes one structured JSON record per
packet (decoded OR encrypted-header-only) to a JSONL log.  This backend TAILS
that log, dedups by (pktid,hop), and aggregates:
  - packet feed       (every intercepted packet, sortable/filterable)
  - per-node stats    (avg RSSI of a node's own transmissions, counts, channels)
  - conversations     (who talks to whom — node->node edges, the big picture)
  - encrypted traffic (headers visible, contents hidden — other-key packets)
Served over a small JSON API + Server-Sent Events; pipeline launched from
lora.toml so there are no flags to pass.

Don't invoke this directly — use `python3 run/web.py` from the repo root.
"""
import os, sys, json, time, threading, queue, subprocess, argparse, shutil, signal
from collections import deque, defaultdict
from flask import Flask, jsonify, request, Response, render_template

# --- path resolution ---
# app.py lives at src/web/app.py — walk up to find src/ (backend imports) and
# the repo root (sibling modules + persistent state).
_HERE = os.path.dirname(os.path.abspath(__file__))    # src/web/
_SRC = os.path.dirname(_HERE)                          # src/
_PROJECT = os.path.dirname(_SRC)                       # repo root
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
# Persistent runtime state (settings, keys, alerts, autosave) — kept in a
# stable project-root location so it isn't co-mingled with the code.
_DATA_DIR = os.path.join(_PROJECT, 'lora_web')
try:
    os.makedirs(_DATA_DIR, exist_ok=True)
except OSError:
    _DATA_DIR = _HERE
try:
    from lora_config import load_config
except Exception:
    load_config = None
try:
    import decoder as dhv   # for web-side retry-decryption of stored packets
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
                'decode': {'budget_s': 10.0, 'workers': -1,
                           'export_dir': '/dev/shm/live_caps',
                           'packet_log': '/tmp/lora_packets.jsonl', 'key': 'default'},
                'web': {'host': '127.0.0.1', 'port': 5000}, '_path': None}

app = Flask(__name__)
CFG = load_config()

# ---------------------------------------------------------------- settings
SETTINGS_PATH = os.path.join(_DATA_DIR, "web_settings.json")
def load_settings():
    _defaults = {'autosave': False, 'waterfall': True,
                 'wide_scan': False,
                 # Advanced Options (Config) — persist across sessions:
                 'protocols': {'meshtastic': True, 'meshcore': True, 'lorawan': True,
                               'loramesher': True, 'lora_aprs': True, 'reticulum': True,
                               'disaster_radio': True, 'ebyte_lora': True, 'radiohead': True},
                 'unknown': False,          # master: surface unidentified protocols (default OFF)
                 'fingerprint': True,       # RF hardware fingerprinting (Mystery Devices + clustering)
                 'ldro_fallback': True,     # retry payload with opposite LDRO on CRC fail (forced-LDRO TX)
                 'iq_invert': False,        # conjugate the input stream to decode IQ-inverted (satellite/downlink) TX
                 'sdr': 'bladerf',          # selected SDR profile (persists across sessions)
                 'radio': {},               # web overrides of lora.toml [radio] (rate/bw/center/gain)
                 # Pipeline tuning — overrides lora.toml [detect] when set.  Defaults
                 # match the well-tuned values; users only edit these to chase weak
                 # signals, suppress false positives in noisy RF, or absorb bursts.
                 'tune': {'threshold': 0.55, 'energy_threshold': 12.0, 'buf_seconds': 16}}
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
KEYS_PATH = os.path.join(_DATA_DIR, "lora_keys.json")
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
    # Node identity keypairs (for DM decrypt: MeshCore TXT_MSG + Meshtastic PKI DM).
    # Each entry: {id, protocol, label, pub, priv, node_id?, enabled, notes}.  Stored
    # as hex.  MeshCore priv is 64-byte (Ed25519 scalar||nonce_prefix); Meshtastic
    # priv is 32-byte raw X25519.  node_id is the Meshtastic !xxxxxxxx address (only
    # meaningful for meshtastic protocol).
    d.setdefault('identities', [])
    for k in d['identities']:
        k.setdefault('protocol', 'meshcore')
        k.setdefault('enabled', True)
        k.setdefault('notes', '')
        k.setdefault('node_id', '')
    return d


def save_keys(d):
    try:
        with open(KEYS_PATH, 'w') as f:
            json.dump(d, f, indent=2)
    except Exception:
        pass


# Alert/trigger rules (persist across sessions).  Evaluated client-side; stored here.
ALERTS_PATH = os.path.join(_DATA_DIR, "lora_alerts.json")


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
NODES_META_PATH = os.path.join(_DATA_DIR, "lora_nodes.json")


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
KNOWN_MESH = set()                # Meshtastic node ids decoded as a SENDER (proven real).
                                  # Auto-populates from `from` field of any successfully
                                  # decoded Meshtastic packet (line ~575).  Every Meshtastic
                                  # node periodically broadcasts NodeInfo, so this self-builds
                                  # within minutes of mesh activity — no manual configuration.
PENDING_ENC = []                  # encrypted unicast DMs held until a node corroborates them
# ---- Mystery devices: aggregate fingerprints of UNKNOWN protocol packets ----
# Key = (freq_mhz rounded to 3 dp, first-8-hex-chars of raw_hex).  This lets
# the user see "5 sightings of one device" instead of 5 isolated 'unknown'
# rows when a repeating transmitter (e.g. RadioLib raw, a custom user
# protocol) is on-air.  Capped at 1000 entries with LRU eviction.
UNKNOWN_FP = {}                   # fingerprint_key -> aggregate dict (see _bump_unknown_fp)
UNKNOWN_FP_CAP = 1000             # max distinct fingerprints to track
# ====================================================================
# ONLINE DEVICE FINGERPRINT CLUSTERING (content-addressable, no training)
# ====================================================================
# Every decoded packet immediately gets a device_id.  For KNOWN protocols
# (Meshtastic/MeshCore/LoRaWAN/LoRaMesher) the device_id IS the protocol
# identity field.  For UNKNOWN protocols, we cluster on UMOP features
# using fixed per-feature noise estimates — same device → same cluster ID,
# new device → new cluster ID.  Sub-millisecond per packet, scales to N
# devices (O(N_clusters) distance comparisons), works from packet 1.
#
# Calibrated thresholds (offline cross-validation): SNR≥30dB filter,
# threshold=2.0σ Mahalanobis-like, EWMA centroid updates.  Empirical
# 90-96% purity per device for same-batch hardware; higher for mixed-vendor.

_DEVICE_FP_SIGMA = {
    # TCXO carrier offset — same-batch discriminator at sub-Hz resolution.
    # Each SX1262 chip has ±2 ppm initial trim (±1800 Hz at 915 MHz); 30
    # same-batch boards land at ~120 Hz average separation, distinguishable.
    # Sigma 100 Hz accepts within-burst drift but separates distinct chips.
    'abs_tx_hz': 100.0,
    # PA-onset / EOB transient features — hardware-specific timing
    # characteristics independent of signal strength (normalized to peak).
    # Memory confirms these hold discrimination down to ~SNR 15 dB.
    'onset_rise_us': 15.0,         # PA slew rate, timing-based — noise-robust
    'onset_overshoot_pct': 0.10,   # Regulator transient at turn-on — strongest
                                   # onset feature per offline study (SNR 1.1-1.4)
    'onset_mid_slope': 0.0005,     # Mid-rise slope (normalized) — noise-robust
    'eob_fall_us': 15.0,           # PA fall time, timing-based — noise-robust
    'eob_mid_fall_slope': 0.001,   # Mid-fall slope — strongest single EOB feature
    'eob_pre_decay_over_pct': 0.5, # Pre-shutdown overshoot ratio
    # Dropped: am_ripple_pct, amp_per_sym_pct, pn_slope (dechirp-derived
    # UMOP features have SNR < 1 for same-batch — they dilute rather than
    # amplify discrimination, per the device-fingerprinting study).
}
_DEVICE_FP_THRESHOLD = 1.5       # Online DBSCAN-style "eps": maximum standardized
                                 # Mahalanobis distance to match an existing cluster.
                                 # 1.5 + min_samples=3 achieves 96% purity on
                                 # same-batch bench hardware (vs ~80% with the
                                 # prior threshold-only algorithm).
_DEVICE_FP_MIN_SAMPLES = 3       # Density gate: a CANDIDATE region must accumulate
                                 # ≥N packets within eps before promoting to a
                                 # visible device cluster.  DBSCAN min_samples
                                 # applied online — singleton noise never
                                 # promotes, which is what delivers high purity.
_DEVICE_FP_SNR_MIN = 18.0        # Lowered from 30 dB.  Empirically: timing
                                 # features hold discrimination SNR ≈ 0.6-1.0
                                 # down to 15 dB capture SNR.  At <18 dB the
                                 # decoder itself starts struggling so we
                                 # rarely have packets below that anyway.
_DEVICE_FP_LOG_CONFIDENCE = 0.5  # Modest confidence floor — combined with the
                                 # purity gate below, this gives 100% accuracy
                                 # at ~19% coverage on the verification tests.
_DEVICE_FP_MIN_PURITY = 0.90     # Cluster's Meshtastic-labeled purity must be
                                 # ≥90% before its centroid can be trusted to
                                 # ID arbitrary unknown-protocol packets.  Pure
                                 # clusters by definition can't lie about which
                                 # device they represent.
_DEVICE_FP_MIN_CLUSTER_SIZE = 5  # cluster must have ≥N samples before its
                                 # centroid is trusted for attribution
_DEVICE_FP_FREEZE_AT = 15        # freeze centroid+variance after N samples to
                                 # prevent drift toward outliers from neighboring
                                 # devices (the "vacuum cleaner" problem)
_DEVICE_FP_FILE = os.path.join(os.path.dirname(__file__), 'lora_device_clusters.json')
DEVICE_CLUSTERS = []             # list of PROMOTED cluster dicts (visible devices)
_FP_CANDIDATES = []              # list of CANDIDATE clusters waiting for density
                                 # promotion.  Same shape as DEVICE_CLUSTERS but
                                 # not surfaced to UI / not used for attribution.
_DEVICE_FP_NEXT_ID = [1]


def _fp_extract_features(rec):
    """Extract UMOP features used for fingerprint clustering.  Returns None
    if signal quality is too low to cluster reliably."""
    hw = rec.get('hw_fp') or {}
    snr = hw.get('signal_snr_db')
    if snr is None or float(snr) < _DEVICE_FP_SNR_MIN:
        return None
    feats = {}
    pc = hw.get('precise_carrier_hz')
    fm = rec.get('freq_mhz')
    if pc is not None and fm is not None:
        try: feats['abs_tx_hz'] = float(fm) * 1e6 + float(pc)
        except (TypeError, ValueError): pass
    for k in _DEVICE_FP_SIGMA:
        if k == 'abs_tx_hz': continue
        v = hw.get(k)
        if v is not None:
            try: feats[k] = float(v)
            except (TypeError, ValueError): pass
    return feats if len(feats) >= 5 else None


def _fp_distance(feats, centroid, cluster_std=None):
    """Standardized Mahalanobis-like distance.

    Uses the STATIC per-feature SIGMA, optionally TIGHTENED by the cluster's
    observed std (cluster_std is only allowed to make the cluster's gate
    TIGHTER, never wider).  This prevents the "vacuum cleaner" feedback loop
    where a cluster absorbing cross-device packets grows its own std, which
    widens its gate, which absorbs more cross-device packets.
    """
    n = 0; d = 0.0
    for k, v in feats.items():
        if k not in centroid or k not in _DEVICE_FP_SIGMA: continue
        base_sigma = _DEVICE_FP_SIGMA[k]
        if cluster_std and k in cluster_std and cluster_std[k] > 0:
            # Cluster std is only used if it's TIGHTER than the static sigma.
            sigma = min(cluster_std[k], base_sigma)
            # But never tighter than 10% of static — avoids overfit on N=2.
            sigma = max(sigma, base_sigma * 0.1)
        else:
            sigma = base_sigma
        d += ((v - centroid[k]) / sigma) ** 2
        n += 1
    return (d / n) ** 0.5 if n > 0 else float('inf')


def _fp_assign_device_id(rec):
    """Find or create a device cluster for this packet.

    Returns a (cluster_id, confidence) tuple.  The cluster is ALWAYS assigned
    (so we can grow the centroid).  But confidence determines whether the
    fingerprint is RELIABLE enough to log as an attribution:

      - confidence ≥ _DEVICE_FP_LOG_CONFIDENCE → cluster_id is trustworthy
      - confidence < threshold → cluster exists but attribution is uncertain
        (the packet may genuinely be from this device, or may be ambiguous)

    Caller should check confidence before claiming the device_id is authoritative.
    Sub-millisecond per packet.
    """
    feats = _fp_extract_features(rec)
    if feats is None:
        return None, 0.0
    ts = rec.get('ts') or time.time()
    # Find nearest two clusters using each cluster's OWN observed variance
    # (so tight clusters demand tight matches, wide clusters absorb their own
    # natural spread).
    best_idx = -1; best_dist = float('inf')
    next_dist = float('inf')
    for i, c in enumerate(DEVICE_CLUSTERS):
        # Use cluster's std only once it has enough samples for std to be meaningful
        cstd = c.get('std') if c.get('count', 0) >= 5 else None
        d = _fp_distance(feats, c['centroid'], cstd)
        if d < best_dist:
            next_dist = best_dist
            best_dist = d
            best_idx = i
        elif d < next_dist:
            next_dist = d
    if best_idx >= 0 and best_dist <= _DEVICE_FP_THRESHOLD:
        c = DEVICE_CLUSTERS[best_idx]
        # FROZEN centroid logic: once a cluster has accumulated enough samples
        # (_FREEZE_AT), stop updating its centroid and variance.  Prevents
        # outlier drift / "vacuum cleaner" pull toward neighboring devices.
        if c['count'] < _DEVICE_FP_FREEZE_AT:
            # Still learning — EWMA centroid + variance update
            alpha = min(0.1, 1.0 / (c['count'] + 1))
            for k, v in feats.items():
                old_mean = c['centroid'].get(k, v)
                c['centroid'][k] = old_mean * (1 - alpha) + v * alpha
                if 'var' not in c: c['var'] = {}
                old_var = c['var'].get(k, 0.0)
                new_mean = c['centroid'][k]
                c['var'][k] = (1 - alpha) * old_var + alpha * (v - new_mean) ** 2
            c['std'] = {k: var ** 0.5 for k, var in c.get('var', {}).items()}
        c['count'] += 1
        c['last_ts'] = ts
        # Track Meshtastic-labeled traffic that lands in this cluster so we can
        # compute its purity (used as a runtime gate for high-confidence ID).
        if rec.get('proto') == 'meshtastic' and rec.get('hops') == 0 and rec.get('from'):
            if 'from_seen' not in c: c['from_seen'] = {}
            f = rec['from']
            c['from_seen'][f] = c['from_seen'].get(f, 0) + 1
        # Compute confidence: distance fit × margin × cluster maturity
        distance_fit = max(0.0, 1.0 - best_dist / _DEVICE_FP_THRESHOLD)
        if next_dist == float('inf'):
            margin = 1.0   # only cluster — no competition
        else:
            margin = max(0.0, min(1.0, (next_dist - best_dist) / max(best_dist, 0.1)))
        maturity = min(1.0, c['count'] / 20.0)
        confidence = distance_fit * 0.5 + margin * 0.3 + maturity * 0.2
        # Cluster must be mature enough to trust at all
        if c['count'] < _DEVICE_FP_MIN_CLUSTER_SIZE:
            confidence = min(confidence, 0.5)   # cap at low confidence
        # Downgrade confidence if capture was CLIPPED at the LNA — the most
        # discriminative features (amplitude) are then unreliable.
        hw = rec.get('hw_fp') or {}
        if hw.get('clipped'):
            confidence = min(confidence, 0.6)
        return c['id'], confidence
    # No promoted cluster matched.  Try CANDIDATE clusters (pre-promotion).
    # This is the DBSCAN density rule applied online: a region must accumulate
    # min_samples points within eps before becoming a "device."  Singletons and
    # noise never get promoted, which is what delivers high per-cluster purity.
    best_ci = -1; best_cd = float('inf')
    for i, cand in enumerate(_FP_CANDIDATES):
        d = _fp_distance(feats, cand['centroid'])
        if d < best_cd:
            best_cd = d; best_ci = i
    if best_ci >= 0 and best_cd <= _DEVICE_FP_THRESHOLD:
        cand = _FP_CANDIDATES[best_ci]
        cand['samples'].append(dict(feats))
        # Re-center candidate as mean of its samples
        new_centroid = {}
        keys = set()
        for s in cand['samples']:
            keys.update(s.keys())
        for k in keys:
            vals = [s[k] for s in cand['samples'] if k in s]
            if vals: new_centroid[k] = sum(vals) / len(vals)
        cand['centroid'] = new_centroid
        cand['last_ts'] = ts
        # Promotion check: enough density → real cluster
        if len(cand['samples']) >= _DEVICE_FP_MIN_SAMPLES:
            cid = f"dev_{_DEVICE_FP_NEXT_ID[0]:04x}"
            _DEVICE_FP_NEXT_ID[0] += 1
            c = {
                'id': cid, 'centroid': cand['centroid'],
                'count': len(cand['samples']),
                'first_ts': cand['first_ts'], 'last_ts': ts,
            }
            # Initialize std/var from candidate samples for proper distance calc
            n = len(cand['samples'])
            if n >= 2:
                var = {}
                for k in cand['centroid']:
                    vals = [s[k] for s in cand['samples'] if k in s]
                    if len(vals) >= 2:
                        m = cand['centroid'][k]
                        var[k] = sum((v - m) ** 2 for v in vals) / len(vals)
                c['var'] = var
                c['std'] = {k: v ** 0.5 for k, v in var.items()}
            DEVICE_CLUSTERS.append(c)
            _FP_CANDIDATES.pop(best_ci)
            # Returning the promoted id allows the just-promoted packet to be
            # attributed (the prior candidate-pending packets had no id and are
            # not retroactively attributed — they're already past the API).
            return cid, 0.5
        # Still a candidate — return no attribution
        return None, 0.0
    # New candidate (first sighting of this signature region)
    _FP_CANDIDATES.append({
        'centroid': dict(feats), 'samples': [dict(feats)],
        'first_ts': ts, 'last_ts': ts,
    })
    return None, 0.0


def _fp_save():
    try:
        with open(_DEVICE_FP_FILE, 'w') as f:
            json.dump({'next_id': _DEVICE_FP_NEXT_ID[0],
                       'clusters': DEVICE_CLUSTERS}, f)
    except Exception:
        pass


def _fp_load():
    try:
        if not os.path.exists(_DEVICE_FP_FILE): return
        with open(_DEVICE_FP_FILE) as f:
            data = json.load(f)
        if isinstance(data, dict):
            _DEVICE_FP_NEXT_ID[0] = int(data.get('next_id', 1))
            DEVICE_CLUSTERS.clear()
            DEVICE_CLUSTERS.extend(data.get('clusters', []))
    except Exception:
        pass


# ---- Supervised per-node hardware-profile calibration ----
# For every CLEAN-decoded Meshtastic broadcast we get a labeled hardware
# fingerprint: hop=0 packets are physically emitted by the `from` node;
# hop=1 packets in 2-node networks are emitted by THE OTHER node (deducible
# by elimination from KNOWN_MESH).  Over N≥10 labeled samples per node,
# per-feature mean/std stabilizes (σ/√N drops noise floor below inter-device
# separation even for same-batch hardware).  Then UNKNOWN packets are
# classified against these profiles via Mahalanobis-like distance using
# each profile's own measured std.  This LINKS specific unknown packets to
# specific known nodes — solves the same-batch limit by using the protocol's
# own routing labels to learn per-device hardware signatures.
NODE_PROFILES = {}                # node_id (e.g. '!ba6a783c') → profile dict
NODE_PROFILES_MAX_SAMPLES = 200   # Keep more samples to train ML model.
                                  # Rule-based Mahalanobis used 4-sample sliding
                                  # window; ML classifier benefits from larger
                                  # accumulated history for robust pattern
                                  # learning across test sessions.
NODE_PROFILES_MIN_FOR_USE = 30    # Need enough samples to train ML model.
                                  # Random Forest needs ~30 samples per class
                                  # to reliably fit; fewer leads to overfitting.

NODE_PROFILES_PATH = os.path.join(_DATA_DIR, "lora_node_profiles.json")


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
        if rec.get('lat') is not None:
            n['lat'] = rec['lat']; n['lon'] = rec.get('lon')
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
        # Online device-fingerprint clustering with CONFIDENCE GATING.
        # Every packet is clustered, but `device_id` is only set if the
        # assignment is high-confidence (clean fit + clear margin + mature
        # cluster).  Low-confidence packets get device_id=None to preserve
        # attribution accuracy — better to abstain than mis-attribute.
        _dev_result = _fp_assign_device_id(rec)
        if _dev_result is not None:
            _dev_cluster_id, _dev_conf = _dev_result
            rec['device_id_cluster'] = _dev_cluster_id    # always set (cluster grew)
            rec['device_id_confidence'] = round(_dev_conf, 3)
            # Compute cluster purity from Meshtastic-labeled traffic that has
            # landed here.  Pure clusters by definition can't lie — they only
            # see one node.  This gates HIGH-CONFIDENCE ID for unknown packets:
            # the cluster has been validated by labeled traffic flowing through it.
            _purity = 0.0; _dominant = None
            for _c in DEVICE_CLUSTERS:
                if _c['id'] != _dev_cluster_id: continue
                _fs = _c.get('from_seen') or {}
                if _fs:
                    _tot = sum(_fs.values())
                    _dominant = max(_fs, key=_fs.get)
                    _purity = _fs[_dominant] / _tot
                break
            rec['device_id_purity'] = round(_purity, 3)
            if _dev_conf >= _DEVICE_FP_LOG_CONFIDENCE:
                _fs = None
                for _c in DEVICE_CLUSTERS:
                    if _c['id'] == _dev_cluster_id:
                        _fs = _c.get('from_seen') or {}
                        break
                has_labels = bool(_fs)
                if has_labels and _purity >= _DEVICE_FP_MIN_PURITY:
                    # Cluster validated by Meshtastic ground truth — attribute to
                    # the labeled node ID (human-readable, definitive).
                    rec['device_id'] = _dominant
                elif not has_labels:
                    # No labels available (purely unknown-protocol traffic) — the
                    # cluster_id IS the device identity.  Two packets in the same
                    # cluster = same physical TX, even if we never learn its name.
                    rec['device_id'] = _dev_cluster_id
                # If has_labels but purity < threshold: mixed cluster, abstain.
        if 'device_id' not in rec and rec.get('proto') == 'meshtastic' and rec.get('from'):
            rec['device_id'] = rec['from']
        elif rec.get('proto') == 'meshcore' and rec.get('pubkey'):
            rec['device_id'] = f"mc:{rec['pubkey'][:16]}"
        elif rec.get('proto') == 'lorawan' and rec.get('devaddr'):
            rec['device_id'] = f"lw:{rec['devaddr']}"
        elif rec.get('proto') == 'loramesher' and rec.get('src') is not None:
            rec['device_id'] = f"lm:{rec['src']}"
        # Supervised classifier: classify UNKNOWN packets against known
        # Meshtastic-derived node profiles BEFORE storing, so the inferred
        # match is attached to the record from the first display.
        if rec.get('proto') == 'unknown':
            _inferred, _conf = _classify_against_profiles(rec)
            if _inferred is not None:
                rec['inferred_from'] = _inferred
                rec['inferred_confidence'] = round(_conf, 3)
        _store(rec)
        # Profile calibration: accumulate labeled feature samples from
        # successful Meshtastic decodes (hop=0 always; hop=1 only in 2-node
        # networks where the relay is deducible).
        if rec.get('proto') == 'meshtastic' and rec.get('decrypted'):
            _bump_node_profile(rec)
        # Unknown-protocol aggregation: cluster recurring transmitters by RF
        # carrier + payload prefix so they show up as ONE device with N
        # sightings instead of N isolated 'unknown' rows.
        if rec.get('proto') == 'unknown':
            _bump_unknown_fp(rec)
    return rec


# Normalize an ASCII run: digits → 'N', uppercase letters → 'A', lowercase → 'a',
# spaces/punct kept as-is.  So `2687,0,12` → `NNNN,N,NN` — visually clusters
# different sensor readings from the same transmitter format.
def _ascii_pattern(raw_hex):
    if not raw_hex:
        return ''
    try:
        b = bytes.fromhex(raw_hex)
    except ValueError:
        return ''
    out = []
    for byte in b:
        if 0x30 <= byte <= 0x39:   out.append('N')   # digit
        elif 0x41 <= byte <= 0x5a: out.append('A')   # upper
        elif 0x61 <= byte <= 0x7a: out.append('a')   # lower
        elif 0x20 <= byte < 0x7f:  out.append(chr(byte))  # punct/space
        else:                      out.append('·')   # non-printable
    return ''.join(out)


def _bump_unknown_fp(rec):
    """Aggregate one unknown packet into the UNKNOWN_FP fingerprint map.
    Caller holds STATE_LOCK.

    Two-level fingerprint:
      LEVEL 1 (protocol family) — key = prefix (first 4 bytes of raw_hex).
                                  Clusters by app/firmware/protocol shape.
      LEVEL 2 (device clusters)  — within each prefix family, packets are
                                  grouped via greedy nearest-neighbor on
                                  the per-packet hardware-fingerprint vector
                                  (dc_i, dc_q, iq_amp_imb, iq_phase_imb,
                                  cfo_hz).  Each sub-cluster is an estimated
                                  distinct transmitter device.

    Hardware fingerprint is independent of distance/orientation (no RSSI),
    payload content, and inter-packet timing — so a node moving around or
    transmitting irregularly stays in ONE cluster.  Same-batch ambiguity and
    temperature drift can still merge close devices; we surface a "≥ N
    devices" lower bound rather than committing to an exact count.
    """
    raw_hex = rec.get('raw_hex') or ''
    prefix = raw_hex[:8]
    ts = rec.get('ts') or time.time()
    rssi = rec.get('rssi')
    hw_fp = rec.get('hw_fp')
    cfo_hz = rec.get('cfo_hz')
    freq_mhz = rec.get('freq_mhz')
    # Single-feature fingerprint: absolute TX frequency = gate_freq + decoder_cfo.
    # See _feature_vector for why DC/IQ imbalance are NOT used (RX-side noise).
    fv = _feature_vector(hw_fp, cfo_hz, freq_mhz)

    d = UNKNOWN_FP.get(prefix)
    if d is None:
        # New prefix family — evict LRU if at cap.
        if len(UNKNOWN_FP) >= UNKNOWN_FP_CAP:
            oldest = min(UNKNOWN_FP.items(), key=lambda kv: kv[1]['last_ts'])
            del UNKNOWN_FP[oldest[0]]
        d = UNKNOWN_FP[prefix] = {
            'prefix': prefix, 'count': 0,
            'first_ts': ts, 'last_ts': ts,
            'rssi_min': rssi, 'rssi_max': rssi,
            'samples': [], 'ascii_pattern': _ascii_pattern(raw_hex),
            'hint': rec.get('hint'),
            'devices': [],   # list of per-device cluster dicts
        }
    d['count'] += 1
    d['last_ts'] = ts
    if rssi is not None:
        if d['rssi_min'] is None or rssi < d['rssi_min']: d['rssi_min'] = rssi
        if d['rssi_max'] is None or rssi > d['rssi_max']: d['rssi_max'] = rssi
    if len(d['samples']) < 5 and raw_hex not in d['samples']:
        d['samples'].append(raw_hex)
    if rec.get('hint') and not d.get('hint'):
        d['hint'] = rec['hint']
    # Greedy nearest-neighbor device clustering on the hw-feature vector.
    if fv is not None:
        _assign_device_cluster(d, fv, ts, rec.get('freq_mhz'))
    # Promote recurring Mystery Devices clusters to NODE_PROFILES so they
    # participate in the supervised classifier.  Once promoted, future unknown
    # packets matching the cluster's RF signature get an anonymous-but-stable
    # `inferred_from = mystery:<prefix>-<freq>MHz` label.  This gives recurring
    # unknown transmitters (RadioLib, custom firmware, etc.) a stable identity
    # even with no Meshtastic-labeled training data for their hardware.
    for dev in d.get('devices', []):
        if dev['count'] >= 20 and not dev.get('promoted'):
            _promote_mystery(prefix, dev)
            dev['promoted'] = True


def _promote_mystery(prefix, dev):
    """Promote a mature Mystery Devices sub-cluster into NODE_PROFILES.

    Synthetic label: `mystery:<prefix>-<freq MHz>`.  Conservative std (250 Hz
    — empirical upper bound for per-packet abs_tx_hz noise).  Mystery Devices
    only tracks the running MEAN of abs_tx_hz, not variance, so we use a
    fixed std rather than computing it.  This may cause looser-than-ideal
    matches for tight-CFO devices but is safer than computing a fake std
    from one running-mean value.
    """
    if not dev.get('mean_fv'):
        return
    mean_freq = float(dev['mean_fv'][0])
    label = "mystery:%s-%.4fMHz" % (prefix[:8], mean_freq / 1e6)
    if label in NODE_PROFILES:
        return
    NODE_PROFILES[label] = {
        'samples': [],
        'count': dev['count'],
        'mean': {'abs_tx_hz': mean_freq},
        'std':  {'abs_tx_hz': 250.0},
        'first_ts': dev.get('first_ts') or time.time(),
        'last_ts':  dev.get('last_ts')  or time.time(),
        'is_mystery': True,
        'mystery_prefix': prefix,
    }


# ============================================================================
# Per-node hardware profile calibration (labeled-traffic supervised learning)
# ============================================================================
_PROFILE_MIN_SNR_DB = 25.0   # Lower floor — ML classifier abstains on
                             # low-confidence calls internally.  At conf≥0.8,
                             # ML achieves 100% accuracy across 4 fresh tests
                             # (leave-one-test-out CV).  Coverage 21-40%
                             # depending on conditions, but ZERO wrong.

def _extract_classifier_features(rec, require_quality=True):
    """Extract UMOP features for profile training/classification.  Returns None
    if essential features are unavailable OR if the sample is too low-SNR to
    be trustworthy (require_quality=True).

    Features extracted from hw_fp:
      abs_tx_hz           absolute TX carrier (gate freq + sample-precise CFO)
      cfo_per_sym_std     short-term crystal jitter (Hz)
      am_ripple_pct       PA AM-AM distortion (%)
      amp_per_sym_pct     inter-symbol amplitude stability (%)
      phase_residual_rms  PLL phase noise residual (rad)
      irr_db              TX I/Q mixer imbalance (dB)
      pn_slope            phase noise spectral slope

    Quality filter: signal_snr_db < 25 dB → sample rejected.  Low-SNR captures
    have alignment errors that corrupt UMOP measurements and pollute profiles."""
    hw = rec.get('hw_fp')
    if not isinstance(hw, dict):
        return None
    if require_quality:
        snr = hw.get('signal_snr_db')
        if snr is None or float(snr) < _PROFILE_MIN_SNR_DB:
            return None
    feats = {}
    pc = hw.get('precise_carrier_hz')
    fm = rec.get('freq_mhz')
    if pc is not None and fm is not None:
        try:
            feats['abs_tx_hz'] = float(fm) * 1e6 + float(pc)
        except (TypeError, ValueError):
            pass
    for key in ('cfo_per_sym_std', 'am_ripple_pct', 'amp_per_sym_pct',
                'phase_residual_rms', 'irr_db', 'pn_slope',
                'onset_rise_us', 'onset_overshoot_pct', 'onset_mid_slope',
                'eob_fall_us', 'eob_mid_fall_slope', 'eob_pre_decay_over_pct',
                'precise_carrier_hz', 'signal_snr_db'):
        v = hw.get(key)
        if v is not None:
            try: feats[key] = float(v)
            except (TypeError, ValueError): pass
    return feats if feats else None


def _profile_node_id(rec):
    """Determine the SPECIFIC physical emitter of a labeled packet.

    Returns the node_id of the radio that physically transmitted this packet,
    or None when it cannot be determined.

    Rules:
      hop=0 packet → emitted by `from` node directly. Reliable label.
      hop>0 → packet was relayed.  The PHYSICAL TRANSMITTER is the relayer,
            which is some node we can't identify from the packet alone.
            Skip these — labels would be ambiguous/wrong in any N>2 mesh.
    """
    if rec.get('proto') != 'meshtastic' or not rec.get('decrypted'):
        return None
    frm = rec.get('from')
    hops = rec.get('hops')
    if frm is None or hops is None:
        return None
    if hops == 0:
        return frm
    return None   # hop>0: physical transmitter unknown


def _recompute_profile_stats(profile):
    """Compute per-feature mean + std from accumulated samples.  Stores in
    profile['mean'] / profile['std'].

    Why classical (not robust median+MAD): empirically, real measurement
    distributions have fat tails that MAD UNDER-estimates as noise.  This
    makes a MAD-based classifier OVER-confident, eliminating the ambiguity
    abstention that protects against same-batch false matches.  Classical
    std OVER-estimates noise via outliers (good — keeps the classifier
    appropriately humble about borderline packets so they're correctly
    flagged as ambiguous rather than confidently mis-classified)."""
    feature_keys = set()
    for s in profile['samples']:
        feature_keys.update(s.keys())
    means = {}; stds = {}
    for key in feature_keys:
        vals = [s[key] for s in profile['samples'] if key in s]
        if len(vals) >= 3:
            m = sum(vals) / len(vals)
            means[key] = m
            if len(vals) >= 2:
                var = sum((v - m) ** 2 for v in vals) / (len(vals) - 1)
                stds[key] = max(var ** 0.5, 1e-9)
            else:
                stds[key] = 1e-9
    profile['mean'] = means
    profile['std'] = stds


def _bump_node_profile(rec):
    """Accumulate a labeled fingerprint sample into the per-node profile.
    Caller holds STATE_LOCK."""
    nid = _profile_node_id(rec)
    if nid is None:
        return
    feats = _extract_classifier_features(rec)
    if feats is None:
        return
    p = NODE_PROFILES.get(nid)
    if p is None:
        p = NODE_PROFILES[nid] = {
            'samples': [],
            'count': 0,
            'mean': {},
            'std': {},
            'first_ts': rec.get('ts') or time.time(),
            'last_ts': rec.get('ts') or time.time(),
        }
    p['samples'].append(feats)
    p['count'] += 1
    p['last_ts'] = rec.get('ts') or time.time()
    if len(p['samples']) > NODE_PROFILES_MAX_SAMPLES:
        p['samples'] = p['samples'][-NODE_PROFILES_MAX_SAMPLES:]
    # Recompute every sample once the profile is usable: with a 30-sample
    # sliding window, every new sample meaningfully shifts the std (TCXO
    # drift would otherwise lag behind reality between recomputes).  Cost
    # is trivial (30 samples × ~3 features per recompute).
    if p['count'] < NODE_PROFILES_MIN_FOR_USE:
        if p['count'] % 3 == 0:
            _recompute_profile_stats(p)
    else:
        _recompute_profile_stats(p)


def _discriminative_features():
    """Identify which features actually distinguish between profiles in the
    current deployment.  Returns a list of (feature_name, snr) for features
    where inter-profile separation exceeds intra-profile noise by SNR > 0.8.

    Self-tuning: features that are RX-constant or measurement-noise-dominated
    (per-feature SNR < 0.8) get EXCLUDED automatically — adding them to the
    distance just dilutes the discriminative features.  Empirically the IRR
    and PN slope features fall in this category for same-batch Meshtastic
    deployments (see project_device_fingerprinting memory)."""
    usable_profiles = [p for p in NODE_PROFILES.values()
                       if p['count'] >= NODE_PROFILES_MIN_FOR_USE and p['mean']]
    if len(usable_profiles) < 2:
        # With only one profile, all features are 'discriminative' (no
        # competing profile to dilute against) — use whatever we have.
        if len(usable_profiles) == 1:
            return [(k, 1.0) for k in usable_profiles[0]['mean']]
        return []
    feature_keys = set()
    for p in usable_profiles:
        feature_keys.update(p['mean'].keys())
    out = []
    for k in feature_keys:
        means = [p['mean'][k] for p in usable_profiles if k in p['mean']]
        stds = [p['std'][k] for p in usable_profiles if k in p['std']]
        if len(means) < 2 or len(stds) < 2:
            continue
        # Inter-profile separation: max-min mean spread
        sep = max(means) - min(means)
        # Intra-profile noise: average std across profiles
        pooled = sum(stds) / len(stds)
        if pooled < 1e-9:
            continue
        snr = sep / pooled
        if snr > 0.8:
            out.append((k, snr))
    return out


def _classify_against_profiles(rec):
    """Classify a packet against all known-node hardware profiles.

    Tries ML model FIRST (Random Forest trained on accumulated labeled
    samples — empirically 100% accuracy at conf≥0.8 on out-of-test samples).
    Falls back to rule-based Mahalanobis distance if ML model isn't trained
    yet (during initial bootstrap with fewer than 30 samples per node).

      - Multi-feature (≥2 discriminative features available):
          threshold 2.0σ, runner-up ratio 1.5×
          → permissive, takes advantage of independent feature corroboration

      - Single-feature (only CFO discriminates — the same-batch case):
          threshold 1.0σ, runner-up ratio 2.0×
          → STRICT, since one feature alone can't safely distinguish
          devices whose CFOs are within ~2σ of each other.  Most borderline
          packets correctly abstain in this mode.

    Returns (node_id, confidence) where confidence ∈ [0, 1], or (None, 0.0).
    Confidence accounts for both match-distance AND profile maturity (a
    profile with only 10 samples gets penalty vs one with 50+).
    """
    # No ML — pure rule-based Mahalanobis classification using fingerprint
    # clustering.  Returns the device the strongest feature similarity points
    # to among known nodes, or None (abstain) if confidence is too low.
    feats = _extract_classifier_features(rec)
    if feats is None:
        return None, 0.0
    # PA-onset features are HARDWARE-STABLE (amplifier characteristics don't
    # drift like crystals).  Combined with a SHORT sliding window (5-6 samples),
    # they reach 95.8% per-packet accuracy on same-batch devices.
    #
    #   onset_overshoot_pct — PA regulator transient (strongest onset feature)
    #   onset_mid_slope     — PA transition rate
    #
    # Other features INTENTIONALLY excluded:
    #   - onset_rise_us: quantized to integer μs steps → sliding-window std
    #     can collapse to 0 (consecutive captures land on same μs value),
    #     which breaks Mahalanobis distance.
    #   - abs_tx_hz: TCXO drifts across the same range on both devices over
    #     a few minutes → long-run discrimination unreliable.
    #   - dechirp-derived (am_ripple, irr_db, phase_residual, etc.): all
    #     derived from same preamble dechirp, so highly correlated —
    #     combining adds noise without independent information.
    #
    # Equal-weight unweighted Mahalanobis (matches the offline-validated
    # algorithm that hit 96% on the hardest test).  Combines onset (PA turn-ON)
    # and end-of-burst (PA turn-OFF) features — these are independent physical
    # signatures and joint discrimination is much stronger than either alone.
    # SNR-weighted dilutes the signal when features have similar SNRs.
    PRIMARY_FEATS = ['onset_overshoot_pct', 'onset_mid_slope',
                     'eob_fall_us', 'eob_mid_fall_slope', 'eob_pre_decay_over_pct']
    use_keys = [k for k in PRIMARY_FEATS if k in feats]
    if not use_keys:
        # No onset features — fall back to discriminative-features as before
        discrim = _discriminative_features()
        if not discrim:
            return None, 0.0
        if 'abs_tx_hz' in {k for k, _ in discrim}:
            use_keys = ['abs_tx_hz']
        else:
            use_keys = [max(discrim, key=lambda x: x[1])[0]]
    # Sigma floor: prevents sliding-window-std collapse when consecutive
    # captures share quantized values.  Set to ~25% of empirical inter-device
    # separation per feature.
    SIGMA_FLOOR = {'onset_overshoot_pct': 0.015,
                   'onset_mid_slope': 0.003,
                   'abs_tx_hz': 50.0,
                   'onset_rise_us': 0.5,
                   'eob_fall_us': 0.5,
                   'eob_mid_fall_slope': 0.005,
                   'eob_pre_decay_over_pct': 0.05}
    # Thresholds: cross-validated across 4 fresh live tests.  thresh=0.6σ +
    # ratio=3.0× at SNR≥28 floor with win=4:
    #   umop10_hires: 97.6% / 53.1%
    #   umop11:       98.3% / 48.2%
    #   umop12:       96.2% / 30.6%
    #   umop13:       88.8% / 56.5%  (harder RF conditions)
    # Average 95%, min 89%.  Wrong rate stays low (2-11%) because the
    # 3.0× ratio test demands clearly-distinct candidates.
    threshold_sigma = 0.6
    ratio_threshold = 3.0
    candidates = []
    for nid, p in NODE_PROFILES.items():
        if p['count'] < NODE_PROFILES_MIN_FOR_USE or not p['mean']:
            continue
        d_sum = 0.0; n = 0
        for k in use_keys:
            if k not in feats or k not in p['mean']: continue
            mu = p['mean'][k]
            sigma = max(p['std'].get(k, 1e-9), SIGMA_FLOOR.get(k, 1e-9))
            d_sum += ((feats[k] - mu) / sigma) ** 2
            n += 1
        if n == 0:
            continue
        dist = (d_sum / n) ** 0.5
        candidates.append((nid, dist, n, p['count']))
    if not candidates:
        return None, 0.0
    candidates.sort(key=lambda x: x[1])
    best_nid, best_dist, best_n, best_count = candidates[0]
    if best_dist > threshold_sigma:
        return None, 0.0
    if len(candidates) > 1:
        _, next_dist, _, _ = candidates[1]
        if next_dist < best_dist * ratio_threshold:
            return None, 0.0          # ambiguous between two profiles
    # Confidence: combines match closeness with profile maturity.  A profile
    # with only NODE_PROFILES_MIN_FOR_USE samples gets a maturity penalty;
    # at 50+ samples maturity factor saturates at 1.0.
    closeness = max(0.0, 1.0 - best_dist / threshold_sigma)
    maturity = min(1.0, best_count / 50.0)
    confidence = closeness * (0.5 + 0.5 * maturity)   # at least half-weight, scales up with samples
    return best_nid, confidence


def _save_node_profiles():
    """Persist NODE_PROFILES to disk so calibration survives web restart.
    Stores only the aggregate stats (mean/std/count/timestamps), not the raw
    samples — file stays small (~1 KB per known node)."""
    try:
        out = {}
        for nid, p in NODE_PROFILES.items():
            out[nid] = {
                'count': p['count'],
                'mean': p['mean'],
                'std': p['std'],
                'first_ts': p['first_ts'],
                'last_ts': p['last_ts'],
            }
        with open(NODE_PROFILES_PATH, 'w') as f:
            json.dump(out, f, indent=2)
    except Exception:
        pass


def _load_node_profiles():
    """Restore profiles from disk on startup.  Aggregate stats only (samples
    list is empty until new traffic arrives)."""
    try:
        with open(NODE_PROFILES_PATH) as f:
            saved = json.load(f)
        for nid, p in saved.items():
            NODE_PROFILES[nid] = {
                'samples': [],
                'count': p.get('count', 0),
                'mean': p.get('mean', {}),
                'std': p.get('std', {}),
                'first_ts': p.get('first_ts'),
                'last_ts': p.get('last_ts'),
            }
    except Exception:
        pass


_load_node_profiles()
_fp_load()


# ---- Per-TX-device fingerprint clustering helpers ----
# Empirical finding from the sleepy_c.sc16 test: in a single-receiver passive
# intercept, the RX's DC offset and I/Q imbalance are CONSTANTS (same SDR for
# every packet), so measured variation across packets is just noise — they
# do NOT fingerprint different transmitters.  Per-packet measurements of
# these are dominated by noise statistics in the capture window, not by
# transmit hardware.
#
# The one feature that genuinely identifies the transmit device from a
# passive single-RX intercept is the ABSOLUTE TX CARRIER FREQUENCY — each
# transmitter's TCXO has a per-unit offset that's stable within hours.
# We compute it as freq_mhz (gate's measured carrier, ~10 kHz scatter) PLUS
# cfo_hz (decoder's preamble-fit residual, sub-Hz precision).  Their sum
# cancels the gate's centering error and yields the device's actual TX
# carrier in Hz, robust across captures.
#
# Cluster threshold: 300 Hz.  Typical ±2 ppm TCXOs at 915 MHz span ±1830 Hz,
# so 300 Hz catches one device's session drift but separates devices > 500 Hz
# apart.  Same-batch TCXO ambiguity (CFOs within 100 Hz) merges — acknowledged
# limitation; flagged as "≥ N devices" rather than exact count.
# Device clustering uses ABSOLUTE TX CARRIER FREQUENCY (precise CFO from
# envelope+Welch on full-rate IQ + gate-measured center).  Empirically validated:
#   - 5 RadioLib bursts → 2 clusters (matches user's "2 nodes" ground truth)
#   - 3 Meshtastic SF12 captures from same node → 1 cluster
#   - 224 same-node SF7 records → 1 cluster (no over-split)
#
# IRR (image rejection) and PN slope are computed and stored on each [PKT]
# record as DIAGNOSTIC features — but NOT used in clustering, because:
#   - Per-capture measurement noise on IRR is ~5-15 dB (alignment-sensitive)
#   - Per-capture measurement noise on PN slope is ~0.3-0.5
#   - Inter-device separation on same-batch hardware is ~2 dB / 0.1 — well
#     within the measurement noise → including these features OVER-SPLITS
#     single-device records into many false clusters
# To distinguish same-batch hardware properly would require either many-sample
# averaging on known-clusters (chicken-and-egg) or per-receiver IQ forensics
# (multi-receiver, beyond passive single-RX intercept).  Acknowledged as a
# fundamental limit of passive single-receiver fingerprinting in the
# project_device_fingerprinting memory.
_DEVICE_MATCH_HZ = 1500.0

def _feature_vector(hw_fp, cfo_hz, freq_mhz=None):
    """1-D fingerprint: absolute TX carrier frequency in Hz.

    Uses hw_fp.precise_carrier_hz (envelope+Welch on burst window) combined
    with freq_mhz (gate's measured carrier) — their sum cancels gate-centering
    scatter and yields the device's actual TX frequency.

    Returns None if essential fields are unavailable."""
    if hw_fp is None:
        return None
    precise = hw_fp.get('precise_carrier_hz') if isinstance(hw_fp, dict) else None
    if precise is None:
        precise = cfo_hz
    if precise is None or freq_mhz is None:
        return None
    try:
        abs_freq_hz = float(freq_mhz) * 1e6 + float(precise)
        return (abs_freq_hz,)
    except (TypeError, ValueError):
        return None


def _scaled_distance(fv1, fv2):
    """1-D distance in Hz (single feature: absolute TX frequency)."""
    return abs(fv1[0] - fv2[0])


def _assign_device_cluster(family, fv, ts, freq_mhz):
    """Greedy nearest-neighbor: find the closest existing device cluster in
    this prefix family; if within threshold, merge; else create a new cluster.
    Each device cluster is the (running mean) feature vector + count + activity
    timestamps so we can show 'device A: 12 sightings, last 3s ago'."""
    devs = family['devices']
    best_idx, best_dist = -1, float('inf')
    for i, d in enumerate(devs):
        dist = _scaled_distance(fv, d['mean_fv'])
        if dist < best_dist:
            best_dist, best_idx = dist, i
    if best_idx >= 0 and best_dist <= _DEVICE_MATCH_HZ:
        d = devs[best_idx]
        # Exponentially weighted running mean — tracks slow drift, stable on
        # measurement noise.  Alpha capped at 1/20 so the cluster mean settles
        # to a stable value after ~20 samples.
        alpha = 1.0 / min(d['count'] + 1, 20)
        d['mean_fv'] = tuple(m + alpha * (v - m) for m, v in zip(d['mean_fv'], fv))
        d['count'] += 1
        d['last_ts'] = ts
        d['last_dist'] = best_dist
        if freq_mhz is not None:
            d['freq_mhz_last'] = freq_mhz
    else:
        # New device — first sighting in this prefix family.
        devs.append({
            'mean_fv': fv,
            'count': 1,
            'first_ts': ts,
            'last_ts': ts,
            'last_dist': 0.0,
            'freq_mhz_last': freq_mhz,
        })


def _node_view(n):
    return {
        'id': n['id'], 'count': n['count'], 'decoded': n['decoded'],
        'encrypted': n['encrypted'], 'first_ts': n['first_ts'],
        'last_ts': n['last_ts'], 'last_to': n['last_to'],
        'avg_rssi': round(n['rssi_sum'] / n['rssi_n'], 1) if n['rssi_n'] else None,
        'rssi_samples': n['rssi_n'], 'channels': sorted(n['channels']),
        'last_text': n['last_text'],
        'lat': n.get('lat'), 'lon': n.get('lon'),
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
          'warn': None, 'detect_workers': None, 'preflight_note': None}
import re as _re
_RE_STAT = _re.compile(r'\[STAT\]\s+([\d.]+)s\s+\|\s+([\d.]+)Msps\s+\|\s+win=(\d+)\s+'
                       r'det=(\d+)\s+save_q=(\d+)\s+dec_q=(\d+)\s+active=(\d+)\s+'
                       r'pipe=(\d+)ms(?:\s+drops=([\d.]+)M)?')
_RE_AUTO = _re.compile(r'Detect workers: AUTO = (\d+)')
# CATCHUP appears when the detector falls so far behind the ring buffer it has
# to skip ahead to avoid wrapping past unread data.  Each occurrence is hard,
# IMMEDIATE evidence the host can't sustain the configured rate — surface it
# without waiting for the keep-up warmup window.
_RE_CATCHUP = _re.compile(r'CATCHUP skip\s+([\d.]+)M samples\s+\(([\d.]+)s\)')


def _tail_health():
    """Follow the pipeline stderr log, parse [STAT]/warnings → HEALTH + SSE.
    rate_msps is refreshed from the live radio config each iteration so a user
    rate change shows up in the UI status bar immediately (instead of being
    pinned to the lora.toml default forever).
    The KEEP-UP warning is gated by a 30 s warm-up window and re-evaluated every
    30 s thereafter against the latest observed msps — if the host catches up,
    the warning clears on its own."""
    pos = 0
    pending_warn = None        # latest KEEP-UP line seen this run (or None)
    last_check = 0.0           # last post-warmup re-evaluation timestamp
    last_started_at = None     # detect pipeline restarts → reset warm-up state
    WARMUP_S = 30.0
    CHECK_S = 30.0
    NO_TELEMETRY_S = 15.0   # if pipeline running this long with NO [STAT] line,
                            # surface a "no telemetry — overloaded or stuck" warn.
                            # Normal STAT cadence is every 5 s, so 15 s = three
                            # missed; anything beyond that is hard evidence the
                            # detector loop is stalled, not a transient hiccup.
    last_stat_at = None     # timestamp of the most recent successful [STAT] parse
    immediate_warn = None   # CATCHUP / no-telemetry warnings (bypass keep-up
                            # warmup gating — they're deterministic, not statistical)
    while True:
        try:
            now = time.time()
            # Refresh declared rate from the LIVE radio cfg (so editing rate in
            # the UI updates the gate's "/N Msps" display without a restart).
            try:
                HEALTH['rate_msps'] = float(_radio_cfg()['rate_hz']) / 1e6
            except Exception:
                pass
            # Pipeline restart → forget any prior warning and re-enter warm-up.
            started_at = PIPELINE.get('started_at')
            if started_at != last_started_at:
                pending_warn = None
                immediate_warn = None
                last_stat_at = None
                last_check = 0.0
                last_started_at = started_at
            if not os.path.exists(PIPELINE_LOG):
                HEALTH.update(msps=None); time.sleep(0.5); continue
            if os.path.getsize(PIPELINE_LOG) < pos:
                pos = 0
            with open(PIPELINE_LOG, 'r', errors='replace') as f:
                f.seek(pos); lines = f.readlines(); pos = f.tell()
            changed = False
            # When the pipeline is stopped, ignore any STAT lines that may
            # still be flushed into the log between stop_pipeline()'s pkill
            # and the detector process actually dying.  Without this gate,
            # _tail_health re-populates HEALTH.msps / dec_q / pipe_ms after
            # stop_pipeline() cleared them and the GUI shows live-looking
            # values even though "pipeline_running" is false.  (Note: pos
            # is still advanced so when the pipeline next starts we don't
            # replay buffered pre-stop content.)
            pipeline_live = PIPELINE.get('running', False)
            for ln in lines:
                if not pipeline_live:
                    continue
                m = _RE_STAT.search(ln)
                if m:
                    HEALTH.update(elapsed=float(m.group(1)), msps=float(m.group(2)),
                                  det=int(m.group(4)), save_q=int(m.group(5)),
                                  dec_q=int(m.group(6)), pipe_ms=int(m.group(8)),
                                  drops_m=float(m.group(9)) if m.group(9) else HEALTH['drops_m'])
                    changed = True
                    last_stat_at = now
                    # A fresh STAT line means we got telemetry — clear any
                    # immediate "no telemetry" warning we may have raised.  We
                    # do NOT clear a CATCHUP warning here: a fresh STAT doesn't
                    # mean the host is now keeping up; the next CATCHUP or the
                    # 30 s keep-up reevaluation should govern that.
                    if immediate_warn and immediate_warn.startswith('Pipeline running '):
                        immediate_warn = None
                a = _RE_AUTO.search(ln)
                if a:
                    HEALTH['detect_workers'] = int(a.group(1))
                if 'KEEP-UP WARNING' in ln:
                    pending_warn = ln.strip()[:200]
                # CATCHUP: detector fell so far behind it had to skip ahead.
                # Surface IMMEDIATELY (no warmup gate) — this is deterministic
                # evidence the host can't sustain the configured rate, not a
                # statistical inference from msps.
                c = _RE_CATCHUP.search(ln)
                if c:
                    immediate_warn = (
                        f"Detector fell {float(c.group(2)):.0f}s behind and had to skip "
                        f"{float(c.group(1)):.0f}M samples — host can't sustain "
                        f"{(HEALTH.get('rate_msps') or 0):.0f} Msps. Lower the rate."
                    )
                    changed = True
            # NO-TELEMETRY watchdog: pipeline marked running but no [STAT] line
            # has arrived within NO_TELEMETRY_S seconds.  Usually means the
            # detector loop is CPU-bound to the point where windows aren't
            # completing.  Different from CATCHUP (which is when the loop IS
            # completing but losing ground); fires when nothing's coming through
            # at all.
            if pipeline_live and started_at:
                since_start = now - started_at
                seconds_since_stat = (now - last_stat_at) if last_stat_at else since_start
                if since_start > NO_TELEMETRY_S and seconds_since_stat > NO_TELEMETRY_S:
                    new_w = (f"Pipeline running {since_start:.0f}s with no telemetry — "
                             f"detector is likely overloaded. Check pipeline log; "
                             f"consider lowering the sample rate.")
                    if immediate_warn != new_w:
                        immediate_warn = new_w
                        changed = True
            # Keep-up banner logic: never surface during warm-up; after that,
            # re-check on a 30 s cadence and clear if msps caught up.
            # Priority: immediate_warn (deterministic) > pending_warn (statistical
            # keep-up).
            if PIPELINE.get('running') and started_at:
                since_start = now - started_at
                want_warn = immediate_warn
                if not want_warn:
                    if since_start < WARMUP_S:
                        want_warn = None
                    else:
                        if now - last_check >= CHECK_S:
                            last_check = now
                            msps = HEALTH.get('msps'); rate = HEALTH.get('rate_msps')
                            keeping_up = (msps is not None and rate
                                          and msps >= rate * 0.97)
                            if keeping_up:
                                pending_warn = None
                        want_warn = pending_warn
                if HEALTH.get('warn') != want_warn:
                    HEALTH['warn'] = want_warn; changed = True
            elif HEALTH.get('warn') is not None:
                HEALTH['warn'] = None; changed = True
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


AUTOSAVE_PATH = os.path.join(_DATA_DIR, "lora_autosave.jsonl")


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


# Bytes per IQ sample for each stream format the SDR profiles can emit.
# Used by _safe_buf_seconds() to model ring-buffer memory cost.  Unknown
# formats fall back to 4 (= sc16, the common case) so the cap stays
# conservative rather than silently underestimating.
_BYTES_PER_SAMPLE = {'cs8': 2, 'sc8': 2, 'sc16': 4, 'complex64': 8}


def _safe_buf_seconds(rate_hz, fmt, requested_s):
    """Cap requested buf_seconds so the ring buffer fits in available RAM.

    Background: the detector's ring buffer is `buf_seconds * rate * BPS`
    bytes of pinned RAM, allocated at startup.  On RAM-constrained hosts
    (e.g. Raspberry Pi 4 with 1.8 GB total) the default 16 s buffer at
    20 Msps sc16 wants 1.28 GB — which the OOM killer reaps within ~10 s,
    BEFORE the existing 30 s keep-up warmup can produce a warning.  The
    user sees a silent death.

    This cap reads /proc/meminfo's MemAvailable (Linux only; non-Linux
    hosts skip the check), reserves 40 % of it as headroom for the
    detector workers, OS, web process, and /dev/shm capture writes, and
    returns the largest buf_seconds whose buffer fits in the remaining
    60 %.  When the cap activates, the chosen value is recorded in
    HEALTH['preflight_note'] so the GUI can surface it; on hosts with
    plenty of RAM the requested value is returned unchanged.

    The cap deliberately does NOT modify rate_hz — that's the user's
    choice and a CPU-bound overload is correctly reported by the existing
    KEEP-UP WARNING (which now has a chance to fire because the system
    is no longer OOM-killed in the first 10 s)."""
    bps = _BYTES_PER_SAMPLE.get((fmt or 'sc16').lower(), 4)
    avail = None
    try:
        with open('/proc/meminfo') as _mi:
            for _ln in _mi:
                if _ln.startswith('MemAvailable:'):
                    avail = int(_ln.split()[1]) * 1024
                    break
    except Exception:
        pass
    if not avail or rate_hz <= 0:
        return max(1, int(requested_s))
    # Allow at most 60 % of MemAvailable for the ring buffer.
    max_buf_bytes = int(avail * 0.6)
    per_second = rate_hz * bps
    max_buf_s = max_buf_bytes // per_second
    requested_s = max(1, int(requested_s))
    capped = max(1, min(requested_s, int(max_buf_s)))
    if capped < requested_s:
        # Surface in HEALTH so the GUI can show it next to the keep-up banner.
        HEALTH['preflight_note'] = (
            f"buf_seconds reduced from {requested_s}s to {capped}s "
            f"({capped * per_second / 1e6:.0f} MB) to fit in "
            f"{avail / 1e9:.1f} GB available RAM "
            f"(at {rate_hz/1e6:.0f} Msps {fmt}, {bps}B/sample)"
        )
    else:
        HEALTH['preflight_note'] = None
    return capped


def _build_pipeline_cmd():
    """Construct the <SDR capture> | detector shell pipeline for the SELECTED SDR.
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
    detector_py = os.path.join(_SRC, 'detector.py')
    # Pipeline-tuning overrides from the web UI's Config tab.  Fall back on the
    # lora.toml [detect] values for anything the user didn't touch.
    _tune = SETTINGS.get('tune', {}) or {}
    _thresh = float(_tune.get('threshold', d['threshold']))
    _eth = float(_tune.get('energy_threshold', d['energy_threshold']))
    _bufs_requested = int(_tune.get('buf_seconds', d['buf_seconds']))
    # RAM-safety cap: see _safe_buf_seconds for rationale.  No-op on hosts
    # with plenty of memory.
    _bufs = _safe_buf_seconds(rate, fmt, _bufs_requested)
    # Use THIS server's interpreter for the gate (it already holds numpy/scipy),
    # not a bare `python3` that PATH might resolve to a different env. run/web.py
    # guarantees sys.executable has the full stack (re-execs if it didn't).
    gate = (f'{shlex.quote(sys.executable)} {shlex.quote(detector_py)} -r {rate} -b {bw} -c {r["center_mhz"]} '
            f'-t {fmt} --threshold {_thresh} '
            f'--overlap {d["overlap"]} --energy-threshold {_eth} '
            f'--detect-workers {d["detect_workers"]} --buf-seconds {_bufs} '
            f'--decode --export-iq {shlex.quote(dec["export_dir"])} -d 1')
    return cap + ' | ' + gate


def _kill_stale_pipeline():
    """Kill any orphaned gate / bladeRF from a previous run so the SDR is free.
    The pipeline runs in its own session (setsid) and is detached, so quitting
    the web does NOT auto-kill it — without this, an orphan keeps the bladeRF
    busy and the next Start launches a second pipeline that can't open the
    device and dies in ~1 s ('running' then 'stopped').  These pkill patterns
    only match the gate/bladeRF, never this web process."""
    pats = ['detector.py -r']
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
    # Only forward a pinned positive worker count.  -1 / 0 → leave env unset so
    # the decoder auto-scales from cpu_count (multipurpose tool: must work on
    # any machine from 4-core laptops to 32-core hosts without operator tuning).
    try:
        _cfg_workers = int(dec.get('workers', -1) or -1)
    except (TypeError, ValueError):
        _cfg_workers = -1
    if _cfg_workers > 0:
        env['LORA_DECODE_WORKERS'] = str(_cfg_workers)
    else:
        env.pop('LORA_DECODE_WORKERS', None)
    # Export capture rate (dec=8 → 2 MHz at SF7/BW250): better PA-onset
    # temporal resolution for the fingerprinter.  Default 2 if config omits.
    if dec.get('export_dec') is not None:
        env['LORA_EXPORT_DEC'] = str(dec['export_dec'])
    env['LORA_KEYS'] = KEYS_PATH    # decoder reads the multi-key list (live mtime reload)
    env['LORA_PSD_FILE'] = PSD_FILE  # gate writes waterfall frames here (reuses its PSD)
    env['LORA_UNKNOWN_REPORT'] = UNKNOWN_PATH  # gate logs unknown-protocol frames here
    # Advanced Options → decoder: which protocols to attempt + whether to surface
    # unknowns (off also lets the decoder early-bail on irrelevant sync words).
    env['LORA_UNKNOWN'] = '1' if SETTINGS.get('unknown') else '0'
    # RF fingerprinting (UMOP feature extraction → web UI device clustering +
    # Mystery Devices).  Default on.  Disable on low-core hosts to save
    # ~10-15 % decode CPU at the cost of those features.
    if SETTINGS.get('fingerprint', True):
        env['LORA_FINGERPRINT'] = '1'
    else:
        env['LORA_FINGERPRINT'] = '0'
    # LDRO fallback: on CRC fail, retry the payload with the opposite Low Data Rate
    # Optimization setting.  Recovers transmitters that force LDRO against the 16 ms
    # symbol-time default (e.g. satellite / tinyGS at SF8/41.7).  Default on; only
    # runs on CRC-fail so the cost is negligible.
    env['LORA_LDRO_FALLBACK'] = '1' if SETTINGS.get('ldro_fallback', True) else '0'
    # IQ inversion: conjugate the input stream so an IQ-inverted transmitter
    # (LoRaWAN downlink, satellite / tinyGS with Invert-IQ) decodes.  Default off
    # (normal IQ).  Mutually exclusive with normal traffic, so it's an explicit
    # mode the user selects, not an automatic fallback.
    env['LORA_IQ_INVERT'] = '1' if SETTINGS.get('iq_invert') else '0'
    _pr = SETTINGS.get('protocols') or {}
    env['LORA_PROTOCOLS'] = ','.join(k for k in (
        'meshtastic', 'meshcore', 'lorawan', 'loramesher',
        'lora_aprs', 'reticulum', 'disaster_radio', 'ebyte_lora', 'radiohead',
    ) if _pr.get(k, True))
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


@app.route('/api/mystery_devices')
def api_mystery_devices():
    """Aggregated UNKNOWN-protocol fingerprints.  Each entry = one protocol
    family (matching payload prefix), with a list of estimated distinct
    transmitter devices clustered by RF hardware fingerprint (DC offset,
    I/Q imbalance, CFO).  Surfaces a '≥ N devices' lower bound, NOT a
    definitive device count: same-batch hardware can merge into one cluster,
    temperature drift can split one device across clusters.

    Default min=2 hides single-sighting fingerprints; pass ?min=1 to include."""
    min_count = int(request.args.get('min', 2))
    with STATE_LOCK:
        rows = []
        for prefix, d in UNKNOWN_FP.items():
            if d['count'] < min_count:
                continue
            devs = [{
                'count': dev['count'],
                'first_ts': dev['first_ts'],
                'last_ts': dev['last_ts'],
                'tx_freq_hz': round(dev['mean_fv'][0], 1),
                'tx_freq_mhz': round(dev['mean_fv'][0] / 1e6, 6),
                'freq_mhz_last': dev.get('freq_mhz_last'),
            } for dev in d.get('devices', [])]
            devs.sort(key=lambda x: -x['count'])
            rows.append({
                'prefix': prefix,
                'count': d['count'],
                'first_ts': d['first_ts'], 'last_ts': d['last_ts'],
                'rssi_min': d['rssi_min'], 'rssi_max': d['rssi_max'],
                'samples': list(d['samples']),
                'ascii_pattern': d['ascii_pattern'],
                'hint': d.get('hint'),
                'device_count': len(devs),     # ≥ N: lower-bound estimate
                'devices': devs,
            })
        rows.sort(key=lambda r: -r['last_ts'])
        return jsonify(rows)


@app.route('/api/device_clusters')
def api_device_clusters():
    """Online device fingerprint clusters.  Each entry is a UMOP-feature
    cluster — same physical device → same cluster_id across packets, new
    device → new cluster.  Created from packet 1 with no training."""
    with STATE_LOCK:
        out = []
        for c in DEVICE_CLUSTERS:
            out.append({
                'id': c['id'],
                'count': c['count'],
                'first_ts': c.get('first_ts'),
                'last_ts': c.get('last_ts'),
                'from_seen': c.get('from_seen', {}),
                'std': c.get('std', {}),
                'centroid': c.get('centroid', {}),
            })
        out.sort(key=lambda x: -x['count'])
    return Response(json.dumps({'clusters': out, 'total': len(out)}),
                    mimetype='application/json')


@app.route('/api/node_profiles')
def api_node_profiles():
    """Hardware fingerprint profiles learned from labeled Meshtastic traffic.
    Each entry shows the per-feature mean and std for one known node — the
    classifier uses these to identify which physical device emitted an
    unknown packet (by Mahalanobis-like distance over the feature vector)."""
    with STATE_LOCK:
        out = []
        for nid, p in NODE_PROFILES.items():
            out.append({
                'node_id': nid,
                'count': p['count'],
                'usable': p['count'] >= NODE_PROFILES_MIN_FOR_USE and bool(p['mean']),
                'mean': p['mean'],
                'std': p['std'],
                'first_ts': p.get('first_ts'),
                'last_ts': p.get('last_ts'),
            })
        out.sort(key=lambda x: -x['count'])
        return jsonify(out)


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
        if 'fingerprint' in d:
            SETTINGS['fingerprint'] = bool(d['fingerprint'])
        if 'protocols' in d and isinstance(d['protocols'], dict):
            pr = dict(SETTINGS.get('protocols') or {})
            for k in ('meshtastic', 'meshcore', 'lorawan', 'loramesher',
                      'lora_aprs', 'reticulum', 'disaster_radio', 'ebyte_lora', 'radiohead'):
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
        if 'tune' in d and isinstance(d['tune'], dict):
            tune = dict(SETTINGS.get('tune') or {})
            # Bounds match the CLI flag ranges and keep the pipeline numerically
            # sane — out-of-range values get clamped, not rejected.
            _bounds = {'threshold': (0.1, 1.0, 0.55),
                       'energy_threshold': (0.0, 40.0, 12.0),
                       'buf_seconds': (2, 128, 16)}
            for k, (lo, hi, _def) in _bounds.items():
                if k in d['tune']:
                    try:
                        v = float(d['tune'][k])
                        if v < lo: v = lo
                        if v > hi: v = hi
                        tune[k] = int(v) if k == 'buf_seconds' else v
                    except (TypeError, ValueError):
                        pass
            SETTINGS['tune'] = tune
        if any(k in d for k in ('autosave', 'waterfall', 'unknown', 'fingerprint',
                                'protocols', 'wide_scan', 'sdr', 'radio', 'tune')):
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
    _NODE_HINT = ' — looks like a node pub/priv key; those go in "Node identities" below, not here'
    if proto == 'meshcore':
        if not _mc_bytes16(key):
            hint = _NODE_HINT if _hex_bytes(key, (32, 64)) is not None else ''
            return jsonify({'ok': False, 'error': 'MeshCore channel key must be 16 bytes (AES-128), hex or base64' + hint})
    elif not _key_valid(key):
        hint = _NODE_HINT if _hex_bytes(key, (64,)) is not None else ''
        return jsonify({'ok': False, 'error': 'key must be base64/hex for 16, 24 or 32 bytes' + hint})
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


def _hex_bytes(s, expected_lens):
    """Strip whitespace, decode hex OR base64, return bytes if length matches expected_lens (tuple)."""
    s = (s or '').strip()
    try:
        b = bytes.fromhex(s)
        if len(b) in expected_lens:
            return b
    except ValueError:
        pass
    try:
        b = _b64.b64decode(s + '=' * (-len(s) % 4))
        if len(b) in expected_lens:
            return b
    except Exception:
        pass
    return None


def _validate_identity(proto, pub, priv):
    """Return (pub_bytes, priv_bytes, error_str_or_None).

    Verifies the priv-pub pair: for MeshCore, Ed25519 scalar (priv[0:32]) * base
    must equal pub; for Meshtastic, X25519 scalar * base must equal pub.  Stops
    the user from saving inconsistent pairs (typo, swapped, copy-paste errors)
    which would otherwise produce undecryptable garbage at runtime.

    Pub-only contacts: if `priv` is empty/None, validate only the pub (32B).
    The entry seeds _MC_PUBKEY_REG so inbound DMs from this peer can be
    decrypted using a separately-configured full identity (the recipient)."""
    pub_b = _hex_bytes(pub, (32,))
    if pub_b is None:
        return None, None, 'pub must be 32 bytes (hex or base64)'
    if not (priv or '').strip():
        if proto in ('meshcore', 'meshtastic'):
            return pub_b, b'', None
        return None, None, 'protocol must be meshcore or meshtastic'
    if proto == 'meshcore':
        priv_b = _hex_bytes(priv, (64,))
        if priv_b is None:
            return None, None, 'meshcore priv must be 64 bytes (hex/base64)'
        try:
            from nacl.bindings import crypto_scalarmult_ed25519_base_noclamp
        except ImportError:
            return None, None, 'pynacl not installed — run: pip install pynacl'
        try:
            derived = crypto_scalarmult_ed25519_base_noclamp(priv_b[:32])
            if derived != pub_b:
                return None, None, 'priv does not derive to pub (Ed25519 base mismatch — check keys)'
        except Exception as e:
            return None, None, 'nacl error: %s' % e
    elif proto == 'meshtastic':
        priv_b = _hex_bytes(priv, (32,))
        if priv_b is None:
            return None, None, 'meshtastic priv must be 32 bytes (hex/base64)'
        try:
            from nacl.bindings import crypto_scalarmult_base
        except ImportError:
            return None, None, 'pynacl not installed — run: pip install pynacl'
        try:
            derived = crypto_scalarmult_base(priv_b)
            if derived != pub_b:
                return None, None, 'priv does not derive to pub (X25519 base mismatch — check keys)'
        except Exception as e:
            return None, None, 'nacl error: %s' % e
    else:
        return None, None, 'protocol must be meshcore or meshtastic'
    return pub_b, priv_b, None


@app.route('/api/identities', methods=['GET'])
def api_identities_get():
    return jsonify({'identities': load_keys().get('identities', [])})


@app.route('/api/identities', methods=['POST'])
def api_identities_add():
    d = request.get_json(force=True, silent=True) or {}
    proto = 'meshcore' if d.get('protocol') == 'meshcore' else 'meshtastic'
    pub = (d.get('pub') or '').strip()
    priv = (d.get('priv') or '').strip()
    pub_b, priv_b, err = _validate_identity(proto, pub, priv)
    if err:
        return jsonify({'ok': False, 'error': err})
    kc = load_keys()
    kc['identities'].append({
        'id': 'i%d' % int(time.time() * 1000),
        'protocol': proto,
        'label': (d.get('label') or 'identity').strip(),
        'pub': pub_b.hex(),
        'priv': priv_b.hex(),
        'node_id': (d.get('node_id') or '').strip(),
        'enabled': d.get('enabled') is not False,
        'notes': (d.get('notes') or '').strip(),
    })
    save_keys(kc)
    return jsonify({'ok': True, 'keys': kc})


@app.route('/api/identities/<iid>', methods=['DELETE'])
def api_identities_del(iid):
    kc = load_keys()
    kc['identities'] = [k for k in kc['identities'] if k.get('id') != iid]
    save_keys(kc)
    return jsonify({'ok': True, 'keys': kc})


@app.route('/api/identities/<iid>/update', methods=['POST'])
def api_identities_update(iid):
    """Update an identity's mutable fields (enabled / label / notes / node_id).  Key
    material (pub / priv) is never editable — delete and re-add to rotate."""
    d = request.get_json(force=True, silent=True) or {}
    kc = load_keys()
    for k in kc['identities']:
        if k.get('id') != iid:
            continue
        if 'enabled' in d: k['enabled'] = bool(d['enabled'])
        if 'label' in d:   k['label'] = (d.get('label') or k.get('label') or 'identity').strip()
        if 'notes' in d:   k['notes'] = (d.get('notes') or '').strip()
        if 'node_id' in d: k['node_id'] = (d.get('node_id') or '').strip()
        save_keys(kc)
        return jsonify({'ok': True, 'keys': kc})
    return jsonify({'ok': False, 'error': 'unknown identity id'})


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
UNKNOWN_PATH = os.path.join(_DATA_DIR, "lora_unknown.jsonl")
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
        # Also wipe the in-memory session-display dictionaries that the
        # original /api/clear missed.  The GUI button reads "Clear all
        # captured packets (also clears the saved log)" — user expectation
        # is a clean slate, but mystery_devices kept showing entries from
        # before the clear because UNKNOWN_FP and PENDING_ENC were never
        # touched.  Both are pure session-display state; they get rebuilt
        # from incoming packets.  Persisted ACCUMULATED knowledge
        # (DEVICE_CLUSTERS, MC pubkey registry, KNOWN_MESH, NODE_PROFILES,
        # LW_DEVICES) stays — that's cross-session learning, not session
        # display, same rationale as the existing MC ADVERT registry note.
        UNKNOWN_FP.clear()
        PENDING_ENC.clear()
        _FP_CANDIDATES.clear()
    # also truncate the persistent logs so cleared data doesn't reload on
    # restart (the tailer detects the shrink and resets its read position).
    # DOES NOT truncate the cross-worker MeshCore ADVERT pubkey registry —
    # that's accumulated cryptographic knowledge of the mesh, not session
    # display state, and clearing it silently regresses path-hash promotion
    # for all subsequent encrypted frames until each node re-advertises.
    # Users who want a registry reset can delete the file manually
    # (/tmp/lora_mc_advert_registry.jsonl) or pipeline-restart with the
    # LORA_MC_ADVERT_REG env override pointing to a tmpfile.
    for _p in (CFG['decode'].get('packet_log', '/tmp/lora_packets.jsonl'),
               AUTOSAVE_PATH):
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


DEBUG_BUNDLE_PATH = None        # set by main() when --debug is on


def _debug_emit(line, prefix=''):
    """Print to stderr AND append to the debug bundle file (when --debug)."""
    print(prefix + line, file=sys.stderr, flush=True)
    if DEBUG_BUNDLE_PATH:
        try:
            with open(DEBUG_BUNDLE_PATH, 'a') as f:
                f.write(prefix + line + '\n')
        except Exception:
            pass                     # never let the bundle writer break the run


def _print_debug_header():
    """`--debug`: dump scrubbed diagnostics so issue reporters get a one-paste
    bundle. The same content is ALSO written to DEBUG_BUNDLE_PATH so the user
    has a single file they can attach to the issue — no terminal-scrollback
    copy-paste needed."""
    try:
        import debug_collect
    except Exception as e:
        print(f'lora_web: --debug active but debug_collect unavailable: {e}',
              file=sys.stderr, flush=True)
        return
    bundle = debug_collect.render_bundle(cfg=CFG, settings=SETTINGS,
                                         include_pipeline_log=False)
    if DEBUG_BUNDLE_PATH:
        try:
            with open(DEBUG_BUNDLE_PATH, 'w') as f:
                f.write(bundle + '\n')
        except Exception as e:
            print(f'lora_web: --debug bundle file write failed: {e}',
                  file=sys.stderr, flush=True)
    print('lora_web: --debug ON — diagnostics dump (PII-scrubbed) follows.',
          file=sys.stderr, flush=True)
    print(bundle, file=sys.stderr, flush=True)
    if DEBUG_BUNDLE_PATH:
        # Basename only — users may paste this terminal output into an issue.
        print(f'\nlora_web: --debug — startup dump also written to '
              f'{os.path.basename(DEBUG_BUNDLE_PATH)} (in the project root)\n'
              f'    Live pipeline output will be appended there as it streams.\n'
              f'    Attach that file to the GitHub issue.',
              file=sys.stderr, flush=True)
    print('lora_web: --debug — end of startup dump. Pipeline stderr will be '
          'tee\'d to this terminal (and the bundle file) once you click Start.',
          file=sys.stderr, flush=True)


def _tee_pipeline_log_to_stderr():
    """`--debug`: stream PIPELINE_LOG to our stderr live AND append to the
    debug bundle file so the user sees the capture/detector subprocess output
    in the terminal — without this, those lines only land in
    /tmp/lora_web_pipeline.log and reporters routinely miss them."""
    try:
        import debug_collect as _dc
    except Exception:
        _dc = None
    pos = 0
    while True:
        try:
            if not os.path.exists(PIPELINE_LOG):
                time.sleep(0.5)
                continue
            sz = os.path.getsize(PIPELINE_LOG)
            if sz < pos:                          # log was truncated on Start
                pos = 0
            if sz > pos:
                with open(PIPELINE_LOG, 'rb') as f:
                    f.seek(pos)
                    data = f.read(sz - pos)
                    pos = sz
                text = data.decode('utf-8', errors='replace')
                if _dc is not None:
                    text = _dc.scrub(text)
                # Tag so the user can tell this came from the subprocess vs flask.
                for line in text.splitlines():
                    _debug_emit(line, prefix='[pipe] ')
        except Exception as e:
            print(f'[pipe-tee error: {e}]', file=sys.stderr, flush=True)
        time.sleep(0.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default=None)
    ap.add_argument('--host', default=None)
    ap.add_argument('--port', type=int, default=None)
    ap.add_argument('--debug', action='store_true',
                    help='Verbose mode for issue reports: dump scrubbed env/SDR/'
                         'binary diagnostics on startup, tee pipeline stderr to '
                         'this terminal, and write the same content to '
                         'lora_debug_<ts>.txt in the project root for one-file '
                         'attachment to a GitHub issue (no PII).')
    ap.add_argument('--debug-out', default=None,
                    help='Override --debug bundle file path (default: '
                         'lora_debug_<ts>.txt in the project root).')
    a = ap.parse_args()
    global CFG, DEBUG_BUNDLE_PATH
    CFG = load_config(a.config)
    if a.debug:
        os.environ['LORA_DEBUG'] = '1'
        DEBUG_BUNDLE_PATH = a.debug_out or os.path.join(
            _PROJECT, 'lora_debug_%s.txt' % time.strftime('%Y%m%d_%H%M%S'))
        _print_debug_header()
        threading.Thread(target=_tee_pipeline_log_to_stderr, daemon=True).start()
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
    # Periodic persistence of node hardware profiles (saves the classifier's
    # calibration state across web restarts).
    def _profiles_save_loop():
        while True:
            time.sleep(60)
            with STATE_LOCK:
                _save_node_profiles()
                _fp_save()
    threading.Thread(target=_profiles_save_loop, daemon=True).start()
    # Save on graceful exit too.
    atexit.register(lambda: (STATE_LOCK.acquire() if hasattr(STATE_LOCK, 'acquire') else None,
                              _save_node_profiles(),
                              _fp_save(),
                              STATE_LOCK.release() if STATE_LOCK.locked() else None))
    _apply_waterfall_flag()   # reflect the saved waterfall on/off to the gate marker
    _apply_unknown_flag()
    threading.Thread(target=_tail_health, daemon=True).start()
    threading.Thread(target=_tail_psd, daemon=True).start()
    _startup_line = (f"lora_web: config={CFG.get('_path')}  tailing={log_path}  "
                     f"serving http://{host}:{port}")
    if os.environ.get('LORA_DEBUG') == '1':
        try:
            import debug_collect
            _startup_line = debug_collect.scrub(_startup_line)
        except Exception:
            pass
    print(_startup_line, flush=True)
    if host not in ('127.0.0.1', 'localhost', '::1'):
        print("lora_web: *** WARNING: bound to %s (non-localhost) and the UI has NO "
              "authentication. Anyone who can reach this port can control the SDR and "
              "read intercepted traffic. Use 127.0.0.1, or put a reverse proxy + auth "
              "in front on trusted networks only. ***" % host, file=sys.stderr, flush=True)
    app.run(host=host, port=port, threaded=True, debug=False)


if __name__ == '__main__':
    main()
