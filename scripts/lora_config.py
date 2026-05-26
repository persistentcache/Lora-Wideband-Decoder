"""Central config loader for the LoRa receiver (lora.toml).

Both lora_web.py and lora_detect.py use this so settings live in one file
instead of long CLI flag strings.  Missing keys fall back to defaults; explicit
CLI flags (in lora_detect) still override the config.
"""
import os
try:
    import tomllib  # Python 3.11+
except ImportError:
    tomllib = None

_DEFAULTS = {
    'radio':  {'rate_hz': 28000000, 'bandwidth_hz': 28000000,
               'center_mhz': 915.0, 'format': 'sc16'},
    'detect': {'threshold': 0.55, 'energy_threshold': 5.0, 'overlap': 0.5,
               'detect_workers': -1, 'buf_seconds': 16, 'commit_lag': 4},
    'decode': {'budget_s': 10.0, 'workers': 10, 'export_dir': '/dev/shm/live_caps',
               'packet_log': '/tmp/lora_packets.jsonl', 'key': 'default'},
    'web':    {'host': '0.0.0.0', 'port': 5000},
}


def find_config(path=None):
    """Locate lora.toml: explicit arg → LORA_CONFIG env → cwd → project root."""
    cands = [path, os.environ.get('LORA_CONFIG'),
             os.path.join(os.getcwd(), 'lora.toml'),
             os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          'lora.toml')]
    for c in cands:
        if c and os.path.isfile(c):
            return c
    return None


def load_config(path=None):
    """Return the merged config dict (defaults + lora.toml overrides)."""
    cfg = {k: dict(v) for k, v in _DEFAULTS.items()}
    found = find_config(path)
    if found and tomllib:
        try:
            with open(found, 'rb') as f:
                user = tomllib.load(f)
            for sec, vals in user.items():
                if isinstance(vals, dict):
                    cfg.setdefault(sec, {}).update(vals)
                else:
                    cfg[sec] = vals
        except Exception:
            pass
    cfg['_path'] = found
    return cfg
