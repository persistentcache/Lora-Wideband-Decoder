#!/usr/bin/env python3
"""Standalone diagnostic collector — for issue reports.

Runs every check (env / SoapySDR / SDR binaries / USB / config / pipeline log)
PLUS a ~5 s capture probe against the currently-configured SDR, and writes ONE
PII-scrubbed text file to the project root.  No web server needed — use this
when the pipeline dies before you can interact with the UI.

  python3 run/collect_debug.py

Output: lora_debug_<timestamp>.txt in the project root (attach to the issue).
PII policy: $HOME→~, hostname, IPs, MACs scrubbed; SDR serials / versions kept.
"""
import os
import sys
import time
import subprocess

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_SRC = os.path.join(_ROOT, 'src')
for _p in (_SRC, os.path.join(_SRC, 'web')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_PROBE_SECONDS = 5


def _load_state():
    """Best-effort load lora.toml and web_settings.json — both optional, since
    the user may run this collector before either exists."""
    cfg = None
    settings = None
    try:
        from lora_config import load_config
        cfg = load_config()
    except Exception as e:
        cfg = {'_error': 'load_config failed: %s' % e}
    try:
        import json
        with open(os.path.join(_ROOT, 'lora_web', 'web_settings.json')) as f:
            settings = json.load(f)
    except Exception:
        settings = None
    return cfg, settings


def _probe_pipeline(cfg, settings, seconds=_PROBE_SECONDS):
    """Run the actual capture|detector pipeline briefly and capture stderr.
    Uses the same builder the web UI does, so this surfaces THE exact failure
    the user would hit clicking Start."""
    try:
        import sdr_profiles
    except Exception as e:
        return 'sdr_profiles import failed: %s' % e
    try:
        radio = dict(cfg.get('radio') or {})
        if settings and settings.get('radio'):
            radio.update(settings['radio'])
        sdr = (settings or {}).get('sdr') or 'bladerf'
        cen = int(float(radio.get('center_mhz', 915.0)) * 1e6)
        rate = int(radio.get('rate_hz', 16000000))
        bw = int(radio.get('bandwidth_hz', rate))
        gain = radio.get('gain', 'auto')
        soapy_driver = radio.get('soapy_driver', 'hackrf')
        cap = sdr_profiles.build_capture(sdr, cen, rate, bw, gain=gain,
                                         soapy_driver=soapy_driver)
    except Exception as e:
        return 'build_capture failed: %s' % e
    # Run capture ALONE (no detector) for the briefest viable test of the
    # capture half — that's where K4KDR-style failures live.
    cmd = '( %s ) > /dev/null 2>/tmp/lora_probe.err & PID=$!; sleep %d; kill $PID 2>/dev/null; wait $PID 2>/dev/null; cat /tmp/lora_probe.err' % (cap, seconds)
    env = dict(os.environ)
    env['LORA_DEBUG'] = '1'
    try:
        p = subprocess.run(['bash', '-c', cmd], capture_output=True, text=True,
                           timeout=seconds + 10, env=env)
        out = 'COMMAND: %s\n\nRC: %d\n\nSTDERR:\n%s' % (cap, p.returncode,
                                                       p.stdout + p.stderr)
    except subprocess.TimeoutExpired:
        out = 'COMMAND: %s\n\n[probe timed out]' % cap
    except Exception as e:
        out = 'COMMAND: %s\n\n[probe error: %s]' % (cap, e)
    return out


def main():
    import debug_collect
    cfg, settings = _load_state()
    print('collect_debug: running ~%ds pipeline probe…' % _PROBE_SECONDS,
          file=sys.stderr, flush=True)
    extras = [('PIPELINE PROBE (capture half, %ds)' % _PROBE_SECONDS,
               _probe_pipeline(cfg, settings, _PROBE_SECONDS))]
    bundle = debug_collect.render_bundle(cfg=cfg, settings=settings,
                                         include_pipeline_log=True,
                                         extra_sections=extras)
    out_path = os.path.join(
        _ROOT, 'lora_debug_%s.txt' % time.strftime('%Y%m%d_%H%M%S'))
    with open(out_path, 'w') as f:
        f.write(bundle)
    # Print just the basename so users can paste the script's own output into
    # an issue without leaking their home-directory path.
    print('collect_debug: wrote %s  (%d bytes, in the project root)'
          % (os.path.basename(out_path), len(bundle)))
    print('Attach that file to your GitHub issue.')


if __name__ == '__main__':
    main()
