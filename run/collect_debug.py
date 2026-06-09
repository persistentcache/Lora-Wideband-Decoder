#!/usr/bin/env python3
"""Standalone diagnostic collector — for issue reports.

Runs every check (env / SoapySDR / SDR binaries / USB / config / pipeline log)
and writes ONE PII-scrubbed text file to the project root.  No web server
needed — use this when the pipeline dies before you can interact with the UI.

  python3 run/collect_debug.py
  python3 run/collect_debug.py --probe     # also attempt a ~5 s pipeline test
                                            # for the currently configured SDR

Output: lora_debug_<timestamp>.txt in the project root (attach to the issue).
PII policy: $HOME→~, hostname, IPs, MACs scrubbed; SDR serials / versions kept.
"""
import os
import sys
import time
import shlex
import argparse
import subprocess

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_SRC = os.path.join(_ROOT, 'src')
for _p in (_SRC, os.path.join(_SRC, 'web')):
    if _p not in sys.path:
        sys.path.insert(0, _p)


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


def _probe_pipeline(cfg, settings, seconds=5):
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
    ap = argparse.ArgumentParser()
    ap.add_argument('--probe', action='store_true',
                    help='Also briefly launch the capture half of the pipeline '
                         'for the currently-configured SDR (~5 s).')
    ap.add_argument('--probe-seconds', type=int, default=5)
    ap.add_argument('-o', '--output', default=None,
                    help='Output file path (default: lora_debug_<ts>.txt in '
                         'the project root)')
    a = ap.parse_args()

    import debug_collect
    cfg, settings = _load_state()
    extras = []
    if a.probe:
        print('collect_debug: running ~%ds pipeline probe…' % a.probe_seconds,
              file=sys.stderr, flush=True)
        extras.append(('PIPELINE PROBE (capture half, %ds)' % a.probe_seconds,
                       _probe_pipeline(cfg, settings, a.probe_seconds)))
    bundle = debug_collect.render_bundle(cfg=cfg, settings=settings,
                                         include_pipeline_log=True,
                                         extra_sections=extras)
    out_path = a.output or os.path.join(
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
