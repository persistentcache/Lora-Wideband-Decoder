"""Diagnostic collectors + PII scrubber for `--debug` and `collect_debug.py`.

One source of truth so the in-server `--debug` mode and the standalone collector
emit IDENTICAL sections — when a user reports an issue you get the same fields
either way.

PII policy (standard):
  - $HOME paths      → ~
  - hostname         → <host>
  - external IPs     → <ip>           (127.0.0.1 / ::1 kept)
  - MAC addresses    → <mac>
KEPT (needed to diagnose hardware-specific bugs):
  - SDR serials, USB IDs, SoapySDR module versions, kernel/distro strings.
"""
import os
import re
import sys
import json
import shutil
import socket
import platform
import subprocess


# ----------------------------------------------------------------- PII scrubber

_HOSTNAME = socket.gethostname() or ''
_USER = os.environ.get('USER') or os.environ.get('LOGNAME') or ''
_HOME = os.path.expanduser('~')

_RE_IP = re.compile(
    r'\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d?\d)(?:\.(?:25[0-5]|2[0-4]\d|[01]?\d?\d)){3})\b')
# IPv6: require ≥3 colons so HH:MM:SS clock strings (only 2 colons) don't match.
# ::1 loopback is then never matched at all — fine, since we wouldn't scrub it anyway.
_RE_IP6 = re.compile(r'\b(?:[A-Fa-f0-9]{1,4}:){3,7}[A-Fa-f0-9]{1,4}\b')
_RE_MAC = re.compile(r'\b(?:[0-9A-Fa-f]{2}[:-]){5}[0-9A-Fa-f]{2}\b')
_LOOPBACK = {'127.0.0.1', '0.0.0.0', '::1'}


def scrub(text):
    """Return `text` with PII scrubbed per the policy above."""
    if not text:
        return text
    s = str(text)
    if _HOME and _HOME != '/':
        s = s.replace(_HOME, '~')
    if _USER:
        s = re.sub(r'\b' + re.escape(_USER) + r'\b', '<user>', s)
    if _HOSTNAME:
        s = re.sub(r'\b' + re.escape(_HOSTNAME) + r'\b', '<host>', s, flags=re.I)
    s = _RE_MAC.sub('<mac>', s)
    s = _RE_IP.sub(lambda m: m.group(0) if m.group(0) in _LOOPBACK else '<ip>', s)
    # ::1 is loopback; everything else IPv6-shaped → scrub
    s = _RE_IP6.sub(lambda m: m.group(0) if m.group(0) == '::1' else '<ip>', s)
    return s


# ------------------------------------------------------------- shell + helpers


def _run(cmd, timeout=8):
    """Run a shell command, return (rc, combined_output_str). Never raises."""
    try:
        p = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                           timeout=timeout)
        return p.returncode, (p.stdout or '') + (p.stderr or '')
    except subprocess.TimeoutExpired:
        return 124, '[timeout after %ds]' % timeout
    except Exception as e:
        return -1, '[error: %s]' % e


def _which(name):
    return shutil.which(name)


# --------------------------------------------------------------- collectors


def collect_env():
    py = sys.executable
    rc, py_v = _run('"%s" --version' % py, timeout=5)
    py3 = _which('python3') or ''
    rc, py3_v = (_run('"%s" --version' % py3, timeout=5) if py3 else (0, ''))
    rc, py_alias = _run('python --version', timeout=5)
    return {
        'platform': platform.platform(),
        'python_executable': py,
        'python_version': sys.version.split('\n', 1)[0],
        'python_dash_v_output': py_v.strip(),
        'sys_path_python3': py3,
        'sys_path_python3_version': py3_v.strip(),
        'system_python_alias_version': py_alias.strip(),
        'in_venv': sys.prefix != getattr(sys, 'base_prefix', sys.prefix),
        'prefix': sys.prefix,
        'distro': _read_os_release(),
        'kernel': platform.release(),
        'cwd': os.getcwd(),
    }


def _read_os_release():
    try:
        with open('/etc/os-release') as f:
            kv = {}
            for line in f:
                if '=' in line:
                    k, _, v = line.partition('=')
                    kv[k.strip()] = v.strip().strip('"')
        return f"{kv.get('NAME','')} {kv.get('VERSION','')}".strip()
    except Exception:
        return ''


def collect_soapy():
    out = {}
    rc, info = _run('SoapySDRUtil --info', timeout=10)
    out['SoapySDRUtil_info_rc'] = rc
    out['SoapySDRUtil_info'] = info
    rc, find = _run('SoapySDRUtil --find', timeout=15)
    out['SoapySDRUtil_find_rc'] = rc
    out['SoapySDRUtil_find'] = find
    # Can OUR python actually import SoapySDR? This is the K4KDR-style mismatch
    # where `python3` is 3.14 but the apt SoapySDR module is built against 3.12.
    rc, imp = _run(
        '"%s" -c "import SoapySDR; '
        'print(\'SoapySDR_import_ok\'); '
        'print(\'SoapySDR_file\', SoapySDR.__file__); '
        'print(\'SoapySDR_version\', getattr(SoapySDR, \'__version__\', '
        '\'(no __version__ attr)\'))"' % sys.executable,
        timeout=10)
    out['python_import_SoapySDR_rc'] = rc
    out['python_import_SoapySDR'] = imp
    rc, imp_np = _run(
        '"%s" -c "import numpy as n; print(\'numpy\', n.__version__)"' % sys.executable,
        timeout=10)
    out['python_import_numpy_rc'] = rc
    out['python_import_numpy'] = imp_np
    return out


def collect_binaries():
    """Resolve each SDR profile's capture binary so missing tools are obvious."""
    bins = ['bladeRF-cli', 'hackrf_transfer', 'hackrf_info', 'rtl_sdr',
            'airspy_rx', 'uhd_find_devices', 'uhd_usrp_probe', 'SoapySDRUtil',
            'python3']
    return {b: (_which(b) or '<missing>') for b in bins}


def collect_hardware():
    out = {}
    # USB enumerate — most SDRs are USB. lsusb is in usbutils, very common.
    rc, lsusb = _run('lsusb', timeout=5)
    out['lsusb_rc'] = rc
    out['lsusb'] = lsusb
    rc, hi = _run('hackrf_info 2>&1 | head -50', timeout=8)
    out['hackrf_info_rc'] = rc
    out['hackrf_info'] = hi
    rc, ud = _run('uhd_find_devices 2>&1 | head -80', timeout=10)
    out['uhd_find_devices_rc'] = rc
    out['uhd_find_devices'] = ud
    rc, br = _run('bladeRF-cli -e info 2>&1 | head -40', timeout=8)
    out['bladeRF_info_rc'] = rc
    out['bladeRF_info'] = br
    # Groups (plugdev for USB SDR access without root)
    rc, idout = _run('id', timeout=3)
    out['id'] = idout.strip()
    return out


def collect_config(cfg=None, settings=None):
    """Stringify lora.toml and SETTINGS. Keys/serials etc. are scrubbed only via
    `scrub()` at write time."""
    out = {}
    if cfg is not None:
        try:
            out['CFG'] = json.dumps(cfg, indent=2, default=str)
        except Exception as e:
            out['CFG'] = '[unstringifiable: %s]' % e
    if settings is not None:
        # Redact channel keys outright — they're literal AES keys, never debug info.
        try:
            s = dict(settings)
            if 'keys' in s:
                s['keys'] = '<redacted: %d entries>' % (
                    len(s['keys']) if hasattr(s['keys'], '__len__') else 0)
            out['SETTINGS'] = json.dumps(s, indent=2, default=str)
        except Exception as e:
            out['SETTINGS'] = '[unstringifiable: %s]' % e
    return out


def collect_pipeline_log(path='/tmp/lora_web_pipeline.log', tail_kb=64):
    """Return the tail of the pipeline stderr log (where the real failure prints).
    This is the file 90% of issue reporters forget to include."""
    try:
        sz = os.path.getsize(path)
    except OSError:
        return {'path': path, 'present': False}
    n = min(sz, tail_kb * 1024)
    try:
        with open(path, 'rb') as f:
            f.seek(max(0, sz - n))
            data = f.read().decode('utf-8', errors='replace')
    except Exception as e:
        return {'path': path, 'present': True, 'size': sz, 'error': str(e)}
    return {'path': path, 'present': True, 'size': sz, 'tail': data}


# --------------------------------------------------------------- bundle render

_SEP = '=' * 78


def _section(title, body):
    return '\n%s\n== %s\n%s\n%s\n' % (_SEP, title, _SEP, body)


def render_bundle(cfg=None, settings=None, include_pipeline_log=True,
                  extra_sections=None):
    """Produce the full scrubbed text bundle. `extra_sections` is a list of
    (title, body) tuples appended at the end (e.g. an attempted pipeline test)."""
    import time as _t
    parts = []
    parts.append('LORA Wideband Decoder — debug bundle  (%s)\n'
                 'PII policy: $HOME→~, hostname, IPs, MACs scrubbed; '
                 'serials/versions kept.' % _t.strftime('%Y-%m-%d %H:%M:%S'))

    env = collect_env()
    parts.append(_section('ENVIRONMENT',
                          json.dumps(env, indent=2, default=str)))

    bins = collect_binaries()
    parts.append(_section('SDR BINARIES (paths)',
                          json.dumps(bins, indent=2)))

    soapy = collect_soapy()
    parts.append(_section('SOAPYSDR', json.dumps(soapy, indent=2)))

    hw = collect_hardware()
    parts.append(_section('HARDWARE', json.dumps(hw, indent=2)))

    if cfg is not None or settings is not None:
        cfgsec = collect_config(cfg, settings)
        parts.append(_section('CONFIG', json.dumps(cfgsec, indent=2)))

    if include_pipeline_log:
        pl = collect_pipeline_log()
        parts.append(_section('PIPELINE LOG (/tmp/lora_web_pipeline.log tail)',
                              json.dumps(pl, indent=2)))

    for title, body in (extra_sections or []):
        parts.append(_section(title, body))

    return scrub('\n'.join(parts))
