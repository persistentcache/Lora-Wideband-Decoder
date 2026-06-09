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


def collect_system():
    """Host capacity + free space — needed to interpret KEEP-UP / 'no space'
    reports. CPU model strings are widely shared; not treated as PII."""
    out = {}
    rc, lscpu = _run('lscpu 2>/dev/null | head -25', timeout=5)
    if rc != 0 or not lscpu.strip():
        rc, lscpu = _run("grep -E 'model name|cpu cores|siblings' /proc/cpuinfo | "
                         "head -6", timeout=5)
    out['cpu'] = lscpu
    rc, mem = _run("grep -E 'MemTotal|MemAvailable|SwapTotal' /proc/meminfo",
                   timeout=3)
    out['memory'] = mem
    # /dev/shm — the gate writes IQ exports there; 'no space left' surfaces here.
    rc, shm = _run('df -h /dev/shm', timeout=3)
    out['dev_shm'] = shm
    return out


def collect_packages():
    """What's actually installed at the apt + pip layer. Catches missing
    SoapySDR modules and apt-vs-pip mismatches (a real, common failure)."""
    out = {}
    rc, dpkg = _run(
        "dpkg -l 2>/dev/null | "
        "grep -E 'soapy|libuhd|libusb|bladerf|hackrf|rtl-sdr|airspy|uhd-host' "
        "| awk '{print $1, $2, $3}'", timeout=8)
    out['dpkg_sdr_packages'] = dpkg or '<none matched>'
    rc, pipl = _run(
        '"%s" -m pip list 2>/dev/null | '
        "grep -i -E 'soapy|numpy|scipy|flask|pyuhd' || true" % sys.executable,
        timeout=10)
    out['pip_relevant'] = pipl or '<none matched>'
    return out


def collect_dmesg():
    """USB enumeration / kernel-side device errors. Most distros restrict dmesg
    to root via kernel.dmesg_restrict=1, so we explicitly probe and report the
    state rather than silently returning empty."""
    rc, dm = _run('dmesg 2>&1', timeout=6)
    if rc != 0 or not dm.strip():
        rc2, sdm = _run('sudo -n dmesg 2>&1', timeout=6)
        if rc2 == 0 and sdm.strip():
            return {'dmesg_tail': '\n'.join(sdm.splitlines()[-120:]),
                    'source': 'sudo -n dmesg'}
        return {'dmesg_tail': '',
                'note': '[dmesg unreadable. On Ubuntu/Debian dmesg requires '
                        'root (kernel.dmesg_restrict=1). To include kernel logs '
                        'in the bundle: sudo dmesg | tail -200 > /tmp/dmesg.txt '
                        'and attach that file too.]',
                'source': 'unreadable'}
    return {'dmesg_tail': '\n'.join(dm.splitlines()[-120:]),
            'source': 'dmesg'}


def collect_udev():
    """udev rules grant non-root access to the SDR. Missing rules = 'works
    as root, permission denied as user' reports."""
    rc, ls = _run(
        "ls -la /etc/udev/rules.d/ /usr/lib/udev/rules.d/ 2>/dev/null | "
        "grep -i -E 'hackrf|rtl|blade|airspy|uhd|sdr|usrp' || "
        "echo '<no SDR udev rules found>'", timeout=5)
    return {'sdr_udev_rules': ls}


def collect_pipeline_command(cfg, settings):
    """The actual shell command the web UI would launch for the currently-
    selected SDR. Highest-signal single item in the bundle: surfaces wrong
    driver, malformed gain clause, wrong rate / bandwidth immediately."""
    out = {}
    try:
        import sdr_profiles
    except Exception as e:
        return {'error': 'sdr_profiles import failed: %s' % e}
    try:
        radio = dict((cfg or {}).get('radio') or {})
        if settings and settings.get('radio'):
            radio.update(settings['radio'])
        sdr = (settings or {}).get('sdr') or 'bladerf'
        out['selected_sdr'] = sdr
        out['effective_radio'] = radio
        cen = int(float(radio.get('center_mhz', 915.0)) * 1e6)
        rate = int(radio.get('rate_hz', 16000000))
        bw = int(radio.get('bandwidth_hz', rate))
        gain = radio.get('gain', 'auto')
        soapy_driver = radio.get('soapy_driver', 'hackrf')
        out['capture_command'] = sdr_profiles.build_capture(
            sdr, cen, rate, bw, gain=gain, soapy_driver=soapy_driver)
        out['format'] = sdr_profiles.fmt_of(sdr)
    except Exception as e:
        out['error'] = 'build_capture failed: %s' % e
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

    sysinfo = collect_system()
    parts.append(_section('SYSTEM (cpu / memory / /dev/shm)',
                          json.dumps(sysinfo, indent=2)))

    pkgs = collect_packages()
    parts.append(_section('PACKAGES (dpkg + pip, SDR-relevant)',
                          json.dumps(pkgs, indent=2)))

    udev = collect_udev()
    parts.append(_section('UDEV rules (SDR-relevant)',
                          json.dumps(udev, indent=2)))

    if cfg is not None or settings is not None:
        cfgsec = collect_config(cfg, settings)
        parts.append(_section('CONFIG', json.dumps(cfgsec, indent=2)))
        # Effective pipeline shell command for the selected SDR — single
        # highest-signal item for 'pipeline won't start' reports.
        cmdsec = collect_pipeline_command(cfg, settings)
        parts.append(_section('PIPELINE COMMAND (effective)',
                              json.dumps(cmdsec, indent=2)))

    dm = collect_dmesg()
    parts.append(_section('DMESG (tail 120)', json.dumps(dm, indent=2)))

    if include_pipeline_log:
        pl = collect_pipeline_log()
        parts.append(_section('PIPELINE LOG (/tmp/lora_web_pipeline.log tail)',
                              json.dumps(pl, indent=2)))

    for title, body in (extra_sections or []):
        parts.append(_section(title, body))

    return scrub('\n'.join(parts))
