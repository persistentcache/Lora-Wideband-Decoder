"""SDR device profiles — capture command, IQ format, and presence-probe per SDR.

Adding support for a new SDR = add an entry here.  The capture template streams
raw interleaved IQ to stdout (piped into detector via -t {format}); the probe
command is run to validate the device is present and read its serial.

Placeholders in `capture`: {bin} (resolved exe path), {freq_hz}, {rate}, {bw},
{gain_cmd}/{gain}.  Only formats detector understands are used: sc16 (int16)
or sc8 (int8).  Devices that don't natively emit those (RTL-SDR cu8, LimeSDR,
USRP, PlutoSDR, …) go through the SoapySDR 'rx_sdr' universal profile with
-F CS16, so no special reader is needed.
"""

# Architecture: bladeRF uses its native CLI (the ONLY path that reliably sustains
# 28 Msps — SoapyBladeRF fails setupStream at 28 Msps).  EVERY other device streams
# through ONE validated path: soapy_rx.py (SoapySDR → CS16 on stdout).  Named
# devices (HackRF/RTL-SDR/Airspy) are just SoapySDR profiles with a fixed driver so
# the dropdown stays friendly; the generic 'soapy' entry covers anything else.
_SOAPY_CAP = '{bin} {soapy_rx} --driver {soapy_driver} -f {freq_hz} -s {rate} -b {bw} {gain_cmd}'


def _soapy_profile(label, driver, tested=False, agc_ok=True, default_gain=40,
                   note='', limits=None):
    """A SoapySDR-backed profile for a named device (fixed Soapy driver).  All such
    devices share the soapy_rx.py capture path and emit CS16 (sc16)."""
    return {
        'label': label,
        'bin': 'python3',
        'format': 'sc16',
        'tested': tested,
        'agc_ok': agc_ok,
        'default_gain': default_gain,
        'note': note,
        'limits': limits or {'rate_hz': (250000, 61440000),
                             'bandwidth_hz': (100000, 56000000),
                             'center_mhz': (1.0, 7250.0)},
        'capture': _SOAPY_CAP,
        'soapy_driver': driver,           # FIXED Soapy driver for this named device
        'gain_auto_cmd': '',              # SoapySDR AGC by default (where supported)
        'gain_manual_cmd': '-g {gain}',
        'probe': 'SoapySDRUtil --find="driver=%s"' % driver,
        'serial_re': r'serial\s*=\s*(\S+)',
        'kill_pat': 'soapy_rx.py',        # NEVER pkill the bare 'python3' interpreter
        'soapy_query_driver': driver,     # live limits via the SoapySDR API
    }


SDR_PROFILES = {
    # ---- bladeRF: the validated default.  Native bladeRF-cli — byte-identical to
    # the long-standing command and the ONLY path proven at 28 Msps (do not change
    # without re-validating). ----
    'bladerf': {
        'label': 'bladeRF (Nuand)',
        'bin': 'bladeRF-cli',
        'format': 'sc16',
        'tested': True,
        'agc_ok': True,            # `set agc rx on` works on bladeRF, but a fixed
                                   # gain (48) beats it for weak-signal sensitivity
        'default_gain': 48,        # manual gain (dB): highest before noise-floor clipping
        # Hz for rate/bandwidth, MHz for center — used to clamp the UI/config so
        # the user can't ask for more than the device supports.  Generous static
        # bounds; the LIVE query (limits_probe) tightens these to the actual unit.
        'limits': {'rate_hz': (520834, 61440000), 'bandwidth_hz': (200000, 56000000),
                   'center_mhz': (47.0, 6000.0)},
        'note': 'Validated, native CLI. Only path proven at full 28 MHz / 28 Msps.',
        'capture': ('{bin} -e "set frequency rx {freq_hz}; set samplerate rx {rate}; '
                    'set bandwidth rx {bw}; {gain_cmd}; '
                    'rx config file=/dev/stdout format=bin n=0 buffers=512 samples=32768 '
                    'xfers=64; rx start; rx wait"'),
        'gain_auto_cmd': 'set agc rx on',
        # AGC must be turned OFF before a manual gain will take ('Operation invalid
        # in current state' otherwise).
        'gain_manual_cmd': 'set agc rx off; set gain rx {gain}',
        'probe': '{bin} -e info',
        'serial_re': r'Serial\s*#?:?\s*([0-9a-fA-F]{4,})',
        # Live capability query: bladeRF-cli prints each setting's "(Range: [min, max])",
        # so we clamp to THIS exact unit (a bladeRF1 caps at 40 Msps / 28 MHz; a
        # 2.0-micro goes higher) instead of the generic profile limits above.
        'limits_probe': '{bin} -e "print samplerate rx; print frequency rx; print bandwidth rx"',
        'soapy_query_driver': 'bladerf',   # SoapySDR fallback if bladeRF-cli is missing
    },
    # ---- Named SoapySDR devices (friendly dropdown entries, one validated path) ----
    'hackrf': _soapy_profile(
        'HackRF One', 'hackrf', tested=True, agc_ok=False, default_gain=40,
        note='Via SoapySDR. Max 20 Msps. No AGC — set gain in dB (default 40).',
        limits={'rate_hz': (1000000, 20000000), 'bandwidth_hz': (1750000, 28000000),
                'center_mhz': (1.0, 6000.0)}),
    'rtlsdr': _soapy_profile(
        'RTL-SDR', 'rtlsdr', tested=False, agc_ok=True, default_gain=40,
        note='Via SoapySDR. RTL2832U, max ~3.2 Msps (too narrow for 500 kHz LoRa wideband).',
        limits={'rate_hz': (225000, 3200000), 'bandwidth_hz': (250000, 3200000),
                'center_mhz': (24.0, 1766.0)}),
    'airspy': _soapy_profile(
        'Airspy (R2/Mini)', 'airspy', tested=False, agc_ok=True, default_gain=40,
        note='Via SoapySDR. Fixed rates (e.g. 10/2.5 Msps).',
        limits={'rate_hz': (2500000, 10000000), 'bandwidth_hz': (2500000, 10000000),
                'center_mhz': (24.0, 1800.0)}),
    # USRP B-series (B200/B205mini/B210) via SoapyUHD.  No hardware AGC — gain is
    # an analog 0-89.75 dB knob on the AD9364, so we force manual gain.  B205mini
    # sustains ~30 Msps over USB 3.0; needs UHD FPGA images (uhd_images_downloader
    # — pulled by the installer) on first connect.
    'usrp': _soapy_profile(
        'USRP B-series (Ettus)', 'uhd', tested=False, agc_ok=False, default_gain=40,
        note='Via SoapyUHD. B200/B205mini/B210. USB3.0 sustains ~30 Msps. '
             'No AGC — set gain in dB (0-89.75; default 40). First connect '
             'downloads FPGA image; run `sudo uhd_images_downloader` if it fails.',
        limits={'rate_hz': (200000, 56000000),
                'bandwidth_hz': (200000, 56000000),
                'center_mhz': (70.0, 6000.0)}),
    # ---- Generic SoapySDR — anything not named above (LimeSDR/USRP/Pluto/…) ----
    'soapy': {
        'label': 'SoapySDR — other device',
        'bin': 'python3',
        'format': 'sc16',
        'tested': True,            # validated path (HackRF 12/12); driver is user-set
        'agc_ok': True,            # driver-dependent — overridden per-driver by _SOAPY_NO_AGC
        'default_gain': 40,
        'note': "For devices not listed above (LimeSDR/USRP/PlutoSDR/…). Set the driver below (lime|uhd|plutosdr|…). Needs python3-soapysdr + that device's Soapy module.",
        'capture': _SOAPY_CAP,     # uses the user-entered soapy_driver
        'gain_auto_cmd': '',
        'gain_manual_cmd': '-g {gain}',
        'probe': 'SoapySDRUtil --find',
        'serial_re': r'serial\s*=\s*(\S+)',
        'kill_pat': 'soapy_rx.py',   # NEVER pkill the bare 'python3' interpreter
        'limits': {'rate_hz': (250000, 61440000), 'bandwidth_hz': (100000, 56000000),
                   'center_mhz': (1.0, 7250.0)},
    },
}

DEFAULT_SDR = 'bladerf'

# SoapySDR drivers whose AGC is unreliable (rails the ADC, destroying the signal)
# or whose device has no hardware AGC → force manual gain.  Value = a sensible
# default gain (dB) for that device.  USRP B-series has no AGC — pure analog gain.
_SOAPY_NO_AGC = {'hackrf': 40, 'uhd': 40}


def gain_ui(sdr, soapy_driver='hackrf'):
    """How the UI should present gain for this SDR (and, for the generic SoapySDR
    profile, the selected driver): {'agc_ok': bool, 'default_gain': 'auto'|number}.
    When agc_ok is False the UI hides the AGC/auto option and shows a numeric gain
    input pre-filled with default_gain."""
    if sdr == 'soapy':
        drv = (soapy_driver or '').strip().lower()
        if drv in _SOAPY_NO_AGC:
            return {'agc_ok': False, 'default_gain': _SOAPY_NO_AGC[drv]}
        return {'agc_ok': True, 'default_gain': 40}
    p = SDR_PROFILES.get(sdr, SDR_PROFILES[DEFAULT_SDR])
    return {'agc_ok': p.get('agc_ok', True), 'default_gain': p.get('default_gain', 'auto')}


def _which(name):
    import shutil
    return shutil.which(name) or name


def safe_soapy_driver(drv, default='hackrf'):
    """Whitelist the SoapySDR driver to a bare identifier.  It's interpolated into a
    shell=True command, so anything with shell metacharacters (`;`, `|`, `$`, space,
    quotes, backticks…) is rejected → injection guard.  SoapySDR driver names are
    short identifiers (hackrf/rtlsdr/lime/uhd/plutosdr/…)."""
    import re
    drv = str(drv or '').strip()
    return drv if re.fullmatch(r'[A-Za-z0-9_+\-]{1,40}', drv) else default


def _gain_clause(prof, gain):
    """Resolve the gain clause: 'auto' → device AGC/default; else manual gain."""
    g = str(gain).strip().lower() if gain is not None else 'auto'
    if g in ('', 'auto', 'agc'):
        return prof.get('gain_auto_cmd', '')
    try:
        gv = int(float(g))
    except ValueError:
        return prof.get('gain_auto_cmd', '')
    return prof.get('gain_manual_cmd', '').format(gain=gv)


def build_capture(sdr, freq_hz, rate, bw, gain='auto', soapy_driver='hackrf'):
    """Return the shell command that streams raw IQ to stdout for the chosen SDR."""
    import os, shlex
    prof = SDR_PROFILES.get(sdr, SDR_PROFILES[DEFAULT_SDR])
    # named profiles fix the driver; the generic 'soapy' one takes user input →
    # whitelist it (this string is interpolated into a shell=True command).
    drv = safe_soapy_driver(prof.get('soapy_driver') or soapy_driver)
    # If this device's AGC doesn't work, never let an 'auto' gain reach it (it would
    # rail the ADC) — substitute the device's sensible manual default instead.
    ui = gain_ui(sdr, drv)
    if not ui['agc_ok'] and str(gain).strip().lower() in ('', 'auto', 'agc'):
        gain = ui['default_gain']
    scripts_dir = os.path.dirname(os.path.abspath(__file__))
    # shell=True consumer → quote any path that may contain spaces (e.g. project
    # under "LORA Project/"). Without this, the shell splits the word and python
    # tries to open the prefix as the script.
    soapy_rx = shlex.quote(os.path.join(scripts_dir, 'soapy_rx.py'))
    return prof['capture'].format(
        bin=shlex.quote(_which(prof['bin'])), freq_hz=int(freq_hz),
        freq_mhz=freq_hz / 1e6,
        rate=int(rate), bw=int(bw), gain_cmd=_gain_clause(prof, gain),
        soapy_driver=drv, soapy_rx=soapy_rx)


def limits_of(sdr):
    return SDR_PROFILES.get(sdr, SDR_PROFILES[DEFAULT_SDR]).get('limits', {})


def clamp_radio(sdr, radio):
    """Clamp rate/bandwidth/center to the SDR's capabilities so the user can't
    request more than the device supports.  Bandwidth is also capped at the sample
    rate (you can't filter wider than you sample).  Returns a new dict."""
    lim = limits_of(sdr)
    out = dict(radio or {})
    for k in ('rate_hz', 'bandwidth_hz', 'center_mhz'):
        if k in lim and out.get(k) is not None:
            lo, hi = lim[k]
            try:
                out[k] = max(lo, min(hi, float(out[k])))
            except (TypeError, ValueError):
                pass
    if out.get('bandwidth_hz') is not None and out.get('rate_hz') is not None:
        out['bandwidth_hz'] = min(float(out['bandwidth_hz']), float(out['rate_hz']))
    return out


def build_probe(sdr):
    prof = SDR_PROFILES.get(sdr, SDR_PROFILES[DEFAULT_SDR])
    return prof['probe'].format(bin=_which(prof['bin']))


def _cli_limits(cmd):
    """Parse a bladeRF-cli 'print' output for each setting's '(Range: [min, max])'."""
    import subprocess
    import re
    try:
        p = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=12)
        txt = (p.stdout or '') + (p.stderr or '')
    except Exception:
        return {}
    out = {}
    m = re.search(r'sample rate:[^\n]*Range:\s*\[\s*(\d+)\s*,\s*(\d+)', txt, re.I)
    if m:
        out['rate_hz'] = [int(m.group(1)), int(m.group(2))]
    m = re.search(r'Bandwidth:[^\n]*Range:\s*\[\s*(\d+)\s*,\s*(\d+)', txt, re.I)
    if m:
        out['bandwidth_hz'] = [int(m.group(1)), int(m.group(2))]
    m = re.search(r'Frequency:[^\n]*Range:\s*\[\s*(\d+)\s*,\s*(\d+)', txt, re.I)
    if m:
        out['center_mhz'] = [int(m.group(1)) / 1e6, int(m.group(2)) / 1e6]
    return out


def _soapy_limits(driver):
    """Universal limit query via the SoapySDR API — works for ANY device with a
    Soapy module (HackRF, RTL-SDR, LimeSDR, USRP, PlutoSDR, Airspy, bladeRF…).
    Returns {} if SoapySDR is missing or the device isn't present/free."""
    try:
        import SoapySDR
        from SoapySDR import SOAPY_SDR_RX
    except Exception:
        return {}
    try:
        dev = SoapySDR.Device({'driver': driver})
    except Exception:
        return {}                       # device absent / busy / no such driver
    out = {}
    try:
        def span(getter):
            try:
                rs = getter(SOAPY_SDR_RX, 0)
                lo = min(r.minimum() for r in rs)
                hi = max(r.maximum() for r in rs)
                return [lo, hi] if hi > 0 else None
            except Exception:
                return None
        r = span(dev.getSampleRateRange)
        if r:
            out['rate_hz'] = [int(r[0]), int(r[1])]
        r = span(dev.getBandwidthRange)
        if r:
            out['bandwidth_hz'] = [int(r[0]), int(r[1])]
        r = span(dev.getFrequencyRange)
        if r:
            out['center_mhz'] = [r[0] / 1e6, r[1] / 1e6]
    finally:
        dev = None                      # release the device (closes on GC)
    return out


def query_limits(sdr, soapy_driver='hackrf'):
    """Query the LIVE device for its actual rate/bandwidth/center ranges so the UI
    can clamp to what THIS unit really supports (e.g. a bladeRF1 = 40 Msps / 28 MHz,
    not the generic profile maxima).  bladeRF uses bladeRF-cli (validated path);
    every other SDR uses the universal SoapySDR API.  Returns {} if unreachable
    (caller falls back to the static profile limits)."""
    prof = SDR_PROFILES.get(sdr, SDR_PROFILES[DEFAULT_SDR])
    probe = prof.get('limits_probe')
    if probe:                                   # native CLI range probe (bladeRF)
        res = _cli_limits(probe.format(bin=_which(prof['bin'])))
        if res:
            return res
    drv = soapy_driver if sdr == 'soapy' else prof.get('soapy_query_driver')
    if drv:                                     # universal SoapySDR fallback / path
        return _soapy_limits(drv)
    return {}


def fmt_of(sdr):
    return SDR_PROFILES.get(sdr, SDR_PROFILES[DEFAULT_SDR])['format']


def kill_pattern(sdr):
    """Process-name fragment to pkill when freeing the device.  NOT the bare 'bin'
    for SoapySDR (that's the python3 interpreter) — use the script name instead."""
    p = SDR_PROFILES.get(sdr, SDR_PROFILES[DEFAULT_SDR])
    return p.get('kill_pat', p['bin'])


def parse_serial(sdr, text):
    import re
    m = re.search(SDR_PROFILES.get(sdr, SDR_PROFILES[DEFAULT_SDR])['serial_re'], text or '')
    return m.group(1) if m else None
