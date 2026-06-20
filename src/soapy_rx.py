#!/usr/bin/env python3
"""Stream IQ from ANY SoapySDR device to stdout as interleaved int16 (CS16).

Lets the 'soapy' SDR profile capture from HackRF / RTL-SDR / LimeSDR / USRP /
PlutoSDR / Airspy etc. through one code path, without needing rx_tools — only the
SoapySDR python bindings (apt: python3-soapysdr) plus the device's Soapy module.
Output is CS16, which detector reads with `-t sc16`.

  python3 soapy_rx.py --driver hackrf -f 920e6 -s 20e6 -b 20e6 [-g 40]

DROP-FREE DESIGN (we can't miss captures): the device's ring buffer overflows the
instant it isn't read, so a dedicated reader thread does nothing but service
readStream back-to-back (sized to the device's MTU), handing finished chunks to a
large in-RAM queue.  A separate writer drains the queue to stdout.  This decouples
device servicing from any downstream (pipe/disk) latency, so a transient stall in
detector or the OS pipe can't cost samples.  Overflows, if they ever occur, are
counted and logged to stderr — never silently swallowed.
"""
import os
import sys
import time
import queue
import argparse
import threading


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--driver', required=True, help='SoapySDR driver, e.g. hackrf/rtlsdr/lime/uhd/plutosdr')
    ap.add_argument('-f', type=float, required=True, help='center frequency (Hz)')
    ap.add_argument('-s', type=float, required=True, help='sample rate (Hz)')
    ap.add_argument('-b', type=float, default=0.0, help='bandwidth (Hz); 0 = device default')
    ap.add_argument('-g', type=float, default=None, help='gain (dB); omit for AGC')
    ap.add_argument('--queue-mb', type=float, default=512.0,
                    help='max in-RAM buffer (MB) that absorbs downstream stalls without dropping')
    a = ap.parse_args()
    DBG = bool(int(os.environ.get('LORA_DEBUG', '0') or '0'))
    if DBG:
        sys.stderr.write('soapy_rx: --debug ON  driver=%s freq=%g rate=%g bw=%g gain=%s '
                         'python=%s\n' % (a.driver, a.f, a.s, a.b, a.g, sys.executable))
        sys.stderr.flush()
    try:
        import numpy as np
        import SoapySDR
        from SoapySDR import SOAPY_SDR_RX, SOAPY_SDR_CS16, SOAPY_SDR_OVERFLOW
    except Exception as e:
        # The apt python3-soapysdr package ships SoapySDR as a SWIG-generated
        # C extension bound to the SYSTEM python interpreter — not pip-
        # installable into a venv.  Detect which scenario this is and point
        # the user at the correct fix.
        _in_venv = (hasattr(sys, 'real_prefix')
                    or sys.prefix != getattr(sys, 'base_prefix', sys.prefix)
                    or bool(os.environ.get('VIRTUAL_ENV')))
        if _in_venv:
            sys.stderr.write(
                'soapy_rx: SoapySDR Python bindings not importable (%s).\n'
                '  python=%s  (running inside a venv)\n'
                '  The apt python3-soapysdr package ships as a C extension\n'
                '  bound to the system python — it is NOT pip-installable.\n'
                '  Fix: recreate the venv with --system-site-packages so\n'
                '  it can fall back to the apt-installed SoapySDR:\n'
                '    deactivate\n'
                '    rm -rf "$VIRTUAL_ENV"\n'
                '    python3 -m venv --system-site-packages "$VIRTUAL_ENV"\n'
                '    "$VIRTUAL_ENV/bin/pip" install -r requirements.txt\n'
                % (e, sys.executable))
        else:
            sys.stderr.write(
                'soapy_rx: SoapySDR Python bindings not importable (%s).\n'
                '  python=%s\n'
                '  Fix: sudo apt install python3-soapysdr\n'
                '  If python=%s is a non-distro interpreter (e.g. /opt/python3.14\n'
                '  or pyenv), the apt SoapySDR is built against the SYSTEM\n'
                '  python and your interpreter cannot see it.  Either switch\n'
                '  to /usr/bin/python3 or build SoapySDR Python bindings from\n'
                '  source against your interpreter.\n'
                % (e, sys.executable, sys.executable))
        return 2
    if DBG:
        sys.stderr.write('soapy_rx: SoapySDR=%s numpy=%s\n'
                         % (getattr(SoapySDR, '__version__', '?'), np.__version__))

    # Pass the args as a "key=val,key=val" string, NOT a dict. On
    # Ubuntu 26.04 (python3-soapysdr 0.8.1-7build1, built with SWIG 4.4
    # against Python 3.14), Device({'driver':'<x>'}) silently dispatches
    # to a different C++ overload — Device_make({}) returns the empty
    # tuple () and Device_make({'driver':'hackrf'}) throws "no match"
    # while Device('driver=hackrf') opens the same device cleanly.
    # Verified locally with the actual 26.04 binding loaded into Python
    # 3.14.4 plus a real HackRF: dict broken, string works. Older stacks
    # (e.g. 24.04 / Python 3.12 / SWIG 4.1) accept both.
    args_str = 'driver=%s' % a.driver
    try:
        dev = SoapySDR.Device(args_str)
    except Exception as e:
        sys.stderr.write('soapy_rx: SoapySDR.Device(%r) failed: %s  '
                         '(device absent / busy / driver module not installed; '
                         'try: SoapySDRUtil --find  and  hackrf_info / rtl_test '
                         'as the same user)\n' % (args_str, e))
        return 3
    if DBG:
        try:
            info = dev.getHardwareInfo()
            sys.stderr.write('soapy_rx: device hw=%s driver=%s\n'
                             % (dict(info), a.driver))
        except Exception:
            pass
    dev.setSampleRate(SOAPY_SDR_RX, 0, a.s)
    dev.setFrequency(SOAPY_SDR_RX, 0, a.f)
    if DBG:
        try:
            sys.stderr.write('soapy_rx: actual rate=%g freq=%g\n'
                             % (dev.getSampleRate(SOAPY_SDR_RX, 0),
                                dev.getFrequency(SOAPY_SDR_RX, 0)))
        except Exception:
            pass
    if a.b > 0:
        try: dev.setBandwidth(SOAPY_SDR_RX, 0, a.b)
        except Exception: pass
    if a.g is None:
        try: dev.setGainMode(SOAPY_SDR_RX, 0, True)    # AGC where supported
        except Exception: pass
    else:
        try: dev.setGainMode(SOAPY_SDR_RX, 0, False)   # manual: AGC must NOT override it
        except Exception: pass
        try: dev.setGain(SOAPY_SDR_RX, 0, a.g)
        except Exception: pass

    st = dev.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CS16)
    try:
        mtu = int(dev.getStreamMTU(st))
    except Exception:
        mtu = 0
    N = max(mtu, 1 << 16)                  # samples per readStream call
    chunk_bytes = N * 4                    # CS16 = 2 * int16 per complex sample
    maxchunks = max(8, int(a.queue_mb * 1e6 / chunk_bytes))
    q = queue.Queue(maxsize=maxchunks)
    stop = threading.Event()
    stats = {'overflow': 0, 'dropped': 0, 'samples': 0}

    dev.activateStream(st)

    def reader():
        """Service the device continuously — this thread must never block long."""
        buf = np.empty(2 * N, np.int16)
        while not stop.is_set():
            sr = dev.readStream(st, [buf], N, timeoutUs=500000)
            n = sr.ret
            if n > 0:
                stats['samples'] += n
                data = buf[:2 * n].tobytes()   # copy out before next read reuses buf
                try:
                    q.put(data, timeout=2.0)
                except queue.Full:
                    stats['dropped'] += 1       # downstream stalled > queue depth
            elif n == SOAPY_SDR_OVERFLOW:
                stats['overflow'] += 1          # device dropped (host too slow) — visible
            # else: timeout / transient (n < 0) → just retry

    th = threading.Thread(target=reader, daemon=True)
    th.start()

    out = sys.stdout.buffer
    last = time.time()
    try:
        while True:
            try:
                data = q.get(timeout=0.5)
            except queue.Empty:
                if stop.is_set():
                    break
                continue
            out.write(data)
            now = time.time()
            if now - last > 5.0:
                last = now
                if stats['overflow'] or stats['dropped']:
                    sys.stderr.write('soapy_rx: overflow=%d dropped=%d samples=%d\n'
                                     % (stats['overflow'], stats['dropped'], stats['samples']))
                    sys.stderr.flush()
    except (BrokenPipeError, KeyboardInterrupt):
        pass
    finally:
        stop.set()
        th.join(timeout=1.0)
        try:
            dev.deactivateStream(st); dev.closeStream(st)
        except Exception:
            pass
        if stats['overflow'] or stats['dropped']:
            sys.stderr.write('soapy_rx: FINAL overflow=%d dropped=%d samples=%d\n'
                             % (stats['overflow'], stats['dropped'], stats['samples']))
    return 0


if __name__ == '__main__':
    sys.exit(main())
