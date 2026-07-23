"""
detector.py — Live LoRa detection from wideband SDR input.

Pipeline:
  1. Welch PSD → find energy peaks
  2. Per peak: extract 1Msps IQ (covers all Meshtastic BWs up to 500kHz)
  3. Multi-lag Schmidl-Cox → preamble fires at lag = 2^SF/BW × 1e6
     Unambiguous lags → SF+BW direct.  Ambiguous lags (2048/4096/8192)
     → resolved via dechirp quality comparison.
  4. CFO correction from dechirp peak bin

Why Schmidl-Cox: fires ON the preamble (16 repeated same-bin chirps give
near-perfect autocorrelation at lag = symbol duration).  Recording always
starts before packet data → decoder always sees full preamble → far fewer
NOT ENOUGH DATA / HEADER CHECKSUM failures than CNN-based detection.

Usage:
  bladeRF-cli -e 'set frequency rx 915000000; set samplerate rx 40000000; \
    set bandwidth rx 28000000; set gain rx 40; \
    rx config file=/dev/stdout format=bin n=0; rx start; rx wait' \
  | python detector.py -r 40000000 -b 28000000 -c 915 -t sc16 -d 1
"""

import sys, os
# ---- Bound glibc malloc arenas BEFORE anything allocates (OOM fix, codex
# review 2026-07-23) ----
# glibc spawns extra malloc arenas under allocation contention (up to ~8×CPU),
# each retaining fragmented free regions after numpy arrays are freed.  With the
# detector's many threads (gate, crop, save, decode-mgr, iq-reader, 2 cvt) doing
# heavy alloc churn, that fragmentation crept RSS up ~200 MB/min at 20 Msps with
# EVERY logical queue bounded and empty — the residual OOM after the joint memory
# plan.  MALLOC_ARENA_MAX=2 flattened it (measured: 20 Msps + decode went from
# OOM-at-13s to a stable MemAvailable plateau).  glibc reads this only at process
# start, so os.environ alone is too late for THIS process — re-exec once (the
# env-presence guard prevents a loop and honours an operator override).  Must
# precede numpy import, threads, arg parsing, and any stdin read; execve keeps
# the pid/pgid and the inherited stdin/stdout pipe fds intact.
if (__name__ == '__main__' and sys.platform.startswith('linux')
        and 'MALLOC_ARENA_MAX' not in os.environ
        and 'glibc.malloc.arena_max=' not in os.environ.get('GLIBC_TUNABLES', '')):
    os.environ['MALLOC_ARENA_MAX'] = '2'
    _oa = getattr(sys, 'orig_argv', None)
    os.execve(sys.executable,
              [sys.executable, *(_oa[1:] if _oa else sys.argv)], os.environ)
if __name__ == '__main__':
    sys.stderr.write("[MALLOC] arena_max=%s (2=bounded via re-exec; unset=off)\n"
                     % os.environ.get('MALLOC_ARENA_MAX', 'UNSET'))
    sys.stderr.flush()
# Pin BLAS/FFT pools to ONE thread — MUST precede the numpy import.
# Two reasons:
# 1. DEADLOCK (found live 2026-07-09, py-spy): scipy-OpenBLAS registers a
#    pthread_atfork handler that quiesces its worker pool.  This process both
#    runs GEMMs on background threads (polyphase crop feed) AND forks decode
#    workers via subprocess — when a fork lands while a GEMM is in flight, the
#    fork handler suspends the pool the GEMM is waiting on: the GEMM spins at
#    100% of one core forever, fork_exec never returns, the fork's before-
#    hooks keep concurrent.futures' _global_shutdown_lock held, and the main
#    loop freezes at the next executor submit.  Whole pipeline wedges,
#    intermittently (needs the fork↔GEMM race).  No pool → no fork handler →
#    the class is gone.
# 2. OVERSUBSCRIPTION: parallelism here is at the process/thread level
#    (detect workers, decode workers, save/crop workers) — a 24-thread BLAS
#    pool underneath multiplies threads for no gain on our small per-call
#    FFTs/GEMMs (same reasoning as soft_fft_demod_batch's single-thread FFT
#    note, and the A/B harness already pinned these for determinism).
# setdefault: an explicit user override in the environment still wins.
for _v in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
           'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '1')
import time, json, argparse, numpy as np
import threading, queue, io, subprocess, fcntl, collections, itertools

# Make the GIL hand off 5× more often.  Default 5ms means a Python-only
# thread (MainThread doing numpy bookkeeping, _save_worker thread building
# capture extents, manager threads dispatching jobs) can hold the GIL for
# a full 5 ms before yielding — and during that hold window the iq-reader
# thread can't acquire the GIL to drain its kernel pipe.  At 5 Msps wire
# speed the kernel pipe fills in ~1.6 ms (8MB pipe / 20MB/s = 400 µs even
# at the SDR sample width); a 5ms GIL stall is enough to back-pressure
# soapy_rx and overflow the HackRF's internal buffer, dropping samples.
# Live diagnostics caught iq-reader at 0 / 20 GIL-active snapshots during
# burst — it was being shut out by MainThread (40%) and recorder threads
# (60%).  1ms hand-off lets iq-reader interleave 5× more often.  Cost:
# slightly more context-switch overhead for pure-Python loops, negligible
# because most of master's time is in numpy/scipy that releases GIL on
# C-call boundaries (the switchinterval only governs voluntary yield in
# pure Python).
try:
    sys.setswitchinterval(0.001)
except (AttributeError, ValueError):
    pass
from config import BW_LIST
# FFT library is core-count adaptive — measured on actual hardware:
#   Pi 4 (4 cores, NEON FFTW):   pyfftw T=1 wins 4.40 ms vs scipy w=1 4.76 ms
#                                 (scipy w=-1 is 6.58 ms — thread fan-out hurts)
#   Laptop (24 cores, Intel):    scipy w=-1 wins ~3.4 ms vs pyfftw T=2 ~5.0 ms
#                                 (per-call pyfftw dispatch costs more than
#                                  scipy's pocketfft saves at this batch size)
try:
    from scipy.fft import next_fast_len as _next_fast_len
except ImportError:
    def _next_fast_len(n):
        return n

_FFT_LIB_CHOICE = 'pyfftw' if (os.cpu_count() or 1) <= 4 else 'scipy'

if _FFT_LIB_CHOICE == 'pyfftw':
    try:
        import pyfftw
        pyfftw.config.NUM_THREADS = 1   # low-core: extra threads net-negative
        pyfftw.interfaces.cache.enable()
        pyfftw.interfaces.cache.set_keepalive_time(300.0)
        from pyfftw.interfaces.scipy_fft import (fft as _pyfftw_fft,
                                                 ifft as _pyfftw_ifft)
        _FFT_WORKERS = 1
        # MEASURE-quality plans (busy-wall profiling 2026-07-19): the default
        # ESTIMATE plans execute 2.4x slower than MEASURE at the per-peak hot
        # sizes (182 vs 74.5 ms @1M measured on-Pi), and the wisdom persist/
        # reload below makes the one-time planning cost (~24 s/size) a
        # first-run-only expense — every later process reuses the plan via
        # wisdom.  LORA_FFTW_MEASURE=0 restores ESTIMATE.
        # SIZE-CAPPED: MEASURE planning cost grows superlinearly — at the
        # 16M+-point transforms a 20 Msps window produces, planning takes
        # minutes per size and crawled a g32 replay to ~45 s/window.  The
        # measured win came from the 262k-1M-pt per-peak sizes, so MEASURE
        # only at or below the cap; ESTIMATE (old behavior) above it.
        if os.environ.get('LORA_FFTW_MEASURE', '1') != '0':
            try:
                _MEASURE_MAX_N = int(os.environ.get('LORA_FFTW_MEASURE_MAX',
                                                    '4194304'))
            except ValueError:
                _MEASURE_MAX_N = 4194304

            def _fft(a, axis=-1, workers=None):
                if a.shape[axis] <= _MEASURE_MAX_N:
                    return _pyfftw_fft(a, axis=axis,
                                       planner_effort='FFTW_MEASURE')
                return _pyfftw_fft(a, axis=axis)

            def _ifft(a, axis=-1, workers=None):
                if a.shape[axis] <= _MEASURE_MAX_N:
                    return _pyfftw_ifft(a, axis=axis,
                                        planner_effort='FFTW_MEASURE')
                return _pyfftw_ifft(a, axis=axis)
        else:
            _fft, _ifft = _pyfftw_fft, _pyfftw_ifft

        # Persist FFTW wisdom across runs.  Same file decoder.py uses so the
        # gate and decode workers share a single converging plan set.  Atomic
        # rename makes concurrent writes safe (last writer wins).
        import atexit as _atexit
        import pickle as _pickle
        import tempfile as _tempfile
        _WISDOM_PATH = os.path.expanduser('~/.lora-wbd-fftw-wisdom.pkl')

        def _load_wisdom():
            try:
                with open(_WISDOM_PATH, 'rb') as f:
                    pyfftw.import_wisdom(_pickle.load(f))
            except (FileNotFoundError, EOFError, _pickle.UnpicklingError, OSError):
                pass

        def _save_wisdom():
            try:
                try:
                    with open(_WISDOM_PATH, 'rb') as f:
                        pyfftw.import_wisdom(_pickle.load(f))
                except (FileNotFoundError, EOFError, _pickle.UnpicklingError, OSError):
                    pass
                d = os.path.dirname(_WISDOM_PATH) or '.'
                with _tempfile.NamedTemporaryFile(
                        mode='wb', dir=d, delete=False,
                        prefix='.lora-wbd-wisdom-') as f:
                    _pickle.dump(pyfftw.export_wisdom(), f)
                    _tmp = f.name
                os.replace(_tmp, _WISDOM_PATH)
            except Exception:
                pass

        _load_wisdom()
        _atexit.register(_save_wisdom)
    except ImportError:
        _FFT_LIB_CHOICE = 'scipy'

if _FFT_LIB_CHOICE == 'scipy':
    try:
        from scipy.fft import fft as _fft, ifft as _ifft
    except ImportError:
        from numpy.fft import fft as _fft, ifft as _ifft
    _FFT_WORKERS = -1   # high-core hosts: scipy.fft fans out across cores well

MAX_ENERGY_PEAKS = 10
# Spur rejection: drop peaks more than N dB below the strongest peak.  Only
# useful when the ADC saturates and generates harmonic spurs; otherwise it
# silently kills weak-but-real LoRa packets that happen to share a window
# with a stronger one (Meshtastic relays from a nearby node frequently land
# in the same window as packets from distant nodes).  Default is effectively
# disabled; the main loop bumps it to ≈30dB only when saturation is detected.
SPUR_REJECT_DB = 200
DECHIRP_MIN_DB = 15.0
# Channelizer dechirp matched-filter detection threshold.  Despread quality is ~9 dB on
# noise and ~14.5-16.5 dB on a weak packet the energy gate and Schmidl-Cox both miss —
# so 12 dB cleanly separates signal from noise while detecting well below where SC and
# the energy gate give out.
DECHIRP_MF_MIN_DB = 12.0

# NOTE (2026-07-20, busy-wall lever 2 investigation): a "high-SNR resolver fast
# path" was prototyped here — skip the dechirp candidate ladder when the best
# matched-rate SC pair leads its same-lag/alias-chain siblings by a margin.  It
# was REMOVED after measurement: matched-rate SC is a PRESENCE detector, not an
# identity classifier (a strong chirp scores ~1.0 in EVERY sibling's matched
# crop, across DIFFERENT slopes too), so on real signals the top siblings tie at
# ~1.0 — the fast path fired 0/460 peak-lags on the busyslice bed and 88% of
# peak-lags had best−2nd < 0.15.  Forcing it to fire matches busyslice by luck
# but would mis-pick cross-slope/same-slope aliases (which pass the bwq confirm)
# elsewhere.  The dechirp ladder is doing irreducible identity work.  See the
# session note; the real per-peak lever is resolver crop reuse from the already-
# computed _matched_nb (behaviour-conditional, needs its own floors gauntlet).

# Forced-dechirp (channelizer) per-channel cooldown for the cross-batch capture dedup.
# The forced dechirp re-fires every window a packet spans (and on its payload), so the
# same packet would be captured/decoded many times.  Anchored to the channel carrier
# (fix #1), these repeats collapse to one capture per channel per this window.  1.5 s
# covers a packet's airtime + relay hops while staying well under typical inter-packet
# gaps (e.g. a 15 s beacon), so distinct transmissions are preserved.
DECHIRP_DEDUP_WIN_S = 1.5

# Max channels to run the dechirp matched-filter on at once (the channelizer's learned
# channels).  Each is a forced per-window dechirp scan (cheap after the per-channel-
# compute fix, fix #3); the cap bounds worst-case work if the channelizer learns many
# channels.  This is the user-facing "max dechirp channels" (default 10); app.py passes
# the learned set.  Channels beyond this are still energy-gated, just not dechirp-boosted.
# 16 (was 10): the web's SEEDED channels (regional defaults / user-pinned, up to 6) ride
# alongside the learned set (up to 10) — the old cap silently trimmed the learned tail
# whenever seeds were present.
DECHIRP_MAX_CHANS = 16


def _parse_chan_tokens(s):
    """Parse a --channels token string ('center_MHz:bw_kHz[:sf],...') into
    [(center_hz, bw_hz, sf)].  Shared by the startup arg and the live ctl-file
    apply path so both accept exactly the same format."""
    out = []
    for tok in (s or '').split(','):
        tok = tok.strip()
        if not tok:
            continue
        try:
            _cp = tok.split(':')   # center_MHz:bw_kHz[:sf]
            _csf = int(_cp[2]) if len(_cp) > 2 and _cp[2] else 7
            out.append((float(_cp[0]) * 1e6, float(_cp[1]) * 1e3, _csf))
        except (ValueError, IndexError):
            pass
    return out


# ---- Patience gate (channel-acquisition range, phase C) -------------------
# Acquire UNKNOWN channels BELOW the energy gate's floor by trading time for
# SNR: a too-weak beacon still deposits max-hold PSD energy at the same bin,
# periodically, forever.  Count per-bin floor-relative exceedances over a
# long horizon; a bin whose hits pile up (but whose duty cycle is far from
# 100% — that's a CW birdie) gets PROMOTED: a forced peak is injected at its
# carrier every window, which hands it to the same matched-SC + dechirp
# pipeline the channelizer uses (that pipeline identifies SF/BW from the
# signal itself, so promotion needs only a carrier).  Wrong promotions cost
# bounded CPU and expire; they cannot mask anything (additive, like the
# channelizer).  All knobs env-tunable for calibration.
# DEFAULT ON (validated 2026-07-09): end-to-end synthetic A/B on a FULL
# 24-beacon file — an unknown carrier at 0 dB in-band SNR that the energy
# gate NEVER detects (baseline 0/24) is promoted within ~30 s and every
# subsequent beacon detects (12 dets, sc=1.00, bwq 24-26 dB, correct
# SF11/125k).  Pure-noise guard: zero promotions at production defaults
# (margin calibrated at 6 dB).  An earlier "experimental-off" verdict was
# based on a TRUNCATED test file (writer killed mid-generation — only 8 of
# 24 beacons existed) plus two integration bugs since fixed: promotions now
# join dechirp_chans with a rotating (sf,bw) trial ladder, and the detect
# pool receives channels per-task instead of frozen spawn-time params.
PATIENCE_ON = os.environ.get('LORA_PATIENCE', '1') != '0'
# 6.0 dB CALIBRATED on pure noise (2026-07-09): the max-hold PSD's upper tail
# is fat — at 3 dB noise bins exceed ~1%/window (≈12 hits/horizon, ABOVE the
# promotion bar: 1634/4096 bins would false-promote), at 5 dB a few bins still
# cross, at 6 dB zero exceedances in 100 windows x 4096 bins.  The gain over
# the energy gate is NOT a lower instantaneous bar — it's that a single-bin,
# occasional poke (which the gate's >=3-bin width floor and per-window logic
# deliberately ignore) accumulates here across many beacon periods.
PATIENCE_MARGIN_DB = float(os.environ.get('LORA_PATIENCE_MARGIN_DB', '6.0'))
PATIENCE_MIN_HITS = float(os.environ.get('LORA_PATIENCE_HITS', '8'))
PATIENCE_HORIZON_WIN = int(os.environ.get('LORA_PATIENCE_HORIZON', '1200'))
# 0.35 (was 0.6): a carrier worth patient acquisition is a SPARSE beacon —
# the validated far-node case measures ~11% duty; even a 9 s slow frame at a
# 60 s interval is 15%.  Strong LOCAL slow-family traffic (9 s frames every
# ~18 s ≈ 50-57% duty) slipped under 0.6 and its spur field (beyond the
# ±1 MHz masks) got promoted — g31_sf12 grew 4 junk promotions at 53-57%
# duty whose trial jobs destabilized that entry's knife-edge decodes.
# 0.30 (was 0.35, was 0.6): two60 (dense band) grew four junk promotions —
# near-DC skirt + intermod artifacts — ALL at exactly 33% duty, sneaking
# under 0.35; their trial jobs perturbed the decode ordering of knife-edge
# decodes (nondeterministically LOSING MT-9 in some runs and GAINING seq
# 3+5 in others).  The legitimate acquisition case measures 11% and the
# theoretical ceiling (9 s frame @ 60 s interval) is 15% — 0.30 keeps 2x
# headroom while excluding the artifact band.
PATIENCE_DUTY_MAX = float(os.environ.get('LORA_PATIENCE_DUTY_MAX', '0.30'))
PATIENCE_CAP = 4          # max simultaneously-promoted carriers
PATIENCE_TTL_WIN = int(os.environ.get('LORA_PATIENCE_TTL', '1200'))
PATIENCE_SCAN_EVERY = 16  # promotion scan cadence (windows)
# (sf, bw) trial ladder for promoted carriers.  The despread matched filter
# (CHAN-DECHIRP, the channelizer's sensitive path — measured 8/8 beacons at
# 0 dB in-band where the plain SC chain floors at ~+2 dB) needs the channel's
# SF/BW, which patience does not know.  Each promoted carrier trials TWO
# ladder combos per window (rotating), so the full ladder sweeps in 6 windows
# (~3 s) — a beacon spanning 2-3 windows meets its true combo within a few
# transmissions.  Ordered most-likely-first for long-range use.
PATIENCE_TRIALS = [(11, 125e3), (12, 125e3), (10, 125e3), (9, 125e3),
                   (12, 62.5e3), (11, 62.5e3), (10, 62.5e3), (11, 250e3),
                   (10, 250e3), (12, 250e3), (9, 250e3), (8, 125e3)]
PATIENCE_TRIALS_PER_WIN = 2


def patience_trial_family(sf, bw):
    """Same-slope in-ladder rung family of a fired patience-trial rung.

    A strong LoRa burst clears the despread bar under SAME-SLOPE aliases
    (slope bw^2/2^sf; partners (sf±2, bw×2^±1)) by +10-18 dB — sometimes
    BEATING the true combo (item 8a lesson 2) — so a fired trial's (sf,bw)
    label names a FAMILY, not an identity.  Everything that must contain or
    identify the TRUE signal (capture crop width, tail length, decode-job
    labels) sizes for the whole family.  Bounded: the 12-rung ladder holds
    families of at most 2 (e.g. {(10,62.5k),(12,125k)}); a rung with no
    in-ladder partner returns just itself."""
    _s0 = (float(bw) ** 2) / float(1 << int(sf))
    _fam = [(s, b) for (s, b) in PATIENCE_TRIALS
            if abs((b * b) / float(1 << s) - _s0) <= _s0 * 0.02]
    return _fam or [(int(sf), float(bw))]


def patience_cap_params(d):
    """(crop_bw_hz, sym_time_s) a capture of detection `d` must be sized for.

    Non-trial detections: their own label (byte-identical to the historic
    per-label sizing).  patience_trial detections: the same-slope family
    max — an alias-BW crop around a wider true signal contains no usable
    preamble (measured: a 62.5k crop of a -6 dB SF12/125k carrier gave the
    decoder NO PREAMBLE 5/5, so a promotion could never confirm and burned
    the forced dechirp until TTL/futility), and an alias-sized tail budget
    (half the true symbol time for the 2-step partner) truncates the true
    payload → CRC fail even when the preamble survives."""
    _sf, _bw = int(d['sf']), float(d['bw'])
    if not d.get('patience_trial'):
        return _bw, float(1 << _sf) / _bw
    _fam = patience_trial_family(_sf, _bw)
    return (max(b for _, b in _fam),
            max(float(1 << s) / b for s, b in _fam))

# PRESCREEN — outcome-learned persistent-junk suppression (2026-07-14,
# measured low-core ceiling item 3).  On a junk-heavy host (Pi at 20 Msps,
# high gain) ambient spurs/images that survive the width floor fire the FULL
# detect path nearly every window (chunk-FFT build + per-peak extract + SC),
# holding detect= at 1100-2700 ms/window with det=0.  Measured separability
# (junk20 + sdr_dcspur + two60): a cheap short-slice SC score does NOT
# separate junk from real weak LoRa (junk SC p99 0.41 vs real floors at
# 0.41-0.60) — but the FULL-PATH OUTCOME does, perfectly (0 detections in
# ~3000 junk peaks).  So the prescreen learns junk from outcomes: a bin whose
# gate peak fires at >=90% duty for MINUTES while the full path yields ZERO
# detections anywhere near it is a spur, and its peaks are suppressed BEFORE
# detect_preamble (an all-junk window then takes the lazy quiet path — no
# window materialization, no 160 MB chunk FFT).  DEFAULT-PERMISSIVE, layered
# unlearn/breakthrough guards (sensitivity is sacred — floors must not move):
#   - detections (ANY path, incl. slow-pass/pooled) purge entries within
#     ±15 bins immediately and veto learning within ±1 MHz for ~5 min —
#     a real repeating beacon DETECTS, so it can never stay learned;
#     beacon duty (airtime/period, <=65% even for 9 s slow frames) also
#     sits under the 90% duty bar;
#   - probe windows: every PRESCREEN_PROBE_EVERY-th window ALL suppressed
#     peaks run the full path anyway (aligned, so 15/16 junk windows stay
#     quiet); a probe detection unlearns instantly;
#   - breakthrough: a peak markedly stronger (>= +margin dB) or much wider
#     than the learned spur is processed, not suppressed — a real signal
#     rising over a spur bin is seen the same window it appears;
#   - fast unlearn: a junk bin that stops firing for PRESCREEN_GONE_WIN
#     windows is dropped (no stale entries waiting to eat a future signal);
#   - never learns: near recent detections, inside channelizer learned
#     channels, or on patience-promoted carriers (their forced peaks are
#     appended AFTER this hook and are never touched by it).
# LORA_PRESCREEN=0 kill switch: restores current behavior verbatim.
# LORA_PRESCREEN_VERIFY=1: learning + would-suppress bookkeeping run but
# NOTHING is suppressed; every real detection is checked against recently
# would-suppressed bins and any match increments the DEAFNESS counter
# (printed in the end-of-run summary — must be 0 across the corpus).
PRESCREEN_ON = os.environ.get('LORA_PRESCREEN', '1') != '0'
PRESCREEN_VERIFY = os.environ.get('LORA_PRESCREEN_VERIFY') == '1'
# decay horizon (windows) of the duty statistic; cold-start learn time is
# ~1.6x this (seen must reach 80% of horizon), ~2.3x for a spur appearing
# later — minutes at the production 0.5 s hop, so short captures and the
# sensitivity battery never learn anything at all.
def _psenv(name, dflt, cast, lo=None):
    # Per-knob parse with fallback (pattern: RAM-cap _envf): a malformed or
    # out-of-range value warns + uses the default instead of crashing the
    # detector at import (int('x')) or later (PROBE=0 -> ZeroDivisionError).
    _v = os.environ.get(name)
    if not _v:
        return dflt
    try:
        _x = cast(_v)
        if lo is not None and _x < lo:
            raise ValueError('below minimum %r' % lo)
        return _x
    except ValueError:
        print('[GATE] ignoring malformed %s=%r (using %r)' % (name, _v, dflt),
              file=sys.stderr, flush=True)
        return dflt
PRESCREEN_LEARN_WIN = _psenv('LORA_PRESCREEN_LEARN_WIN', 120, int, lo=2)
PRESCREEN_DUTY = _psenv('LORA_PRESCREEN_DUTY', 0.90, float, lo=0.05)
PRESCREEN_PROBE_EVERY = _psenv('LORA_PRESCREEN_PROBE', 16, int, lo=1)
PRESCREEN_MARGIN_DB = _psenv('LORA_PRESCREEN_MARGIN_DB', 6.0, float)
PRESCREEN_GONE_WIN = _psenv('LORA_PRESCREEN_GONE_WIN', 20, int, lo=1)
PRESCREEN_CAP = 64        # max simultaneously-learned junk bins

# PATIENCE FUTILITY RETIRE (measured low-core ceiling item 4, 2026-07-16):
# the HackRF fs/4 birdie (935.0000 MHz at -r 20M / -c 915 — a 1-bin spur
# FLICKERING across windows) sails under every promotion veto (duty ~28% <
# the 0.30 CW cap, isolated, clean observation) and then burns a forced
# full-window dechirp EVERY window until TTL=1200 (~2.6 s/window on a Pi,
# ~10 min per cycle) confirming NOTHING.  Physics: a promotion that
# repeatedly HAS energy at its bin yet never yields a decode confirm is a
# birdie/spur by definition — a real sub-gate carrier confirms via the
# forced dechirp within its first energetic packet windows (the CHAN-DECHIRP
# despread floor is ~-12 dB in-band, 14-16 dB below the gate).  So count
# FUTILE trials: windows where the forced dechirp ran AND the promoted bin
# showed a max-hold exceedance (the same signal patience accumulated hits
# on).  Energy-ABSENT windows do not count — a 30 s-interval beacon, quiet
# ~59 of every 60 windows, is never retired between packets.  After M futile
# energy-present windows with ZERO confirmations: demote + a long
# re-promotion cooldown on that bin, cleared instantly by any published
# detection or decode confirm near it.  M=12 sizing: a beacon packet spans
# 2-3 windows and the 12-rung (sf,bw) trial ladder sweeps in 6 windows
# (2/window), so a real carrier meets its true combo within ~2 packets
# (~4-6 energetic windows) — 12 keeps 2x headroom.
# LORA_PATIENCE_FUTILE=0 kill switch: exact current (TTL-only) behavior.
PATIENCE_FUTILE_M = _psenv('LORA_PATIENCE_FUTILE', 12, int, lo=0)
PATIENCE_FUTILE_COOLDOWN = _psenv('LORA_PATIENCE_FUTILE_COOLDOWN',
                                  3 * PATIENCE_HORIZON_WIN, int, lo=1)

# IQ inversion: when set, conjugate the input stream (negate Q) so an IQ-inverted
# transmitter — LoRaWAN downlink, satellite / tinyGS configs with Invert-IQ on —
# decodes.  It must happen at the stream input, before detection: the detector's
# carrier/timing refinement uses a directional dechirp, so a post-extraction flip
# is too late.  Read once at import (env is fixed per process).  Default off
# (normal IQ).  Mutually exclusive with normal traffic.  GUI: Config → Advanced.
_IQ_INVERT = os.environ.get('LORA_IQ_INVERT') == '1'

# PREALLOC CONVERSION BUFFERS (2026-07-13, measured low-core ceiling item 1):
# the int->float32 hop conversion used to fresh-alloc ~80 MB per hop (file
# mode: astype + separate /= = two passes; live: np.empty per _convert_span)
# — the first-touch page faults alone cost ~23 ms/hop at 20 Msps.  Reusing
# preallocated buffers (single fused np.multiply into them) measured
# 23.4 -> 8.3 ms/hop file-mode.  OWNERSHIP RULE for reused buffers: a
# buffer returned by IQReader.read()/StreamBuffer.read() with owned=False
# is valid ONLY until the caller's next read() call on that reader; any
# caller that retains the data past its next read (e.g. tail reads kept
# as _carry_tail, or hop reads when hop_n >= win_n where buf becomes a
# view of the returned array) MUST pass owned=True to get a fresh
# allocation.  LORA_PREALLOC=0 restores the old always-fresh behavior.
_PREALLOC = os.environ.get('LORA_PREALLOC', '1') != '0'

# LAZY INT16 QUIET WINDOW (2026-07-14, measured low-core ceiling item 2):
# on windows where the energy gate finds NO candidate peaks (true-quiet band
# — the steady state the Pi-class hosts must sustain), the old pipeline
# still converted every hop to complex64 and slid a full complex64 window
# (~320 MB of memory traffic per hop at 20 Msps sc16) even though the only
# consumers of a quiet window are a handful of Welch PSD segments (50 gate
# + ~10 fresh short-sweep x 4096 samples), a 1/32-strided saturation probe,
# and the per-hop owned copy (_hop_own).  Instead: keep the sliding window
# as RAW int16/int8 (40 MB slide), convert ON DEMAND only the samples those
# fixed-path consumers actually touch (int->float32 conversion is EXACT —
# the scale is a power of two — so every lazily-converted segment is
# bit-identical to a slice of a fully-converted window), and materialize
# the full complex64 window ONCE (into a persistent prealloc buffer, item-1
# pattern) the moment the final per-window peak list is non-empty (real
# peaks, channelizer forced peaks, patience placeholders — they all land in
# _gate_peaks before detect) or a detect-pool slot needs the window.
# pre_hop / _hop_own / feed_tail / slow-scan assemblies stay complex64
# (owned copies, deep pipeline dependency) — only the SLIDING WINDOW and
# the hop/tail reads feeding it move to the raw domain.
#   LORA_LAZY=0        kill switch: restores the previous always-complex64
#                      pipeline byte-for-byte (all old code paths intact).
#   LORA_LAZY_VERIFY=1 paranoia mode: maintains a shadow complex64 window
#                      via the OLD pipeline's exact ops and asserts every
#                      lazily-computed product (segments, strided probe,
#                      Welch PSDs, materialization) bit-identical to it.
_LAZY = os.environ.get('LORA_LAZY', '1') != '0'
_LAZY_VERIFY = os.environ.get('LORA_LAZY_VERIFY') == '1'
# CROP-CENTER FIX (2026-07-21): exported captures were mis-centered
# +6..45 kHz because a burst-correlated CW spur (measured at carrier
# +57 kHz) hijacks the Welch-argmax carrier centroid in detect_preamble.
# Fix lives ENTIRELY in the recorder save path: after the narrowband
# slice exists, measure the packet's max-hold plateau ON THE SLICE
# (_nb_plateau_offset — the investigation's ±150 Hz-validated meter) and,
# if it says the crop is >5 kHz off DC, recenter the exported bytes and
# add the correction to the FILENAME frequency only.  The det dicts'
# freq_hz, DETECTED lines, dedup keys and channel learning are
# byte-identical by construction (nothing detection-side reads the
# correction).  Estimator abstains (None) whenever no bw-wide plateau
# qualifies → original center kept, fail-safe.
# LORA_CROP_CENTER_FIX=0 → exact legacy bytes + filenames.
_CROP_CENTER_FIX = os.environ.get('LORA_CROP_CENTER_FIX', '1') != '0'


# ============================================================================
# Instrumentation (VALIDATION-ONLY, additive — no behavior change).  Emits
# structured RUN_EVENT lines to stderr for the paced-replay validation harness
# (RF-time + monotonic clocks so pre-input-EOF decodes can be separated from
# post-input-EOF drain).  run_id comes from the runner via LORA_RUN_ID so it
# matches the runner-generated manifest; detector does NOT call git.
# ============================================================================
_RUN_ID = os.environ.get('LORA_RUN_ID') or ('pid%d-%d' % (os.getpid(),
                                                          time.monotonic_ns()))
_RUN_EVENT_LOCK = threading.Lock()

def _run_event(event, **kw):
    # Caller may pass mono_ns to log the EXACT stored timestamp (so the field
    # matches the value saved on the object, not a second clock read).
    if 'mono_ns' not in kw:
        kw['mono_ns'] = time.monotonic_ns()
    rec = {'run_event': event, 'run_id': _RUN_ID, **kw}
    # NO silent swallow: for a VALIDATION build a lost event (esp. input_eof)
    # would make an invalid run look scoreable.  One os.write() is atomic for
    # a <PIPE_BUF line, so events from different threads never interleave; a
    # write failure propagates and fails the run loudly rather than silently.
    line = json.dumps(rec, separators=(',', ':'), default=str) + '\n'
    with _RUN_EVENT_LOCK:
        os.write(2, line.encode('utf-8', 'replace'))


# ============================================================================
# Candidate-lifecycle audit (VALIDATION-ONLY, gated OFF by default).  Emits ONE
# batched cand_audit RUN_EVENT per non-empty gate window so raw-detection,
# admission, and preamble-detection recall can be scored SEPARATELY offline
# against the frozen occurrence oracle.  Energy candidates are Welch-PSD peaks
# (freq bin + width + power); SF/BW are NOT known until detect_preamble, so
# actual SF/BW/conf are attached only to 'detected' results (forced/patience
# candidates carry a HYPOTHESIS sf/bw, kept distinct).  When _CAND_AUDIT is
# False every hook is a single boolean skip — near-zero hot-path cost.
# ============================================================================
_CAND_AUDIT = os.environ.get('LORA_CAND_AUDIT', '0') == '1'
_CAND_AUDIT_SCHEMA = 3

# Per-peak cost-breakdown profiler (channelizer feasibility, 2026-07-23): times
# the three per-peak categories a channelizer would/wouldn't amortize — EXTRACT
# (narrowband crop+IFFT, amortizable) vs SC (schmidl-cox, per-signal) vs DECHIRP
# (per-signal).  Gated by LORA_PP_PROF (near-zero cost off).  Run --detect-workers 0
# so it accumulates in main and prints at the Done line.
_PP_PROF_ON = os.environ.get('LORA_PP_PROF', '0') == '1'
_PP_PROF = {'extract': 0.0, 'sc': 0.0, 'dechirp': 0.0,
            'n_extract': 0, 'n_sc': 0, 'n_dechirp': 0}
_EXT_CALLS = [0]         # exact-duplicate extraction meter (bit-identical memo potential)
_EXT_KEYS = set()        # distinct (F, center_bin, n_out) keys seen


def _pp_time(_cat, _ncat):
    def _deco(_fn):
        if not _PP_PROF_ON:
            return _fn

        def _wrap(*a, **k):
            import time as _t
            _t0 = _t.perf_counter()
            try:
                return _fn(*a, **k)
            finally:
                _PP_PROF[_cat] += _t.perf_counter() - _t0
                _PP_PROF[_ncat] += 1
        return _wrap
    return _deco

# PSD diagnostic (validation-only, gated OFF).  Dumps pre-rejection gate inputs
# for windows containing target rf-times (hop-0 controls + hop-1 packets).
# Ring-gate ablation (TEST-ONLY, default OFF, codex-agreed first causal probe):
# when set, the slow scan's ring-pressure deferral (is_live && ring>0.3) is NOT
# enforced.  Busy + rate gates are UNCHANGED so the intervention is isolated.
# The ring-pressure condition is still COMPUTED and logged (would_have) so the
# ablation's effect on coherent coverage is attributable.
_SLOW_NO_RING_DEFER = os.environ.get('LORA_SLOW_NO_RING_DEFER', '0') == '1'
# PROF/measurement kill-switch: hard-DISABLE slow-scan EXECUTION (never launches
# the bg scan), so its CPU/memory is fully removed for isolation runs. Distinct
# from drop-if-busy (which still fires when idle). Default off.
_NO_SLOW = os.environ.get('LORA_NO_SLOW', '0') == '1'
# Per-peak coverage-based candidate COALESCING (throughput, 2026-07-23): the
# overlapping mean/max-hold/short-sweep gate passes emit several energy peaks for
# ONE physical signal; the per-peak SF/BW resolution then runs redundantly on
# each (the measured detect-worker bottleneck at 20 Msps).  Process peaks
# strongest-first and skip any whose carrier already sits inside a CONFIRMED
# detection's freq±bw/2 — LOSSLESS (an adjacent channel outside the detected
# bandwidth is never covered, so it is still processed; the post-loop
# one-signal-one-detection dedup stays the final arbiter).  Kill-switch.
# DEFAULT OFF — measured LOSSY 2026-07-23: freq/BW coverage cannot distinguish a
# redundant over-split candidate from a DISTINCT same-band packet at a different
# TIME (a broadcast on the beacon's channel was suppressed and LOST), and the
# time isn't knowable until the per-peak work runs.  Freq-only pre-skip is
# inherently lossy; kept off pending a truly-lossless design.
_COALESCE = os.environ.get('LORA_COALESCE', '0') != '0'
_PSD_DIAG = os.environ.get('LORA_PSD_DIAG', '0') == '1'
def _parse_psd_diag_times(s):
    out = []
    for tok in (s or '').replace(' ', '').split(','):
        if tok:
            try:
                out.append(float(tok))
            except ValueError:
                pass
    return out
_psd_diag_times = _parse_psd_diag_times(os.environ.get('LORA_PSD_DIAG_TIMES', ''))
try:
    _psd_diag_freq_hz = float(os.environ.get('LORA_PSD_DIAG_FREQ_MHZ', '914.54')) * 1e6
except ValueError:
    _psd_diag_freq_hz = 914.54e6
try:
    _psd_diag_halfbins = max(4, int(os.environ.get('LORA_PSD_DIAG_HALFBINS', '160')))
except ValueError:
    _psd_diag_halfbins = 160


class _IdAlloc:
    """Central, thread-safe, per-run id allocator (Phase-1 lineage).  ONE lock,
    separate monotonic counters per id type (candidate/detection/work/scan).
    'producer' is a SEPARATE field on each event, never an id prefix (codex Q1).
    Pool workers and the slow background thread run concurrently, so every id
    is allocated through this lock — no worker-local counters."""
    __slots__ = ('_lock', '_c')

    def __init__(self):
        self._lock = threading.Lock()
        self._c = {'candidate': 0, 'detection': 0, 'work': 0, 'scan': 0}

    def next(self, kind):
        with self._lock:
            self._c[kind] += 1
            return self._c[kind]


class _WorkLedger:
    """Thread-safe submitted-vs-terminal work reconciliation.  A work item is
    'submitted' once (work_id); requeue/re-tier does NOT re-submit (same logical
    work) and is NOT terminal.  A submitted work item must terminate EXACTLY
    ONCE via note_terminal; a duplicate terminal or a terminal for an
    un-submitted work_id is an INVARIANT FAILURE (dup_terminal / unknown_terminal
    -> audit_valid=false).  Outstanding = submitted - terminated at clean drain."""
    __slots__ = ('_lock', '_sub', '_done', 'outcomes', 'dup_terminal',
                 'unknown_terminal')

    def __init__(self):
        self._lock = threading.Lock()
        self._sub = set()          # work_ids submitted
        self._done = {}            # work_id -> terminal outcome
        self.outcomes = {}
        self.dup_terminal = 0
        self.unknown_terminal = 0

    def note_submit(self, wid):
        with self._lock:
            self._sub.add(wid)

    def note_terminal(self, wid, outcome):
        """Returns True if this is the FIRST terminal for wid (emit the event);
        False on a duplicate (already terminalized) -> caller suppresses."""
        with self._lock:
            if wid in self._done:
                self.dup_terminal += 1
                return False
            if wid not in self._sub:
                self.unknown_terminal += 1
            self._done[wid] = outcome
            self.outcomes[outcome] = self.outcomes.get(outcome, 0) + 1
            return True

    def snapshot(self):
        with self._lock:
            outstanding = sorted(self._sub - set(self._done))
            return (len(self._sub), len(self._done), dict(self.outcomes),
                    self.dup_terminal, self.unknown_terminal, outstanding)


# Module-level singletons (only live when audit is on).
_LID = _IdAlloc() if _CAND_AUDIT else None
_WLED = _WorkLedger() if _CAND_AUDIT else None

class _CandWin:
    """One gate window's candidate lifecycle.  Created at window begin, populated
    during that window's candidate handling, ATTACHED to the _inflight item, and
    emitted at _commit_oldest when the (async, lagged) pool result for its seq
    returns — or emitted inline (pool_seq=None) when detection is synchronous.
    Strong refs to every tracked peak tuple are held until emit so Python cannot
    reuse an id() and corrupt attribution.  Reconciliation is by ordinal + exact
    bin signature, never float freq."""
    __slots__ = ('mgr', 'awid', 'pool_seq', 'tot_s', 'cen_hz', 'fres',
                 'center_bin', 'rate', 'dispatch_mono',
                 'raw', 'forced', 'patience', 'drops', 'admitted', '_ord_by_id')

    def __init__(self, mgr, awid, tot_s, cen_hz):
        self.mgr = mgr; self.awid = awid; self.pool_seq = None
        self.tot_s = int(tot_s); self.cen_hz = float(cen_hz)
        self.fres = mgr.fres; self.center_bin = mgr.center_bin; self.rate = mgr.rate
        self.dispatch_mono = None
        self.raw = []            # (ord, peakobj, bin, width, pwr)  STRONG REF
        self.forced = []         # (ord, peakobj, bin, width, pwr, hyp_sf, hyp_bw)
        self.patience = []       # (ord, peakobj, bin, width, pwr, hyp_sf, hyp_bw)
        self.drops = {}          # id(peak) -> reason
        self.admitted = []       # STRONG REFS to final dispatched peaks
        self._ord_by_id = {}     # id(peak) -> ordinal

    def _bin_hz(self, b):
        return self.cen_hz + (b - self.center_bin) * self.fres

    def snapshot_raw(self, peaks):
        for p in peaks:
            o = len(self.raw)
            self._ord_by_id[id(p)] = o
            self.raw.append((o, p, int(p[0]), int(p[1]), float(p[2])))

    def add(self, peak, tier, hyp_sf=None, hyp_bw=None):
        o = -(len(self.forced) + len(self.patience)) - 1
        self._ord_by_id[id(peak)] = o
        row = (o, peak, int(peak[0]), int(peak[1]), float(peak[2]), hyp_sf, hyp_bw)
        (self.forced if tier == 'forced' else self.patience).append(row)

    def drop(self, peaks, reason):
        for p in peaks:
            self.drops[id(p)] = reason

    def set_admitted(self, final_peaks, dispatch_mono, pool_seq=None):
        self.admitted = list(final_peaks)       # strong refs
        self.dispatch_mono = dispatch_mono
        self.pool_seq = pool_seq

    def emit(self, dets, buf_len, status, commit_mono):
        adm_ids = set(id(p) for p in self.admitted)
        # invariant canaries
        inv = {'dropped_unknown': 0, 'untracked_final': 0,
               'drop_admit_conflict': 0, 'drop_unknown_cand': 0, 'duplicate_id': 0}
        seen = set()
        for coll in (self.raw, self.forced, self.patience):
            for row in coll:
                pid = id(row[1])
                if pid in seen:
                    inv['duplicate_id'] += 1
                seen.add(pid)
        for pid in self.drops:
            if pid not in seen:
                inv['drop_unknown_cand'] += 1
            if pid in adm_ids:
                inv['drop_admit_conflict'] += 1
        for p in self.admitted:
            if id(p) not in self._ord_by_id:
                inv['untracked_final'] += 1

        def _disp(peakobj):
            pid = id(peakobj)
            if pid in self.drops:
                return self.drops[pid]
            if pid in adm_ids:
                return 'admitted'
            inv['dropped_unknown'] += 1
            return 'dropped_unknown'
        from collections import Counter as _C
        drop_reasons = dict(_C(self.drops.values()))

        # --- Phase-1 lineage: producer + window_id + per-candidate side-table id.
        # window_id == awid (assigned at new_window, BEFORE inline/pool choice).
        # producer is a SEPARATE field, derived from whether a pool_seq was set.
        producer = 'main_inline' if self.pool_seq is None else 'main_pool'
        window_id = self.awid
        # Assign a monotonic side-table candidate_id to each window candidate
        # WITHOUT modifying the (pickled, id()-tracked) peak tuple (codex g3).
        _cand_id = {}                    # ordinal -> candidate_id
        _bin_candids = {}                # bin -> [candidate_id, ...] (for annotation)
        for _coll in (self.raw, self.forced, self.patience):
            for _row in _coll:
                _o, _b0 = _row[0], _row[2]
                _cid = _LID.next('candidate') if _LID is not None else None
                _cand_id[_o] = _cid
                _bin_candids.setdefault(_b0, []).append(_cid)

        det_rows = []
        _link_counts = {}
        for d in (dets or ()):
            try:
                _b = int(round((float(d.get('freq_hz', 0.0)) - self.cen_hz)
                               / self.fres)) + self.center_bin
            except Exception:
                _b = None
            _off = d.get('preamble_t_s')
            # detection_id is assigned UNCONDITIONALLY (candidate-independent,
            # codex g1): every returned detection is emitted regardless of
            # whether candidate reconciliation resolves.
            _did = _LID.next('detection') if _LID is not None else None
            # HEURISTIC candidate ANNOTATION by bin signature (±8 bins) — a
            # DIAGNOSTIC HINT, NOT a causal parent (codex A).  Never gates
            # emission.  Dedupe by candidate_id; record nearest bin delta + the
            # match count so the first run establishes the delta distribution.
            _match_ids, _near_delta = set(), None
            if _b is not None:
                for _bb, _ids in _bin_candids.items():
                    if abs(_bb - _b) <= 8:
                        _match_ids.update(_ids)
                        if _near_delta is None or abs(_bb - _b) < abs(_near_delta):
                            _near_delta = _bb - _b
            _nmatch = len(_match_ids)
            if _nmatch == 1:
                _link, _hint = 'heuristic_unique_8bin', next(iter(_match_ids))
            elif _nmatch > 1:
                _link, _hint = 'heuristic_ambiguous', None
            else:
                _link, _hint = 'unresolved', None
            _link_counts[_link] = _link_counts.get(_link, 0) + 1
            # stamp lineage onto the SHARED detection dict so it flows to the
            # recorder/submit (emit runs BEFORE recorder — placement-audit B).
            # candidate_hint_id is a HINT; parent_candidate_id stays null in
            # Phase-1 (exact linkage is Phase-2).
            if isinstance(d, dict):
                d['_lin_detection_id'] = _did
                d['_lin_producer'] = producer
                d['_lin_window_id'] = window_id
                d['_lin_candidate_hint_id'] = _hint
                d['_lin_candidate_bin_delta'] = _near_delta
                d['_lin_candidate_matches_within_8bins'] = _nmatch
                d['_lin_candidate_link_status'] = _link
            det_rows.append({'freq_hz': d.get('freq_hz'), 'bin': _b,
                             'sf': d.get('sf'), 'bw': d.get('bw'),
                             'detect_conf': d.get('detect_conf'),
                             'preamble_offset_s': _off,
                             'event_rf_s': (None if _off is None
                                            else self.tot_s / self.rate + float(_off)),
                             'patience_trial': bool(d.get('patience_trial')),
                             'forced_dechirp': bool(d.get('forced_dechirp')),
                             'detection_id': _did, 'producer': producer,
                             'window_id': window_id,
                             'candidate_hint_id': _hint,
                             'candidate_bin_delta': _near_delta,
                             'candidate_matches_within_8bins': _nmatch,
                             'candidate_link_status': _link})
        n_inv = sum(inv.values())
        _run_event('cand_audit',
                   audit_window_id=self.awid, window_id=window_id, producer=producer,
                   pool_seq=self.pool_seq, status=status,
                   window_start_sample=self.tot_s,
                   window_end_sample=self.tot_s + int(buf_len),
                   rf_start_s=self.tot_s / self.rate,
                   rf_end_s=(self.tot_s + int(buf_len)) / self.rate,
                   fres_hz=self.fres, center_hz=self.cen_hz, center_bin=self.center_bin,
                   dispatch_mono_ns=self.dispatch_mono, commit_mono_ns=commit_mono,
                   n_raw=len(self.raw), n_forced=len(self.forced),
                   n_patience=len(self.patience), n_admitted=len(self.admitted),
                   n_detected=len(det_rows), invariant_failures=n_inv, invariants=inv,
                   candidate_link_counts=_link_counts,
                   raw=[{'ord': o, 'bin': b, 'width': w, 'pwr': pw,
                         'freq_hz': self._bin_hz(b), 'disposition': _disp(pk),
                         'candidate_id': _cand_id.get(o)}
                        for (o, pk, b, w, pw) in self.raw],
                   forced=[{'ord': o, 'bin': b, 'width': w, 'pwr': pw,
                            'freq_hz': self._bin_hz(b), 'hyp_sf': hs, 'hyp_bw': hb,
                            'disposition': _disp(pk), 'candidate_id': _cand_id.get(o)}
                           for (o, pk, b, w, pw, hs, hb) in self.forced],
                   patience=[{'ord': o, 'bin': b, 'width': w, 'pwr': pw,
                              'freq_hz': self._bin_hz(b), 'hyp_sf': hs, 'hyp_bw': hb,
                              'disposition': _disp(pk), 'candidate_id': _cand_id.get(o)}
                             for (o, pk, b, w, pw, hs, hb) in self.patience],
                   drop_reasons=drop_reasons, detections=det_rows)
        self.mgr.note_emit(status, len(self.raw) + len(self.forced) + len(self.patience),
                           len(self.raw) == 0 and not self.forced and not self.patience,
                           n_inv, bool(self.pool_seq is None), len(det_rows), _link_counts)


class _CandAuditMgr:
    """Config + cumulative bookkeeping across all windows; creates per-window
    _CandWin objects and emits the final drained summary."""
    __slots__ = ('rate', 'fres', 'center_bin', '_awid', 'windows_begun',
                 'windows_emitted', 'candidates', 'empty_windows', 'inv_failures',
                 'inline_windows', 'pool_windows', 'status_counts', 'skips',
                 'detections', 'link_counts', 'slow_suppressed', 'slow_started',
                 'slow_consumed', 'slow_failed', 'slow_discarded', 'slow_hits',
                 'slow_nohit', 'slow_max_ring_frac')

    def __init__(self, rate, nfft_gate):
        self.rate = rate; self.fres = rate / nfft_gate; self.center_bin = nfft_gate // 2
        self._awid = 0; self.windows_begun = 0; self.windows_emitted = 0
        self.candidates = 0; self.empty_windows = 0; self.inv_failures = 0
        self.inline_windows = 0; self.pool_windows = 0
        self.status_counts = {}; self.skips = []
        self.detections = 0
        self.link_counts = {}
        # slow-scan lifecycle rollups (per-attempt events are separate)
        self.slow_suppressed = {'busy': 0, 'rate': 0, 'ring': 0, 'would_ring': 0}
        self.slow_started = 0; self.slow_consumed = 0; self.slow_failed = 0
        self.slow_discarded = 0; self.slow_hits = 0; self.slow_nohit = 0
        # max ring fraction observed at a fire-eligible slow-scan decision — shows
        # how close the ring got to the 0.3 defer threshold even when it didn't
        # cross (codex: "did not reproduce" evidence needs the near-miss margin).
        self.slow_max_ring_frac = 0.0

    def new_window(self, tot_s, cen_hz):
        self._awid += 1; self.windows_begun += 1
        return _CandWin(self, self._awid, tot_s, cen_hz)

    def note_emit(self, status, n_cand, is_empty, n_inv, is_inline,
                  n_det=0, link_counts=None):
        self.windows_emitted += 1; self.candidates += n_cand
        if is_empty:
            self.empty_windows += 1
        self.inv_failures += n_inv
        self.status_counts[status] = self.status_counts.get(status, 0) + 1
        if is_inline:
            self.inline_windows += 1
        else:
            self.pool_windows += 1
        self.detections += n_det
        if link_counts:
            for k, v in link_counts.items():
                self.link_counts[k] = self.link_counts.get(k, 0) + v

    def note_skip(self, start_sample, end_sample):
        self.skips.append([int(start_sample), int(end_sample)])

    def note_slow_suppressed(self, busy, rate, ring, would_ring=False):
        if busy: self.slow_suppressed['busy'] += 1
        if rate: self.slow_suppressed['rate'] += 1
        if ring: self.slow_suppressed['ring'] += 1
        if would_ring: self.slow_suppressed['would_ring'] += 1

    def note_slow_started(self):
        self.slow_started += 1

    def note_slow_failed(self):
        self.slow_failed += 1

    def note_slow_discarded(self, n):
        self.slow_discarded += int(n)

    def note_slow_result(self, n_hits):
        self.slow_consumed += 1
        if n_hits:
            self.slow_hits += 1
        else:
            self.slow_nohit += 1

    def summary(self, outstanding):
        # WINDOW-level rollup only (emitted at loop end, BEFORE the decoder
        # drain).  Work reconciliation is NOT here — it must wait for the drain
        # (see lineage_summary(), emitted post-drain), else in-flight work would
        # falsely read as outstanding.  windows_valid is window-scoped.
        windows_valid = (self.windows_begun == self.windows_emitted
                         and outstanding == 0 and self.inv_failures == 0)
        _run_event('cand_audit_summary', schema=_CAND_AUDIT_SCHEMA,
                   windows_begun=self.windows_begun, windows_emitted=self.windows_emitted,
                   outstanding_windows=outstanding, candidates=self.candidates,
                   detections=self.detections, empty_windows=self.empty_windows,
                   invariant_failures=self.inv_failures,
                   inline_windows=self.inline_windows, pool_windows=self.pool_windows,
                   status_counts=self.status_counts, skip_ranges=self.skips,
                   skip_samples=sum(b - a for a, b in self.skips),
                   candidate_link_counts=self.link_counts,
                   slow_suppressed=self.slow_suppressed, slow_started=self.slow_started,
                   slow_consumed=self.slow_consumed, slow_hits=self.slow_hits,
                   slow_nohit=self.slow_nohit, windows_valid=windows_valid,
                   slow_max_ring_frac=round(self.slow_max_ring_frac, 4))

    def lineage_summary(self, outstanding_windows):
        """Authoritative post-drain reconciliation (codex req 1 + ordering).
        Emitted AFTER the decoder queues/requeues/slow-results have drained, so
        outstanding work is real, not in-flight.  audit_valid gates on window
        validity AND exactly-once work termination AND no slow started/consumed
        mismatch."""
        w_sub = w_term = 0
        w_out = {}
        dup_t = unk_t = 0
        outstanding_work = []
        if _WLED is not None:
            (w_sub, w_term, w_out, dup_t, unk_t,
             outstanding_work) = _WLED.snapshot()
        windows_valid = (self.windows_begun == self.windows_emitted
                         and outstanding_windows == 0 and self.inv_failures == 0)
        # every STARTED slow scan must be accounted: consumed, failed, or
        # DISCARDED at shutdown (residual results audit-off would also discard).
        slow_consistent = (self.slow_started
                           == self.slow_consumed + self.slow_failed + self.slow_discarded)
        detection_lineage_valid = bool(
            windows_valid and not outstanding_work and dup_t == 0 and unk_t == 0
            and slow_consistent)
        _run_event('lineage_summary', schema=_CAND_AUDIT_SCHEMA,
                   windows_valid=windows_valid,
                   work_submitted=w_sub, work_terminal=w_term,
                   work_outstanding=len(outstanding_work),
                   outstanding_work_ids=outstanding_work[:64],
                   work_outcomes=w_out, dup_terminal=dup_t, unknown_terminal=unk_t,
                   slow_started=self.slow_started, slow_consumed=self.slow_consumed,
                   slow_failed=self.slow_failed, slow_discarded=self.slow_discarded,
                   slow_hits=self.slow_hits,
                   slow_nohit=self.slow_nohit, slow_consistent=slow_consistent,
                   candidate_lineage_complete=False,
                   detection_lineage_valid=detection_lineage_valid,
                   audit_valid=detection_lineage_valid)


class IQReader:
    def __init__(self, fp, fmt='sc16'):
        self.fp, self.sc16 = fp, (fmt == 'sc16')
        self.bps = 4 if self.sc16 else 2
        self._scale = np.float32(1.0 / (2048.0 if self.sc16 else 128.0))
        self._buf = None    # preallocated float32 target (see _PREALLOC note)
    def read(self, n, owned=False):
        raw = self.fp.read(n * self.bps)
        if len(raw) < n * self.bps: return None
        # Fused scale straight into a reused buffer + zero-copy complex
        # reinterpret — bit-identical to the old astype + /= two-pass form
        # (scale is an exact power of two) but one pass and no 80 MB fresh
        # alloc per hop (page faults dominated: measured 23.4 -> 8.3 ms per
        # 0.5 s hop at 20 Msps). Matters for file replay/regression runs.
        # owned=True callers retain the result past the next read() — give
        # them a private allocation (see _PREALLOC ownership rule above).
        s = np.frombuffer(raw, dtype=np.int16 if self.sc16 else np.int8)
        n_elem = n * 2
        if owned or not _PREALLOC:
            f = np.empty(n_elem, dtype=np.float32)
        else:
            if self._buf is None or len(self._buf) < n_elem:
                self._buf = np.empty(n_elem, dtype=np.float32)
            f = self._buf[:n_elem]
        np.multiply(s, self._scale, out=f, casting='unsafe')
        if _IQ_INVERT:
            f[1::2] *= -1.0
        return f.view(np.complex64)
    def read_raw(self, n):
        """Read n IQ samples as RAW interleaved int16/int8 pairs (2n elems,
        UNCONVERTED — the lazy-window path, see _LAZY).  Returns None at
        EOF.  Each call returns a fresh array (a zero-copy view of the just
        -read bytes), so the result is owned by the caller."""
        raw = self.fp.read(n * self.bps)
        if len(raw) < n * self.bps: return None
        return np.frombuffer(raw, dtype=np.int16 if self.sc16 else np.int8)


class StreamBuffer:
    """Background-threaded IQ reader with ring buffer.

    Keeps the pipe/stdin permanently drained so bladeRF never drops samples.
    Stores raw int16/int8 pairs in a pre-allocated ring — conversion to
    complex64 happens on read(), split across a small thread pool: numpy's
    astype releases the GIL, so two worker threads convert the two halves
    of the hop in parallel, straight from the ring slices into one output
    buffer (no separate copy pass).  Measured on a Pi 4 at 20 Msps this
    takes the on-critical-path read cost from ~380 ms (copy 97 + serial
    convert 286) to ~150 ms per 0.9 s hop; the i9 gains proportionally.
    (A converted-float32-ring variant was tried and REVERTED: it pushed
    3 passes of memory traffic onto the core-0-pinned reader thread and
    made the Pi ~15 % slower overall under detect-worker bus contention.)

    When the consumer can't keep up, the ring overwrites the oldest data.
    read() detects this and skips the read pointer forward, returning a
    `skipped` count so the caller can reset any stateful context.
    """

    def __init__(self, fp, fmt='sc16', rate=40_000_000, buf_seconds=3.0):
        self.sc16 = (fmt == 'sc16')
        self.bps = 4 if self.sc16 else 2
        self.rate = rate
        self._fp = fp
        self._scale = np.float32(1.0 / (2048.0 if self.sc16 else 128.0))
        from concurrent.futures import ThreadPoolExecutor
        self._cvt_pool = ThreadPoolExecutor(max_workers=2,
                                            thread_name_prefix='iq-cvt')

        # Low-RAM guard: the default ring is sized to bridge the
        # measured 5-15 s slow-preset overload bursts (catchup root
        # cause, 2026-07-07 — a 45 s ring produced 0 catchups vs 2-6
        # per leg at 6-16 s), which is multi-GB at wideband rates.
        # Hosts that cannot hold it get the largest ring that fits in
        # 40% of MemAvailable instead of an allocation failure.
        try:
            with open('/proc/meminfo') as _mi:
                _avail_kb = next(int(l.split()[1]) for l in _mi
                                 if l.startswith('MemAvailable'))
            _budget = _avail_kb * 1024 * 0.40
            _need = rate * buf_seconds * (4 if self.sc16 else 2)
            if _need > _budget:
                _fit = max(3.0, _budget / (rate * (4 if self.sc16
                                                   else 2)))
                print(f"[RING] buf_seconds {buf_seconds:.0f}s needs "
                      f"{_need/1e9:.1f} GB — scaling to {_fit:.0f}s "
                      f"to fit 40% of available RAM", file=sys.stderr)
                buf_seconds = _fit
        except Exception:
            pass
        ring_n = int(rate * buf_seconds)
        dtype = np.int16 if self.sc16 else np.int8
        self._ring = np.zeros(ring_n * 2, dtype=dtype)  # ×2 for I + Q
        self._ring_n = ring_n                             # capacity in IQ samples
        self._ring_elems = ring_n * 2                     # capacity in array elements

        self._wp = 0          # write position (absolute, in IQ samples)
        self._rp = 0          # read position (absolute, in IQ samples)
        self._lock = threading.Lock()
        self._eof = False
        # Instrumentation (validation-only): monotonic-ns marks of the stream
        # timebase, same clock the decode-emit records use, so a packet is
        # pre-input-EOF iff result_processed_mono_ns <= input_eof_mono_ns.
        # first_chunk_read_complete = when the first blocking chunk read
        # returned; input_eof = clean source EOF.  A reader ERROR is NOT eof:
        # it leaves input_eof None and emits input_terminated → run invalid.
        self._first_chunk_read_complete_mono_ns = None
        self._input_eof_mono_ns = None
        self._reader_error = None   # set to repr(exc) on a reader-thread failure
        self._total_drops = 0
        # Ingress accounting (validation-only): total BYTES pulled from the
        # source, counted BEFORE any truncation/partial-read handling so every
        # discarded byte stays observable (source_summary invariant, codex).
        self._bytes_read_total = 0
        # CONVERT-AHEAD (2026-07-05): live reads are 100% uniform hop-sized
        # since the deferred-tail work, so a dedicated thread converts the
        # NEXT hop while the main loop processes the current one — read()
        # becomes pure wait (the ~150 ms/hop conversion was the last live
        # stall source). The prepared buffer is keyed by (rp, n); any
        # mismatch (first read, size change, lap) falls back inline.
        self._ahead = None            # (pos, n, complex64 buf) when ready
        self._ahead_want = None       # (pos, n) the thread is working on
        self._ahead_cv = threading.Condition(self._lock)
        self._ahead_thread = None
        # Preallocated conversion targets (see _PREALLOC ownership rule at
        # module top).  Three DISJOINT buffers: one for read()'s inline
        # conversion, two alternating for the ahead worker.  Disjoint
        # because a cancelled ahead prepare can still be mid-write when
        # read() converts inline; two worker slots because the worker
        # prepares hop k+1 while the consumer still holds hop k.  SAFETY
        # ARGUMENT: every prepare is kicked from inside read(), i.e. after
        # the previously returned buffer's lifetime ended (consumer contract:
        # dead at its next read() call), and consecutive prepares alternate
        # slots, so no conversion ever writes a buffer the consumer may
        # still read.  owned=True reads bypass all of this (fresh alloc).
        self._cvt_inline = None
        self._ahead_bufs = [None, None]
        self._ahead_idx = 0
        # Reused raw-copy target for read_raw() (lazy-window path, _LAZY).
        # Same ownership rule as the float buffers: an owned=False read_raw
        # result is valid only until the caller's next read_raw().
        self._raw_buf = None

        # Read in ~20 ms chunks.  Smaller chunks reduce pipe backpressure at
        # high sample rates while keeping per-call numpy conversion overhead low.
        self._chunk_n = max(int(rate // 50), 64_000)
        self._chunk_bytes = self._chunk_n * self.bps

        # Increase kernel pipe buffer to 8 MB to absorb burst writes from
        # bladeRF-cli while the reader thread processes the previous chunk.
        try:
            F_SETPIPE_SZ = 1031
            fcntl.fcntl(fp.fileno(), F_SETPIPE_SZ, 8 * 1024 * 1024)
        except (OSError, AttributeError):
            pass  # not a pipe, or permission denied — ignore

        self._thread = threading.Thread(target=self._reader_loop, daemon=True,
                                        name='iq-reader')
        self._thread.start()

    def _reader_loop(self):
        """Reader thread — runs at full wire speed, never blocks on consumer.

        Self-prioritises at thread start: pins to a dedicated core so it is
        never preempted by master-side gate threads (MainThread / recorders /
        manager threads all share a small affinity), and requests SCHED_FIFO
        priority so the OS schedules it ahead of every other Python thread
        when its read() returns.

        Why this matters: live diagnostics (mpstat + py-spy GIL holder
        snapshots) showed iq-reader holding the GIL ~0% of burst windows
        while master threads held it ~100%.  Sample drops measured at the
        SDR (5.0 → 4.4 Msps during burst) were tracking iq-reader's GIL
        starvation, not actual decode load.  Master's `top` shows cores
        0-7 at 100% during burst (gate + first decode workers + detect
        pool overflow); without explicit pinning Linux had nowhere to
        schedule iq-reader except by preempting another thread on those
        same cores, and SCHED_OTHER doesn't preempt in time at 5Msps wire
        speed.  Pinning to its own otherwise-idle core + SCHED_FIFO makes
        iq-reader's `read() → ring_write` cycle deterministic and removes
        the back-pressure that was overflowing the kernel pipe to soapy_rx
        and dropping samples (MT packets at 200-400ms windows are bigger
        targets for the drop windows than ~100ms MC packets — measured as
        ~14% MT_unique loss in 4×4 vs subprocess baseline).
        """
        # Pin to a dedicated core, distinct from where master's other
        # gate threads tend to run.  Use core 0 — first physical core,
        # claimed exclusively for iq-reader.  Linux's CFS scheduler
        # naturally keeps the other gate threads on whichever cores they
        # were spawned on, so taking core 0 here removes iq-reader from
        # the OS-level run queue that MainThread / recorders / manager
        # threads compete for.
        try: os.sched_setaffinity(0, {0})
        except (AttributeError, OSError): pass
        # Try to raise to SCHED_FIFO real-time scheduling.  Needs
        # CAP_SYS_NICE; if denied (most non-root setups) fall back to
        # the default and rely on affinity isolation alone.
        try:
            param = os.sched_param(50)
            os.sched_setscheduler(0, os.SCHED_FIFO, param)
        except (AttributeError, OSError, PermissionError):
            pass

        while not self._eof:
            try:
                raw = self._fp.read(self._chunk_bytes)
                if not raw:
                    # Empty read = CLEAN EOF.
                    if self._input_eof_mono_ns is None:
                        _t_ns = time.monotonic_ns()
                        self._input_eof_mono_ns = _t_ns
                        _run_event('input_eof', mono_ns=_t_ns, reason='clean_eof')
                    self._eof = True
                    break
                # Count every byte read BEFORE truncation/partial handling so a
                # sub-sample remainder can never be silently unobserved.  Under
                # the SAME lock as source_snapshot() so the post-loop snapshot is
                # truly atomic even if the reader thread is not yet joined.
                with self._lock:
                    self._bytes_read_total += len(raw)
                if len(raw) < self.bps:
                    # A nonempty read shorter than ONE complex sample is
                    # MALFORMED/truncated input, NOT a clean EOF.  Record it as
                    # a reader error (→ main() exits nonzero) and emit
                    # input_terminated so the run is scored invalid.
                    self._reader_error = ('partial_sample got=%d < bps=%d'
                                          % (len(raw), self.bps))
                    _run_event('input_terminated', reason='partial_sample',
                               got=len(raw), bps=self.bps)
                    self._eof = True
                    break
                if self._first_chunk_read_complete_mono_ns is None:
                    _t_ns = time.monotonic_ns()
                    self._first_chunk_read_complete_mono_ns = _t_ns
                    _run_event('first_chunk_read_complete', mono_ns=_t_ns,
                               chunk_bytes=self._chunk_bytes)
                n_samp = len(raw) // self.bps

                if self.sc16:
                    data = np.frombuffer(raw[:n_samp * self.bps],
                                         dtype=np.int16)
                else:
                    data = np.frombuffer(raw[:n_samp * self.bps],
                                         dtype=np.int8)

                n_elem = n_samp * 2
                with self._lock:
                    start = (self._wp * 2) % self._ring_elems
                    end = start + n_elem
                    if end <= self._ring_elems:
                        self._ring[start:end] = data[:n_elem]
                    else:
                        split = self._ring_elems - start
                        self._ring[start:] = data[:split]
                        self._ring[:n_elem - split] = data[split:n_elem]
                    self._wp += n_samp
            except Exception as _re:
                # ANY reader-thread failure must end the stream as EOF: a
                # death without _eof leaves read() spinning forever on a
                # silently-frozen pipeline (UC audit) — as EOF the process
                # exits and the outage is at least visible upstream.
                print(f"[READER] reader thread failed: {_re!r} — "
                      f"treating as EOF", file=sys.stderr, flush=True)
                # A reader ERROR is NOT a clean EOF.  Leave _input_eof_mono_ns
                # None (so no valid pre/post-EOF split exists), record the error
                # so main() can EXIT NONZERO, and emit a distinct terminal event
                # that INVALIDATES the run for scoring.
                self._reader_error = repr(_re)
                _run_event('input_terminated', reason='reader_error',
                           error=repr(_re))
                self._eof = True
                break

    def _convert_span(self, pos, n, out=None):
        """Convert ring span [pos, pos+n) to complex64 WITHOUT the lock.
        Returns None if the writer lapped the span during conversion
        (caller must re-claim). Safe because the writer only ADVANCES
        _wp; we validate no-overwrite after converting.
        `out`: optional preallocated float32 target (>= n*2 elems); when
        None a fresh buffer is allocated (owned by the caller forever)."""
        start = (pos * 2) % self._ring_elems
        n_elem = n * 2
        if out is None:
            out = np.empty(n_elem, dtype=np.float32)
        else:
            out = out[:n_elem]
        end = start + n_elem
        if end <= self._ring_elems:
            segs = [(start, end, 0)]
        else:
            split = self._ring_elems - start
            segs = [(start, self._ring_elems, 0),
                    (0, n_elem - split, split)]
        jobs = []
        for (ra, rb, oa) in segs:
            ln = rb - ra
            if ln > 4_000_000 and len(segs) == 1:
                h = (ln // 2) & ~1
                jobs.append((ra, ra + h, oa))
                jobs.append((ra + h, rb, oa + h))
            else:
                jobs.append((ra, rb, oa))

        def _cvt(job):
            ra, rb, oa = job
            dst = out[oa:oa + (rb - ra)]
            np.multiply(self._ring[ra:rb], self._scale,
                        out=dst, casting='unsafe')
            if _IQ_INVERT:
                dst[1::2] *= -1.0
        if len(jobs) > 1 and n_elem > 4_000_000:
            list(self._cvt_pool.map(_cvt, jobs))
        else:
            for j in jobs:
                _cvt(j)
        with self._lock:
            if self._wp - pos > self._ring_n:
                return None        # overwritten mid-convert — stale
        return out.view(np.complex64)

    def _copy_span_raw(self, pos, n, out=None):
        """Copy ring span [pos, pos+n) as RAW int16/int8 pairs WITHOUT the
        lock — same lap-validation contract as _convert_span (returns None
        if the writer overwrote the span mid-copy; caller must re-claim)."""
        start = (pos * 2) % self._ring_elems
        n_elem = n * 2
        if out is None:
            out = np.empty(n_elem, dtype=self._ring.dtype)
        else:
            out = out[:n_elem]
        end = start + n_elem
        if end <= self._ring_elems:
            out[:] = self._ring[start:end]
        else:
            split = self._ring_elems - start
            out[:split] = self._ring[start:self._ring_elems]
            out[split:] = self._ring[:n_elem - split]
        with self._lock:
            if self._wp - pos > self._ring_n:
                return None        # overwritten mid-copy — stale
        return out

    def read_raw(self, n, owned=False):
        """Read n IQ samples as RAW interleaved int16/int8 pairs (2n elems,
        UNCONVERTED — the lazy-window path, see _LAZY).

        Returns (data, skipped) with the same skip/EOF semantics as read().
        Ownership mirrors read(): owned=False may return a reused buffer
        valid only until this reader's next read_raw(); owned=True callers
        (tail reads retained as raw carry) get a private allocation.
        Lazy mode never wants the convert-ahead machinery (there is nothing
        to convert), so any pending prepare is cancelled."""
        while True:
            pos = None
            with self._lock:
                self._ahead = None
                self._ahead_want = None
                avail = self._wp - self._rp
                if avail >= n:
                    skipped = 0
                    if avail > self._ring_n:
                        lost = avail - self._ring_n
                        self._rp += lost
                        self._total_drops += lost
                        skipped = lost
                    pos = self._rp
                elif self._eof:
                    return None, 0
            if pos is None:
                time.sleep(0.005)
                continue
            # raw copy OUTSIDE the lock (same reasoning as read(): a long
            # copy under the lock stalls the writer thread)
            if owned or not _PREALLOC:
                data = self._copy_span_raw(pos, n)
            else:
                if self._raw_buf is None or len(self._raw_buf) < n * 2:
                    self._raw_buf = np.empty(n * 2, dtype=self._ring.dtype)
                data = self._copy_span_raw(pos, n, out=self._raw_buf)
            with self._lock:
                if data is None or self._wp - pos > self._ring_n:
                    continue          # lapped mid-copy: re-claim
                self._rp = pos + n
            return data, skipped

    @staticmethod
    def _grown(buf, n_elem):
        """Reuse `buf` if it fits n_elem float32 elems, else allocate."""
        if buf is None or len(buf) < n_elem:
            return np.empty(n_elem, dtype=np.float32)
        return buf

    def _ahead_worker(self):
        while True:
            with self._lock:
                want = self._ahead_want
                if want is None:
                    self._ahead_cv.wait(0.5)
                    if self._eof and self._ahead_want is None:
                        return
                    continue
                pos, n = want
                ready = (self._wp - pos) >= n
                if self._eof and not ready:
                    self._ahead_want = None
                    continue
            if not ready:
                time.sleep(0.005)
                continue
            out = None
            if _PREALLOC:
                # Alternate the two worker slots per prepare: the slot of
                # the LAST successful prepare may be the buffer the consumer
                # is holding right now (returned via the fast path), so this
                # prepare must target the other one (safety argument at the
                # slot declarations in __init__).
                self._ahead_idx = 1 - self._ahead_idx
                self._ahead_bufs[self._ahead_idx] = self._grown(
                    self._ahead_bufs[self._ahead_idx], n * 2)
                out = self._ahead_bufs[self._ahead_idx]
            buf = self._convert_span(pos, n, out=out)
            with self._lock:
                if self._ahead_want == (pos, n):
                    self._ahead = (pos, n, buf)   # buf None if lapped
                    self._ahead_want = None
                    self._ahead_cv.notify_all()

    def _kick_ahead(self, pos, n):
        """Ask the worker to prepare [pos, pos+n). Caller holds _lock."""
        self._ahead = None
        self._ahead_want = (pos, n)
        if self._ahead_thread is None:
            self._ahead_thread = threading.Thread(
                target=self._ahead_worker, daemon=True,
                name='iq-ahead')
            self._ahead_thread.start()
        self._ahead_cv.notify_all()

    def read(self, n, owned=False):
        """Read n IQ samples as complex64.

        Returns (data, skipped) where:
          data:    np.complex64 array of length n, or None on EOF
          skipped: number of samples that were lost (ring overwrite)

        Each call returns temporally contiguous data. If samples were lost,
        `skipped` > 0 indicates a temporal gap since the previous read.

        OWNERSHIP (see _PREALLOC at module top): with owned=False the
        returned array may be a reused preallocated buffer, valid only
        until this reader's next read(); pass owned=True when the caller
        retains the data past that (a private copy/allocation is returned).
        """
        # convert-ahead fast path: the prepared buffer matches our rp/n
        while True:
            pos = None
            with self._lock:
                if self._ahead is not None:
                    apos, an, abuf = self._ahead
                    if apos == self._rp and an == n and abuf is not None:
                        self._ahead = None
                        self._rp += n
                        self._kick_ahead(self._rp, n)
                        # prepared buffers live in the reused worker slots —
                        # an owned read must hand out a private copy
                        return (abuf.copy() if owned and _PREALLOC
                                else abuf), 0
                    self._ahead = None    # stale (skip/size change)
                if (self._ahead_want is not None
                        and self._ahead_want != (self._rp, n)):
                    self._ahead_want = None   # cancel stale prepare
                avail = self._wp - self._rp
                if avail >= n:
                    skipped = 0
                    if avail > self._ring_n:
                        lost = avail - self._ring_n
                        self._rp += lost
                        self._total_drops += lost
                        skipped = lost
                    pos = self._rp
                elif self._eof:
                    return None, 0
            if pos is None:
                time.sleep(0.005)
                continue
            # inline conversion OUTSIDE the lock — holding it for the
            # whole ~150 ms stalled the WRITER thread (pipe backpressure
            # at exactly the moments the ring most needed draining)
            if owned or not _PREALLOC:
                data = self._convert_span(pos, n)
            else:
                self._cvt_inline = self._grown(self._cvt_inline, n * 2)
                data = self._convert_span(pos, n, out=self._cvt_inline)
            with self._lock:
                if data is None or self._wp - pos > self._ring_n:
                    continue          # lapped mid-convert: re-claim
                self._rp = pos + n
                self._kick_ahead(self._rp, n)
            return data, skipped

    def available(self):
        """Samples available for reading."""
        with self._lock:
            return min(self._wp - self._rp, self._ring_n)

    def skip_to_latest(self, keep_n):
        """Skip ahead so only keep_n samples remain to be read."""
        with self._lock:
            avail = self._wp - self._rp
            if avail > keep_n:
                skip = avail - keep_n
                self._rp += skip
                return skip
            return 0

    @property
    def drops(self):
        with self._lock:
            return self._total_drops

    def source_snapshot(self):
        """Atomic (locked) ingress snapshot for the post-loop source_summary.
        Returns _wp/_rp/drops/bytes together so residual and partial-byte
        accounting are coherent (no reliance on the production-only
        tot_s == _rp equivalence).  samples_dequeued + residual == ingress."""
        with self._lock:
            wp, rp, drops, bread = (self._wp, self._rp, self._total_drops,
                                    self._bytes_read_total)
        return {
            'bytes_per_iq_sample': self.bps,
            'bytes_read_total': bread,
            'samples_ingested': wp,
            'samples_dequeued': rp,
            'residual_samples': wp - rp,
            'ring_dropped_samples': drops,
            'partial_bytes_discarded': bread - wp * self.bps,
            'reader_error': self._reader_error,
            'saw_clean_eof': self._input_eof_mono_ns is not None,
        }


# ---- Energy detection ----

# welch_psd runs ~21x per main-loop iteration (1 long pass + ~20 short multireso
# slices) so caching the Hanning window saves that many small alloc+computes per
# window.  The returned array is read-only by convention — welch_psd only does
# `segs *= win` which mutates segs, never win.
_HANNING_C64_CACHE = {}


def _hanning_c64(nfft):
    w = _HANNING_C64_CACHE.get(nfft)
    if w is None:
        w = np.hanning(nfft).astype(np.complex64)
        _HANNING_C64_CACHE[nfft] = w
    return w


def _welch_psd_from(n, getseg, nfft=4096, n_avg=64, also_max=False):
    """welch_psd core over an abstract segment source: `n` total samples,
    `getseg(start, count)` returns complex64 samples [start, start+count).
    Lets the lazy raw window (_LazyWindow, see _LAZY) feed segments that
    are converted on demand — for an ndarray source (welch_psd below) the
    getseg is a plain slice, so this refactor is bit-identical to the old
    inline form (same segs array, same ops)."""
    step = max(nfft, n // n_avg)
    n_segs = max(1, min(n_avg, (n - nfft) // step + 1))
    if n_segs <= 0:
        z = np.zeros(nfft, dtype=np.float64)
        return (z, z) if also_max else z
    win = _hanning_c64(nfft)
    # Vectorised: stack the segments and FFT them in one batched call so
    # scipy.fft's pocketfft can use multiple threads (workers=-1).
    starts = np.arange(n_segs) * step
    segs = np.empty((n_segs, nfft), dtype=np.complex64)
    for i in range(n_segs):
        s = starts[i]
        if s + nfft > n:
            n_segs = i
            segs = segs[:n_segs]
            break
        segs[i] = getseg(s, nfft)
    # SDR-agnostic DC/LO-leak removal: subtract the MEASURED stationary DC
    # component (the segments' mean — whatever this particular SDR's spike is,
    # no hardcoded magnitude or width) before windowing.  A constant is pure-DC,
    # so this changes ONLY the DC bin of each segment's spectrum — every other
    # bin is untouched, so off-centre detection is unaffected — while a real
    # signal sitting at the centre frequency (whose chirp averages to ~0 mean)
    # keeps its energy and is no longer zeroed the way a fixed dc-notch would.
    if n_segs > 0:
        segs -= segs.mean()
    segs *= win
    # _FFT_WORKERS picked at import time to match the chosen FFT lib's sweet
    # spot for this host: pyfftw=1 on low-core (per-call fan-out hurts),
    # scipy=-1 on high-core (pocketfft scales).
    try:
        F = _fft(segs, axis=1, workers=_FFT_WORKERS)
    except TypeError:
        F = _fft(segs, axis=1)
    F = np.fft.fftshift(F, axes=1)
    pw = np.abs(F) ** 2                                  # |FFT|^2 per segment
    psd = pw.sum(axis=0) / max(1, n_segs)               # mean (validated path)
    if also_max:
        # Max-hold over segments — same FFTs, ~free.  A short packet lands in only
        # a few of the n_segs sparse segments; the mean dilutes it ~/n_segs (up to
        # ~18 dB), but max-hold keeps the catching segment's full power, so weak
        # short bursts (SF8-12, ~+4-5 dB) surface.  Noise is concentrated under a
        # max (0 false peaks even at 3 dB), so this only ADDS real candidates.
        pmax = pw.max(axis=0)
        return 10.0 * np.log10(psd + 1e-15), 10.0 * np.log10(pmax + 1e-15)
    return 10.0 * np.log10(psd + 1e-15)


def welch_psd(iq, nfft=4096, n_avg=64, also_max=False):
    return _welch_psd_from(len(iq), lambda s, m: iq[s:s + m],
                           nfft=nfft, n_avg=n_avg, also_max=also_max)


class _LazyWindow:
    """RAW-domain sliding analysis window (low-core ceiling item 2, _LAZY).

    Owns the main loop's sliding window as raw interleaved int16/int8 and
    hands the per-window fixed path bit-exact complex64 on demand:

      seg(start, n)     complex64 of window samples [start, start+n)
                        (view of the materialization when present, else a
                        fresh conversion of just those samples)
      seg_owned(...)    same but ALWAYS a private allocation (retainers:
                        pre_hop fallback, _hop_own)
      strided(step)     complex64 equivalent of buf[::step] (sat probe)
      welch(off, n, ..) welch_psd over window samples [off, off+n) built
                        from lazily-converted segments (gate PSD + short
                        sweep slices — only ~n_avg*nfft samples convert)
      materialize()     the FULL complex64 window, converted once per
                        window into a persistent prealloc buffer (item-1
                        pattern).  REUSED across windows — any consumer
                        retaining data across iterations must copy, exactly
                        as with the old in-place-slid `buf`.

    Bit-exactness: int->float32 conversion is exact (power-of-two scale,
    same np.multiply + _IQ_INVERT ops as IQReader/_convert_span), so a
    lazily-converted segment is bit-identical to the same slice of a fully
    converted window.  LORA_LAZY_VERIFY=1 proves it at runtime: a shadow
    complex64 window is maintained with the OLD pipeline's exact ops
    (convert each fresh hop, slide in the complex domain) and every product
    above is asserted np.array_equal against it."""

    def __init__(self, win_n, sc16):
        self.win_n = win_n
        self._dtype = np.int16 if sc16 else np.int8
        self.raw = np.zeros(win_n * 2, dtype=self._dtype)
        self._scale = np.float32(1.0 / (2048.0 if sc16 else 128.0))
        self._matf = None      # persistent float32 materialization target
        self._c64 = None       # cached complex64 view for THIS window
        self._conv_samples = 0  # PROF: cumulative complex samples converted
        self._mat_count = 0     # PROF: cumulative materialize() calls
        self.verify_fails = 0
        self._shadow = (np.zeros(win_n, dtype=np.complex64)
                        if _LAZY_VERIFY else None)

    def empty_raw(self):
        return np.zeros(0, dtype=self._dtype)

    def _convert(self, src, out=None):
        """Raw interleaved ints -> complex64, bit-identical to the reader
        conversions (fused power-of-two multiply + optional Q negate)."""
        self._conv_samples += len(src) // 2   # PROF: complex samples converted
        if out is None:
            out = np.empty(len(src), dtype=np.float32)
        else:
            out = out[:len(src)]
        np.multiply(src, self._scale, out=out, casting='unsafe')
        if _IQ_INVERT:
            out[1::2] *= -1.0
        return out.view(np.complex64)

    def convert_owned(self, raw):
        """Standalone raw->complex64 (fresh allocation, caller owns) — for
        tail reads that must become recorder-visible complex64."""
        return self._convert(raw)

    # ---- window mutation -------------------------------------------------
    def slide(self, raw_fresh):
        """Slide raw_fresh (interleaved elems, < window) into the window."""
        n_el = len(raw_fresh)
        w_el = len(self.raw)
        if n_el >= w_el:
            self.raw[:] = raw_fresh[-w_el:]
        else:
            self.raw[:w_el - n_el] = self.raw[n_el:]
            self.raw[w_el - n_el:] = raw_fresh
        self._c64 = None
        if self._shadow is not None:
            self._shadow_slide(raw_fresh)

    def set_full(self, raw_iq):
        """Full replacement (>= one window of fresh samples).  COPIES, so
        reused reader buffers are safe to hand in."""
        self.raw[:] = raw_iq[-len(self.raw):]
        self._c64 = None
        if self._shadow is not None:
            self._shadow[:] = self._convert(self.raw)

    def reset(self):
        self.raw[:] = 0
        self._c64 = None
        if self._shadow is not None:
            self._shadow[:] = 0

    def _shadow_slide(self, raw_fresh):
        # mirror of the OLD pipeline: convert the fresh hop, slide the
        # complex64 window — the reference every lazy product must equal
        k = len(raw_fresh) // 2
        f = self._convert(raw_fresh)
        if k >= self.win_n:
            self._shadow[:] = f[-self.win_n:]
        else:
            self._shadow[:self.win_n - k] = self._shadow[k:]
            self._shadow[self.win_n - k:] = f

    def _vfail(self, tag):
        self.verify_fails += 1
        print(f"[LAZY-VERIFY] MISMATCH ({tag}) — lazy product is NOT "
              f"bit-exact!", file=sys.stderr, flush=True)

    # ---- lazy consumers ----------------------------------------------------
    def seg(self, start, n):
        """complex64 of window samples [start, start+n).  May be a VIEW of
        the materialized window — transient use only (see seg_owned)."""
        if self._c64 is not None:
            out = self._c64[start:start + n]
        else:
            out = self._convert(self.raw[start * 2:(start + n) * 2])
        if self._shadow is not None and not np.array_equal(
                out, self._shadow[start:start + n]):
            self._vfail(f'seg {start}+{n}')
        return out

    def seg_owned(self, start, n):
        """Like seg() but always a private allocation (retainer-safe)."""
        if self._c64 is not None:
            out = self._c64[start:start + n].copy()
        else:
            out = self._convert(self.raw[start * 2:(start + n) * 2])
        if self._shadow is not None and not np.array_equal(
                out, self._shadow[start:start + n]):
            self._vfail(f'seg_owned {start}+{n}')
        return out

    def strided(self, step):
        """complex64 equivalent of buf[::step] (saturation probe)."""
        if self._c64 is not None:
            out = self._c64[::step]
        else:
            pairs = self.raw.reshape(self.win_n, 2)[::step]
            f = np.empty(pairs.shape, dtype=np.float32)
            np.multiply(pairs, self._scale, out=f, casting='unsafe')
            if _IQ_INVERT:
                f[:, 1] *= -1.0
            out = f.reshape(-1).view(np.complex64)
        if self._shadow is not None and not np.array_equal(
                out, self._shadow[::step]):
            self._vfail(f'strided {step}')
        return out

    def welch(self, off, n, nfft=4096, n_avg=64, also_max=False):
        """welch_psd over window samples [off, off+n) — converts only the
        n_avg segment starts the Welch actually reads (the quiet-window
        enabler: 50x4096 samples instead of the whole window)."""
        if self._c64 is not None:
            res = welch_psd(self._c64[off:off + n], nfft=nfft,
                            n_avg=n_avg, also_max=also_max)
        else:
            res = _welch_psd_from(n, lambda s, m: self.seg(off + s, m),
                                  nfft=nfft, n_avg=n_avg, also_max=also_max)
        if self._shadow is not None:
            ref = welch_psd(self._shadow[off:off + n], nfft=nfft,
                            n_avg=n_avg, also_max=also_max)
            ok = (np.array_equal(res[0], ref[0])
                  and np.array_equal(res[1], ref[1])) if also_max else \
                np.array_equal(res, ref)
            if not ok:
                self._vfail(f'welch {off}+{n}')
        return res

    def materialize(self):
        """The FULL complex64 window — converted at most once per window
        into the persistent prealloc target (returns the cached view on
        repeat calls within the same window)."""
        if self._c64 is None:
            self._mat_count += 1        # PROF: full-window materialization
            if self._matf is None:
                self._matf = np.empty(self.win_n * 2, dtype=np.float32)
            self._c64 = self._convert(self.raw, out=self._matf)
            if self._shadow is not None and not np.array_equal(
                    self._c64, self._shadow):
                self._vfail('materialize')
        return self._c64


def _emit_psd_frame(path, psd_db, out_bins=512, span_db=55.0):
    """Downsample a full-band dB PSD to `out_bins` (max-pool, so narrow LoRa peaks
    survive), quantize to uint8 relative to the per-frame noise floor (SDR-gain
    agnostic), and atomically write the frame for the web waterfall.  Reuses the
    gate's already-computed PSD → no extra FFT; ~512 bytes per frame.  Never raises."""
    try:
        psd = np.asarray(psd_db, dtype=np.float64)
        n = psd.size
        if n > out_bins:
            trim = n - (n % out_bins)
            ds = psd[:trim].reshape(out_bins, trim // out_bins).max(axis=1)
        else:
            ds = psd
        floor = np.percentile(ds, 20.0)
        q = np.clip((ds - floor) / span_db * 255.0, 0, 255).astype(np.uint8)
        tmp = path + '.tmp'
        with open(tmp, 'wb') as f:
            f.write(q.tobytes())
        os.replace(tmp, path)
    except Exception:
        pass


def find_peaks(psd_db, thresh_db=8.0, min_bins=3, max_peaks=MAX_ENERGY_PEAKS,
               wide_run_bins=None, min_sep_bins=20,
               base_contour_db=4.0, base_min_bins=4, fres_hz=None):
    """Find energy peaks in a PSD.

    Each contiguous above-threshold run normally maps to one signal — but
    the old "centre = midpoint of the run" rule landed off-carrier when the
    run was wide (signal plateau is several dB wide and the centre bin was
    rarely the strongest one).  We use argmax of the run as the centre.

    For runs WIDER than `wide_run_bins` (≈ 685 kHz at 28 Msps / nfft=4096
    — larger than any single Meshtastic channel's 500 kHz BW), we assume
    two or more signals have merged and decompose the run into ≥2 local
    maxima separated by `min_sep_bins`.  Narrow runs (single signal) emit
    one peak only, keeping the per-window SC cost bounded.

    Hysteresis width rescue (bandwidth-aware): a run NARROWER than min_bins
    is normally dropped as a CW spur — but a weak wide-BW LoRa signal
    (SF7/500k especially) spreads its energy so thin that only 1-2 bins
    clear (median + thresh_db) while its true footprint is many bins wide
    at a lower contour.  Measured live: real SF7/500k beacons ~13 dB SNR
    poke 2 bins above +12 dB (rejected, packet lost) but span 4+ bins above
    +4-5 dB.  So for narrow runs ONLY, re-measure the footprint width at
    (median + base_contour_db); if it spans >= base_min_bins the peak is
    LoRa-shaped and is accepted, centred on the footprint's power centroid.
    A CW spur / front-end overload comb tooth stays 1-3 bins wide at ANY
    contour (verified at max-gain overload: full-band ~100 kHz spur comb,
    every tooth narrow), so spur immunity is preserved.  ADDITIVE ONLY:
    runs that already pass min_bins are emitted exactly as before.

    Returns: list of (center_bin, width_bins, peak_db) tuples, sorted by
    peak_db descending, truncated to `max_peaks`.
    """
    # wide_run_bins was a FIXED 100 - calibrated as '685 kHz at 28 Msps'.
    # At the production 20 Msps a single 500 kHz signal spans 102 bins and
    # was silently DECOMPOSED into shoulder fragments with arbitrary
    # sub-centroids (the origin of the 500k-family center scatter and
    # duplicate shoulder peaks). Scale by frequency: 700 kHz - wider than
    # any single LoRa channel - whatever the sample rate.
    if wide_run_bins is None:
        wide_run_bins = int(round(700e3 / fres_hz)) if fres_hz else 100
    nf = np.median(psd_db)
    n = len(psd_db)
    peaks = []
    # Above-threshold runs via vectorised edge detection (was a per-bin Python
    # loop over all 4096 bins — ~2.9x slower, and it runs ~21x/window once the
    # max-hold pass is on).  Byte-for-byte identical runs → identical peaks.
    a = (psd_db > (nf + thresh_db)).astype(np.int8)
    d = np.diff(a)
    starts = np.flatnonzero(d == 1) + 1
    ends = np.flatnonzero(d == -1) + 1
    if a[0]:
        starts = np.concatenate(([0], starts))
    if a[-1]:
        ends = np.concatenate((ends, [n]))
    base_thr = nf + min(base_contour_db, thresh_db)
    rescued = []   # (lo, hi) footprints already emitted, to fold split runs
    for s, e in zip(starts.tolist(), ends.tolist()):
        if e - s < min_bins and base_min_bins > 0:
            # Narrow at the detection threshold — check the low-contour
            # footprint before rejecting.  Per-bin expansion is cheap: it
            # only runs for narrow runs and stops at the contour edge.
            pk = s + int(np.argmax(psd_db[s:e]))
            if any(lo <= pk <= hi for lo, hi in rescued):
                continue   # split-run twin of an already-rescued footprint
            lo = pk
            while lo > 0 and psd_db[lo - 1] > base_thr:
                lo -= 1
            hi = pk
            while hi < n - 1 and psd_db[hi + 1] > base_thr:
                hi += 1
            w = hi - lo + 1
            if w >= base_min_bins:
                peaks.append((_power_centroid(psd_db, lo, hi + 1), w,
                              float(psd_db[pk]), nf))
                rescued.append((lo, hi))
            continue
        _emit_run_peaks(psd_db, s, e, min_bins, wide_run_bins, min_sep_bins, peaks, nf_ref=nf)
    peaks.sort(key=lambda p: p[2], reverse=True)
    return peaks[:max_peaks]


def _power_centroid(seg_db, lo, hi):
    """Power-weighted centroid (float bin index) of seg_db[lo:hi], with the
    run's noise floor subtracted so only signal energy is weighted.

    A LoRa chirp's PSD is ~rectangular across its bandwidth, so the argmax of
    the above-threshold run lands anywhere on the flat top — typically a band
    edge, and (under scipy.fft workers=-1's ~1e-6 noise) a DIFFERENT bin each
    run.  Centring the extraction there leaves the capture ~BW/2 off-carrier:
    the downstream ±BW/2 crop discards half the chirp, the dechirp-based CFO
    correction then sees only half the signal and can't recover the true
    carrier, so the packet decodes one run but not the next.  The run's
    power-weighted centroid IS the carrier — accurate and deterministic.
    """
    sub = seg_db[lo:hi]
    # Subtract the run's own floor so the weighting reflects signal power, not
    # an additive pedestal (a flat pedestal would pull the centroid toward the
    # geometric midpoint instead of the energy centre).
    lin = 10.0 ** ((sub - sub.min()) / 10.0)
    idx = np.arange(lo, hi)
    s = float(np.sum(lin))
    if s <= 0:
        return float(lo + (hi - lo) / 2.0)
    return float(np.sum(idx * lin) / s)


def _emit_run_peaks(psd_db, start, end, min_bins, wide_run_bins, min_sep_bins, peaks, nf_ref=0.0):
    """Emit one peak per contiguous above-threshold run, or multiple peaks if
    the run is suspiciously wide (likely two overlapping signals).  Each peak
    is centred on the power-weighted centroid of its band (see _power_centroid)
    — NOT the argmax — so the downstream narrowband extraction lands on the
    actual carrier accurately and deterministically.
    """
    width = end - start
    if width < min_bins:
        return
    seg = psd_db[start:end]
    rel_seed = int(np.argmax(seg))
    peak_db = float(seg[rel_seed])
    if width <= wide_run_bins:
        # Single signal: centroid over the whole run = carrier.
        peaks.append((start + _power_centroid(seg, 0, len(seg)), width, peak_db, nf_ref))
        return
    # Wide run: two (or more) merged signals.  Find a second clearly-separate
    # local max, split the run at the dip between the two maxima, and centroid
    # each side over its own band so neither carrier biases the other.
    seg2 = seg.copy()
    lo_z = max(0, rel_seed - min_sep_bins)
    hi_z = min(len(seg2), rel_seed + min_sep_bins + 1)
    seg2[lo_z:hi_z] = -1e9
    rel2_seed = int(np.argmax(seg2))
    peak_db2 = float(seg2[rel2_seed])
    if not (peak_db2 > peak_db - 6.0 and peak_db2 > -1e8):
        # Only one real signal in a wide run — centroid the whole run.
        peaks.append((start + _power_centroid(seg, 0, len(seg)), width, peak_db, nf_ref))
        return
    a, b = sorted((rel_seed, rel2_seed))
    dip = a + int(np.argmin(seg[a:b + 1]))   # split at the trough between them
    peaks.append((start + _power_centroid(seg, 0, dip), width, peak_db, nf_ref))
    peaks.append((start + _power_centroid(seg, dip, len(seg)), width, peak_db2, nf_ref))


# ---- Narrowband extraction ----

@_pp_time('extract','n_extract')
def extract_nb_fft_multi_bw(iq, wb_fs, offset_hz, target_bws, chunk=65536,
                            fft_cache=None):
    """
    Batched FFT-crop: forward FFTs computed ONCE, then crop at each BW.
    Returns list of (nb_iq, nb_fs) per BW. Not phase-coherent — fine for CNN.

    fft_cache: precomputed list of UNSHIFTED fft(chunk) arrays.  When provided,
    the forward FFT pass is skipped entirely — pass the same cache to every peak
    in a multi-peak window to avoid recomputing the same 28 M-sample FFT N times.

    PERF: the forward FFTs are stored UNSHIFTED (DC at bin 0).  The old code
    np.fft.fftshift'd every chunk (a full array roll, ~42 % of this function's
    time on the batched 65536-chunk cache), then cropped a centered slice.  We
    instead crop directly from the unshifted spectrum with a contiguous (≤2-piece
    wrap) slice and keep the cheap ifftshift on the small crop — bit-identical
    output, no large roll.  Zero-padding at the ±Nyquist edges is preserved.
    """
    # Normalize the cache to ONE 2-D (n_chunks, chunk) array.  The per-peak
    # crop below then runs as two 2-D slice assignments + ONE batched ifft
    # instead of a Python loop of ~305 tiny allocs/rolls/iffts per peak —
    # measured on a Pi 4 that loop cost 0.3-1 s PER PEAK in busy windows
    # (pure per-call overhead); the batched form is tens of ms and
    # bit-identical (same crop, same ifftshift, same per-row ifft).
    if fft_cache is None or (isinstance(fft_cache, list) and not fft_cache):
        n_chunks = len(iq) // chunk
        if n_chunks >= 1:
            FF = _fft(iq[:n_chunks * chunk].reshape(n_chunks, chunk), axis=1)
        else:
            FF = None
    elif isinstance(fft_cache, np.ndarray):
        FF = fft_cache                    # already the batched 2-D cache
    else:
        FF = np.asarray(fft_cache)        # legacy list-of-rows input

    cb = chunk // 2 + int(round(offset_hz * chunk / wb_fs))
    _h = chunk // 2
    results = []
    for tbw in target_bws:
        bw_bins = max(4, int(round(tbw * chunk / wb_fs)))
        lo, hi = cb - bw_bins, cb + bw_bins
        cw = hi - lo
        # Valid range in shifted coords, then map to the unshifted spectrum.
        lo_c, hi_c = max(0, lo), min(chunk, hi)
        dst = max(0, -lo)
        cnt = hi_c - lo_c
        src = (lo_c - _h) % chunk        # shifted bin lo_c → unshifted bin
        if FF is None:
            results.append((np.array([], dtype=np.complex64), tbw * 2))
            continue
        nc = FF.shape[0]
        cropped = np.zeros((nc, cw), dtype=np.complex64)
        if cnt > 0:
            if src + cnt <= chunk:
                cropped[:, dst:dst + cnt] = FF[:, src:src + cnt]
            else:                         # wraps the DC/Nyquist boundary once
                f0 = chunk - src
                cropped[:, dst:dst + f0] = FF[:, src:]
                cropped[:, dst + f0:dst + cnt] = FF[:, :cnt - f0]
        nb = _ifft(np.fft.ifftshift(cropped, axes=1), axis=1)
        results.append((nb.astype(np.complex64).reshape(-1),
                        cw * wb_fs / chunk))
    return results


def _design_lowpass(dec, ntaps=None):
    """Design windowed-sinc lowpass FIR for decimation."""
    if ntaps is None:
        ntaps = max(16, dec * 4) | 1  # odd number, at least 4× decimation factor
    cutoff = 1.0 / dec  # normalized cutoff (0 to 1 = Nyquist)
    n = np.arange(ntaps, dtype=np.float64)
    mid = (ntaps - 1) / 2.0
    x = (n - mid) * cutoff
    # Windowed sinc — handle x=0 case
    h = np.ones(ntaps, dtype=np.float64) * cutoff
    nonzero = np.abs(x) > 1e-12
    h[nonzero] = np.sin(np.pi * x[nonzero]) / (np.pi * x[nonzero]) * cutoff
    # Blackman window for -74 dB sidelobes
    h *= 0.42 - 0.5 * np.cos(2 * np.pi * n / (ntaps - 1)) + 0.08 * np.cos(4 * np.pi * n / (ntaps - 1))
    h /= np.sum(h)  # normalize to unity gain
    return h.astype(np.float64)

# Cache filters by decimation factor
_fir_cache = {}

def _get_fir(dec):
    if dec not in _fir_cache:
        _fir_cache[dec] = _design_lowpass(dec)
    return _fir_cache[dec]


def extract_narrowband(iq, wb_fs, offset_hz, target_bw, chunk_size=1_000_000):
    """Phase-coherent narrowband via freq shift + FIR lowpass + decimate."""
    target_fs = target_bw * 2
    dec = max(1, int(round(wb_fs / target_fs)))
    actual_fs = wb_fs / dec

    if dec == 1:
        return iq.copy(), actual_fs

    fir = _get_fir(dec)
    phase = 0.0
    phase_inc = -2.0 * np.pi * offset_hz / wb_fs

    # Shift to baseband
    n = len(iq)
    phases = phase + np.arange(n, dtype=np.float64) * phase_inc
    shifted = iq * np.exp(1j * phases).astype(np.complex64)

    # FIR filter then decimate
    filtered = np.convolve(shifted, fir, mode='same')
    decimated = filtered[::dec]

    return decimated.astype(np.complex64), actual_fs


def extract_narrowband_stateful(iq, wb_fs, offset_hz, target_bw,
                                 phase=0.0, fir_tail=None):
    """Phase-coherent narrowband with persistent phase AND filter state.
    
    Returns (nb_iq, actual_fs, next_phase, next_fir_tail).
    Pass fir_tail from previous call to maintain filter continuity.
    """
    target_fs = target_bw * 2
    dec = max(1, int(round(wb_fs / target_fs)))
    actual_fs = wb_fs / dec
    phase_inc = -2.0 * np.pi * offset_hz / wb_fs

    n = len(iq)
    if n == 0:
        return np.array([], dtype=np.complex64), actual_fs, phase, fir_tail

    # Shift to baseband with continuous phase
    phases = phase + np.arange(n, dtype=np.float64) * phase_inc
    shifted = iq * np.exp(1j * phases).astype(np.complex64)
    next_phase = phases[-1] + phase_inc

    if dec == 1:
        return shifted, actual_fs, next_phase, None

    # FIR lowpass with overlap from previous call
    fir = _get_fir(dec)
    ntaps = len(fir)

    if fir_tail is not None and len(fir_tail) > 0:
        # Prepend tail from previous call for filter continuity
        padded = np.concatenate([fir_tail, shifted])
        filtered = np.convolve(padded, fir, mode='full')
        # Trim: skip filter group delay from prepended tail
        start = len(fir_tail)
        filtered = filtered[start:start + n]
    else:
        filtered = np.convolve(shifted, fir, mode='same')

    # Save tail for next call (last ntaps-1 samples)
    next_tail = shifted[-(ntaps - 1):].copy() if n >= ntaps - 1 else shifted.copy()

    # Decimate
    decimated = filtered[::dec]

    return decimated.astype(np.complex64), actual_fs, next_phase, next_tail



@_pp_time('extract','n_extract')
def extract_narrowband_fft(iq, wb_fs, offset_hz, target_bw, fft_cache=None):
    """FFT-based narrowband extraction — clean brick-wall filter, no aliasing.

    Takes FFT of wideband IQ, crops around signal frequency, IFFTs back.
    Result: signal centered at DC with fs = target_bw * 2.
    Perfect for recording — chirp structure fully preserved.

    fft_cache: optional [F_shifted] one-element list.  On first call with the
    cache empty, the forward FFT is computed and stored; subsequent calls
    reuse the cached spectrum for additional offsets at the same wideband
    length.  A 28-M-point FFT is the dominant cost in the save_worker
    (>1 s per call); caching across multiple preambles in the same batch
    is the difference between real-time and queue backlog.

    The forward FFT is taken at next_fast_len(N): a raw gate buffer length
    such as 43,065,344 = 2^13·7·751 carries a large prime factor (751) that
    forces pocketfft onto a slow general path (~5.7 s); zero-padding up to a
    5-smooth length (~+0.2 % samples) drops it to ~0.8 s with no change to the
    extracted signal (verified: 0.99998 correlation in the signal region).
    The zero pad only lengthens the near-zero tail of the output by ~1.6 ms.
    """
    N0 = len(iq)
    target_fs = target_bw * 2

    # Forward FFT stored UNSHIFTED (DC at bin 0).  The old code fftshift'd the
    # whole spectrum (a full array roll) so it could take a centered slice; we
    # instead map the centered crop back into the unshifted spectrum with a
    # contiguous (≤2-piece wrap) slice below, dropping the large roll while
    # staying bit-identical.  Pad to a fast FFT length to avoid prime-factor
    # slowdowns; the cached F's length defines N for all crop math so cached and
    # fresh calls stay consistent.
    if fft_cache is not None and fft_cache:
        F = fft_cache[0]
        N = len(F)
    else:
        N = _next_fast_len(N0)
        if N != N0:
            iqp = np.empty(N, dtype=np.complex64)
            iqp[:N0] = iq
            iqp[N0:] = 0
            F = _fft(iqp)
        else:
            F = _fft(iq)
        if fft_cache is not None:
            fft_cache.append(F)

    # Number of output bins = N * (target_fs / wb_fs)
    n_out = int(round(N * target_fs / wb_fs))
    if n_out < 2:
        return np.array([], dtype=np.complex64), target_fs

    # Find the center bin corresponding to offset_hz
    freq_per_bin = wb_fs / N
    center_bin = N // 2 + int(round(offset_hz / freq_per_bin))
    if _PP_PROF_ON:   # exact-duplicate extraction redundancy meter (memo potential)
        _EXT_CALLS[0] += 1
        _EXT_KEYS.add((id(F), center_bin, n_out))

    # Crop n_out bins centered on the signal
    half = n_out // 2
    lo = center_bin - half
    hi = lo + n_out

    # Handle edge cases.  cropped used to be complex128, which forced an
    # upcast from F (complex64) on the slice assignment and then a
    # complex128 IFFT (~2x slower than complex64) and a final downcast
    # back to complex64.  All three were pure waste — F is already
    # complex64, and the downstream pipeline only needs complex64
    # precision.  Staying in complex64 throughout halves IFFT cost and
    # eliminates two array conversions.
    cropped = np.zeros(n_out, dtype=np.complex64)
    src_lo = max(0, lo)
    src_hi = min(N, hi)
    dst_lo = src_lo - lo
    dst_hi = dst_lo + (src_hi - src_lo)
    # F is unshifted: shifted bin src_lo maps to unshifted bin (src_lo - N//2)%N,
    # a contiguous run that wraps the boundary at most once.  Edge zero-padding
    # (dst_lo>0 / dst_hi<n_out) is preserved exactly as before.
    _cnt = src_hi - src_lo
    if _cnt > 0:
        _src = (src_lo - N // 2) % N
        if _src + _cnt <= N:
            cropped[dst_lo:dst_hi] = F[_src:_src + _cnt]
        else:
            _f0 = N - _src
            cropped[dst_lo:dst_lo + _f0] = F[_src:]
            cropped[dst_lo + _f0:dst_hi] = F[:_cnt - _f0]

    # IFFT back to time domain — signal is now centered at DC.
    # _ifft preserves complex64 dtype when given complex64 input.
    result = _ifft(np.fft.ifftshift(cropped))
    if result.dtype != np.complex64:
        result = result.astype(np.complex64)

    # Scale to preserve amplitude (fewer bins → need to scale)
    result *= (n_out / N)

    return result, float(target_fs)


def _nb_plateau_offset(iq, fs, bw):
    """CROP-CENTER FIX: max-hold plateau mis-centering meter.

    Verbatim port of the investigation's validated ground-truth estimator
    (scratch plateau_validated.py, ±150 Hz on the gatecap SF11 set).
    A LoRa chirp sweeps the full bw every symbol, so under MAX-HOLD every
    in-band bin reaches ~full chirp power → a flat, bw-wide plateau that a
    narrow burst-CW spur (1-3 bins, constant envelope, gains nothing from
    max-hold) cannot displace — unlike the Welch-average argmax centroid it
    hijacks in detect_preamble.  Floor is a LOW percentile (p20), NOT the
    median: at fs = 2*bw the plateau itself spans ~half the bins.

    Returns the plateau edge-midpoint offset from DC (Hz) — i.e. how far
    off-center the crop is — or None to ABSTAIN (slice too short, or no
    contiguous floor+6dB run of width 0.6-1.4x bw).  Callers MUST treat
    None as "keep the original center" (fail-safe contract: a correction
    is only ever applied on a positive, bw-wide plateau identification).
    """
    n = 2048
    if len(iq) < 4 * n:
        return None
    mh = None
    w = np.hanning(n)
    for i in range(0, min(len(iq), int(fs * 3.0)) - n, n // 2):
        S = np.abs(np.fft.fftshift(np.fft.fft(iq[i:i + n] * w))) ** 2
        mh = S if mh is None else np.maximum(mh, S)
    if mh is None:
        return None
    frq = np.fft.fftshift(np.fft.fftfreq(n, 1.0 / fs))
    floor = np.percentile(mh, 20)
    thr = floor * 3.981  # floor + 6 dB
    abv = mh > thr
    d = np.diff(abv.astype(np.int8))
    st = np.where(d == 1)[0] + 1
    en = np.where(d == -1)[0]
    if abv[0]:
        st = np.r_[0, st]
    if abv[-1]:
        en = np.r_[en, len(abv) - 1]
    best, be = None, -1.0
    for s, e in zip(st, en):
        wd = (e - s + 1) * (fs / n)
        if 0.6 * bw <= wd <= 1.4 * bw:
            es = float(mh[s:e + 1].sum())
            if es > be:
                be, best = es, (s, e)
    if best is None:
        return None
    return 0.5 * (frq[best[0]] + frq[best[1]])



# ---- Dechirp ----

# Downchirp template cache.  generate_downchirp(sf, bw, fs) is deterministic
# in its three integer args, and gets called repeatedly with the same
# arguments — e.g. dechirp_peak_quality regenerates the template every
# invocation, and the resolver iterates several (sf, bw) candidates per
# hit_lag.  Profile measurements (2026-06-22): SF7 = 6.5us per call,
# SF12 = 118us per call; with ~25 calls per peak the SF12 case alone
# costs ~3 ms / peak before this cache.  Pure memoization, bit-identical
# output — output arrays are read-only views from the cache so callers
# can't mutate the cached template (e.g. an in-place multiply elsewhere
# would corrupt every later use).
_DOWNCHIRP_CACHE = {}


def generate_downchirp(sf, bw, fs):
    """Module-level cached.  See _DOWNCHIRP_CACHE comment above."""
    key = (int(sf), int(bw), int(fs))
    cached = _DOWNCHIRP_CACHE.get(key)
    if cached is not None:
        return cached
    N = 2 ** sf
    osf = int(round(fs / bw))
    sps = N * osf
    t = np.arange(sps, dtype=np.float64) / fs
    Ts = N / bw
    phase = 2.0 * np.pi * (-bw / 2.0 * t + bw / (2.0 * Ts) * t * t)
    dc = np.exp(-1j * phase).astype(np.complex64)
    dc.setflags(write=False)   # immutable view — callers MUST NOT mutate
    _DOWNCHIRP_CACHE[key] = dc
    return dc


def _hybrid_sc_locate_window(iq, fs, sf, bw, top_k=1):
    """Time-localise LoRa preambles inside a wideband IQ buffer using a
    frequency-agnostic Schmidl-Cox autocorrelation.  Returns a list of
    (sample_index, confidence_norm) tuples, top_k entries sorted by
    confidence descending.  Confidence is the SC sliding-sum magnitude
    over 6 symbols divided by the median across all candidate windows;
    > ~5-6 means a clean preamble is present at that position.

    Multiple preambles in one buffer (hop0 from Node A + hop1 from Node B,
    or successive packets within a long recording) each peak at their own
    time index; top_k lets the caller iterate over them.
    """
    osf = int(round(fs / bw))
    lag = (1 << sf) * osf
    if len(iq) <= lag + 64:
        return [(0, 0.0)]
    auto = iq[lag:] * np.conj(iq[:-lag])
    n_blocks = len(auto) // lag
    if n_blocks < 6:
        return [(0, 0.0)]
    auto_blocks = auto[:n_blocks * lag].reshape(n_blocks, lag).mean(axis=1)
    pwr = (np.abs(iq[:n_blocks * lag]) ** 2).reshape(n_blocks, lag).mean(axis=1)
    norm = np.abs(auto_blocks) / np.maximum(pwr, 1e-30)
    win_b = 6
    cs = np.cumsum(norm)
    sums = cs[win_b - 1:] - np.concatenate(([0], cs[:-win_b]))
    noise = float(np.median(sums)) + 1e-9
    if top_k <= 1:
        best_b = int(np.argmax(sums))
        return [(best_b * lag, float(sums[best_b] / win_b) / noise)]
    # Top-K with non-maximum suppression: each retained peak masks ±win_b
    # neighbours so we don't return overlapping copies of the same
    # preamble.  Plateau width = 6 symbols, so masking ±6 is conservative.
    candidates = []
    masked = sums.copy()
    for _ in range(top_k):
        idx = int(np.argmax(masked))
        val = float(masked[idx])
        if val <= 0:
            break
        candidates.append((idx * lag, val / win_b / noise))
        lo = max(0, idx - win_b * 2)
        hi = min(len(masked), idx + win_b * 2 + 1)
        masked[lo:hi] = 0
    return candidates if candidates else [(0, 0.0)]


def _hybrid_fft_recenter(iq, fs, shift_hz, out_bw):
    """FFT-shift iq by -shift_hz then crop to 2*out_bw sample rate.  Matches
    extract_narrowband_fft semantics in production; kept inline so the
    re-centre routine doesn't depend on having a contiguous narrowband
    extraction step.
    """
    n = len(iq)
    target_fs = out_bw * 2
    n_out = int(round(n * target_fs / fs))
    F = np.fft.fftshift(_fft(iq))
    freq_per_bin = fs / n
    center_bin = n // 2 + int(round(shift_hz / freq_per_bin))
    half = n_out // 2
    lo = center_bin - half
    hi = lo + n_out
    cropped = np.zeros(n_out, dtype=np.complex128)
    src_lo = max(0, lo)
    src_hi = min(n, hi)
    cropped[src_lo - lo:src_lo - lo + (src_hi - src_lo)] = F[src_lo:src_hi]
    out = _ifft(np.fft.ifftshift(cropped)) * (n_out / n)
    return out.astype(np.complex64), float(target_fs)


def _hybrid_dechirp_pmr(iq, sf, bw, fs, n_syms=8):
    """Best n_syms-symbol PMR (dB) + peak bin from those symbols, used by
    the hybrid re-centre routine.  Equivalent to dechirp_peak_quality below
    but returns the peak bin from the *best* sliding window so the answer
    is stable even when the preamble doesn't start at sample 0."""
    N = 1 << sf
    osf = max(1, int(round(fs / bw)))
    sps = N * osf
    n_total = len(iq) // sps
    if n_total < 1:
        return -99.0, 0
    dc = generate_downchirp(sf, bw, fs)
    iq_syms = iq[:n_total * sps].reshape(n_total, sps)
    dechirped = iq_syms * dc
    if osf > 1:
        _r = dechirped.reshape(n_total, N, osf)   # slice-add: see line ~1466
        dechirped = ((_r[:, :, 0] + _r[:, :, 1]) * np.complex64(0.5)
                     if osf == 2 else _r.mean(axis=2))
    spectra = np.abs(_fft(dechirped, axis=1)) ** 2
    peaks = np.max(spectra, axis=1)
    means = np.mean(spectra, axis=1)
    ratios = np.where(means > 0, peaks / means, 0)
    win = min(n_syms, n_total)
    if n_total <= win:
        best_avg = float(np.mean(ratios))
        top_idx = np.arange(n_total)
    else:
        cs = np.cumsum(ratios)
        sums = cs[win - 1:] - np.concatenate(([0], cs[:-win]))
        best_start = int(np.argmax(sums))
        best_avg = float(sums[best_start] / win)
        top_idx = np.arange(best_start, best_start + win)
    avg_spec = np.mean(spectra[top_idx], axis=0)
    peak_bin = int(np.argmax(avg_spec))
    return 10.0 * np.log10(best_avg + 1e-15), peak_bin


def _hybrid_locate_one(iq, fs, sf, bw, time_sample):
    """Run the frequency-search portion of the hybrid recenter for a single
    SC-time-localised window.  Returns (offset_hz, pmr_db, pb_residual,
    status).  Caller is responsible for providing time_sample from the SC
    locator (so multi-preamble scanning can reuse the localiser result).
    """
    osf = int(round(fs / bw))
    lag = (1 << sf) * osf
    N = 1 << sf
    slice_n = lag * 16
    st = max(0, time_sample - lag * 2)
    en = min(len(iq), st + slice_n)
    slc = iq[st:en]
    if len(slc) < lag * 8:
        return 0.0, 0.0, 999, 'NO_LOCK'
    # ----- Carrier = energy centroid of the preamble band -----
    # A LoRa chirp's PSD is ~rectangular across [fc-BW/2, fc+BW/2], so the
    # energy centroid of its band IS the carrier fc — unambiguously (no cyclic
    # dechirp ±BW alias) and deterministically.  The previous coarse-grid +
    # alias-resolve (10 dechirp+FFTs) keyed off PMR, but a cyclic dechirp
    # peaks at EVERY trial offset so PMR is ~equal across candidates; the
    # |pb| centring tiebreak was then decided by the ~1e-6 non-determinism of
    # scipy.fft(workers=-1).  The chosen centre flipped run-to-run and often
    # landed BW/2 off (signal half-cropped by the downstream ±BW/2 crop →
    # confident demod errors that chase can't fix).  Centroid is 1 PSD +
    # 1 dechirp: less compute AND robust/deterministic.
    win = np.hanning(len(slc)).astype(np.complex64)
    S = np.abs(np.fft.fftshift(_fft(slc * win))) ** 2
    fr = np.fft.fftshift(np.fft.fftfreq(len(slc), 1.0 / fs))
    # Search only within ±BW of the anchor — the energy detector already put
    # the carrier within ~BW/2, so distant spurs can't pull the estimate.
    inb = np.abs(fr) <= bw
    if not inb.any():
        return 0.0, 0.0, 999, 'NO_LOCK'
    Sb = np.where(inb, S, 0.0)
    peak = int(np.argmax(Sb))
    # Centroid over the contiguous above-threshold plateau CONTAINING the peak,
    # so an adjacent-channel hop (one BW away, possibly in this slice) can't
    # bias the estimate — only the dominant channel's own band is averaged.
    thr = Sb[peak] * 0.25
    lo = peak
    while lo > 0 and Sb[lo - 1] > thr:
        lo -= 1
    hi = peak
    while hi < len(Sb) - 1 and Sb[hi + 1] > thr:
        hi += 1
    w = Sb[lo:hi + 1]
    f_centroid = float(np.sum(fr[lo:hi + 1] * w) / np.sum(w))
    # ----- Fine refine + quality gate (single dechirp at the centroid) -----
    iq_lo, lo_fs = _hybrid_fft_recenter(slc, fs, f_centroid, bw)
    pmr_final, pb = _hybrid_dechirp_pmr(iq_lo, sf, bw, int(lo_fs), n_syms=8)
    pb_signed = pb if pb <= N // 2 else pb - N
    f_final = f_centroid + pb_signed * bw / N
    pb_abs = abs(pb_signed)
    # Status from PMR only.  pb is just the sub-bin refine of an already-good
    # centroid (folded into f_final), so it must NOT gate the lock.
    if pmr_final < 10.0:
        status = 'NO_LOCK'
    elif pmr_final >= 15.0:
        status = 'LOCK'
    else:
        status = 'WEAK'
    return f_final, pmr_final, pb_abs, status


def find_all_preambles(iq, fs, sf, bw, top_k=10):
    """Find every LoRa preamble in a wideband IQ capture, regardless of
    carrier offset.  Returns a list of dicts:
        [{'offset_hz', 'time_sample', 'pmr_db', 'status'}, ...]
    sorted by time_sample ascending.

    Architectural note: this is the principled answer to "one recording can
    contain multiple LoRa signals at different carriers."  Each preamble's
    sample position is found by a frequency-agnostic Schmidl-Cox sweep;
    each found position is then refined to a precise carrier offset via
    the hybrid frequency-search.  Clusters of candidates within ±2 symbol-
    times AND ±1 LoRa bin (BW/N) are collapsed to a single canonical
    answer (the highest-PMR member).  SF-agnostic — works for any SF and
    BW because lag, bin width, and clustering thresholds all scale.
    """
    osf = int(round(fs / bw))
    lag = (1 << sf) * osf
    N = 1 << sf
    sc_candidates = _hybrid_sc_locate_window(iq, fs, sf, bw, top_k=top_k)
    found = []
    for time_sample, sc_conf in sc_candidates:
        if sc_conf < 3.0:    # SC plateau dropping into the noise floor
            continue
        f_off, pmr, pb_abs, status = _hybrid_locate_one(
            iq, fs, sf, bw, time_sample)
        if status == 'NO_LOCK':
            continue
        found.append({
            'offset_hz':   float(f_off),
            'time_sample': int(time_sample),
            'pmr_db':      float(pmr),
            'status':      status,
        })
    if not found:
        return []
    # Cluster: same physical signal appears as multiple SC candidates
    # within a preamble's plateau and within one bin in frequency.
    # The SC autocorrelation stays high across all 8 preamble symbols, so
    # one preamble can produce SC peaks several symbol-times apart.
    # With the old 2-sym window, two SC peaks ~4 ms apart at the same
    # carrier survived dedup, save_worker saved both → both decoded as
    # the same packet, wasting decoder CPU and inflating dec_q.  Use a
    # 12-sym window — covers the full 8-sym preamble plateau plus a
    # safety margin on each side — so the cluster collapses correctly.
    sym_time_samp = lag                         # samples per symbol
    bin_hz = bw / N                              # carrier resolution
    found.sort(key=lambda x: (-x['pmr_db']))     # strongest first
    kept = []
    for cand in found:
        is_dup = False
        for k in kept:
            if (abs(cand['time_sample'] - k['time_sample']) < 16 * sym_time_samp
                    and abs(cand['offset_hz'] - k['offset_hz']) < bin_hz):
                is_dup = True
                break
        if not is_dup:
            kept.append(cand)
    kept.sort(key=lambda x: x['time_sample'])
    return kept


def hybrid_recenter(iq, fs, sf, bw):
    """Single-preamble convenience wrapper around find_all_preambles —
    returns (offset_hz, pmr_db, status, preamble_sample) for the highest-
    PMR preamble in the buffer.  Kept for backward compatibility; new
    code should call find_all_preambles directly when multi-signal
    recordings matter.

    Find the true carrier offset of a LoRa preamble inside a wideband IQ
    capture.  Returns (offset_hz, pmr_db, status, preamble_sample) where
    status is one of 'LOCK', 'WEAK', 'NO_LOCK', and preamble_sample is the
    approximate sample index of the preamble start (from the SC time-
    localise step) so callers can compute symbol indices for blanking.

    Algorithm (subagent-validated against 6 real captures, 6/6 within 2 kHz
    of ground truth at ~45 ms per call):
      1. SC time-localise: one-symbol-lag autocorrelation across the whole
         wideband buffer to find where the preamble is in TIME (frequency-
         agnostic — chirps have constant magnitude so the autocorrelation
         is bright wherever there's a preamble, regardless of carrier).
      2. Coarse frequency grid: at the located time slice, try f0 in
         {-BW, -BW/2, 0, +BW/2, +BW}, dechirp at each, pick the f0 with
         the highest PMR.  Five candidates cover ±BW with overlap.
      3. Sub-bin refine: at the chosen f0, the dechirp peak bin gives a
         BW/N (~3.9 kHz) resolution offset.
      4. Alias resolve: re-dechirp at f_raw and at f_raw ± BW, ± 2BW;
         pick the candidate with highest PMR and smallest residual peak
         bin.  Cyclic dechirp aliases mean the true signal may report at
         any of these; preferring the smallest |f| picks the canonical
         answer that the downstream FFT crop will actually keep.
      5. Confidence gate: PMR ≥ 15 dB AND |pb_residual| ≤ 2 → LOCK;
         PMR 10-15 dB → WEAK; anything lower → NO_LOCK.  The recorder
         must NOT save junk when NO_LOCK — fall back or skip instead.
    """
    preambles = find_all_preambles(iq, fs, sf, bw, top_k=1)
    if not preambles:
        return 0.0, 0.0, 'NO_LOCK', 0
    p = max(preambles, key=lambda x: x['pmr_db'])
    return (p['offset_hz'], p['pmr_db'], p['status'], p['time_sample'])


# Per-peak dechirp memo (UC audit C, measured): the resolver evaluates the
# SAME nb slice under the same (sf, bw, agg) up to several times per peak —
# primary vote, positions loop, alias-chain re-checks — 657/2108 calls in the
# busy bench were byte-identical (dechirp_peak_quality = 39.5 s of 102.4 s
# total serial runtime).  The memo key hashes the actual slice CONTENT, so a
# hit is byte-exact by construction; the dict lives in a thread-local set up
# per peak (each pool worker / per-peak thread gets its own), so nothing
# leaks across windows or peaks.  Hashing costs ~0.1-1 ms vs 10-40 ms per
# avoided dechirp.
_DPQ_MEMO = threading.local()


@_pp_time('dechirp','n_dechirp')
def dechirp_peak_quality(iq, sf, bw, agg='best8'):
    _memo = getattr(_DPQ_MEMO, 'd', None)
    if _memo is None:
        return _dechirp_peak_quality_core(iq, sf, bw, agg)
    # GUARD: the pointer key below is exact ONLY for contiguous complex64
    # buffers — nbytes is stride-blind and ctypes.data is the base address, so
    # a strided view (nb[::2]) or a dtype-reinterpret view COLLIDES with a
    # contiguous slice's key while holding different bytes (reproduced: stale
    # 20.8dB/bin0 served for a strided view whose true result is 7.6dB/bin16).
    # Route anything the key is not exact for to the unmemoized core (~100ns
    # check; branch unreachable from all current callers — every call site
    # passes fresh contiguous complex64 basic slices).  NOT covered: in-place
    # mutation of a pinned buffer within one peak's memo lifetime — do not
    # mutate nb arrays inside _process_one_peak.
    if not (iq.flags.c_contiguous and iq.dtype == np.complex64):
        return _dechirp_peak_quality_core(iq, sf, bw, agg)
    # EXACT O(1) memo key: buffer start address + byte length.  The nb slice
    # handed in is always a contiguous view into a base array the caller pins
    # alive for the whole peak (nb_cache / _nb_by_bw dicts) and the memo is
    # reset per peak (_process_one_peak wrapper), so within one memo's lifetime
    # an address is NEVER reused by a different array.  Storing iq in the memo
    # VALUE additionally pins the buffer for the entry's lifetime, making that
    # invariant self-enforcing (a future refactor that stopped caching nb still
    # could not alias).  (data_ptr, nbytes) == identical bytes by construction:
    # no hashing, no subsample-collision risk.  Verified: 0 in-scope collisions
    # over 32 052 real dechirp calls across 5 corpus files (g62, two60,
    # g500_sf11, g42_sf12, gsf12_500k); the 2131 cross-peak address reuses seen
    # are all harmless because the memo is cleared between peaks.
    _mk = (sf, bw, agg, iq.nbytes, iq.ctypes.data)
    _hit = _memo.get(_mk)
    if _hit is None:
        _hit = (_dechirp_peak_quality_core(iq, sf, bw, agg), iq)
        _memo[_mk] = _hit
    return _hit[0]


def _dechirp_peak_quality_core(iq, sf, bw, agg='best8'):
    """Measure dechirp quality and carrier frequency offset.

    Returns (quality_dB, peak_bin) where:
      quality_dB — peak-to-mean ratio of the dechirped spectrum (dB).
      peak_bin   — FFT bin of the averaged preamble spectrum [0, N).
                   Converts to CFO: if bin <= N/2: cfo = +bin*BW/N
                                    if bin >  N/2: cfo = -(N-bin)*BW/N
    The Welch energy-scan estimate can be off by ±BW/2 for short packets.
    Using peak_bin * BW/N corrects that to ±BW/(2N) ≈ ±4kHz for SF7/500k.
    """
    fs = bw * 2
    N = 2 ** sf
    osf = int(round(fs / bw))
    sym_samples = N * osf
    if len(iq) < sym_samples: return -99.0, 0
    dc = generate_downchirp(sf, bw, fs)
    n_total_syms = len(iq) // sym_samples
    if n_total_syms < 1: return -99.0, 0

    # Vectorized: reshape into symbols, dechirp all at once, batch FFT.
    # Document step 2: "resampled at the chirp rate, normalizing to exactly
    # 2^SF samples."  At 2×BW rate each symbol is 2N samples.  We must
    # DECIMATE (average pairs) to N samples before the N-point FFT — not
    # truncate.  Truncating discards the second half of every symbol,
    # losing ~3 dB of coherent gain vs. proper decimation.
    iq_syms = iq[:n_total_syms * sym_samples].reshape(n_total_syms, sym_samples)
    dechirped = iq_syms * dc
    if osf > 1:
        # Proper decimation: average osf consecutive samples → N samples/symbol.
        # ndarray.mean(axis=2) over the tiny inner axis hits numpy's generic
        # reduce loop and is ~15x slower than an explicit add for the osf==2
        # production case (measured 85% of this function's compute).  osf==2 is
        # the ONLY value on this path (fs=2*bw here) and (a+b)*0.5 == (a+b)/2 is
        # BIT-identical to mean in IEEE754.  osf!=2 (only reachable from the
        # other two dechirp sites) keeps mean() so it stays bit-identical too —
        # sum/osf rounds ~1e-7 differently for ODD osf.
        _r = dechirped.reshape(n_total_syms, N, osf)
        if osf == 2:
            dechirped = (_r[:, :, 0] + _r[:, :, 1]) * np.complex64(0.5)
        else:
            dechirped = _r.mean(axis=2)
    spectra = np.abs(_fft(dechirped, axis=1)) ** 2
    peaks = np.max(spectra, axis=1)
    means = np.mean(spectra, axis=1)
    all_ratios = np.where(means > 0, peaks / means, 0)

    if agg == 'median':
        # WHOLE-SLICE consistency metric for (sf, bw) IDENTITY
        # arbitration: the true hypothesis dechirps cleanly on preamble
        # AND payload, while a same-slope alias is clean ONLY on the
        # preamble (its windows straddle two payload symbols and split).
        # best8 would let the alias hide in its clean preamble stretch;
        # the median exposes it. CFO bin from the strongest window's
        # spectrum, same as below.
        _med = float(np.median(all_ratios))
        _bi = int(np.argmax(all_ratios))
        peak_bin = int(np.argmax(spectra[_bi]))
        return 10.0 * np.log10(_med + 1e-15), peak_bin

    # Quality metric: best 8-symbol window by sum of per-symbol PMRs.
    win = min(8, n_total_syms)
    if n_total_syms <= win:
        best_start = 0
        best_avg = np.mean(all_ratios)
        avg_spec = np.mean(spectra, axis=0)
    else:
        cs = np.cumsum(all_ratios)
        sums = cs[win - 1:] - np.concatenate(([0], cs[:-win]))
        best_start = int(np.argmax(sums))
        best_avg = sums[best_start] / win

        # CFO bin from the BEST (preamble) window's AVERAGED spectrum.  The
        # preamble's same-bin upchirps add to a sharp peak at the CFO bin, so
        # the averaged spectrum over just those symbols gives a stable, accurate
        # carrier.  The OLD global per-symbol-argmax vote (across ALL symbols,
        # incl. random-bin payload + noise) let the payload and sub-bin preamble
        # spread corrupt the estimate → the carrier scattered ±BW/2 run-to-run,
        # producing off-centre captures (slow recovery / decrypt-fail garbage /
        # lost hops in live).  Averaging the preamble window fixes that.
        avg_spec = np.mean(spectra[best_start:best_start + win], axis=0)

    peak_bin = int(np.argmax(avg_spec))

    return 10.0 * np.log10(best_avg + 1e-15), peak_bin


@_pp_time('dechirp','n_dechirp')
def _dechirp_scan(nb, sf, bw, nbfs):
    """Dechirp matched-filter preamble detection for a single channel.

    Scans sub-symbol offsets; at each, dechirps the channel into symbols and finds the
    best 8-symbol window — the preamble's identical upchirps all dechirp to the SAME
    FFT bin, so they sum to a sharp peak.  The detection metric is that despread
    peak-to-mean ratio (full SF processing gain), which detects well BELOW where
    Schmidl-Cox and the energy gate go deaf.  Mirrors dechirp_peak_quality's dechirp
    exactly, but also returns the preamble TIMING so a capture can be cut.

    Returns (quality_db, preamble_t_s_from_channel_start, peak_bin) for the best
    offset, or None.  nb must be the channel at nbfs ≈ 2*bw.
    """
    N = 2 ** sf
    osf = max(1, int(round(nbfs / bw)))
    sym = N * osf
    if len(nb) < 8 * sym:
        return None
    dc = generate_downchirp(sf, bw, nbfs)
    best = None
    for off in range(0, sym, max(1, sym // 8)):
        seg = nb[off:]
        ns = len(seg) // sym
        if ns < 8:
            continue
        d = seg[:ns * sym].reshape(ns, sym) * dc
        if osf > 1:
            _r = d.reshape(ns, N, osf)            # slice-add: see line ~1466
            d = ((_r[:, :, 0] + _r[:, :, 1]) * np.complex64(0.5)
                 if osf == 2 else _r.mean(axis=2))
        spec = np.abs(_fft(d, axis=1)) ** 2
        mn = spec.mean(axis=1)
        ratios = np.where(mn > 0, spec.max(axis=1) / mn, 0.0)
        cs = np.cumsum(ratios)
        sums = cs[7:] - np.concatenate(([0.0], cs[:-8]))
        bs = int(np.argmax(sums))
        q = 10.0 * np.log10(sums[bs] / 8.0 + 1e-15)
        if best is None or q > best[0]:
            pbin = int(np.argmax(spec[bs:bs + 8].mean(axis=0)))
            best = (q, (off + bs * sym) / float(nbfs), pbin)
    return best


def find_preamble_bin(iq, sf, bw, fs=None):
    """Find the preamble carrier frequency offset.
    
    Returns (lora_bin, cfo_hz):
      - lora_bin: LoRa bin 0..N-1
      - cfo_hz:   CFO in Hz (bin * BW/N, wrapped to ±BW/2)
    
    FFT-crops to 1×BW then dechirps — matching the decoder and
    analysis script exactly.  LoRa chirps wrap cyclically at 1x rate
    so the dechirp bin is correct regardless of how far off-center
    the signal is in the extraction.
    """
    if fs is None:
        fs = bw * 2
    N = 2 ** sf
    
    # FFT crop to 1×BW (matching decoder lines 624-628)
    dec = int(round(fs / bw))
    if dec > 1:
        F = np.fft.fftshift(_fft(iq))
        keep = len(F) // dec
        start_f = len(F) // 2 - keep // 2
        iq = _ifft(np.fft.ifftshift(
            F[start_f:start_f + keep])) * (keep / len(F))
        iq = iq.astype(np.complex64)
    
    # Dechirp at 1x (matching decoder lines 630-633, 661-664)
    n_syms = len(iq) // N
    if n_syms < 4:
        return 0, 0.0
    
    t = np.arange(N, dtype=np.float64) / bw
    Ts = N / bw
    downchirp = np.conj(np.exp(1j * np.pi * bw / Ts * t**2)).astype(
        np.complex64)
    
    iq_syms = iq[:n_syms * N].reshape(n_syms, N)
    dechirped = iq_syms * downchirp
    spectra = np.abs(_fft(dechirped, axis=1)) ** 2
    
    # Find preamble bin via weighted vote histogram (same approach as
    # dechirp_peak_quality).  Preamble symbols all vote for bin K;
    # payload and noise scatter uniformly.
    peaks = np.max(spectra, axis=1)
    means = np.mean(spectra, axis=1)
    ratios = np.where(means > 0, peaks / means, 0)
    _best_bins = np.argmax(spectra, axis=1)
    _hist = np.zeros(N, dtype=np.float64)
    np.add.at(_hist, _best_bins, ratios)
    avg_spectrum = _hist
    best_start = 0  # unused below, kept for peak_snr reference
    peak_bin = int(np.argmax(avg_spectrum))
    peak_snr = 10 * np.log10(avg_spectrum[peak_bin] /
                              (np.mean(avg_spectrum) + 1e-15) + 1e-15)
    
    # Hz: bin * BW/N without wrapping.
    # LoRa bins are cyclic mod N. The ambiguity (bin*BW/N vs (bin-N)*BW/N)
    # is resolved by choosing the value closest to +BW/2, since the CNN
    # typically detects the lower edge of the chirp and the signal center
    # is ~BW/2 above.
    pos_hz = float(peak_bin * bw / N)
    neg_hz = float((peak_bin - N) * bw / N)
    expected_offset = bw / 2.0  # CNN at lower edge → signal ~+BW/2 above
    if abs(pos_hz - expected_offset) <= abs(neg_hz - expected_offset):
        cfo_hz = pos_hz
    else:
        cfo_hz = neg_hz
    
    print(f"    find_preamble: bin={peak_bin} snr={peak_snr:.1f}dB "
          f"cfo={cfo_hz/1000:+.1f}kHz window={best_start}..{best_start+win-1}",
          file=sys.stderr, flush=True)
    
    return peak_bin, cfo_hz


# ---- Schmidl-Cox preamble detection ----

# At fs=1Msps each Meshtastic preset maps to one lag (samples = 2^SF/BW × 1e6).
# Ambiguous lags share a symbol duration across two different SF×BW presets;
# dechirp quality resolves which one is present.
_SC_LAGS_1M = {
    # Curated SF×BW set — covers all Meshtastic + MeshCore presets AND every
    # SF7-11 / 62.5k combination (added so MeshCore@62.5k traffic is detected
    # by default — see issue #2).  SF11/500, SF12/500, SF12/250 are deliberately
    # omitted from this default set (see notes below).  Same-lag candidates are
    # resolved by dechirp quality in _resolve_sf_bw_ambig.
    256:   [(7, 500000)],
    512:   [(7, 250000), (8, 500000)],
    1024:  [(7, 125000), (8, 250000), (9, 500000)],
    2048:  [(7, 62500),  (8, 125000), (9, 250000), (10, 500000)],
    # NOTE: SF11/500 (LONG_TURBO) is ALSO lag 4096, but it is deliberately NOT
    # listed here: these radios transmit LONG_TURBO as SF11/250 on-air (verified
    # by measurement — config reads bw=500 but the signal is ~250 kHz), so no
    # real SF11/500 traffic exists, and adding it made every SF9/250 signal's
    # 2-symbol-lag periodicity mis-resolve to SF11/500 → ~386 spurious
    # detections per MEDIUM_FAST run (493 -> 878), bloating decode time. If a
    # genuine SF11/500 source is ever needed, re-add it AND tighten the dechirp
    # resolver so SF9 2-symbol content can't pass as SF11/500.
    # SF11/500 RE-ADDED 2026-07-05: ground-truthed on the CURRENT MeshCore
    # radio (RAK4631) — it emits GENUINE SF11/500k (dechirp 30.2 dB at
    # (11,500k) vs 4-9 dB at every alternative). The old 'transmits as
    # ~250 kHz on-air' measurement was the earlier hardware; the historical
    # mis-resolution fear (SF9/250 2-symbol harmonics) is addressed by the
    # matched-primary grouped resolver + peak-width decider, as with
    # SF12/500k. Soak evidence: 0/10 catch while omitted.
    4096:  [(7, 31250),  (8, 62500),  (9, 125000), (10, 250000),
            (11, 500000)],
    # SF12/500 omitted here for the same architectural reason as SF11/500: it
    # would share lag 8192 with SF10/125 + SF11/250, and the resolver may
    # mis-vote on 2-symbol harmonics of those fundamentals.  Re-add only after
    # the resolver is hardened against same-lag fundamental-vs-harmonic confusion.
    # SF12/500 RE-ADDED 2026-07-04: the historic omission ('resolver may
    # mis-vote on same-lag harmonics') is addressed by the matched-primary
    # resolver (grouped same-slope votes, half-lag candidate chains,
    # bin-normalized within-group quality). Ground-truthed against a real
    # MC beacon at SF12/500k config: the radio emits GENUINE SF12/500k
    # (dechirp 35.0 dB at (12,500) vs 7-9.5 dB at every alternative —
    # unlike SF11/500, which these radios transmit as ~250 kHz on-air).
    # Without this entry, hits resolved to (10,250)-family junk with
    # scattered carriers and zero decodes (measured live).
    8192:  [(8, 31250),  (9, 62500),  (10, 125000), (11, 250000), (12, 500000)],
    # SF12/250 omitted on the same harmonic-confusion grounds (would share
    # lag 16384 with SF11/125).
    # SF12/250 RE-ADDED 2026-07-05: ground-truthed genuine (34.4 dB at
    # (12,250k) vs <=9.5 at alternatives); no same-slope partner in this
    # family. Soak evidence: 0/5 catch while omitted.
    16384: [(9, 31250),  (10, 62500), (11, 125000), (12, 250000)],
    32768: [(10, 31250), (11, 62500), (12, 125000)],
    65536: [(11, 31250), (12, 62500)],
    # 31.25 kHz family ADDED 2026-07-04: was entirely absent (the rig's own
    # sweep transmits 30 beacons/cycle here, all invisible). Same-slope
    # partners (sf+2, 62.5k) exist for every entry — the matched-primary
    # grouped resolver is the prerequisite hardening, as with SF12/500.
    131072: [(12, 31250)],
    # 41.67 kHz LoRa (Semtech-defined BW=0110) — used by long-range hobby
    # deployments + satellite TT&C (e.g. tinyGS).  Each SF lands on its own
    # unique lag bucket (no collision with the Meshtastic/MeshCore-focused
    # buckets above), so adding these costs only the per-bucket SC compute
    # for the new lags and creates ZERO risk of harmonic mis-resolution.
    # Added in response to Issue #4 — K4KDR reported 41.7k signals visible
    # on waterfall but no decodes; root cause was this curated table not
    # including 41.67 kHz combinations.
    3072:  [(7, 41667)],
    6144:  [(8, 41667)],
    12288: [(9, 41667)],
    24576: [(10, 41667)],
    49152: [(11, 41667)],
    98303: [(12, 41667)],
}
if os.environ.get('LORA_SCAN_FULL'):
    # General LoRa scan (opt-in): add EVERY SF×BW combination to its Schmidl-Cox
    # lag bucket so non-Meshtastic configs the curated table omits (e.g. SF7/125k,
    # SF9/500k, SF12/31.25k) can be detected.  TRADE-OFF: this stacks more
    # (SF,BW) candidates onto the common ambiguous lags (2048/4096/8192/16384),
    # which the dechirp resolver can mis-vote — exactly the failure noted above
    # for SF11/500 (caused ~386 spurious detections in a MEDIUM_FAST run).  So it
    # widens coverage at the cost of more spurious/mis-resolved detections and
    # extra decode work.  Off by default; the web "Wide LoRa scan" setting sets it.
    for _sf in range(7, 13):
        for _bw in BW_LIST:
            _lag = int(round((2 ** _sf) / _bw * 1e6))
            _SC_LAGS_1M.setdefault(_lag, [])
            if (_sf, _bw) not in _SC_LAGS_1M[_lag]:
                _SC_LAGS_1M[_lag].append((_sf, _bw))
_ALL_LAGS_1M = sorted(_SC_LAGS_1M.keys())


@_pp_time('sc','n_sc')
def _schmidl_cox_curve_multilag(iq_1m, lags, n_sym=8):
    """Compute SC curves for many lags on the same signal in one pass.

    Per-peak profiling during a B+C burst load showed `_schmidl_cox_curve`
    dominated detect_preamble's per-peak cost at 130-170 ms / peak (~30%
    of total), almost all of it spent on redundant energy sums across the
    15 lags in the default scan.  Each lag recomputed |iq|^2 windowed
    energy from scratch even though every lag operates on the same
    downconverted 1Msps signal.

    Strategy: compute |iq|^2 cumulative-sum ONCE per signal.  Then for
    ANY lag L the windowed energy E[k] = cum[(k+1)*L] - cum[k*L] is two
    O(1) array lookups.  Correlation P = sum(iq_b * conj(iq_a)) still
    needs per-lag computation (it depends on the lag-shift) but the
    energy portion (~60% of per-lag work) is now amortised across all
    lags.  Measured 7.48x speedup vs the per-lag loop on a synthetic
    500k-sample × 8-lag benchmark, max relative diff 5e-8 vs original
    (well inside float32 precision).

    Returns dict {lag: curve_or_None}.  The single-lag
    `_schmidl_cox_curve(iq, lag)` below stays available for callers that
    only need one lag (e.g. find_all_preambles).
    """
    N = len(iq_1m)
    e_pow = np.abs(iq_1m) ** 2
    cum = np.empty(N + 1, dtype=np.float64)
    cum[0] = 0.0
    np.cumsum(e_pow, out=cum[1:])

    out = {}
    for L in lags:
        if N < L * (n_sym + 2):
            out[L] = None
            continue
        n_wins = (N - L) // L
        if n_wins < n_sym:
            out[L] = None
            continue
        n_full = n_wins * L
        # Correlation must still be per-lag (depends on shift L) — but the
        # old one-shot `np.sum(iq_b * np.conj(iq_a), axis=1)` materialized
        # two full-signal complex64 temporaries per lag (~48 B/element of
        # DRAM traffic; 400-500 MB/peak across the lag scan).  On a
        # memory-bandwidth-bound host (Pi 4: ~4 GB/s, 1 MB shared L2) that
        # alone was ~100-125 ms/peak — the busy-wall profile's #1 ARM-
        # hostile stage.  Stream it through an L2-resident scratch instead
        # (~16 B/element): conj+multiply in place over window-aligned
        # chunks, reduce per window.  Per-call scratch keeps the >=8-core
        # per-peak threadpool safe (np.empty, no zero-fill — cheap).
        # Float accumulation order changes at the chunk seams; difference
        # is float32-rounding-level, same class as the 5e-8 the multilag
        # rewrite itself was validated at.
        flat_a = iq_1m[:n_full]
        flat_b = iq_1m[L:L + n_full]
        P = np.empty(n_wins, dtype=np.complex64)
        _CH = 65536
        _scr = np.empty(min(_CH, max(L, 1)), dtype=np.complex64) \
            if L > _CH else np.empty(min(_CH // L, n_wins) * L,
                                     dtype=np.complex64)
        if L <= _CH:
            rows = max(1, _CH // L)
            for w0 in range(0, n_wins, rows):
                r = min(rows, n_wins - w0)
                seg = r * L
                off = w0 * L
                t = _scr[:seg]
                np.conjugate(flat_a[off:off + seg], out=t)
                np.multiply(t, flat_b[off:off + seg], out=t)
                P[w0:w0 + r] = t.reshape(r, L).sum(axis=1)
        else:
            for w in range(n_wins):
                acc = 0.0 + 0.0j
                base = w * L
                for off in range(0, L, _CH):
                    m = min(_CH, L - off)
                    t = _scr[:m]
                    np.conjugate(flat_a[base + off:base + off + m], out=t)
                    np.multiply(t, flat_b[base + off:base + off + m], out=t)
                    acc += complex(t.sum())
                P[w] = acc
        # Energy from precomputed cumsum — O(1) lookups per window.
        a_starts = np.arange(n_wins, dtype=np.int64) * L
        b_starts = a_starts + L
        E_a = cum[a_starts + L] - cum[a_starts]
        E_b = cum[b_starts + L] - cum[b_starts]
        s = np.abs(P) / (np.sqrt(E_a * E_b) + 1e-15)
        out[L] = np.convolve(s, np.ones(n_sym) / n_sym, mode='valid')
    return out


def _schmidl_cox_curve(iq_1m, lag, n_sym=8):
    """Compute the Schmidl-Cox sliding-mean curve (mean_s) for one lag at 1Msps.

    During the LoRa preamble (16 consecutive same-bin upchirps), each window of
    `lag` samples is a phase-rotated copy of the previous one, so
    |auto_corr|/energy → 1.0; data symbols use random bins → ≈ 0.  The sliding
    mean over n_sym windows shows a plateau during the preamble.  Returns the
    curve (np.ndarray) or None if the signal is too short.  Shared by
    schmidl_cox_score (argmax) and schmidl_cox_peaks (NMS) so the gate computes
    it ONCE per lag instead of twice (score then peaks recomputed the same math).
    """
    L = lag
    N = len(iq_1m)
    if N < L * (n_sym + 2):
        return None
    n_wins = (N - L) // L
    if n_wins < n_sym:
        return None
    n_full = n_wins * L
    iq_a = iq_1m[:n_full].reshape(n_wins, L)            # windows 0…n-1
    iq_b = iq_1m[L:L + n_full].reshape(n_wins, L)       # windows 1…n (lag-shifted)
    P   = np.sum(iq_b * np.conj(iq_a), axis=1)          # (n_wins,) correlation
    E_a = np.sum(np.abs(iq_a) ** 2, axis=1)             # (n_wins,) energy of a
    E_b = np.sum(np.abs(iq_b) ** 2, axis=1)             # (n_wins,) energy of b
    # Bounded [0,1] by Cauchy-Schwarz: |P|^2 <= E_a * E_b
    s = np.abs(P) / (np.sqrt(E_a * E_b) + 1e-15)       # per-window score 0…1
    # Sliding mean over n_sym windows → plateau visible during preamble
    return np.convolve(s, np.ones(n_sym) / n_sym, mode='valid')


@_pp_time('sc','n_sc')
def schmidl_cox_score(iq_1m, lag, n_sym=8, _curve=None):
    """Schmidl-Cox score for one lag: argmax of the SC sliding-mean curve.
    Returns (score, preamble_sample_idx) where score ∈ [0, 1].
    `_curve` reuses a precomputed curve from _schmidl_cox_curve (gate caching)."""
    mean_s = _curve if _curve is not None else _schmidl_cox_curve(iq_1m, lag, n_sym)
    if mean_s is None:
        return 0.0, 0
    best_idx = int(np.argmax(mean_s))
    return float(mean_s[best_idx]), best_idx * lag


@_pp_time('sc','n_sc')
def schmidl_cox_peaks(iq_1m, lag, n_sym=8, thr=0.5, max_peaks=6, min_sep_win=16,
                      _curve=None):
    """Like schmidl_cox_score, but returns ALL time-separated preamble plateaus
    above `thr` — not just the single strongest.

    A capture/window often holds more than one packet at the SAME carrier and
    SF/BW: a relay hop1 ~0.5 s after hop0, or simply another packet shortly
    after the first.  schmidl_cox_score takes argmax of the sliding-mean SC
    curve, so it sees only the stronger/earlier plateau and the later packet is
    never detected → never captured → lost (this was the last live miss root).
    Here we non-max-suppress the SC curve: take the top plateau, blank ±min_sep
    windows around it, repeat — so every distinct preamble in the window gets
    its own (score, sample_pos).  Returns a list sorted by score desc.
    min_sep_win (≈ one preamble length in symbol-windows) keeps us from
    splitting a single plateau while still resolving back-to-back packets.
    `_curve` reuses a precomputed curve (gate caching) — avoids recomputing the
    same SC math that schmidl_cox_score already did for this lag."""
    mean_s = _curve if _curve is not None else _schmidl_cox_curve(iq_1m, lag, n_sym)
    if mean_s is None:
        return []
    ms = mean_s.copy()
    out = []
    for _ in range(max_peaks):
        i = int(np.argmax(ms))
        if ms[i] < thr:
            break
        out.append((float(ms[i]), i * lag))
        lo = max(0, i - min_sep_win)
        hi = min(len(ms), i + min_sep_win + 1)
        ms[lo:hi] = -1.0
    return out


def schmidl_cox_multi(iq_1m, lags=None, n_sym=8):
    """Run Schmidl-Cox for all lags.  Returns dict {lag: (score, sample_pos)}."""
    if lags is None:
        lags = _ALL_LAGS_1M
    return {lag: schmidl_cox_score(iq_1m, lag, n_sym) for lag in lags}


def _resolve_sf_bw_ambig(iq_1m, candidates, fft_cache=None, pos=None,
                         nb_cache=None, pos_map=None, width_hz=None,
                         off_hz=0.0):
    """Pick best (sf, bw) from ambiguous Schmidl-Cox candidates via dechirp quality.
    fft_cache: reuses a single forward FFT across candidates with different BWs
    (same buffer, different narrowband extraction).  ~25 % per-call speedup when
    a bucket has 3+ candidates.
    pos: optional SC preamble position (samples at 1 Msps).  When given, the
    dechirp vote runs on a 16-symbol slice AT the preamble instead of the whole
    1 s buffer — the same alignment the global resolver already applies, and
    the quality of THIS preamble (not of whatever else shares the buffer) is
    what the vote is supposed to measure.  ~4-60x less dechirp work per
    candidate depending on SF.
    nb_cache: optional {bw: (nb, fs)} shared with the CALLER's per-peak loop.
    The narrowband extraction depends only on bw — without the shared cache a
    peak with 4 hit lags extracted the same 125/250/500 kHz slices up to 3x
    (once per resolver call, once again in the positions loop): the full-length
    crop+IFFT per extraction was the dominant resolver cost on low-core hosts,
    not the dechirp vote."""
    best_sf, best_bw, best_q = candidates[0][0], candidates[0][1], -99.0
    _quals = []
    for sf, bw in candidates:
        if nb_cache is not None and bw in nb_cache:
            nb = nb_cache[bw][0]
        else:
            # off_hz: occupied-band centroid measured by the caller — see
            # the 2026-07-08 note at the hit_lags loop (PSD nomination can
            # sit ±BW/2 off-carrier; a mis-centred crop halves the chirp
            # and collapses the dechirp vote to noise level).
            nb, _fs = extract_narrowband_fft(iq_1m, 1_000_000, off_hz, bw,
                                             fft_cache=fft_cache)
            if nb_cache is not None:
                nb_cache[bw] = (nb, _fs)
        nb_use = nb
        _pos_c = (pos_map or {}).get((sf, bw), pos)
        if _pos_c:
            _p = int(round(_pos_c * (bw * 2) / 1_000_000.0))
            if 0 < _p < len(nb) - (2 ** sf) * 4:
                # POSITION-ROBUST slice (2026-07-08): the old 16-sym slice
                # assumed the SC plateau position lands ON the preamble,
                # but plateau centers sit up to ~50 ms off the preamble
                # start (lag-dependent skew; measured on the MC-SF7/125k
                # ground-truth recording).  A candidate evaluated on the
                # wrong 16 syms dechirps at junk level (3-7 dB vs 19-22 dB
                # at the right spot), so NO candidate reached
                # DECHIRP_MIN_DB, rule 1 emptied _viable, and the vote
                # fell back to the raw argmax — whose +3 dB/SF-step
                # bin-count bias then crowned same-slope harmonics
                # (17 of 24 windows classified SF11/500k+SF12/500k on a
                # pure SF7/125k burst train → captures saved with wrong
                # SF/BW in the fname → 2/12 frames decoded).  Widen the
                # slice to [-16, +112] symbols around the plateau:
                # dechirp_peak_quality's internal best-8 run finds the
                # preamble anywhere inside, restoring sane measurements
                # for every candidate at bounded cost (still sliced —
                # ~13% of the buffer at SF7, whole-buffer at SF12 which
                # simply matches the pre-optimization behaviour).
                _n2 = (2 ** sf) * 2
                _s0 = max(0, _p - 16 * _n2)
                nb_use = nb[_s0:_p + 112 * _n2]
        if len(nb_use) < (2 ** sf) * 4:
            continue
        q, _ = dechirp_peak_quality(nb_use, sf, bw)
        _quals.append((q, sf, bw))
        if q > best_q:
            best_q, best_sf, best_bw = q, sf, bw
    # CROSS-SF COMPARABILITY: dechirp PMR is peak-to-mean over 2^sf bins,
    # so a higher-SF hypothesis gets a systematic +3 dB/SF-step advantage
    # on the SAME tone (more bins -> lower mean noise per bin). That bias
    # is exactly why same-slope aliases (sf+2, 2x bw — mathematically
    # degenerate under dechirp on preamble content) kept WINNING the raw
    # vote. Compare on bin-count-normalized quality; the winner's
    # REPORTED quality stays raw (the DECHIRP_MIN_DB gate downstream
    # keeps its meaning).
    if len(_quals) > 1:
        import math as _m
        # IDENTITY DECISION, three rules (each earned the hard way):
        # 1. Only candidates that could actually be emitted compete —
        #    raw best-8 q >= DECHIRP_MIN_DB. (Normalizing junk-level
        #    scores inverts votes toward low SF; the winner then fails
        #    the gate and a REAL detection dies silently.)
        # 2. Same-slope groups — identical bw^2/2^sf, where dechirp on
        #    preambles is mathematically near-degenerate and PMR's
        #    bin-count bias (+3 dB per SF step) makes the (sf+2, 2bw)
        #    alias win RAW votes — are decided WITHIN the group by
        #    bin-normalized quality (measured: picks the truth by 1-6 dB
        #    on the injected SF9/125 control and a real 75 dB beacon).
        # 3. ACROSS groups raw best-8 decides (different slopes separate
        #    by ~12 dB raw; global normalization would wreck this).
        _viable = [t for t in _quals if t[0] >= DECHIRP_MIN_DB]
        if not _viable:
            # NO candidate dechirps like LoRa (2026-07-08): a real preamble
            # measures >= DECHIRP_MIN_DB at its own (sf, bw) here; when
            # EVERY hypothesis sits at noise level the content is not a
            # LoRa preamble at any of them — reject the peak instead of
            # returning the raw argmax, whose +3 dB/SF-step bin-count bias
            # crowns the widest/highest-SF candidate.  Ground truth: the
            # HackRF ±5 MHz CW birdies SC-hit at EVERY lag (0.74-0.97 flat
            # across four octaves — CW autocorrelates at any lag), resolved
            # to SF11/SF12/500k whose ±500 kHz crops swallowed a REAL
            # neighbouring beacon's edge energy (910.525 sits +525 kHz
            # from the 910.000 birdie), squeaked past the bwq gate on that
            # stolen energy, and emitted junk detections at ~910.53 that
            # out-competed the real SF7/125k frame's own detection — the
            # MC-SF7 leg lost ~65% of its frames this way.  The downstream
            # bwq gate still protects the accepted path; this just stops
            # known-junk from being handed a preset label.
            if os.environ.get('LORA_RESOLVE_DEBUG'):
                print(f"    RESOLVE: all {len(_quals)} candidates < "
                      f"{DECHIRP_MIN_DB} dB — peak rejected (no viable)",
                      file=sys.stderr)
            return None, None, best_q
        if _viable:
            _groups = {}
            for q, sf, bw in _viable:
                _key = round(bw * bw / (2.0 ** sf) / 1e3)   # slope, kHz^2/s-ish
                _groups.setdefault(_key, []).append((q, sf, bw))
            _winners = []
            for _g in _groups.values():
                _g.sort(key=lambda t: t[0] - 10.0 * _m.log10(2.0 ** t[1]),
                        reverse=True)
                if width_hz and width_hz >= 20000 and len(_g) > 1:
                    # width gate (2026-07-08): only trust the width
                    # tie-break when the measurement is physically
                    # plausible for LoRa (narrowest preset 31.25 kHz;
                    # skirt-narrowed measurements bottom out ~0.6x bw).
                    # The MC-SF7/125k ground truth measured w=0.2-0.5 kHz
                    # — a junk contour — while the position-robust
                    # normalized order (4+ dB separation within the
                    # same-slope group) picks the truth; let it.
                    # same-slope members are dechirp- AND SC-degenerate BY
                    # CONSTRUCTION (identical bw^2/2^sf) — their normalized
                    # qualities differ only by noise, so no q-window: the
                    # PEAK-RELATIVE occupied width (strength-invariant,
                    # passed only when measurable at >= +18 dB) decides the
                    # WHOLE group. Absent a trustworthy width, the
                    # normalized order stands.
                    _gw = sorted(_g, key=lambda t: abs(
                        _m.log2(t[2] / width_hz)))
                    _winners.append(_gw[0])
                    continue
                _winners.append(_g[0])
            _winners.sort(reverse=True)          # raw q across groups
            best_q, best_sf, best_bw = _winners[0]
        if os.environ.get('LORA_RESOLVE_DEBUG'):
            print(f"    RESOLVE(w={width_hz and round(width_hz/1e3,1)}k): "
                  + " | ".join(f"SF{sf}/{bw/1e3:.0f}k q={q:.1f}"
                               for q, sf, bw in sorted(_quals, reverse=True))
                  + f" -> SF{best_sf}/{best_bw/1e3:.0f}k",
                  file=sys.stderr)
    return best_sf, best_bw, best_q


# Full SF×BW grid → lag map, for the wide-scan global resolver (built once).
_FULL_LAGS = {}
for _sf in range(7, 13):
    for _bw in BW_LIST:
        _l = int(round((2 ** _sf) / _bw * 1e6))
        _FULL_LAGS.setdefault(_l, []).append((_sf, _bw))


def _resolve_sf_bw_global(iq_1m, lag, pos=None, sc_scores=None, fft_cache=None):
    """Wide-scan resolver — resolves (sf,bw) in two robust steps:

      1. Pick the FUNDAMENTAL lag.  A signal autocorrelates at its symbol period
         and at 2×/4× it, so the observed lag may be a harmonic.  Among the lag and
         its power-of-2 divisors, the true fundamental is the one with the strongest
         Schmidl-Cox score (harmonics score lower).  This collapses every harmonic
         detection of one signal onto the SAME lag → they dedup, no spurious.
      2. Dechirp-vote among the candidates ON that lag.  Same-lag (sf,bw) pairs have
         maximally distinct chirp rates (BW 62.5/125/250/500 kHz, each 2× apart) —
         the exact discrimination the validated 2-candidate preset vote relies on,
         just up to 4-way.  SF-normalized (raw quality grows ~10·log10(2^sf)), and
         aligned to the preamble start.  Returns (sf, bw, raw_q) or None."""
    div_lags = []
    L = lag
    while L >= 256:
        if L in _FULL_LAGS:
            div_lags.append(L)
        if L % 2:
            break
        L //= 2
    if not div_lags:
        return None
    if sc_scores is not None:
        fund = max(div_lags, key=lambda l: (sc_scores.get(l) or (-1.0,))[0])
    else:
        fund = div_lags[0]
    nb_by_bw = {}
    best = None
    for sf, bw in _FULL_LAGS[fund]:
        if bw not in nb_by_bw:
            nb_by_bw[bw] = extract_narrowband_fft(iq_1m, 1_000_000, 0.0, bw,
                                                  fft_cache=fft_cache)[0]
        nb = nb_by_bw[bw]
        nb_use = nb
        if pos:
            _p = int(round(pos * (bw * 2) / 1_000_000.0))
            if 0 < _p < len(nb) - (2 ** sf) * 4:
                # 16-symbol slice at the preamble (see _resolve_sf_bw_ambig).
                nb_use = nb[_p:_p + 32 * (2 ** sf)]
        if len(nb_use) < (2 ** sf) * 4:
            continue
        q, _ = dechirp_peak_quality(nb_use, sf, bw)
        qn = q - 10.0 * np.log10(2 ** sf)
        if best is None or qn > best[0]:
            best = (qn, sf, bw, q)
    return (best[1], best[2], best[3]) if best else None


def detect_preamble(iq, wb_fs, wb_bw, center_mhz, sc_threshold=0.7,
                    ethresh=8.0, spur_db=SPUR_REJECT_DB,
                    dc_notch_mhz=0.0, spur_notch_hz=None, debug=0,
                    cached_psd=None, cached_peaks=None, dechirp_chans=None,
                    only_bws=None):
    """Schmidl-Cox LoRa preamble detector.  Replaces the CNN detect() pipeline.

    1. Welch PSD → find energy peaks
    2. Per peak: extract 1Msps IQ (covers all Meshtastic BWs ≤ 500kHz)
    3. Multi-lag Schmidl-Cox → score > sc_threshold means preamble present
    4. Unambiguous lags → direct SF+BW.  Ambiguous → dechirp-resolve.
    5. CFO correction from dechirp peak_bin.

    Returns list of detection dicts with the same fields as the old detect():
      freq_hz, freq_mhz, sf, bw, detect_conf, sf_conf, bw_quality_db, peak_power_db

    cached_psd: pre-computed, pre-notched PSD from the main loop gate scan (same
    buffer, same parameters).  When provided, Stage 1 PSD is skipped entirely.
    """
    if os.environ.get('LORA_STUB_DETECT'):
        return []   # diagnostic: measure producer-only throughput
    center_hz = center_mhz * 1e6
    nfft_c = 4096
    fres = wb_fs / nfft_c

    # --- Stage 1: Energy scan ---
    if cached_psd is not None:
        # Reuse the gate PSD already computed in the main loop (same buffer,
        # same notches already applied) — avoids duplicate computation.
        psd = cached_psd
    else:
        # n_avg=50 on a 1s/28Msps buffer: step≈20ms.  A 10ms SF7/500k frame
        # lands in 1 segment → SNR_psd ≈ 11dB above noise floor.  See the
        # matching block in the main loop for the full margin analysis.
        _psd_n = min(len(iq), int(wb_fs))
        psd = welch_psd(iq[-_psd_n:], nfft=nfft_c, n_avg=50)

        if dc_notch_mhz > 0:
            dc_notch_bins = max(1, int(round(dc_notch_mhz * 1e6 / fres)))
            dc_c = nfft_c // 2
            lo = max(0, dc_c - dc_notch_bins)
            hi = min(nfft_c, dc_c + dc_notch_bins + 1)
            psd[lo:hi] = np.median(psd)

        if spur_notch_hz:
            for spur_abs_hz, half_w_hz in spur_notch_hz:
                spur_off = spur_abs_hz - center_hz
                spur_bin = nfft_c // 2 + int(round(spur_off / fres))
                nb_spur = max(1, int(round(half_w_hz / fres)))
                lo = max(0, spur_bin - nb_spur)
                hi = min(nfft_c, spur_bin + nb_spur + 1)
                psd[lo:hi] = np.median(psd)

    # If the caller pre-computed the peak set (e.g., the main loop's multi-
    # resolution Welch sweep), use it directly — find_peaks on the cached
    # 1s PSD alone would miss short-burst peaks that only surface in the
    # short-window pass.
    if cached_peaks is not None:
        peaks = list(cached_peaks)
    else:
        peaks = find_peaks(psd, thresh_db=ethresh, fres_hz=wb_fs / 4096.0)
    # window floor for elevation math — needed on BOTH branches (the
    # cached-peaks path is the main live/gate path)
    _psd_floor_db = float(np.median(psd))
    if not peaks:
        return []

    if len(peaks) > 1:
        max_pwr = max(p[2] for p in peaks)
        kept = [p for p in peaks if p[2] >= max_pwr - spur_db]
        # Never cull silently: a real weak co-window signal dropped here is
        # otherwise indistinguishable from a quiet band.  This only ever
        # fires when a tightened gate (saturation path, or a user-set
        # --spur-reject) actually eats a peak — at the 200 dB default the
        # condition is unreachable.
        if len(kept) < len(peaks):
            culled = [p for p in peaks if p[2] < max_pwr - spur_db]
            _desc = " ".join(
                f"{(center_hz + (p[0] - nfft_c / 2) * fres) / 1e6:.3f}MHz"
                f"({p[2] - max_pwr:+.0f}dB)" for p in culled[:6])
            print(f"  [SPUR-CULL] {len(culled)} peak(s) >{spur_db:.0f}dB "
                  f"below strongest dropped: {_desc}",
                  file=sys.stderr, flush=True)
        peaks = kept

    if debug >= 2:
        print(f"  Energy: {len(peaks)} peaks after spur-reject", file=sys.stderr)

    # --- Stage 2: Per-peak 1Msps extraction + Schmidl-Cox ---
    dets = []

    # Pre-compute the chunk FFT cache ONCE for the full buffer, then reuse it
    # for every energy peak.  We batch-FFT 2D — scipy.fft on a (n_chunks, N)
    # array runs all chunks through pocketfft in one call with `workers=-1`
    # using every CPU core, which is several×faster than a Python loop of
    # 1-D FFTs.  chunk=65536 is the sweet spot for L2 cache + pocketfft.
    _NB_CHUNK = 65536
    _nc = len(iq) // _NB_CHUNK
    _fftw = int(os.environ.get('LORA_FFT_WORKERS', '-1'))  # -1 = all cores
    if _nc > 0:
        try:
            _iq_2d = iq[: _nc * _NB_CHUNK].reshape(_nc, _NB_CHUNK)
            try:
                _ffts = _fft(_iq_2d, axis=1, workers=_fftw)
            except TypeError:
                # numpy.fft fallback: no `workers`, no batched FFT speedup
                _ffts = _fft(_iq_2d, axis=1)
            # Store UNSHIFTED — extract_nb_fft_multi_bw crops the unshifted
            # spectrum directly.  The old fftshift(axes=1) here was a full
            # per-chunk roll costing ~42 % of this batched-FFT step; removing it
            # is bit-identical (verified) and the dominant per-peak-window win.
            # Kept as the 2-D (n_chunks, chunk) array so each peak's crop is
            # two 2-D slice assignments + one batched ifft (see the extractor).
            _nb_fft_cache = _ffts
        except MemoryError:
            # Heap-fragmented allocation failure (seen on 28Msps + buf-seconds=16
            # ring buffers). Empty cache → per-peak FFTs computed on-demand;
            # slower but correct, and we don't crash the pipeline.
            _nb_fft_cache = []
    else:
        _nb_fft_cache = []

    # Per-peak work (1Msps extract + multi-lag SC + dechirp confirm) is
    # INDEPENDENT across peaks, so run peaks concurrently.  The heavy steps
    # are numpy/scipy FFTs that release the GIL, so a thread pool gives real
    # parallelism.  This is the gate's dominant cost at 28 Msps (serial it
    # ran ~870 ms for a 10-peak window — over the 0.5 s hop budget).  Results
    # are identical: each peak builds its own dets and the final dedup sorts
    # by bw_quality, so peak order doesn't matter.  Per-peak FFTs stay
    # single-threaded (workers default) so the pool doesn't oversubscribe.
    def _process_one_peak(peak):
        cb, bw_bins, pwr = peak[0], peak[1], peak[2]
        if only_bws is not None and bw_bins * fres > 100e3:
            return []        # slow pass: narrow signals only
        # the emitting pass's own noise floor (4th element when present) —
        # the ONLY correct reference for this peak's elevation; absolute
        # peak_db stays in [2] because the cross-pass [SPUR-CULL] compares
        # absolute signal power (per-pass elevations are NOT comparable —
        # converting them broke cull decisions, lost anchor decodes and
        # resurrected mirror-image ghosts before this was understood)
        _src_floor = (peak[3] if len(peak) > 3 and peak[3] is not None
                      else _psd_floor_db)
        off_hz = (cb - nfft_c / 2) * fres
        _ld = []

        # MATCHED-PRIMARY: extract this peak at EVERY LoRa bandwidth in
        # one batched crop of the cached wideband FFT.  SC then runs at
        # each bandwidth's own rate (fs = 2xBW) instead of a single fixed
        # 1 Msps pass: a narrow signal no longer correlates against 8-16x
        # its noise bandwidth (that mismatch alone cost ~10 dB of floor,
        # measured), and the total correlation math DROPS ~4x
        # (125k*6 + 250k*6 + 500k*6 + 1M*3 sample-lags vs 1M*15).
        _bw_targets = ([500_000] + sorted(only_bws, reverse=True)
                       if only_bws is not None else
                       [500_000, 250_000, 125_000, 62_500, 41_667, 31_250])
        iq_1m_parts = extract_nb_fft_multi_bw(
            iq, wb_fs, off_hz, _bw_targets,
            chunk=_NB_CHUNK, fft_cache=_nb_fft_cache)
        if not iq_1m_parts:
            return _ld
        iq_1m, _rate1m = iq_1m_parts[0]

        # SELF-RECENTERING: the gate's center bin is the run's argmax —
        # on a chirp's flat-topped spectrum that is a ripple lottery
        # anywhere within +/-bw/2, and a miscentered matched crop CLIPS
        # the chirp edge (measured: a clean SF10/500k beacon reading
        # sc 0.59 / bwq 7 at the pipeline's center vs sc 1.00 / bwq 26
        # at the true carrier — also the long-standing 'SF12/500k
        # captures 140-165 kHz off' mystery). The 1 Msps crop always
        # contains the whole signal, and a chirp plateau has steep
        # edges: the occupied-band edge-midpoint IS the carrier. Only
        # engages on a solid (+6 dB) contour, so weak-signal floors
        # (battery-validated) are untouched.
        _nseg = min(len(iq_1m) // 4096, 64)
        if bw_bins * fres <= 150_000.0:
            _nseg = 0   # narrow peak: centroid accurate, skip the FFT
        # Never self-recenter a peak inside a dechirp CHANNEL: the channel
        # carrier is ground truth (learned/seeded/promoted), while at low SNR
        # the +6 dB contour pass can only latch OTHER energy in the crop and
        # drag off_hz 15-400 kHz off the known carrier — disqualifying the
        # peak from its own channel's despread branch (audit mechanism #2).
        if _nseg and dechirp_chans:
            if any(abs(off_hz - _dc[0]) < _dc[2] for _dc in dechirp_chans):
                _nseg = 0
        
        # No recentering in clip/overload-guard windows (spur_db pinned
        # 35): a spur forest merges +6 dB contour runs and the bogus
        # midpoint MIScenters captures that decoded fine at the gate
        # centroid (cost a railed-window beacon decode in corpus g62).
        if _nseg >= 4 and spur_db >= 100.0:
            _ps = np.abs(np.fft.fft(
                iq_1m[:_nseg * 4096].reshape(_nseg, 4096), axis=1)) ** 2
            _ps = np.fft.fftshift(_ps.mean(axis=0))
            _flr = float(np.median(_ps))
            _abv = _ps > _flr * 3.98          # +6 dB
            if _abv.any():
                # the contiguous above-contour run nearest the crop center
                _d = np.diff(_abv.astype(np.int8))
                _st = list(np.flatnonzero(_d == 1) + 1)
                _en = list(np.flatnonzero(_d == -1) + 1)
                if _abv[0]:
                    _st = [0] + _st
                if _abv[-1]:
                    _en = _en + [4096]
                _best = None
                for _a, _b in zip(_st, _en):
                    _mid = (_a + _b) / 2.0
                    if _best is None or abs(_mid - 2048) < abs(_best[0] - 2048):
                        _best = (_mid, _a, _b)
                if _best is not None:
                    _delta = (_best[0] - 2048.0) * ((_rate1m or 1e6) / 4096.0)
                    if 15_000.0 < abs(_delta) < 400_000.0:
                        off_hz += _delta
                        iq_1m_parts = extract_nb_fft_multi_bw(
                            iq, wb_fs, off_hz, _bw_targets,
                            chunk=_NB_CHUNK, fft_cache=_nb_fft_cache)
                        if iq_1m_parts:
                            iq_1m, _rate1m = iq_1m_parts[0]

        _matched_nb = {500000.0: iq_1m_parts[0]}
        for _bwx, _part in zip(_bw_targets[1:], iq_1m_parts[1:]):
            _matched_nb[float(_bwx)] = _part

        # Minimum length: 3 × worst-case symbol duration (SF12/125k at 1Msps = 32768)
        if len(iq_1m) < 32768 * 3:
            return _ld

        # One forward-FFT cache for EVERY narrowband re-extraction of iq_1m
        # below (trial rungs, resolver hypotheses, per-lag nb).  Created here —
        # after the self-recenter above may have REPLACED iq_1m — because each
        # cold ~1M-pt forward FFT costs ~93 ms on a Pi 4 and the trial-rung
        # loop used to pay one per rung (the cache lived below the trial
        # block, so N rungs = N identical FFTs — pure waste).
        _ffc = []

        # === Channelizer dechirp matched-filter detection ===
        # If this peak is one of the channelizer's channels, despread it and detect on
        # the dechirp peak-to-noise — sensitive BELOW where Schmidl-Cox and the energy
        # gate go deaf.  Emits a detection the decoder then confirms (CRC rejects false
        # positives).  When it fires we skip SC for this peak (dechirp is more sensitive).
        _dech_cands = []
        for _dc in (dechirp_chans or ()):
            if abs(off_hz - _dc[0]) >= _dc[2]:  # _dc = (offset_hz, sf, bw[, trial])
                continue
            if not (len(_dc) > 3 and _dc[3]):
                # Verified/learned channel: first within-bw match wins,
                # exactly as before (channelizer channels sit at distinct
                # offsets and precede the trial rungs in dechirp_chans).
                _dech_cands = [_dc]
                break
            # TRIAL PARITY FIX (2026-07-16): _win_chans appends BOTH of this
            # window's trial rungs at the SAME promoted-carrier offset, but
            # the old first-match break only ever attempted the k=0 rung —
            # and the rotation index (wc*TRIALS_PER_WIN + pi + k) steps by an
            # EVEN stride per window, so a promotion's slot-index parity
            # permanently locked it to 6 of the 12 rungs.  Measured: a -6 dB
            # SF12/125k carrier fired ONLY its same-slope alias (10,62.5k)
            # 5/5 and could never trial the true rung.  Collect ALL matching
            # trial rungs and attempt each below; the per-window pairs
            # (i, i+1) then cover the full ladder across the 6-window sweep.
            _dech_cands.append(_dc)
        if _dech_cands and (len(_dech_cands[0]) > 3 and _dech_cands[0][3]):
            # Patience TRIAL channel (4th field): the (sf, bw) is a rotating
            # HYPOTHESIS, not a learned identity — same-slope aliases clear
            # the 12 dB despread bar by +10-18 dB and can even beat the true
            # combo (measured 2026-07-09), so a trial's label is untrustworthy
            # by construction.  Trial detections are ACQUISITION-INTERNAL:
            # they cut a capture and feed the decoder (whose CRC/parse
            # arbitrates the identity), but are never published as DETECTED
            # and never counted (soundness review: rotating labels turned one
            # strong carrier into N phantom detections and starved real
            # decodes on 10/14 corpus entries).
            # A TRIAL may only process the patience PLACEHOLDER peak (floor-
            # level power at the promoted bin) — never a peak with real
            # elevation.  Otherwise a promotion near live traffic eats every
            # real peak in a predator-prey loop: trial despread fires, the
            # (unpublished) trial det early-returns, the real detection is
            # never published, so the retire-on-detection guard never fires
            # and the promotion holds the carrier until TTL (two60: the
            # continuous 'seq' transmitter lost dets + a decode this way,
            # deterministically).  Real energy falls through to the normal
            # SC chain, which publishes, which retires the promotion.
            if (pwr - _src_floor) >= 5.0:
                _dech_cands = []
        for _dech in _dech_cands:
            _ldoff, _ldsf, _ldbw = _dech[0], _dech[1], _dech[2]
            _ltrial = bool(len(_dech) > 3 and _dech[3])
            try:
                _nbq, _nbqfs = extract_narrowband_fft(iq_1m, _rate1m, 0.0,
                                                      _ldbw, fft_cache=_ffc)
                _r = _dechirp_scan(_nbq, _ldsf, _ldbw, _nbqfs)
                if _r is not None and _r[0] >= DECHIRP_MF_MIN_DB:
                    _q, _ts, _pbin = _r
                    # Fix #1 (anchor): tag the detection with the CHANNEL carrier, not this
                    # peak's offset.  The forced dechirp fires on EVERY energy peak within
                    # ±bw of the channel, so using each peak's own offset smears one packet
                    # across the band.  Channel granularity is what the channelizer tracks.
                    _fhz = center_hz + _ldoff
                    if debug >= 1:
                        print("  CHAN-DECHIRP SF%d q=%.1fdB t=%.0fms f=%.4fMHz" % (
                            _ldsf, _q, _ts * 1000, _fhz / 1e6), file=sys.stderr)
                    _ld.append({
                        'freq_hz': _fhz, 'freq_mhz': _fhz / 1e6,
                        'sf': _ldsf, 'bw': int(_ldbw), 'bw_cnn': int(_ldbw),
                        'detect_conf': float(min(1.0, _q / 20.0)),
                        'sf_conf': float(min(1.0, _q / 20.0)),
                        'bw_quality_db': float(_q),
                        'peak_power_db': pwr,
                        'preamble_t_s': float(_ts),
                        'forced_dechirp': True,   # Fix #2: collapse repeats via wide cooldown dedup
                        'patience_trial': _ltrial,  # acquisition-internal: never published/counted
                    })
                    return _ld
            except Exception:
                pass
        # === end channelizer dechirp ===

        # Multi-lag Schmidl-Cox across all Meshtastic symbol durations.
        # The FFT extraction's rate is ~999756 Hz, not exactly 1 Msps, so a
        # symbol's true length is nominal_lag*(rate/1e6).  SC autocorrelation of
        # a CHIRP is razor-sensitive to lag: at the long lags (SF11/12) the
        # few-sample error cancels the correlation (SF12: sc 0.997 -> 0.091 ->
        # undetectable).  Only the LONG lags need correcting, so refine each
        # nominal lag to the actual rate and (for the long ones) take the best
        # SC over a tiny ±2-sample neighborhood — this leaves the short-lag
        # (SF7-10) behaviour byte-identical to the original integer-lag scan
        # while recovering SF11/12.
        # Compute the SC curve ONCE per lag and cache it (sc_cache[lag] =
        # (used_lag, curve)).  The argmax score is taken here; the per-plateau
        # NMS below reuses the SAME cached curve instead of recomputing the
        # identical SC math (was computed twice: score then peaks).
        # Compute SC curves for ALL lags in one batched pass.  The cumsum-
        # based energy share across lags cuts per-peak sc_curve time from
        # ~150 ms to ~20 ms (7.5x on a synthetic benchmark; ~22% off the
        # per-peak total which was 30% sc_curve).  Long lags (SF11/12)
        # still use the rate-scaled lag value the original loop computed
        # to keep SF12 sc=0.997 at the scaled lag (vs 0.091 at the
        # integer lag — the difference between detecting SF12 and not).
        # Group the curated (sf, bw) set by bandwidth; each group gets a
        # multilag pass on ITS OWN matched extraction.  Same-lag 1 Msps
        # ambiguities (e.g. 2048 = SF7/62.5k or SF8/125k...) separate
        # naturally at matched rates; scores fold back to the nominal-lag
        # key (max wins) so everything downstream — hit_lags, plateau NMS,
        # resolve's dechirp disambiguation — keeps its existing contract.
        _by_bw = {}
        for _lag in _ALL_LAGS_1M:
            for _sf, _bwc in _SC_LAGS_1M[_lag]:
                if only_bws is not None and int(_bwc) not in only_bws:
                    continue     # long-window pass: slow families only
                _by_bw.setdefault(float(_bwc), []).append((_sf, _lag))
        sc = {}
        sc_cache = {}   # nominal_lag -> (lag_used, curve, fs of that curve)
        sc_pairs = {}   # (sf, bw) -> (score, pos_1m, lag_used, curve, fs):
                        # EVERY pair's own analysis. The nominal-lag max
                        # can be won by a different-BW pair whose crop
                        # sees sweep-throughs of a wider signal (score
                        # ~0.9 at a MID-SYMBOL position) — analyses for a
                        # resolved identity must come from ITS OWN pair
                        # or dechirp runs misaligned and real detections
                        # die at the gate (bwq ~7 flat, seen on SF10/500k).
        _wh_gate = bw_bins * fres
        # PEAK-RELATIVE occupied width of this candidate (strength-
        # invariant, unlike the gate's floor-relative width): PSD of the
        # 1 Msps crop, walk from center while above peak-12 dB. Used by
        # the resolver to separate same-slope families (e.g. (9,62.5k)
        # vs (7,31.25k) — dechirp-degenerate AND SC-degenerate; the
        # 31.25k table add made these flips real). Only trusted when the
        # peak clears floor+18 (weak signals' width collapses).
        _wh_peak = None
        _nseg_tot = len(iq_1m) // 4096
        if _nseg_tot >= 4:
            # STRIDE segments across the WHOLE window: consecutive-only
            # sampling covered just the first 0.2 s, so bursts starting
            # later measured as noise (w=None -> tie-break silently off)
            # or as a lone CW spur (w=0.2 kHz -> wrong family won; cost
            # 11% of live decodes as alternating same-slope flips).
            # CONSECUTIVE segments AT THE BURST: a chirp paints its full
            # bandwidth only over >= 1 symbol of CONTIGUOUS time (each
            # 4.1 ms segment sees ~16 kHz of sweep; strided/max-hold
            # sampling hit phase-locked spots and measured 11-25 kHz on
            # a 125 kHz signal). Find the burst by power profile, then
            # mean-PSD over 48 consecutive segments (~196 ms >= 6 SF11
            # symbols) inside it.
            _pp_blk = np.abs(
                iq_1m[:_nseg_tot * 4096].reshape(_nseg_tot, 4096)) ** 2
            _pp_blk = _pp_blk.mean(axis=1)
            _nwin = min(48, _nseg_tot)
            _cum = np.concatenate(([0.0], np.cumsum(_pp_blk)))
            _sums = _cum[_nwin:] - _cum[:-_nwin]
            _b0 = int(np.argmax(_sums))
            _segs_w = iq_1m[_b0 * 4096:(_b0 + _nwin) * 4096].reshape(
                _nwin, 4096)
            _psw = np.abs(np.fft.fft(_segs_w, axis=1)) ** 2
            _psw = np.fft.fftshift(_psw.mean(axis=0))
            _flw = float(np.median(_psw))
            # search near the CROP CENTER (+/-100 kHz — covers shoulder
            # centroid error), not the global argmax (grabs unrelated
            # spurs elsewhere in the +/-500 kHz crop).
            _wlo, _whi = 2048 - 410, 2048 + 410
            _pkw = float(_psw[_wlo:_whi].max())
            if _pkw > _flw * 63.0:            # >= +18 dB: measurable
                _thw = _pkw / 15.85           # peak - 12 dB
                _c0 = _wlo + int(np.argmax(_psw[_wlo:_whi]))
                _lo_w2 = _c0
                while _lo_w2 > 0 and _psw[_lo_w2 - 1] > _thw:
                    _lo_w2 -= 1
                _hi_w2 = _c0
                while _hi_w2 < 4095 and _psw[_hi_w2 + 1] > _thw:
                    _hi_w2 += 1
                _wh_peak = (_hi_w2 - _lo_w2 + 1) * ((_rate1m or 1e6) / 4096.0)
        for _bwc, _pairs in _by_bw.items():
            # width-gated BW evaluation, STRONG PEAKS ONLY: measured
            # width upper-bounds the true BW within ~2.5x when the peak
            # is well above the floor, but UNDERESTIMATES wildly at low
            # SNR (a floor-level SF7/500k pokes 2-6 bins = 10-30 kHz —
            # an unconditional gate cost that preset +8 dB of battery
            # floor). Weak peaks evaluate every class.
            # pwr is ABSOLUTE PSD dB — subtract the window floor for
            # true elevation (using pwr raw mis-gated floor-level
            # battery signals whose absolute values sat above 15).
            # true per-provenance elevation: absolute peak power minus
            # the EMITTING pass's own floor. 15 dB is an honest bar:
            # floor-level battery signals never gate; splatter does.
            if (pwr - _src_floor) >= 15.0 and _bwc > _wh_gate * 2.5:
                continue
            _nbp = _matched_nb.get(_bwc)
            if _nbp is None:
                continue
            _nbx, _fsx = _nbp
            _lagx = {}
            for _sf, _lag in _pairs:
                _lx = int(round((2 ** _sf) / _bwc * _fsx))
                if len(_nbx) >= _lx * 10:
                    _lagx[(_sf, _lag)] = _lx
            if not _lagx:
                continue
            _curves = _schmidl_cox_curve_multilag(
                _nbx, list(set(_lagx.values())), n_sym=8)
            for (_sf, _lag), _lx in _lagx.items():
                _curve = _curves.get(_lx)
                _scv, _posv = schmidl_cox_score(_nbx, _lx, n_sym=8,
                                                _curve=_curve)
                _pos1m = int(round(_posv * (1e6 / _fsx))) if _fsx else _posv
                sc_pairs[(_sf, _bwc)] = (_scv, _pos1m, _lx, _curve, _fsx)
                if _lag not in sc or _scv > sc[_lag][0]:
                    sc[_lag] = (_scv, _pos1m)
                    sc_cache[_lag] = (_lx, _curve, _fsx)

        # Acceptance bar: the shipped rescue's relaxed threshold
        # (threshold - 0.15, floored at 0.30) — matched-rate SC scores are
        # cleaner, and the dechirp confirm behind it is the stronger test
        # (0 FPs measured on pure noise at this bar). EXCEPTION: in
        # clip/overload-guard windows (spur_db pinned to the flat-35
        # value) rail splatter can dechirp convincingly — keep the full
        # user threshold there; weak-signal recovery in a clipped window
        # is a lost cause anyway.
        _thr_m = (max(0.30, sc_threshold - 0.15)
                  if spur_db >= 100.0 else sc_threshold)
        hit_lags = [(lag, sc[lag][0], sc[lag][1])
                    for lag in sc if sc[lag][0] >= _thr_m]

        # NOTE: a pre-loop "drop 2L when L also hits" arbitration lived
        # here and was REMOVED: it assumed only fundamentals score at L,
        # but a strong signal ALSO scores at HALF its own lag in matched
        # crops (fact 1), so for a true (sf, bw) signal the phantom
        # half-lag hit survived and the REAL hit was dropped — its
        # downward-only candidate chain could then never recover the
        # truth (halved SF10/500k beacon decode live). Alias handling
        # lives entirely in the grouped resolve + window arbitration.

        if debug >= 2 and sc:
            best_lag = max(sc, key=lambda l: sc[l][0])
            print(f"  Peak {off_hz / 1e6:+.3f}MHz pwr={pwr:.0f}dB  "
                  f"SC best lag={best_lag} score={sc[best_lag][0]:.3f}  "
                  f"hits={len(hit_lags)}",
                  file=sys.stderr)

        if not hit_lags:
            return _ld

        if os.environ.get('LORA_SLOW_DEBUG'):
            print(f"  [TRACE] hit_lags={[(l, round(sv,2), pv) for l, sv, pv in hit_lags]}",
                  file=sys.stderr)
        _wide = bool(os.environ.get('LORA_SCAN_FULL'))

        hit_lags.sort(key=lambda x: x[1], reverse=True)

        # ---- Occupied-band centroid for this peak (2026-07-08) ----
        # The PSD peak that nominated this candidate can sit up to ±BW/2
        # off-carrier: the max-hold is built from ~0.2 ms FFT snapshots, and
        # a slow-sweep short burst (SF7/125k: 1.024 ms/sweep, ~110 ms burst)
        # gets caught at a handful of random instantaneous chirp positions
        # (measured: jagged 24 kHz contour on a 125 kHz chirp, peak +61 kHz
        # off-carrier).  Every narrowband crop below (resolver vote AND the
        # bw_quality gate) was extracted at that bad centre, so the TRUE
        # preset lost half its chirp and dechirped at noise level while
        # same-slope harmonics in wider crops kept the whole signal —
        # 17 of 24 windows of an MC-SF7/125k ground-truth burst train
        # classified as SF11/500k+SF12/500k (fname SF/BW wrong → 2/12
        # frames decoded).  Fix: Welch-average a ~260 ms slice at the
        # strongest plateau, take the dominant contiguous region's
        # centroid, and centre every crop there.  Guarded: only applied
        # when the region is clearly elevated (>9 dB over the median), so
        # floor-level signals keep the exact old behaviour.
        # LAZY: computed on first rescue-eligibility check only — the
        # unconditional per-peak version cost +36% CPU on the dense spur
        # band (~254 FFTs per peak, dozens of junk peaks per window) while
        # its value is consumed only by the rescue branch.
        _off_c_memo = [None]

        def _band_centroid():
            if _off_c_memo[0] is not None:
                return _off_c_memo[0]
            _off_c_memo[0] = 0.0
            try:
                _cs = int(max(0, hit_lags[0][2] - 8192))
                _seg_c = iq_1m[_cs:_cs + 262144]
                if len(_seg_c) >= 8192:
                    _wc = np.hanning(2048).astype(np.complex64)
                    _accc = None
                    for _oc in range(0, len(_seg_c) - 2048, 1024):
                        _Sc = np.abs(_fft(_seg_c[_oc:_oc + 2048] * _wc)) ** 2
                        _accc = _Sc if _accc is None else _accc + _Sc
                    if _accc is not None:
                        _accc = np.fft.fftshift(_accc)
                        _fqc = np.fft.fftshift(np.fft.fftfreq(2048, 1.0 / 1e6))
                        _pkc = int(np.argmax(_accc))
                        if _accc[_pkc] > 8.0 * float(np.median(_accc)):
                            _thrc = _accc[_pkc] * 0.25
                            _loc = _pkc
                            while _loc > 0 and _accc[_loc - 1] > _thrc:
                                _loc -= 1
                            _hic = _pkc
                            while (_hic < len(_accc) - 1
                                   and _accc[_hic + 1] > _thrc):
                                _hic += 1
                            _wcv = _accc[_loc:_hic + 1]
                            _c = float(np.sum(_fqc[_loc:_hic + 1] * _wcv)
                                       / _wcv.sum())
                            # Correction band for the RESCUE pass (the
                            # centroid is only consulted AFTER a primary
                            # resolve at the nominated centre comes back
                            # dead or marginal — see the call site).
                            # 10 kHz floor: if the primary died, a
                            # sub-10 kHz nudge won't revive it (birdie
                            # peaks measure ±105 Hz — they stay dead).
                            # 100 kHz cap: larger pulls mean the dominant
                            # energy is a DIFFERENT signal — latching it
                            # made junk peaks viable on stolen energy
                            # (harness: new spurious dets 270-1400 kHz
                            # off-carrier), and sub-kHz nudges flipped a
                            # marginal two60 decode in the always-on
                            # design.
                            if 10e3 <= abs(_c) <= 100e3:
                                _off_c_memo[0] = _c
            except Exception:
                _off_c_memo[0] = 0.0
            if os.environ.get('LORA_SLOW_DEBUG'):
                print(f"  [TRACE] peak {off_hz/1e6:+.3f}MHz pwr={pwr:.0f}dB "
                      f"w={_wh_peak and round(_wh_peak/1e3,1)}k "
                      f"centroid={_off_c_memo[0]:+.0f}Hz "
                      f"lags={[(l, round(s,2)) for l, s, _ in hit_lags[:5]]}",
                      file=sys.stderr)
            return _off_c_memo[0]

        seen = set()
        tried = set()      # exact (sf, bw, pos) dechirps already evaluated —
                           # different lags often resolve to the same candidate
        _nb_by_bw = {}     # bw -> (nb, fs): the narrowband extraction depends
                           # only on bw, so share it across the hit-lag loop
                           # (4 lags resolving to the same bw re-extracted the
                           # SAME slice 4x — pure waste, identical values)
        for lag, sc_score_best, _pos_best in hit_lags[:4]:
            # Resolve SF/BW.  Presets: the curated per-lag vote (validated, byte-
            # identical).  Wide-scan: global divisor-lag resolution so any SF/BW is
            # identified AND harmonics collapse onto the true fundamental.
            if _wide:
                _res = _resolve_sf_bw_global(iq_1m, lag, pos=_pos_best,
                                             sc_scores=sc, fft_cache=_ffc)
                if _res is None:
                    continue
                sf, bw, _ = _res
            else:
                # Matched-rate SC is a PRESENCE detector, not an identity
                # classifier: a strong chirp in its tight crop scores ~1.0
                # at MANY lags (measured 0.999/0.999/0.875 at half/true/
                # double lag on a real 75 dB beacon), so identity comes
                # from the dechirp vote — run it across this lag's family
                # PLUS the half-lag's (the 2-symbol-alias fundamentals),
                # slicing each candidate at ITS OWN family's plateau
                # position (a candidate sliced at another family's
                # position dechirps unfairly low — that bug flipped the
                # vote toward same-slope aliases).
                candidates = list(_SC_LAGS_1M[lag])
                # alias chains run BOTH DIRECTIONS: a strong fundamental
                # scores at 2L/4L/8L (harmonic aliases — pull in the lower
                # families), AND at L/2 (identical preamble symbols
                # correlate at HALF the symbol lag — fact 1), so a hit at
                # THIS lag may be the half-lag phantom of a fundamental
                # living at 2L (found live: (10,62.5k) truths emitting as
                # (8,31.25k) via their lag-8192 half-lag hits once the
                # 31.25k family gave that lag a table entry).
                for _div in (2, 4, 8):
                    _fam = _SC_LAGS_1M.get(lag // _div)
                    if _fam:
                        candidates += [c for c in _fam
                                       if c not in candidates]
                _fam_up = _SC_LAGS_1M.get(lag * 2)
                if _fam_up:
                    candidates += [c for c in _fam_up
                                   if c not in candidates]
                if only_bws is not None:
                    # Restrict which ALIAS-CHAIN families the slow-pass vote
                    # may pull in, but NEVER filter the hit lag's OWN family
                    # or its UP-LAG (2L) family: the slow families share lags
                    # with faster presets whose preambles ALSO straddle the
                    # 1 s window and are therefore only ever seen HERE, and a
                    # hit at THIS lag may be the half-lag phantom of a
                    # fundamental at 2L (fact 1) whose true preset must stay
                    # votable.  Ground truth 2026-07-08 matrix: SF12/62.5k
                    # (preamble 1.05 s, lag 65536) hit this filter — the
                    # 65536-bucket scans were FORCED onto the same-slope
                    # half-lag alias SF10/31.25k, and the 32768-bucket scans
                    # (where the truth arrives via the 2L chain) likewise —
                    # every capture mislabeled, 0/5 decoded, the only zero in
                    # the 36-combo matrix; the mislabeled copies then STOLE
                    # the RFDEDUP keep-2 slots from correct-label siblings.
                    _keep_fams = list(_SC_LAGS_1M.get(lag, []))
                    _keep_fams += _SC_LAGS_1M.get(lag * 2, [])
                    candidates = [c for c in candidates
                                  if int(c[1]) in only_bws
                                  or c in _keep_fams] or candidates[:1]
                if len(candidates) == 1:
                    sf, bw = candidates[0]
                else:
                    _pos_map = {}
                    for _csf, _cbw in candidates:
                        _pp = sc_pairs.get((_csf, float(_cbw)))
                        if _pp is not None:
                            _pos_map[(_csf, _cbw)] = _pp[1]
                    # PRIMARY resolve at the nominated centre — byte-
                    # identical inputs to the pre-fix behaviour.
                    sf, bw, _q1 = _resolve_sf_bw_ambig(iq_1m, candidates,
                                                       fft_cache=_ffc,
                                                       pos=_pos_best,
                                                       nb_cache=_nb_by_bw,
                                                       pos_map=_pos_map,
                                                       width_hz=_wh_peak)
                    if ((sf is None or _q1 < DECHIRP_MIN_DB + 6.0)
                            and _band_centroid()):
                        # RESCUE: the primary verdict is DEAD (all
                        # candidates junk) or MARGINAL, and the occupied-
                        # band centroid says the energy sits 10-100 kHz
                        # off the nomination.  The snapshot max-hold
                        # nominates a slow-sweep short burst up to ±BW/2
                        # off its carrier; a crop there starves the TRUE
                        # preset below the viable bar while a same-slope
                        # harmonic's wider crop keeps the signal and can
                        # win a MARGINAL vote at 15-18 dB (measured:
                        # SF7/125k MC bursts classified SF11/500k in 17
                        # of 24 windows).  Re-vote once on crops centred
                        # at the centroid, in a PRIVATE cache so the
                        # shared nb (and every healthy detection's
                        # bw_quality) is untouched; keep whichever vote
                        # MEASURED higher.  Well-nominated real signals
                        # resolve at 25-35 dB and never enter this path.
                        _nb_rescue = {}
                        _sf2, _bw2, _q2 = _resolve_sf_bw_ambig(
                            iq_1m, candidates, fft_cache=_ffc,
                            pos=_pos_best, nb_cache=_nb_rescue,
                            pos_map=_pos_map, width_hz=_wh_peak,
                            off_hz=_band_centroid())
                        if _sf2 is not None and (sf is None or _q2 > _q1):
                            sf, bw = _sf2, _bw2
                            _nb_by_bw = _nb_rescue   # downstream bw_quality
                                                     # must see the crop the
                                                     # vote was won on
                    if sf is None:
                        continue    # no candidate dechirps like LoRa — junk
                                    # (CW birdie / flat-lag class); next lag

            # SAME-SLOPE FAMILY MATCHED-SC OVERRIDE (2026-07-09, matrix
            # counter-5 class): at weak SNR the dechirp vote between
            # same-slope family siblings (bw <-> 2bw @ sf+2) COIN-FLIPS
            # scan to scan — the 62.5k-slice replay shows lag-65536
            # resolves alternating SF10/31.25k vs SF12/62.5k all session,
            # and whichever label the frame's single preamble-covering
            # scan happened to draw became the save (a wrong draw = a
            # 62ksps/8s crop that can never decode a 62.5kHz/12s frame).
            # The MATCHED SC score separates the pair decisively where the
            # vote cannot: true SF12/62.5k frames measure 0.94-0.98 under
            # the 62.5k matched extraction vs 0.67-0.72 under the 31.25k
            # one (and vice versa for true 31.25k frames).  When the vote
            # winner's own matched score is materially below a same-slope
            # sibling's, trust the matched filter over the marginal vote.
            # Slow pass only — the 1 s path has its own frozen arbitration.
            if only_bws is not None and sf is not None:
                _own_pp = sc_pairs.get((sf, float(bw)))
                if _own_pp is not None:
                    for _ssf, _sbw in ((sf + 2, bw * 2), (sf - 2, bw // 2)):
                        # Evidence bar is an sc_pairs entry (the sibling's
                        # matched analysis ran and scored) — NOT membership
                        # in `candidates`: the up-family sibling reaches a
                        # half-lag bucket only via the down-lag pull, which
                        # the only_bws filter strips, so a candidates check
                        # can never pass in exactly the case that matters.
                        if not (7 <= _ssf <= 12):
                            continue
                        _sib_pp = sc_pairs.get((_ssf, float(_sbw)))
                        if _sib_pp is not None and _sib_pp[0] > _own_pp[0] + 0.15:
                            if os.environ.get('LORA_SLOW_DEBUG'):
                                print(f"  [TRACE] lag={lag} family SC override "
                                      f"SF{sf}/{bw} (sc={_own_pp[0]:.2f}) -> "
                                      f"SF{_ssf}/{_sbw} (sc={_sib_pp[0]:.2f})",
                                      file=sys.stderr)
                            sf, bw = _ssf, _sbw
                            break

            if os.environ.get('LORA_SLOW_DEBUG'):
                print(f"  [TRACE] lag={lag} resolved sf={sf} bw={bw}",
                      file=sys.stderr)
            # Find ALL time-separated preamble plateaus at this lag — so a
            # SECOND packet that shares this carrier + SF/BW (a relay hop1
            # ~0.5 s later, or simply another packet shortly after the first)
            # gets its OWN detection instead of being hidden behind the
            # strongest plateau.  The single-argmax SC saw only one of them, so
            # the later packet was never detected → never captured → lost (the
            # last live-miss root).  Reuse the SC curve already computed for this
            # lag in the scan (cached) — same math, no recompute.
            _pp_res = sc_pairs.get((sf, float(bw)))
            if _pp_res is not None and _pp_res[3] is not None:
                # the RESOLVED identity's own matched analysis
                _lag_used, _curve, _fs_used = _pp_res[2], _pp_res[3], _pp_res[4]
            else:
                _lag_used, _curve, _fs_used = sc_cache[lag]
            _positions = schmidl_cox_peaks(iq_1m, _lag_used, n_sym=8,
                                           thr=max(0.30, sc_threshold - 0.15),
                                           max_peaks=6, _curve=_curve)
            # curve positions are in the MATCHED extraction's samples —
            # convert to 1 Msps coordinates for everything downstream
            if _fs_used and _fs_used != 1e6:
                _positions = [(_s, int(round(_p * (1e6 / _fs_used))))
                              for _s, _p in _positions]
            if not _positions:
                _positions = [(sc_score_best, _pos_best)]

            # Narrowband extraction depends only on bw (not the plateau
            # position), so compute it ONCE per hit lag instead of per plateau.
            # Pass `_ffc` so the 1Msps forward FFT computed by an earlier
            # `_resolve_sf_bw_ambig` / `_resolve_sf_bw_global` call (or by a
            # prior hit_lag iteration) is REUSED here.  Each fresh FFT of
            # a 1Msps signal is 8-25 ms; on a peak with 2+ hit_lags this
            # reuse cuts the per-call cost from ~14 ms (cold) to ~3 ms
            # (cached) at bw=125k.  Profiling on 2026-06-21 measured
            # nb_extract at ~130 ms / peak after the v2.2 sc_curve fix —
            # the dominant cost was these redundant forward FFTs across
            # the hit_lags loop.
            if bw in _nb_by_bw:
                nb, _nbfs = _nb_by_bw[bw]
            else:
                # 0.0 = the nominated centre: the shared cache must stay
                # byte-identical to pre-fix behaviour; a rescue that WINS
                # swaps in its own centred cache above.
                nb, _nbfs = extract_narrowband_fft(iq_1m, 1_000_000, 0.0, bw,
                                                    fft_cache=_ffc)
                _nb_by_bw[bw] = (nb, _nbfs)

            for sc_score, _pos in _positions:
                # Merge the SAME packet re-found at another lag (≈20 ms time
                # bucket); genuinely distinct packets — including close
                # back-to-back ones >20 ms apart — keep separate detections.
                _tb = int((_pos / _rate1m) / 0.02) if _rate1m else int(_pos)
                if os.environ.get('LORA_SLOW_DEBUG'):
                    print(f"  [TRACE] plateau sc={sc_score:.2f} pos={_pos} tb-check",
                          file=sys.stderr)
                if (sf, bw, _tb) in seen:
                    continue

                # ---- Carrier estimate: dechirp the preamble ALIGNED to the SC
                # position.  dechirp_peak_quality chunks symbols from sample 0,
                # so any sub-symbol pre-roll couples a TIME offset into the
                # dechirp peak bin (LoRa time/freq coupling).  Slicing nb at the
                # preamble start aligns the symbol boundaries so peak_bin ≈ pure
                # CFO → stable carrier → one well-centred capture per packet.
                N_sf = 2 ** sf
                _pos_nb = int(round(_pos * _nbfs / 1_000_000.0))
                # Exact-input dedup: different lags resolving to the same
                # (sf, bw) yield plateaus at the same sample positions — the
                # dechirp below would produce byte-identical results.
                if (sf, bw, _pos_nb) in tried:
                    continue
                tried.add((sf, bw, _pos_nb))
                if 0 < _pos_nb < len(nb) - N_sf * 4:
                    # 16-symbol slice at the preamble: the quality/CFO of THIS
                    # SC-aligned preamble lives in its first 8-12 symbols; the
                    # old open-ended tail scan mostly measured payload/noise
                    # (and could latch a DIFFERENT later packet's window).
                    nb_al = nb[_pos_nb:_pos_nb + 32 * N_sf]
                else:
                    nb_al = nb[:32 * N_sf]
                if len(nb_al) >= N_sf * 4:
                    bw_quality, peak_bin = dechirp_peak_quality(nb_al, sf, bw)
                else:
                    bw_quality, peak_bin = -99.0, 0

                # Dechirp quality is the reliable LoRa indicator.  SC alone can
                # false-positive on any narrowband/CW signal.  Require both SC
                # and dechirp confirmation (also rejects wrong-SF lag aliases
                # and any spurious extra SC plateau, so multi-peak adds no FP).
                # Relaxed-tier hits (sc below the user's threshold — the
                # matched-SC low bar) must clear the dechirp gate with
                # margin: genuine weak LoRa measures 31+ dB PMR even at
                # +2 dB SNR, while sidelobes of a strong same-window
                # signal squeak by at 15-16 dB with borderline SC — the
                # one FP class the low bar admits.
                # +2 dB over the legacy gate: matched-crop SC scores run
                # high even for intermod ghosts of strong co-transmissions
                # (measured sc 0.89-0.90 with bwq exactly 15-16 at 907.4 in
                # two60, spawned by 60+ dB parents), so dechirp is the only
                # real gate and needs margin. Every verified-real signal
                # today measures bwq >= 19; ghosts sit at 15-17. Floor-level
                # battery signals measure 30+ (unaffected).
                _need_bwq = (DECHIRP_MIN_DB + 2.0 if sc_score >= sc_threshold
                             else DECHIRP_MIN_DB + 4.0)
                if spur_db < 100.0:
                    # clip/overload-guard window (flat-35): rail splatter
                    # is chirp-like enough to pass matched SC at 0.9+ AND
                    # ordinary dechirp gates (measured bwq 17-23 junk in
                    # the railed g62 capture, where the old wideband SC's
                    # dilution hid it by accident). Real signals strong
                    # enough to matter in a railed window measure 30+.
                    _need_bwq = max(_need_bwq, DECHIRP_MIN_DB + 9.0)
                if bw_quality < _need_bwq:
                    if debug >= 2 or os.environ.get('LORA_SLOW_DEBUG'):
                        print(f"    Rejected SF{sf}/BW{bw/1000:.0f}k: "
                              f"sc={sc_score:.2f} bwq={bw_quality:.1f}dB "
                              f"(need bwq>={_need_bwq:.0f}dB)",
                              file=sys.stderr)
                    continue

                # Carrier = PACKET-MATCHED Welch-averaged energy centroid of the
                # preamble (slice the SC-localised preamble out of iq_1m, Welch-
                # average with sub-symbol windows, take the plateau centroid).
                # Time-invariant AND packet-matched → tight, accurate carrier.
                _sym1m = max(1, int(round(N_sf * _rate1m / bw)))
                # Skip EDGE detections: if the 16-symbol preamble doesn't fully
                # fit in this window the centroid is taken on a partial slice →
                # off-centre capture.  50% overlap means a neighbouring window
                # has the full preamble, so dropping the edge copy loses nothing.
                # Slow-pass exception: at SF12/31.25k, 16 symbols = 2.1 s —
                # demanding them beyond _pos overruns even a 4 s assembly,
                # and unlike the 1 s path there is NO neighbouring window
                # holding the full preamble (this assembly IS the retry).
                # 10 symbols (preamble 8 + sync 2) centroid fine.
                _need_syms = 16 if only_bws is None else 10
                if (only_bws is not None
                        and int(_pos) + _need_syms * _sym1m > len(iq_1m)):
                    # SLOW-PASS EDGE RELAXATION (2026-07-09, matrix ctr-5
                    # kill chain): a late-in-assembly slow preamble has NO
                    # neighbouring window to re-catch it — this assembly IS
                    # the only look.  Ground truth: the true SF12/62.5k det
                    # (plateau sc 0.99) missed the 10-sym bar by 816 samples
                    # (0.05 sym) and the frame fell to its half-lag alias.
                    # 8 symbols is the preamble core — centroid quality is
                    # marginally worse than 10 but the alternative is TOTAL
                    # loss of the frame.  Every downstream slice is already
                    # length-guarded.
                    _need_syms = 8
                if (int(_pos) + _need_syms * _sym1m > len(iq_1m)
                        or int(_pos) < 0):
                    if os.environ.get('LORA_SLOW_DEBUG'):
                        print(f"  [TRACE] EDGE-DROP SF{sf}/{bw} pos={int(_pos)} "
                              f"need={_need_syms * _sym1m} len={len(iq_1m)}",
                              file=sys.stderr)
                    continue
                _ps = int(_pos)
                _pre = iq_1m[_ps:_ps + 16 * _sym1m]
                cfo_hz = 0.0
                if len(_pre) >= 4 * _sym1m:
                    _subw = max(64, _sym1m // 4)
                    _wn = np.hanning(_subw).astype(np.complex64)
                    _acc = None
                    for _ss in range(0, len(_pre) - _subw, _subw // 2):
                        _S = np.abs(np.fft.fftshift(_fft(_pre[_ss:_ss + _subw] * _wn))) ** 2
                        _acc = _S if _acc is None else _acc + _S
                    if _acc is not None:
                        _frq = np.fft.fftshift(np.fft.fftfreq(_subw, 1.0 / _rate1m))
                        _inb = np.abs(_frq) <= bw
                        _Sb = np.where(_inb, _acc, 0.0)
                        _pk = int(np.argmax(_Sb))
                        _thr = _Sb[_pk] * 0.25
                        _lo = _pk
                        while _lo > 0 and _Sb[_lo - 1] > _thr:
                            _lo -= 1
                        _hi = _pk
                        while _hi < len(_Sb) - 1 and _Sb[_hi + 1] > _thr:
                            _hi += 1
                        _cw = _Sb[_lo:_hi + 1]
                        if _cw.sum() > 0:
                            cfo_hz = float(np.sum(_frq[_lo:_hi + 1] * _cw) / _cw.sum())
                freq_hz = center_hz + off_hz + cfo_hz

                if debug >= 1:
                    print(f"  PREAMBLE lag={lag} SF{sf} BW={fmt_bw(bw)} "
                          f"sc={sc_score:.2f} bwq={bw_quality:.1f}dB "
                          f"freq={freq_hz / 1e6:.4f}MHz t={_pos / _rate1m * 1000:.0f}ms "
                          f"[off={off_hz/1e3:+.0f}k cfo={cfo_hz/1e3:+.0f}k]",
                          file=sys.stderr)

                # fs/4 image-spur carrier veto: interleaved-ADC SDRs put
                # constant tones at EXACTLY +/-rate/4; nearby candidates'
                # analysis crops inherit the tone and the CFO snap lands
                # the final carrier ON it (measured: phantom SF12/500k at
                # center+5.0002 MHz from a peak 313 kHz away — PSD
                # blanking cannot reach that). Vetoing a +/-25 kHz sliver
                # at the two deterministic artifact frequencies costs
                # 0.25% of band coverage.
                if abs(abs(freq_hz - center_hz) - wb_fs / 4.0) < 25e3:
                    if debug >= 2:
                        print(f"    FS4-VETO {freq_hz/1e6:.4f}MHz",
                              file=sys.stderr)
                    continue
                # Whole-slice consistency metric for the cross-preset
                # contested-bucket arbitration below (2026-07-08): the true
                # (sf, bw) dechirps cleanly on preamble AND payload, while
                # an alias/sliver-steal is clean only on its preamble
                # stretch (or nowhere) — its MEDIAN collapses.  Raw best-8
                # bwq cannot arbitrate cross-preset: bin-count bias hands
                # up-aliases +3 dB/SF-step (a CW-birdie sliver-steal at
                # (11,500k) measured bwq 25 vs the REAL SF7/125k's honest
                # 19-20 and displaced it in the tight carrier+time dedup —
                # burst-1 class of the MC-SF7 ground truth), while a
                # normalized key hands DOWN-aliases the same handicap
                # (the dedup comment's SF9/62.5-vs-SF11/125 case).  The
                # median metric kills both directions categorically.
                _med_end = _pos_nb + 192 * N_sf
                _nb_med = (nb[_pos_nb:_med_end]
                           if 0 < _pos_nb < len(nb) - N_sf * 8 else nb_al)
                if len(_nb_med) >= N_sf * 8:
                    _bwq_med, _ = dechirp_peak_quality(_nb_med, sf, bw,
                                                       agg='median')
                else:
                    _bwq_med = bw_quality
                _ld.append({
                    'freq_hz':       freq_hz,
                    'freq_mhz':      freq_hz / 1e6,
                    'sf':            sf,
                    'bw':            bw,
                    'bw_cnn':        bw,        # compatibility field
                    'detect_conf':   float(sc_score),
                    'sf_conf':       float(sc_score),
                    'bw_quality_db': bw_quality,
                    'bwq_med_db':    float(_bwq_med),
                    'width_hz':      (float(_wh_peak) if _wh_peak else None),
                    'peak_power_db': pwr,
                    # Preamble start time WITHIN this window (seconds), combined
                    # downstream with the window sample position for an absolute
                    # RF timestamp (identical across overlapping windows that
                    # re-detect the same packet) — used to dedup captures.
                    'preamble_t_s':  (float(_pos) / _rate1m) if _rate1m else 0.0,
                })
                seen.add((sf, bw, _tb))
        return _ld

    _process_one_peak_core = _process_one_peak

    def _process_one_peak(_pk):
        # Arm the per-peak dechirp memo for this peak's resolver work
        # (thread-local: safe under both the serial and threaded paths).
        _DPQ_MEMO.d = {}
        try:
            return _process_one_peak_core(_pk)
        finally:
            _DPQ_MEMO.d = None

    if (len(peaks) <= 1 or os.environ.get('LORA_DETECT_SERIAL')
            or (os.cpu_count() or 8) < 8):
        # Serial per-peak — used when a higher level already parallelises
        # (e.g. a pool of single-threaded detect worker processes), so the
        # thread pool here would only oversubscribe.  ALSO forced on <8-core
        # hosts (UC audit, measured A/B on a 4-core taskset: the thread pool
        # is pure GIL contention there — serial 91.3s/3.29GB vs threaded
        # 111.4s/4.04GB, identical detections).
        if _COALESCE and len(peaks) > 1:
            # Coverage-based coalescing (see _COALESCE at module top): strongest
            # first; skip a peak already covered by a confirmed detection's band.
            _cov = []   # (carrier_hz, bw) of confirmed detections this window
            _n_skipped = 0
            for _pk in sorted(peaks, key=lambda p: -p[2]):
                _pkhz = center_hz + (_pk[0] - nfft_c / 2) * fres
                if any(abs(_pkhz - _c) <= _b * 0.5 for (_c, _b) in _cov):
                    _n_skipped += 1
                    continue
                _r = _process_one_peak(_pk)
                dets.extend(_r)
                for _d in _r:
                    _cov.append((_d['freq_hz'], _d['bw']))
            if _n_skipped and debug >= 1:
                print(f"  [COALESCE] skipped {_n_skipped}/{len(peaks)} "
                      f"redundant per-peak jobs (same-signal)", file=sys.stderr)
        else:
            for _pk in peaks:
                dets.extend(_process_one_peak(_pk))
    else:
        from concurrent.futures import ThreadPoolExecutor
        # Cap workers at the PHYSICAL core count: on a 4-core host the old
        # min(peaks, 8) ran 8 threads on 4 cores — pure oversubscription
        # (GIL churn + context switches) since the per-peak numpy work is
        # CPU-bound. Fast hosts with >=8 cores are unchanged.
        _nw = min(len(peaks), 8, os.cpu_count() or 8)
        with ThreadPoolExecutor(max_workers=_nw) as _ex:
            for _r in _ex.map(_process_one_peak, peaks):
                dets.extend(_r)

    # ONE PHYSICAL SIGNAL -> ONE DETECTION (window level — the variants
    # of one burst can come from DIFFERENT gate peaks: the fundamental's
    # peak, a splatter shoulder, or a same-slope alias that survived its
    # own resolve). Greedy best-first by bin-normalized dechirp quality;
    # accept only detections not overlapping an accepted one in time
    # (within the longer preamble span) AND frequency (within the wider
    # bandwidth).
    if len(dets) > 1:
        _center_hz_arb = center_mhz * 1e6
        # RAW quality sort — DELIBERATELY unchanged (the pre-2026-07-08
        # order): same-preset copies of one packet must keep their
        # historical survivor (a marginal two60 Meshtastic decode flips
        # deterministically under ANY reordering of its copies).  Cross-
        # preset contests are arbitrated pairwise in the keep-loop below
        # via WIDTH-CONSISTENCY, because no per-window PMR metric can do
        # it: raw best-8 hands UP-aliases the +3 dB/SF-step bin-count
        # bias (a sliver-steal at (11,500k) measured bwq 23-25 vs the
        # real SF7/125k's honest 19-20 and displaced it here);
        # SF-normalizing hands DOWN-aliases the same handicap (observed:
        # SF9/62.5 beating a real SF11/125); the whole-slice median is
        # DEGENERATE for same-slope up-aliases (k tones per wider window
        # lose exactly the 10*log10(k) the bin count gains).
        dets.sort(key=lambda d: d['bw_quality_db'], reverse=True)

        def _width_cls(d):
            # 2 = claimed bw consistent with the peak's measured occupied
            # width; 1 = width unmeasurable; 0 = inconsistent.  Ground
            # truth: the real SF7/125k det carries w=121.8k (consistent);
            # every sliver-steal SF11/500k det carried w=None or a
            # sub-kHz junk contour.
            _w = d.get('width_hz')
            if not _w:
                return 1
            import math as _m2
            return 2 if abs(_m2.log2(d['bw'] / _w)) <= 0.85 else 0
        _kept = []
        for _d in dets:
            _span = 8.0 * (2.0 ** _d['sf']) / _d['bw']
            _dup = False
            for _ki, _k in enumerate(_kept):
                _kspan = 8.0 * (2.0 ** _k['sf']) / _k['bw']
                if (abs(_d['preamble_t_s'] - _k['preamble_t_s'])
                        < max(_span, _kspan)
                        and abs(_d['freq_hz'] - _k['freq_hz'])
                        < max(_d['bw'], _k['bw'], 60_000.0)):
                    # CROSS-PRESET width arbitration (2026-07-08): when a
                    # kept det and a later (lower raw-bwq) det claim
                    # DIFFERENT (sf, bw) for the same carrier+time, the
                    # one whose claim matches the measured occupied width
                    # wins regardless of raw bwq — the raw order is
                    # bin-count-biased across presets (see sort note).
                    # Within the consistent class the CLOSER width wins:
                    # a 125 kHz burst whose width over-measured at 163 kHz
                    # made BOTH 125k (0.39) and 250k (0.61) 'consistent',
                    # and the raw-bwq tie-break crowned the SF9/250k
                    # mislabel (ctr-5 class of the ground truth).
                    # Same-preset copies keep the historical order.
                    if (_d['sf'], _d['bw']) != (_k['sf'], _k['bw']):
                        _cd, _ck = _width_cls(_d), _width_cls(_k)
                        _swap = _cd > _ck
                        if (not _swap and _cd == _ck == 2
                                and _d.get('width_hz')):
                            import math as _m3
                            _swap = (abs(_m3.log2(_d['bw'] / _d['width_hz']))
                                     < abs(_m3.log2(_k['bw'] / _k['width_hz']))
                                     - 0.15)
                        if _swap:
                            _kept[_ki] = _d
                    _dup = True
                    break
                # MID-PACKET duplicate: at strong SNR the payload's
                # quasi-periodicity spawns a second detection later in
                # the SAME burst with a corrupted carrier (measured
                # cfo +396 kHz on an SF10/500k beacon). Distinguish from
                # real back-to-back packets on one channel (DM
                # handshakes, protected downstream): genuine ones sit
                # within a few kHz of each other; the duplicate is
                # >60 kHz off. Same sf/bw + within 2 s + off by
                # (60 kHz .. 0.6 bw) => duplicate of the kept one.
                if (_d['sf'] == _k['sf'] and _d['bw'] == _k['bw']
                        and abs(_d['preamble_t_s'] - _k['preamble_t_s']) < 2.0
                        and 60_000.0 < abs(_d['freq_hz'] - _k['freq_hz'])
                        < 0.6 * _d['bw']):
                    _dup = True
                    break
                # IQ MIRROR-IMAGE ghost: receiver I/Q imbalance images a
                # strong signal to the CONJUGATE frequency (2*center - f) at
                # ~-40 dB (HackRF IRR ~41-47 dB measured). The image passes
                # SC + dechirp (it IS the signal, conjugated) — seen live as
                # a phantom SF12/31.25k at 919.464 mirroring the real
                # 910.535 beacon, and as the dcspur +/-491 kHz pairs. Same
                # sf/bw, overlapping time, mirrored carriers (sum of offsets
                # ~0), and the kept (better-bwq) one wins.
                if (_d['sf'] == _k['sf'] and _d['bw'] == _k['bw']
                        and abs(_d['preamble_t_s'] - _k['preamble_t_s']) < 2.0
                        and abs((_d['freq_hz'] - _center_hz_arb)
                                + (_k['freq_hz'] - _center_hz_arb)) < 30e3):
                    _dup = True
                    break
            if not _dup:
                _kept.append(_d)
        dets = _kept

    # Deduplicate by (SF, BW, freq-bucket).
    # PRIMARY SORT IS preamble_t_s ASC (earliest first) — critical for DM
    # 3-packet handshake clusters (DM + ROUTING ack + ack-of-ack, ~50-400 ms
    # apart at essentially the same carrier).  The previous bw_quality DESC
    # sort tended to pick the LATER ack/ack-of-ack when its quality was
    # higher, and the resulting capture started at THAT preamble — so DM
    # hop0 (which transmitted earlier) was outside the 1.5 s capture window
    # entirely and its bytes were unrecoverable even at unlimited offline
    # budget (proven 2026-05-26: ~10/50 DM hop0 pktids absent from the
    # entire offline log; broadcasts hit 100/100 because their hop0 is also
    # the earliest preamble).  Picking the EARLIEST det per freq bucket
    # anchors the capture at DM hop0; the decoder's multi-packet extraction
    # (validated on broadcast hop0+hop1) then pulls ack + ack-of-ack out of
    # the same 1.5 s capture.  Same save-worker load as the bw_quality sort
    # (still one capture per cluster), no live regression.  bw_quality is
    # used as a tiebreak so a clean preamble beats a marginal one at the
    # same time.
    GRID_HZ = 15000
    dets.sort(key=lambda d: (d.get('preamble_t_s', 0.0), -d.get('bw_quality_db', 0.0)))
    out, used = [], {}
    for d in dets:
        key = (d['sf'], d['bw'])
        used.setdefault(key, set())
        k = round(d['freq_hz'] / GRID_HZ)
        if k not in used[key]:
            out.append(d)
            used[key].update([k - 1, k, k + 1])

    # ---- Harmonic collapse (wide-scan only) ----
    # After BW-aware resolution, residual harmonics of one signal occasionally
    # resolve to a slightly-wrong (sf,bw) and so survive the (sf,bw,freq) dedup as
    # extra detections that never decode but burn a full decode budget each.  A
    # harmonic shares the fundamental's carrier AND preamble time; distinct packets
    # never share both (hop0/hop1 differ ~0.5 s; concurrent ones differ in carrier).
    # So within a tight carrier+time bucket keep only the best-dechirp detection
    # (the correctly-resolved fundamental).  Gated to wide mode → presets unchanged.
    if os.environ.get('LORA_SCAN_FULL') and len(out) > 1:
        _hb = {}
        for _d in out:
            _g = (round(_d['freq_hz'] / 15000.0), round(_d['preamble_t_s'] / 0.010))
            _b = _hb.get(_g)
            if _b is None or _d['bw_quality_db'] > _b['bw_quality_db']:
                _hb[_g] = _d
        out = list(_hb.values())

    # ---- Harmonic collapse (preset mode) ----
    # A LoRa preamble autocorrelates at its symbol period AND at every power-of-2
    # multiple/divisor of it, and the chirp slope BW²/2^SF is INVARIANT along that
    # chain (SF±2, BW×/÷2).  So one packet fires Schmidl-Cox at several lags with
    # the SAME slope — SF7/62.5 (lag 2048) also throws SF9/125 (4096) and SF11/250
    # (8192); a genuine SF11/250 (8192) also throws an SF9/125 (4096) ghost.  These
    # can land in SEPARATE energy peaks, so they only meet here, with every peak's
    # detections pooled.  All pass the dechirp gate — and the ghosts even score
    # HIGHER on raw dechirp quality (peak-to-mean grows with FFT size 2^SF), which
    # is why bw_quality must NOT be the selector.  Group by carrier + preamble time
    # + chirp-rate family and keep the strongest-SCHMIDL-COX detection (the true
    # fundamental; harmonics autocorrelate weaker).  Only same-family, same-carrier,
    # same-time detections merge, so genuinely distinct signals are never collapsed.
    # Wide-scan already collapses harmonics at the lag stage, so preset-mode only.
    if not os.environ.get('LORA_SCAN_FULL') and len(out) > 1:
        # Time tolerance is WIDE (0.25 s) on purpose: a signal's harmonic
        # detections localize their preamble ~10-30 ms apart (the SC peak sits at
        # a different sample for each lag), so a tight bucket would split them and
        # miss the collapse.  This does NOT risk merging genuine close packets: the
        # (sf,bw,freq) dedup above already collapsed same-(sf,bw) clusters (incl.
        # DM/ack handshakes ~50-400 ms apart) to one entry each, so the only things
        # that can share a (carrier, chirp-family) bucket here are DIFFERENT-(sf,bw)
        # harmonics of one signal — exactly what we want to fold together.
        _fam = {}
        for _d in out:
            _g = (round(_d['freq_hz'] / 15000.0),
                  round(_d.get('preamble_t_s', 0.0) / 0.25),
                  round(_d['bw'] * _d['bw'] / (2 ** _d['sf'])))   # chirp slope key
            _b = _fam.get(_g)
            if _b is None or _d.get('detect_conf', 0.0) > _b.get('detect_conf', 0.0):
                _fam[_g] = _d
        out = list(_fam.values())

    # ---- IQ-image rejection ----
    # bladeRF IQ imbalance mirrors EVERY signal to its conjugate frequency
    # (2*center - f), typically 20-40 dB weaker.  The gate otherwise detects
    # these mirrors as real signals and the decoder burns a full recovery budget
    # on each — they decode to NOTHING.  Measured on MEDIUM_FAST: ~50% of all
    # captures (243/490) were images at 916.9 MHz (mirror of the 913.1 signal),
    # 0 of them decoded a packet.  Drop a detection when a MUCH stronger
    # (>IMG_DB) detection sits at its mirror frequency.  Two genuinely symmetric
    # signals (within IMG_DB of each other) are BOTH kept; only the weak
    # conjugate image is removed.  SF/BW-agnostic, halves decode time.
    if len(out) > 1:
        _IMG_TOL_HZ = 60000.0
        _IMG_DB = 10.0
        _img_kept = []
        for _di in out:
            _mirror = 2.0 * center_hz - _di['freq_hz']
            _is_image = any(
                _dj is not _di
                and abs(_dj['freq_hz'] - _mirror) < _IMG_TOL_HZ
                and _dj['peak_power_db'] > _di['peak_power_db'] + _IMG_DB
                for _dj in out)
            if not _is_image:
                _img_kept.append(_di)
            elif debug >= 1:
                print(f"  [IMG-REJECT] {_di['freq_mhz']:.4f}MHz "
                      f"pwr={_di['peak_power_db']:.0f}dB — mirror of stronger "
                      f"signal, dropped", file=sys.stderr)
        out = _img_kept

    return out


def decimate_by(iq, factor):
    """FFT-crop decimation — proper brick-wall bandlimit before decimating.

    Block-averaging aliases components outside the target band back into the
    chirp's bin space, which degrades dechirp quality for narrow BWs (62.5k,
    125k).  FFT crop gives a clean bandlimit identical to extract_narrowband_fft
    so the chirp fold/wrap behavior is correct at every sample rate.
    """
    if factor <= 1: return iq
    N = len(iq)
    n_out = N // factor
    if n_out < 1: return np.array([], dtype=np.complex64)
    F = np.fft.fftshift(_fft(iq))
    keep = n_out
    start = N // 2 - keep // 2
    cropped = F[start:start + keep]
    result = _ifft(np.fft.ifftshift(cropped)) * (keep / N)
    return result[:n_out].astype(np.complex64)


def determine_bw(nb_widest, fs_widest, sf, debug=0):
    best_bw, best_quality = BW_LIST[3], -99.0
    for bw in BW_LIST:
        target_fs = bw * 2
        dec = max(1, int(round(fs_widest / target_fs)))
        nb = decimate_by(nb_widest, dec)
        if len(nb) < (2 ** sf) * 2: continue
        quality, _ = dechirp_peak_quality(nb, sf, bw)
        if debug >= 2:
            print(f"    Dechirp BW={bw/1000:.2f}k: quality={quality:.1f}dB",
                  file=sys.stderr)
        if quality > best_quality:
            best_quality = quality
            best_bw = bw
    return best_bw, best_quality


def determine_sf_and_bw(nb_widest, fs_widest, sf_hint, debug=0):
    """Test all 9 Meshtastic presets via dechirp. Only valid presets are
    tested, avoiding chirp-rate collisions between arbitrary SF×BW pairs
    (e.g. SF10/250k and SF8/125k have identical chirp rates).
    
    Normalize quality by 10*log10(2^SF) before comparing across SFs,
    since higher SFs have larger FFTs giving a free peak-to-mean boost."""
    from config import MESHTASTIC_PRESETS

    # Build list: hint preset first (if it matches), then all others
    presets = []
    for name, p in MESHTASTIC_PRESETS.items():
        presets.append((name, p['sf'], p['bw']))
    # Sort: hint-SF presets first, then by SF ascending, then by BW ascending.
    # BW ascending ensures SHORT_FAST (250k) is tested before SHORT_TURBO (500k)
    # for the same SF — critical because generate_downchirp(sf, bw, fs=bw*2) is
    # phase-identical for all BWs at the same SF (phase = 2π(-k/4 + k²/1024)),
    # so when nb_fs < 2*bw_wider the wider-BW dec=1 path accidentally dechirps
    # with the same downchirp as the narrower BW and scores equally well.
    presets.sort(key=lambda x: (0 if x[1] == sf_hint else 1, x[1], x[2]))

    best_name, best_sf, best_bw = "?", sf_hint, BW_LIST[3]
    best_q_norm, best_q_raw = -99.0, -99.0
    best_pbin = 0

    # Cache decimated signals by dec factor — 9 presets share only 4 unique
    # dec values (500k→1, 250k→2, 125k→4, 62.5k→8) so without caching 5 of
    # the 9 decimate_by calls recompute an identical FFT(len(nb_widest)).
    _dec_cache = {}

    for name, sf, bw in presets:
        target_fs = bw * 2
        # Skip if nb_fs is too low to properly represent this BW.
        # dechirp_peak_quality uses fs=bw*2 internally; if nb_fs < bw*2 then
        # dec=max(1, round(<1))=1 so nb is passed at the wrong rate.  Because
        # all SF-matched downchirps are phase-identical at the sample level
        # (phase = 2π(-k/4 + k²/1024)), the quality score would be the same as
        # for the correct narrower BW — causing false BW identification.
        if fs_widest < target_fs * 0.9:
            continue
        dec = max(1, int(round(fs_widest / target_fs)))
        if dec not in _dec_cache:
            _dec_cache[dec] = decimate_by(nb_widest, dec)
        nb = _dec_cache[dec]
        if len(nb) < (2 ** sf) * 2:
            continue
        q_raw, pbin = dechirp_peak_quality(nb, sf, bw)
        q_norm = q_raw - 10.0 * np.log10(2 ** sf)
        if debug >= 2:
            print(f"    Preset {name:>17s} SF{sf:2d}/BW{bw/1000:>6.1f}k: "
                  f"q={q_raw:.1f}dB norm={q_norm:.1f}dB",
                  file=sys.stderr)
        if q_norm > best_q_norm:
            best_name, best_sf, best_bw = name, sf, bw
            best_q_norm, best_q_raw = q_norm, q_raw
            best_pbin = pbin
            # Early exit: if normalized quality is very high, no other preset will beat it
            if q_norm > -2.0 and q_raw > 18.0:
                break

    if debug >= 2:
        print(f"    → Best: {best_name} SF{best_sf}/BW{best_bw/1000:.0f}k "
              f"q={best_q_raw:.1f}dB norm={best_q_norm:.1f}dB",
              file=sys.stderr)

    return best_sf, best_bw, best_q_raw, best_pbin




from datetime import datetime


def estimate_and_correct_cfo(nb, nb_fs, sf, bw, debug=0):
    """CFO correction stub — intentionally disabled.

    Both the analysis script (analyze_capture.py) and the decoder
    (decoder.py) measure the preamble bin and apply their own
    CFO correction internally.  Pre-correcting here adds no value and
    has historically caused catastrophic bugs (wrong chirp convention,
    grid misalignment, autocorrelation failures).

    The recording is saved as-is.  Downstream tools handle it.

    Returns (nb, 0.0) — signal unchanged, zero correction.
    """
    return nb, 0.0


class _StreamDecimator:
    """Streaming mix-to-DC + polyphase FIR decimator (integer factor D).

    Purpose: NARROWBAND PENDING (2026-07-06).  Deferred capture jobs used
    to hold WIDEBAND hop references (80 MB per 0.5 s hop at 20 Msps; a
    24-window slow tail retained ~1.9 GB) and every finalize — including
    each break_pending flush at a ring skip — dragged a multi-GB
    assembly through concat + forward FFT in the save worker.  That
    memory-bandwidth storm starved the SDR reader (14.8-17.5 of
    20 Msps), causing the next skip: a feedback loop.  Cropping each
    part to ±export_bw around the detection carrier AS IT ARRIVES makes
    pending tails KB-scale and finalize trivial.

    Chunk-fed but mathematically identical to filtering the whole
    stream at once (FIR locality — no per-chunk spectral edge effects,
    unlike chunked FFT crops, which would plant a seam artifact every
    hop; the capture-splice arc proved how sensitive slow-SF decode is
    to seam defects).  Definition, with X = input mixed to DC and taken
    as zero outside the stream:

        y[m] = sum_k h[k] * X[m*D + H - k],   k in [0, L)

    h = firwin(20*D+1, 1/D, kaiser beta 5.0) — the same kernel family
    scipy.signal.resample_poly uses; H = (L-1)/2 = 10*D exactly, so the
    output grid sits on absolute input multiples of D (group delay
    exactly compensated) and stays aligned across arbitrary chunk
    boundaries.  Output rate = wb_fs / D — exactly 2*bw for every
    standard preset at 20 Msps, matching the decoder's fs = bw*dec
    reconstruction with zero rate error.
    """

    _MIX_BS = 8192   # mixer block size (see _mix)

    def __init__(self, f_off_hz, wb_fs, D):
        from scipy.signal import firwin
        self.D = int(D)
        self.fs = float(wb_fs)
        self.f = float(f_off_hz)
        self.h = firwin(20 * self.D + 1, 1.0 / self.D,
                        window=('kaiser', 5.0)).astype(np.float32)
        self.L = len(self.h)
        self.H = (self.L - 1) // 2          # = 10*D, divisible by D
        # Polyphase-as-GEMM decomposition (the naive per-output dot and
        # scipy's upfirdn both ran ~0.5 s per 10 M-sample hop — too slow
        # for the cropper to keep up with the hop cadence).  With h
        # zero-padded to K*D taps and hp2[j, p] = h_pad[(j+1)*D - 1 - p],
        #     y[m] = sum_j ( seg[c_m - (j+1)D + 1 : c_m - jD + 1] . hp2[j] )
        # and the inner windows for consecutive m are consecutive D-sample
        # rows of seg — so ONE contiguous reshape view of seg matmul'd
        # against hp2.T computes every (m, j) partial in a single BLAS
        # call, and y falls out as K diagonal slice-sums.  Exactly the
        # same arithmetic, ~10x faster.
        self._K = (self.L + self.D - 1) // self.D    # = 21
        _hpad = np.zeros(self._K * self.D, np.float32)
        _hpad[:self.L] = self.h
        self._hp2T = np.ascontiguousarray(
            _hpad.reshape(self._K, self.D)[:, ::-1].T).astype(np.complex64)
        # K*D leading zeros so every GEMM row index stays in range from
        # the first output on (the pad is X's zero-extension, explicit).
        self._hist = np.zeros(self._K * self.D, np.complex64)
        self._n0 = 0        # abs index of next input sample
        self._m_next = 0    # next output index to emit
        # invariant: abs index of _hist[0] is a multiple of D (H and K*D
        # are too), so the reshape rows land on the global output grid.

    def _mix(self, x):
        # Phase-exact heterodyne without a full-rate cos/sin pass: the
        # rotation at abs sample n0+b*BS+i factors into (per-block phasor)
        # x (within-block phasor).  Both factors' phases are computed mod
        # 1 in float64 — the absolute index (up to ~1e11 after hours)
        # never loses fractional phase and block errors cannot accumulate
        # (each block phasor is computed from scratch, not chained).
        n = len(x)
        d = self.f / self.fs
        BS = self._MIX_BS
        nb = (n + BS - 1) // BS
        _bph = (np.arange(nb, dtype=np.float64) * (BS * d)
                + self._n0 * d) % 1.0
        Rb = np.exp((-2j * np.pi) * _bph).astype(np.complex64)
        base = np.exp((-2j * np.pi) *
                      ((np.arange(BS, dtype=np.float64) * d) % 1.0)
                      ).astype(np.complex64)
        xm = np.empty(n, np.complex64)
        full = (n // BS) * BS
        if full:
            _v = xm[:full].reshape(-1, BS)
            np.multiply(x[:full].reshape(-1, BS), base, out=_v)
            _v *= Rb[:n // BS, None]
        if n - full:
            xm[full:] = x[full:] * base[:n - full] * Rb[n // BS]
        return xm

    def feed(self, x):
        """Feed the next contiguous chunk; returns 0+ output samples."""
        if len(x) == 0:
            return np.empty(0, np.complex64)
        xm = self._mix(np.ascontiguousarray(x, dtype=np.complex64))
        s = self._n0 - len(self._hist)      # abs index of seg[0]
        self._n0 += len(x)
        e = self._n0
        seg = np.concatenate((self._hist, xm))
        D, H, L, K = self.D, self.H, self.L, self._K
        m_hi = (e - 1 - H) // D             # last fully-supported output
        if m_hi < self._m_next:
            self._hist = seg
            return np.empty(0, np.complex64)
        n_y = m_hi - self._m_next + 1
        c_min = self._m_next * D + H - s    # seg index of y's first center
        n_rows = n_y - 1 + K
        W = seg[1 + (c_min // D - K) * D:
                1 + (c_min // D - K + n_rows) * D].reshape(n_rows, D)
        G = W @ self._hp2T                  # (n_rows, K) partial sums
        y = np.zeros(n_y, np.complex64)
        for j in range(K):
            y += G[K - 1 - j: K - 1 - j + n_y, j]
        self._m_next = m_hi + 1
        # keep L-1 (+ one extra D so the next GEMM's earliest row exists)
        keep_abs = self._m_next * D + H - (L - 1) - D
        keep_abs = min(keep_abs, e)
        keep_abs -= keep_abs % D            # preserve D | s
        keep_abs = max(keep_abs, s)
        self._hist = seg[keep_abs - s:]
        return y

    def finish(self):
        """Flush: emit every output whose center lies inside the stream."""
        if self._n0 == 0:
            return np.empty(0, np.complex64)
        return self.feed(np.zeros(self.H, np.complex64))


def _slope(sf, bw):
    """Chirp slope family key: bw^2 / 2^sf (rounded to kHz-scale).  Every
    (sf, bw) that shares this value produces the same physical up-chirp rate,
    so a down-alias (e.g. SF7/62.5k) collapses onto its real same-slope parent
    (SF9/125k).  Used by the recorder's cross-batch dedup to keep an alias from
    claiming its own keep-2 slots."""
    return round(bw * bw / (2.0 ** sf) / 1e3)


class SignalRecorder:
    """One file per detection, saved in a background thread.

    update() queues the wideband data immediately (fast, non-blocking) and
    returns.  A daemon thread does the narrowband FFT extraction, writes the
    .cf32 file, then submits to the decoder.  This prevents the main
    detection loop from stalling during the large-array FFT that was causing
    30-60 second blockages (and the resulting 68+ second CATCHUP spikes).

    Recording = wideband_buf + tail, trimmed per SF/BW.
    """

    def __init__(self, export_dir, wb_fs, center_hz, hop_n, debug=0):
        self.export_dir = export_dir
        self.wb_fs = wb_fs
        self.center_hz = center_hz
        self.hop_n = hop_n
        self.debug = debug
        os.makedirs(export_dir, exist_ok=True)
        # Reap orphan captures from a previous crashed run.  The pipeline holds
        # captures live only as long as the decoder refcount is non-zero; if
        # the prior process was killed (OOM, Ctrl-C, etc.) those files have no
        # live refcount and are dead weight.  At a fresh start nothing is in
        # flight, so any pre-existing CAPTURE is by definition orphaned.
        # PATTERN-RESTRICTED (2026-07-09): the export dir is user-configurable
        # now (web Advanced → Capture Storage) — an indiscriminate all-files
        # sweep would DELETE THE CONTENTS of any mistyped/shared directory the
        # user points it at.  Only remove what this pipeline writes: SF*.cf32
        # captures and .spool temporaries.
        try:
            for _f in os.listdir(export_dir):
                if not ((_f.startswith('SF') and _f.endswith('.cf32'))
                        or _f.endswith('.spool')
                        or _f == '.lora_write_test'):
                    continue
                _p = os.path.join(export_dir, _f)
                if os.path.isfile(_p):
                    try:
                        os.unlink(_p)
                    except OSError:
                        pass
        except OSError:
            pass

        self._decoder = None
        # RAM-PROPORTIONAL SPOOL (UC audit (s) re-exam 2026-07-12): the class
        # default (600 MB) is the threshold above which a save job spills its
        # wideband concat to a disk-backed memmap instead of holding it in RAM.
        # With the save queue bounded at 8, up to 8 concurrent wideband jobs can
        # sit in RAM — 8×600 MB ≈ 4.8 GB, an OOM on a small-RAM host.  Lower the
        # threshold on such hosts so ~8 in-flight jobs stay under ~1/4 of system
        # RAM (spilling sooner just trades RAM for disk I/O — never data loss).
        # Big-RAM hosts keep the 600 MB default (min() only lowers, never
        # raises).  Mostly a legacy-path guard: with LORA_NB_PENDING the pool
        # commits queue KB-scale NB streams, not wideband arrays.
        _memtot_b = 8 << 30      # fallback if /proc/meminfo unreadable
        try:
            with open('/proc/meminfo') as _mi:
                _memtot_b = next(int(l.split()[1]) for l in _mi
                                 if l.startswith('MemTotal')) * 1024
            self._SPOOL_RAM_LIMIT = min(type(self)._SPOOL_RAM_LIMIT,
                                        _memtot_b // 32)
        except (OSError, StopIteration, ValueError):
            pass
        self._memtot_b = _memtot_b
        # BOUNDED queue: each item carries a full wideband gate window (~300 MB
        # at 28 Msps).  An unbounded queue OOM-killed the process on a 60-msg
        # run when the save-worker fell behind (queue hit 129 → ~40 GB).  With
        # a small maxsize the detector blocks (backpressure) instead of growing
        # memory without bound.  For offline file replay this only slows the
        # detector (no data lost); for live it caps memory at maxsize×window.
        self._save_queue = queue.Queue(maxsize=8)
        self._pending = []   # deferred-tail jobs (live path)
        # NARROWBAND PENDING (2026-07-06): deferred jobs crop every part
        # to ±export_bw around each detection carrier in this background
        # thread instead of retaining wideband hop refs until finalize.
        # See _StreamDecimator's docstring for the bandwidth-storm
        # mechanism this kills.  LORA_NB_PENDING=0 restores the legacy
        # wideband PARTS path (also the automatic fallback if scipy is
        # unavailable).  Queue items hold hop REFS only until cropped
        # (~0.1-0.2 s), bounded so a drowning cropper can never retain
        # unbounded wideband memory.
        self._nb_mode = False
        self._crop_queue = None
        # HOST-DERIVED crop-queue bound (codex OOM review 2026-07-23): a BASE
        # item can hold a ~wb_fs*8-byte wideband window copy (160 MB @20 Msps);
        # the old fixed maxsize=48 = a 7.7 GB ceiling that OOMs a small-RAM host
        # once the single crop worker starves under detect-worker CPU contention
        # (the 20 Msps Pi's second OOM sink).  Budget a modest RAM slice AFTER
        # the ring+pool+OS reserve, in BYTES, and derive the entry count from the
        # worst-case item size — big-RAM hosts still get up to 48, the Pi ~6.
        # Env override is a kill-switch only (never the primary tuning knob).
        _win_bytes = max(1, int(self.wb_fs * 8))         # ~1 s wideband, complex64
        _crop_budget_b = self._memtot_b // 12            # ~0.66 GB / 8 GB host
        _cq_max = max(2, min(48, _crop_budget_b // _win_bytes))
        try:
            _cq_max = max(1, int(os.environ.get('LORA_CROP_QUEUE_MAX', _cq_max)))
        except ValueError:
            pass
        self._crop_overflow_drops = 0        # telemetry: hard-dropped jobs at queue-full
        self._crop_overflow_last_warn = 0.0
        if os.environ.get('LORA_NB_PENDING', '1') != '0':
            try:
                from scipy.signal import firwin, upfirdn  # noqa: F401
                self._crop_queue = queue.Queue(maxsize=_cq_max)
                print(f"[RING] decode crop queue: maxsize={_cq_max} "
                      f"(~{_cq_max * _win_bytes / 1e9:.1f} GB ceiling; "
                      f"host-derived from {self._memtot_b/1e9:.0f} GB RAM, "
                      f"{_win_bytes/1e6:.0f} MB/window)", file=sys.stderr)
                t = threading.Thread(target=self._crop_worker,
                                     daemon=True, name='recorder-crop')
                t.start()
                self._crop_thread = t
                self._nb_mode = True
            except Exception:
                self._nb_mode = False
        # Parallel save-worker pool.  Each gate-batch needs a 28-M-sample
        # forward FFT (~1 s wall-clock) plus find_all_preambles + per-
        # preamble extracts.  With a single worker thread, bursty gate
        # fires during a sendtest caused save_q to grow to 15-19, the
        # main loop fell behind real-time, and the Welch-window phase
        # drift caused some packets at window edges to miss detection.
        # Running 4 workers in parallel drains the queue ~4× faster
        # (numpy FFT calls release the GIL so threads truly parallelise).
        self._save_threads = []
        for _i in range(4):
            t = threading.Thread(target=self._save_worker, daemon=True,
                                 name=f'recorder-{_i}')
            t.start()
            self._save_threads.append(t)
        # Keep self._save_thread for legacy code paths that .join() it.
        self._save_thread = self._save_threads[0]
        # Post-CFO global carrier registry: keyed by (freq_bucket, sf, bw),
        # value = timestamp of last save.  Persists across batches so that
        # cross-window duplicates (same packet detected in two consecutive
        # wideband windows, different Welch estimates) are caught by corrected
        # carrier frequency rather than the noisy pre-correction estimate.
        self._saved_carriers: dict = {}
        self._saved_carriers_lock = threading.Lock()
        # Raw Schmidl-Cox peak registry: used to eliminate ±BW aliases before
        # the expensive CFO re-extraction work.  The SC autocorrelation always
        # produces a mirror peak at ±BW from every real signal; this cache lets
        # us detect and skip those aliases using the raw SC-estimated frequency
        # (which is reliably ≈BW away from the real signal, unlike the
        # post-correction carrier which can differ due to alias-band CFO maths).
        self._saved_sc_freqs: dict = {}
        self._saved_sc_lock = threading.Lock()
        # Absolute-preamble-time dedup registry: list of (freq_hz, abs_t_s) for
        # recently-saved captures.  abs_t_s = window_sample_pos/wb_fs +
        # preamble_t_s, an RF-accurate timestamp that is IDENTICAL across the
        # overlapping windows that re-detect one packet, so it collapses the
        # ~5× redundant captures (the dominant decode-time cost) WITHOUT merging
        # hop0/hop1 (which are ~0.9 s apart).  Unlike the old wall-clock dedup
        # (disabled — offline replay ran slower than real-time and compressed
        # the window, eating real hop1s), RF time is immune to processing speed.
        self._saved_abs_t: list = []
        self._saved_abs_t_lock = threading.Lock()
        # How many captures to keep per (carrier±BW/2, RF-time±0.35s) packet.
        # LIVE lean path sets LORA_DEDUP_KEEP=1 (decode each packet ONCE — no
        # redundant decodes, the dominant resource cost).  Offline keeps 2 (a
        # fallback capture in case the strongest is off-centre).  Decode-once is
        # safe when the extraction is well-centred on the true carrier (fast
        # single-pass decode); the gate's per-preamble carrier estimate provides
        # that.
        self._dedup_keep = int(os.environ.get('LORA_DEDUP_KEEP', '2'))
        # Option A — slope-family-aware cross-batch dedup.  Collapse a same-slope
        # alias (SF7/62.5k = down-alias of real SF9/125k) onto its real parent
        # cluster so it does NOT get its own keep-2 slots + decode jobs; when a
        # cluster is full, a higher-quality (bw_quality_db) member evicts the
        # weakest slot so the true parent always wins over the 3-6 dB-weaker
        # alias.  LORA_SLOPE_DEDUP=0 → exact per-(sf,bw) HEAD behaviour.
        self._slope_dedup = os.environ.get('LORA_SLOPE_DEDUP', '1') != '0'
        # Option B — load-aware keep-1.  When the decode backlog (dec_q =
        # decoder.pending()) is above threshold, drop effective keep 2→1 so a
        # real beacon's 2nd fallback capture is not queued while the single slow
        # worker is saturated.  LORA_KEEP1_UNDER_LOAD=0 → legacy always-keep-2.
        self._keep1_under_load = os.environ.get('LORA_KEEP1_UNDER_LOAD', '1') != '0'
        self._keep1_load_thresh = int(
            os.environ.get('LORA_KEEP1_LOAD_THRESH', '6'))
        # Record-time same-slope alias gate margin (dB).  A same-slope member
        # whose raw dechirp quality is more than this below a different-label
        # sibling already in its cluster is the down-alias and is not recorded.
        # 2.5 dB sits safely inside the measured 3-6 dB alias-vs-parent gap
        # while never gating a genuinely-strongest member.  0 disables the gate
        # (keeps only the registry eviction).
        try:
            self._slope_margin_db = float(
                os.environ.get('LORA_SLOPE_MARGIN_DB', '2.5'))
        except ValueError:
            self._slope_margin_db = 2.5
        # Strong-carrier registry for GLOBAL IQ-image rejection.  detect_preamble
        # rejects images only when the stronger real signal is co-detected in
        # the SAME 1 s window; with 50 % window overlap an image is often saved
        # from a neighbouring window where the real wasn't in that window's peak
        # list (~2/3 of images leaked through the per-window check).  This
        # cross-batch list of (freq_hz, power_db, abs_t_s) for strong real
        # carriers lets the save-worker drop ANY preamble that is the mirror
        # (2*center - f, ±60 kHz) of a much-stronger (>10 dB) carrier at ~the
        # same RF time (images are simultaneous with their source).
        self._img_carriers: list = []
        self._img_lock = threading.Lock()
        # Per-save filename uniquifier.  Different jobs finalized in the
        # same tight loop (flush_pending at EOF / break_pending) get the
        # same millisecond dt_str; same-carrier packets then collide on
        # one filename and two parallel save workers write the same path
        # CONCURRENTLY — one tofile truncates while the other is mid-
        # write, corrupting the capture (observed: 5.5 s of real samples
        # + a 15 s zero hole where frame 11 should be).  The legacy
        # GB-scale extraction serialized saves by accident; the
        # narrowband path made them fast enough to actually race.
        self._fname_seq = itertools.count()

    def set_decoder(self, decoder):
        self._decoder = decoder

    def _crop_overflow_warn(self):
        """Rate-limited overload notice when decode crop jobs are hard-dropped
        because the crop worker can't keep up (host too slow at this rate)."""
        _now = time.time()
        if _now - self._crop_overflow_last_warn >= 5.0:
            self._crop_overflow_last_warn = _now
            print(f"[OVERLOAD] decode crop queue full — "
                  f"{self._crop_overflow_drops} job(s) hard-dropped so far "
                  f"(detection continues; some decodes lost at this rate)",
                  file=sys.stderr, flush=True)

    def update_deferred(self, detections, wideband_buf, sample_pos=0,
                        pre_hop=None, need_tail_n=0, owned_buf=False,
                        gap_parts=None, tail_parts=None):
        """Register a save whose TAIL arrives later: instead of BLOCKING
        the main loop for an airtime-sized ring read (2.4 s per SF11/125k
        detection — measured as the dominant live-loop stall: CATCHUP
        ring-wrap skips ate ~every 10th beacon in serial mode and still
        bit under load in pool mode), hold the job and let feed_tail()
        complete it from the hops that flow through the loop anyway —
        the SAME samples the blocking read would have consumed, with
        zero loop stall. Memory-bounded: max 3 pending; overflow
        finalizes the oldest with whatever tail it has (== the old
        truncation behavior at worst)."""
        if not detections:
            return []
        # CONFLICTING-LABEL PRE-RESOLVE (2026-07-08): a slow-scan result can
        # carry the SAME burst under two labels (matrix ground truth,
        # SF12/62.5k: true label at sc 0.98 + its half-lag alias SF10/31.25k
        # at sc 0.67 from the same assembly's two lag buckets).  The NB crop
        # is set up at job creation from the det list — BEFORE the finalize
        # batch-dedup picks a winner — so the job was cropped at the FIRST
        # det's rate and the save came out a chimera (winner's freq, loser's
        # label/rate) that could never decode.  Resolve same-carrier label
        # conflicts HERE with the same (power, detect_conf) order the
        # finalize uses; time-separated packets inside one capture are
        # handled downstream by per-det plateaus, not by this list.
        if len(detections) > 1:
            # patience_trial dets sort LAST regardless of their (saturated)
            # placeholder conf/power — a rotating trial HYPOTHESIS must never
            # evict a real detection's job for the same frame (soundness
            # review follow-up: trial jobs beat g31_sf12's real slow-pass
            # jobs in this arbitration and 2 of 3 decodes vanished).
            _srt = sorted(detections,
                          key=lambda d: (bool(d.get('patience_trial')),
                                         -d.get('peak_power_db', 0.0),
                                         -d.get('detect_conf', 0.0)))
            _pre = []
            for _d in _srt:
                if any(abs(_d['freq_hz'] - _k['freq_hz']) < 6e3
                       and (_d['sf'], _d['bw']) != (_k['sf'], _k['bw'])
                       for _k in _pre):
                    continue
                _pre.append(_d)
            detections = _pre
        # ONE PENDING JOB PER PHYSICAL FRAME: overlapping windows re-detect
        # the same long frame every hop, and each registration held a full
        # wideband base (up to 640 MB for slow-pass assemblies) — the
        # 3-job cap then EVICTED older jobs with truncated tails before
        # their 14-19 s fill completed. Soak fingerprint: 131 slow-pass
        # detections, ZERO long-frame decodes. A same-carrier same-sf
        # registration while a matching job is still filling is the same
        # frame — skip it (the save-side keep-2 dedup still sees the
        # finalized captures).
        for _j in self._pending:
            for _jd in _j['dets']:
                for _d in detections:
                    if (_d['sf'] == _jd['sf'] and _d['bw'] == _jd['bw']
                            and abs(_d['freq_hz'] - _jd['freq_hz'])
                            < max(_d['bw'], 60e3)):
                        # FRAME-TIME AWARE: same carrier+sf but arriving
                        # more than ~a frame-length after the filling
                        # job's registration is the NEXT frame, not a
                        # redetect — continuous back-to-back phases
                        # (frame >= beacon interval) otherwise lose
                        # EVERY second frame to this dedup.
                        # REDETECT window, not worst-case airtime: multi-
                        # window redetects of ONE frame register within
                        # ~2 s (hop cadence); genuine next-frames in
                        # continuous phases arrive at the beacon interval.
                        # The old 0.8x max-airtime bound (11.7 s at
                        # 41.67k) swallowed every other REAL frame —
                        # closed-loop synthetic pipeline measured 2/4
                        # captures at 8.5 s spacing.
                        _air_n = int(148.25 * (2 ** _jd['sf'])
                                     / _jd['bw'] * self.wb_fs)
                        if (sample_pos - _j['sample_pos']
                                > min(0.25 * _air_n,
                                      int(3.0 * self.wb_fs))):
                            continue
                        import os as _os_dbg
                        if _os_dbg.environ.get('LORA_SLOW_DEBUG'):
                            print(f"[UPDEF-SKIP] new SF{_d['sf']}/{_d['bw']} vs pending SF{_jd['sf']}/{_jd['bw']} f={_d['freq_hz']/1e6:.4f}", flush=True)
                        return []
        # copy ONLY what will be overwritten: the sliding window buf.
        # pre_hop and slow assemblies are already owned arrays — hold refs.
        _own = (wideband_buf if owned_buf
                else np.array(wideband_buf, copy=True))
        parts = []
        if pre_hop is not None and len(pre_hop) > 0:
            parts.append(pre_hop)
        parts.append(_own)
        if gap_parts:
            # ASYNC-SCAN GAP FILL (2026-07-06): the background slow scan
            # registers 1-2 iterations after its assembly snapshot; the
            # hops that streamed through DURING the scan were never fed
            # (feed_tail predates the job) — every live slow-pass capture
            # carried a 0.5-1.0 s HOLE at the base|tail seam, exactly
            # where slow frames' SFDs land (SEAMDBG-measured: gap =
            # 10,000,000 samples). File mode runs the scan inline (no
            # gap) — this single defect was the file-clean/live-broken
            # decode split. The caller passes the missed hops (still in
            # its deque) as owned refs; they extend the base seamlessly.
            parts.extend(gap_parts)
        pre_hop_n = len(pre_hop) if pre_hop is not None else 0
        if os.environ.get('LORA_SEAM_DEBUG'):
            print(f"[SEAMDBG] register: sample_pos={sample_pos} "
                  f"base_n={sum(len(p) for p in parts)} "
                  f"base_span=[{sample_pos - sum(len(p) for p in parts)},"
                  f"{sample_pos})", flush=True)
        _gap_n = sum(len(g) for g in gap_parts) if gap_parts else 0
        # tail_parts (pool commit) SEEDS the in-flight tail.  'need' is the FULL
        # airtime (need_tail_n): if the seed covers it (short frame) the job
        # finalizes now; if not (long slow-preset frame whose tail exceeds the
        # in-flight window depth) it PENDS and feed_tail completes it from
        # future loop hops (UC audit p — was silently truncated to the
        # n_slots-hop ceiling).  Fallback to the seed length if no need given.
        _need = (int(need_tail_n) if need_tail_n
                 else (sum(len(t) for t in tail_parts)
                       if tail_parts is not None else 0))
        job = {'base_parts': parts, 'dets': list(detections),
               'pre_hop_n': pre_hop_n, 'wb_n': len(wideband_buf) + _gap_n,
               'sample_pos': sample_pos, 'tail': [], 'got': 0,
               'need': _need}
        if self._nb_mode:
            _nb = self._nb_setup(job['dets'])
            if _nb is not None:
                try:
                    job['nb'] = _nb
                    # the cropper owns the base parts from here; the job
                    # itself retains NO wideband references
                    self._crop_queue.put_nowait(('BASE', job, parts))
                    job['base_parts'] = []
                except queue.Full:
                    # Cropper drowning (overload).  HARD-DROP this job — do NOT
                    # fall back to the legacy WIDEBAND-retaining path (the old
                    # `del job['nb']` let job['tail']=wideband land in _pending +
                    # the save queue, so the queue-full path GREW retained memory
                    # instead of shrinking it — the 20 Msps Pi's decode OOM sink,
                    # codex review).  The queue-full path must monotonically
                    # reduce memory: drop the job, free its base copy, count it.
                    # Losing this detection's DECODE is the correct behaviour at
                    # ~4x decode overload; detection + other jobs are unaffected.
                    self._crop_overflow_drops += 1
                    parts = None            # release the ~160 MB base copy now
                    self._crop_overflow_warn()
                    return []
        if os.environ.get('LORA_SLOW_DEBUG'):
            print(f"[UPDEF-JOB] created dets={[(d['sf'], d['bw'], round(d['freq_hz']/1e6,4)) for d in job['dets']]} need={job['need']}", flush=True)
        # POOL-MODE SEED (UC audit q + p): feed the fully-known in-flight tail
        # through the SAME NB cropper the serial live path uses — the save queue
        # then holds KB-scale NB streams, not the ~610 MB wideband array
        # recorder.update() queued (the >=8-core / web detect_workers=-1 OOM
        # driver).  If the seed already covers the airtime, finalize now (short
        # frames — low latency, no pend).  Otherwise the frame is LONGER than
        # the in-flight window depth: PEND, and feed_tail completes it from
        # future loop hops (the seed + future hops are contiguous — the
        # in-flight forward-overlaps ARE the _hop_own's already emitted).
        if tail_parts is not None:
            if 'nb' in job:
                for _h in tail_parts:
                    try:
                        self._crop_queue.put_nowait(('HOP', job, _h))
                        job['got'] += len(_h)
                    except queue.Full:
                        # cropper full mid-tail: truncate here (same
                        # degradation feed_tail takes) rather than stall the
                        # realtime commit path.
                        break
            else:
                job['tail'] = list(tail_parts)
                job['got'] = sum(len(t) for t in tail_parts)
            if job['got'] >= job['need']:
                self._finalize(job)     # seed covers the airtime — done
                return []
            # long frame — feed_tail finishes it (bounded by the 3-job cap)
        self._pending.append(job)
        if len(self._pending) > 3:
            if os.environ.get('LORA_SLOW_DEBUG'):
                print(f"[UPDEF-EVICT] cap overflow — finalizing oldest early", flush=True)
            self._finalize(self._pending.pop(0))
        return []

    def feed_tail(self, hop):
        """Give every pending deferred job the freshly-arrived hop; jobs
        finalize once their tail requirement is met. ZERO-COPY: the hop
        is an owned array (the caller's per-hop copy) — jobs hold
        REFERENCES; trimming and concatenation happen in the save
        worker. (Per-job copies here cost 46 ms median / 1.48 s max on
        the loop during long-frame phases — the top stall driver.)"""
        if not self._pending:
            return
        done = []
        for job in self._pending:
            if job['got'] < job['need']:
                if job['got'] == 0 and os.environ.get('LORA_SEAM_DEBUG'):
                    _ff = getattr(self, '_dbg_tot_s', 0) - len(hop)
                    print(f"[SEAMDBG] first-feed: base_end={job['sample_pos']} "
                          f"feed_start={_ff} gap={_ff - job['sample_pos']} "
                          f"({(_ff - job['sample_pos'])/self.wb_fs*1000:.2f} ms)",
                          flush=True)
                if 'nb' in job:
                    # pre-slice the overshoot (legacy trims via _tneed in
                    # the save worker; the cropper must never see samples
                    # past 'need' or the stream lengths diverge)
                    _room = job['need'] - job['got']
                    _h = hop if len(hop) <= _room else hop[:_room]
                    try:
                        self._crop_queue.put_nowait(('HOP', job, _h))
                        job['got'] += len(_h)
                    except queue.Full:
                        # dropping a mid-stream hop would SPLICE the
                        # capture (the exact defect the break_pending arc
                        # fixed) — finalize now with the contiguous data
                        # already cropped instead
                        print("         [NB-CROP] queue full — early "
                              "finalize of a pending job", flush=True)
                        job['got'] = job['need']
                else:
                    job['tail'].append(hop)
                    job['got'] += len(hop)
            if job['got'] >= job['need']:
                done.append(job)
        for job in done:
            self._pending.remove(job)
            self._finalize(job)

    def flush_pending(self, wait=False):
        for job in self._pending:
            self._finalize(job)
        self._pending = []
        if wait and self._crop_queue is not None:
            # shutdown path only — the live loop must never block here
            self._crop_queue.join()

    def break_pending(self, reason=''):
        """Stream DISCONTINUITY (ring CATCHUP skip, sample drops): the
        hops that arrive next are NOT contiguous with pending tails —
        appending them would SPLICE a multi-second gap invisibly into
        the capture (measured: post-seam symbols land ~0.44 sym off the
        frame grid; every continuous-mode bench capture carried such a
        splice and decoded as structured garbage). Finalize everything
        with the contiguous data collected so far — a truncated real
        capture beats a spliced one."""
        if self._pending:
            print(f"         [PENDING-BREAK] finalizing "
                  f"{len(self._pending)} job(s) at discontinuity "
                  f"{reason}", flush=True)
        self.flush_pending()

    def _finalize(self, job):
        dt_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        if 'nb' in job:
            # narrowband path: the cropper finishes the decimators and
            # forwards a KB-scale NB item to the save queue.  FIFO on the
            # crop queue guarantees every earlier BASE/HOP of this job is
            # cropped first.  Blocking put (rare — FINAL items are one per
            # capture): losing a FINAL would silently drop the capture.
            try:
                self._crop_queue.put(('FINAL', job, dt_str), timeout=30.0)
            except queue.Full:
                # Cropper dead/stuck (UC audit: unsupervised single point of
                # failure): losing this capture beats wedging the gate's
                # finalize path forever behind a blocking put.
                print('[RECORDER] crop queue stuck 30s — DROPPING capture '
                      'finalize (cropper dead?)', file=sys.stderr, flush=True)
            return
        # heavy concat happens in the SAVE WORKER (the queue item carries
        # the parts list); trim the tail to 'need' there too
        self._save_queue.put((('PARTS', job['base_parts'], job['tail'],
                               job['need']), job['dets'], dt_str,
                              job['pre_hop_n'], job['wb_n'],
                              job['sample_pos']))

    def _nb_setup(self, detections):
        """Build per-detection streaming decimators for a deferred job.

        D is anchored on the strongest detection's bw — the same rule the
        save worker uses for export_bw — so every capture from this job
        comes out at the rate the legacy full-array extraction would have
        produced.  Returns None when decimation is degenerate (D < 2);
        the caller then keeps the legacy wideband path.
        """
        anchor = max(detections, key=lambda x: x.get('peak_power_db', 0.0))
        _exp_dec = int(os.environ.get('LORA_EXPORT_DEC', '2'))
        # patience_trial anchors size for the same-slope rung FAMILY (see
        # patience_cap_params): an alias-BW crop cannot contain the true
        # signal.  Non-trial anchors: identical to the old anchor['bw'].
        export_bw = min(int(patience_cap_params(anchor)[0] * _exp_dec / 2),
                        int(self.wb_fs) // 2)
        D = int(round(self.wb_fs / float(export_bw * 2)))
        if D < 2:
            return None
        decs = {}
        for _i, _d in enumerate(detections):
            _d['_nb_ix'] = _i
            decs[_i] = _StreamDecimator(_d['freq_hz'] - self.center_hz,
                                        self.wb_fs, D)
        return {'D': D, 'nb_fs': self.wb_fs / D, 'export_bw': export_bw,
                'decs': decs, 'parts': {i: [] for i in decs}}

    def _crop_worker(self):
        """Background thread: crop deferred-job parts to narrowband as
        they arrive, then hand finalized KB-scale streams to the save
        queue.  Single thread == in-order processing per job (BASE, then
        HOPs, then FINAL), which the decimator state requires."""
        while True:
            item = self._crop_queue.get()
            if item is None:
                self._crop_queue.task_done()
                break
            try:
                kind, job = item[0], item[1]
                nb = job.get('nb')
                if nb is None:
                    pass                      # job degraded to legacy
                elif kind == 'BASE':
                    for _p in item[2]:
                        self._crop_feed(nb, _p)
                elif kind == 'HOP':
                    self._crop_feed(nb, item[2])
                elif kind == 'FINAL':
                    streams = {}
                    for _ix, _dec in nb['decs'].items():
                        _t = _dec.finish()
                        if len(_t):
                            nb['parts'][_ix].append(_t)
                        _ps = nb['parts'][_ix]
                        streams[_ix] = (np.concatenate(_ps) if _ps
                                        else np.empty(0, np.complex64))
                        nb['parts'][_ix] = []
                    nb['decs'] = {}
                    self._save_queue.put(
                        (('NB', {'streams': streams,
                                 'nb_fs': nb['nb_fs'],
                                 'export_bw': nb['export_bw']}),
                         job['dets'], item[2], job['pre_hop_n'],
                         job['wb_n'], job['sample_pos']))
            except Exception as e:
                print(f"[NB-CROP] error: {e}", file=sys.stderr, flush=True)
            finally:
                self._crop_queue.task_done()

    @staticmethod
    def _crop_feed(nb, arr):
        for _ix, _dec in nb['decs'].items():
            _out = _dec.feed(arr)
            if len(_out):
                nb['parts'][_ix].append(_out)

    _SPOOL_RAM_LIMIT = 600_000_000   # bytes; above this, spool to disk

    def update(self, detections, wideband_buf, sample_pos=0, tail=None, pre_hop=None):
        """Queue a save job — returns immediately, work happens in background."""
        if not detections:
            return []

        # Concatenate NOW (fast memcopy) so the caller can recycle its buffers.
        parts = []
        if pre_hop is not None and len(pre_hop) > 0:
            parts.append(pre_hop)
        parts.append(wideband_buf)
        if tail is not None and len(tail) > 0:
            parts.append(tail)
        extended_full = np.concatenate(parts)  # owned copy, safe to hand off

        pre_hop_n = len(pre_hop) if pre_hop is not None else 0
        dt_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        self._save_queue.put((extended_full, list(detections), dt_str,
                              pre_hop_n, len(wideband_buf), sample_pos))
        return []  # file paths reported asynchronously by the save thread

    def _save_worker(self):
        """Background thread: FFT-extract → write file → notify decoder."""
        while True:
            item = self._save_queue.get()
            if item is None:
                self._save_queue.task_done()
                break
            try:
                extended_full, detections, dt_str, pre_hop_n, wb_n, sample_pos = item
                _spool_path = None
                _nb_job = None
                if (isinstance(extended_full, tuple)
                        and extended_full[0] == 'NB'):
                    # NARROWBAND PENDING: the cropper already extracted
                    # one stream per detection — no wideband concat, no
                    # forward FFT.  Everything below that touches
                    # extended_full is skipped for these items.
                    _nb_job = extended_full[1]
                    extended_full = None
                if (isinstance(extended_full, tuple)
                        and extended_full[0] == 'PARTS'):
                    _, _bparts, _tparts, _tneed = extended_full
                    _tp, _tn = [], 0
                    for _t in _tparts:
                        if _tn >= _tneed:
                            break
                        _tp.append(_t[:_tneed - _tn]
                                   if _tn + len(_t) > _tneed else _t)
                        _tn += len(_tp[-1])
                    _all = _bparts + _tp
                    _total_n = sum(len(p) for p in _all)
                    if _total_n * 8 > self._SPOOL_RAM_LIMIT:
                        # MEMMAP SPOOL (2026-07-06): multi-GB RAM concats of
                        # wideband tails were the bandwidth bomb starving
                        # the reader (14.8-17.5 of 20 Msps under load, skip
                        # feedback loop). Spool parts sequentially to a
                        # disk-backed memmap; the chunked extraction reads
                        # from it and RAM traffic collapses to the
                        # narrowband output.
                        import tempfile as _tf
                        _fd, _spool_path = _tf.mkstemp(
                            suffix='.spool', dir=self.export_dir)
                        os.close(_fd)
                        _mm = np.memmap(_spool_path, dtype=np.complex64,
                                        mode='w+', shape=(_total_n,))
                        _o = 0
                        for _p in _all:
                            _mm[_o:_o + len(_p)] = _p
                            _o += len(_p)
                        _mm.flush()
                        extended_full = _mm
                    elif _all:
                        extended_full = np.concatenate(_all)
                    else:
                        extended_full = np.empty(0, np.complex64)
                pre_hop_s = pre_hop_n / self.wb_fs

                # === Architectural shift ===
                # Old: one save per gate-stage detection.  Lost the second
                # packet whenever hop0 + hop1 sat in the same wideband
                # window at different crystal-offset carriers (the
                # extraction landed on one and the other had no chance).
                #
                # New: per BATCH, find ALL LoRa preambles in the recording
                # via the frequency-agnostic SC top-K + hybrid refinement,
                # then save one correctly-centred capture per preamble.
                # SF-agnostic — lag, bin width, and cluster thresholds all
                # scale with sf/bw.
                if not detections:
                    continue   # finally balances task_done — an inner call
                               # double-counts and RAISES, silently killing
                               # the worker thread (UC audit d)

                # Pick anchor sf/bw from the strongest detection.  We rely
                # on the gate stage to have picked the right preset for the
                # band being observed.
                anchor = max(detections,
                             key=lambda x: x.get('peak_power_db', 0.0))
                sf, bw = anchor['sf'], anchor['bw']
                N_sf = 2 ** sf

                # Register every carrier in this batch (freq, power, RF time) for
                # GLOBAL IQ-image rejection, and prune entries >3 s old.  Used in
                # the per-preamble loop below to drop a capture that is the mirror
                # of a much-stronger carrier at ~the same time.
                _batch_t0 = sample_pos / self.wb_fs
                with self._img_lock:
                    for _d in detections:
                        self._img_carriers.append((
                            float(_d['freq_hz']),
                            float(_d.get('peak_power_db', 0.0)),
                            _batch_t0 + float(_d.get('preamble_t_s', 0.0)),
                            float(_d.get('bw_quality_db', -99.0))))
                    if len(self._img_carriers) > 4000:
                        _cut = _batch_t0 - 3.0
                        self._img_carriers = [
                            c for c in self._img_carriers if c[2] > _cut]

                # Capture sizing: patience_trial anchors size for the same-
                # slope rung FAMILY (crop width AND symbol time — the alias
                # label's symbol time is half the true rung's for the 2-step
                # partner, which would truncate the true payload).  Non-trial
                # anchors get exactly the historic (bw, (2**sf)/bw).
                _cap_bw, sym_time = patience_cap_params(anchor)
                # Capture extent past the preamble = preamble (16) + SFD (4.25)
                # + sync (8) + payload-symbol budget.  Originally 120 syms covered
                # the absolute Meshtastic max (PL=237 / CR4/5 / LDRO) — but that
                # tail is mostly silence for the typical 20-90 byte Meshtastic
                # packet.  Slow SFs especially suffer: at SF12 sym_time≈33 ms so
                # the unused tail adds ~3 s per capture, which both raises
                # send→decoded latency by that much AND inflates the decode
                # workload proportionally.  100 syms covers Meshtastic up to
                # ~80 bytes after FEC (>99 % of real traffic incl. all standard
                # text, telemetry, position, nodeinfo, routing); larger packets
                # are truncated and may fail CRC.  Live SF11 4-core: latency
                # p50 18s → 11s with this trim, 20/20 hops preserved.
                # Tunable via LORA_PAYLOAD_SYM_BUDGET for users who need to
                # capture giant packets at the cost of latency.
                _pay_syms = int(os.environ.get('LORA_PAYLOAD_SYM_BUDGET', '100'))
                max_pkt_syms = 16 + 4.25 + 8 + _pay_syms
                max_pkt_s = max_pkt_syms * sym_time
                buf_dur_s = pre_hop_s + wb_n / self.wb_fs
                max_record_n = int((buf_dur_s + max_pkt_s) * self.wb_fs)
                extended = (extended_full[:max_record_n]
                            if _nb_job is None else None)
                # Export sample rate = bw * _exp_dec (nb_fs = export_bw*2).  Was
                # hard-wired to 8*bw (export_bw=4*bw → 4 MHz at BW500), an 8x
                # oversample that made every recenter-requiring decode ~4x slower
                # than necessary.  At the realtime decode budget that timed out
                # the off-centre (recenter-needing) hop1 captures and they were
                # dropped LIVE (they decode fine offline with more time).  Lower
                # oversampling = proportionally faster decode → the recenter
                # finishes inside the live budget.  _exp_dec=4 (2 MHz, span
                # ±1 MHz) keeps full ±BW recenter headroom; lower is faster but
                # tighter.  Tunable via LORA_EXPORT_DEC for validation.
                _exp_dec = int(os.environ.get('LORA_EXPORT_DEC', '2'))
                export_bw = min(int(_cap_bw * _exp_dec / 2), self.wb_fs // 2)

                # SOURCE OF TRUTH: the gate's detections list.  Each entry
                # is a (Welch peak + multi-lag SC + dechirp) confirmed
                # LoRa signal at a known carrier.  Save one file per
                # gate detection, centred on the gate-reported frequency.
                # find_all_preambles' second-pass SC scan was producing
                # alias-frequency saves that didn't match the real signal
                # — verified by offline spectrum analysis (run96 file
                # 926.7217 had its dominant signal at -1.62 MHz from DC,
                # i.e. absolute 925.1 MHz, not the gate-reported 926.77).
                # The gate's freq is already validated; trust it directly.
                _ext_fft_cache = []
                anchor_off = anchor['freq_hz'] - self.center_hz
                # nb_fs and sym_n needed for filename + decoder hint math.
                # NB items carry the cropper's exact realized rate
                # (wb_fs/D — equals export_bw*2 for standard presets at
                # standard wideband rates).
                nb_fs = (float(_nb_job['nb_fs']) if _nb_job is not None
                         else float(export_bw * 2))
                nb_fs_int = int(round(nb_fs))
                sym_n = int(round(N_sf * nb_fs_int / bw))
                # Build preamble list from gate detections, deduped by
                # ±1 LoRa bin (same TX seen in adjacent Welch peaks).
                # Also: suppress BW/2 spectral-fold aliases — a single LoRa
                # chirp's autocorrelation produces a spurious twin peak at
                # carrier ± BW/2 (see Robyns LoRa PHY).  Empirically (SF10
                # BW250 30-msg test, 2026-06-01): 16/30 packets produced
                # alias-paired captures at exactly df = BW/2 ± 5 kHz, same
                # timestamp.  These aren't different real packets (Meshtastic
                # / LoRaWAN channels are BW-spaced, not BW/2-spaced) — they
                # double the decoder load for SF10 and drive worker pool
                # saturation → gate sample drops.  Tolerance ±5 kHz is 10×
                # the post-centering scatter (±12 kHz total) but well inside
                # the BW/2 vs BW channel-spacing gap, so a real adjacent-
                # channel signal at df ≈ BW cannot be misidentified as alias.
                bin_hz = bw / N_sf
                # BW/2 spectral-fold alias tolerance.  Widened from ±5 kHz to
                # ±20 kHz on 2026-06-01 after histogramming residual SF10
                # over-detections: the alias df scatters across 113-145 kHz
                # (centered on BW/2=125 kHz but biased by carrier-centering
                # noise + bin granularity).  ±20 kHz catches the full
                # observed scatter while staying well inside the closest
                # legitimate LoRa channel-grid spacing (LoRaWAN 200 kHz).
                _alias_tol_hz = 20000.0
                _bw2 = bw * 0.5
                # BW chirp-edge artifact: a single LoRa chirp's up-slope and
                # down-slope endpoints can register as separate detections
                # at the same timestamp, separated by exactly BW.  SF10
                # histogram showed 13 such pairs at df = 249-254 kHz.  Safe
                # to dedup because (a) Meshtastic CSMA prevents simultaneous
                # adjacent-channel transmits and (b) extra PMR-delta gate
                # ensures we only fold near-identical-power pairs (real
                # adjacent transmitters almost always have different RSSI).
                _bw_edge_tol_hz = 10000.0
                _bw_pmr_tol_db = 1.5
                # Power-descending; detect_conf breaks POWER TIES (2026-07-08):
                # a slow-pass batch can hold the same burst under two labels
                # from different assemblies (matrix ground truth, SF12/62.5k:
                # true label at sc 0.98 vs its half-lag alias SF10/31.25k at
                # sc 0.67, both pwr 20.0) — the stable sort then kept
                # whichever was ENQUEUED first and the batch dedup below
                # dropped the truth.  Distinct-power cases sort exactly as
                # before.
                _dets_sorted = sorted(detections,
                                      key=lambda d: (-d.get('peak_power_db', 0.0),
                                                     -d.get('detect_conf', 0.0)))
                preambles = []
                for d in _dets_sorted:
                    d_off = d['freq_hz'] - anchor['freq_hz']
                    d_pwr = float(d.get('peak_power_db', 0.0))
                    d_pmr = float(d.get('bw_quality_db', 0.0))
                    if any(abs(d_off - p['offset_hz']) < bin_hz
                           for p in preambles):
                        if self.debug >= 1:
                            _ddt = (sample_pos / self.wb_fs) + d.get('preamble_t_s', 0.0)
                            print(f"         [BATCHDEDUP] drop {d['freq_hz']/1e6:.4f}MHz "
                                  f"abst={_ddt:.2f}s (≈ kept {anchor['freq_hz']/1e6:.4f}+"
                                  f"{[round(p['offset_hz']/1e3) for p in preambles]}kHz in batch)",
                                  flush=True)
                        continue
                    # BW/2 spectral-fold alias check: drop iff this offset is
                    # within ±tol of (stronger_offset ± BW/2).  _dets_sorted is
                    # power-descending, so any prior preamble was stronger.
                    if any(abs(abs(d_off - p['offset_hz']) - _bw2) < _alias_tol_hz
                           for p in preambles):
                        if self.debug >= 1:
                            _ddt = (sample_pos / self.wb_fs) + d.get('preamble_t_s', 0.0)
                            print(f"         [BW2ALIAS] drop {d['freq_hz']/1e6:.4f}MHz "
                                  f"abst={_ddt:.2f}s (BW/2 fold of stronger peak in batch)",
                                  flush=True)
                        continue
                    # BW chirp-edge alias check: drop iff this offset is
                    # within ±tol of (kept_offset ± BW) AND PMRs nearly match
                    # (real adjacent transmitters have different power).
                    if any(abs(abs(d_off - p['offset_hz']) - bw) < _bw_edge_tol_hz
                           and abs(d_pwr - p.get('peak_pwr_db', d_pwr)) < _bw_pmr_tol_db
                           for p in preambles):
                        if self.debug >= 1:
                            _ddt = (sample_pos / self.wb_fs) + d.get('preamble_t_s', 0.0)
                            print(f"         [BWEDGE] drop {d['freq_hz']/1e6:.4f}MHz "
                                  f"abst={_ddt:.2f}s pwr={d_pwr:.1f}dB "
                                  f"(BW chirp-edge of stronger peak in batch)",
                                  flush=True)
                        continue
                    # 3×BW spectral artifact: observed at SF12 BW125 — 416
                    # same-timestamp pairs at df ≈ 371-374 kHz (= 2.97-2.99 ×
                    # BW), tightly clustered.  Likely 3rd-harmonic spectral
                    # content of a SF12 chirp registering as a separate peak
                    # in the gate's PSD.  Structural signature: the alias's
                    # PMR (peak-to-mean-ratio, ≈ bw_quality_db in detection
                    # output) is CONSISTENTLY 6-9 dB lower than the real
                    # signal's PMR (verified at SF12: real pmr ≈ 33-35 dB,
                    # alias pmr ≈ 25-27 dB).  Peak power can be similar but
                    # PMR differentiates because the alias is spectrally
                    # smeared whereas the real chirp has a tight peak.
                    # Real adjacent-channel transmitters at df = 3·BW would
                    # not exhibit this asymmetric PMR relationship (each is
                    # a clean chirp).  Require: df ≈ 3·BW (±5 kHz) AND alias
                    # PMR is 3-15 dB lower than the kept preamble's PMR.
                    _bw3 = bw * 3.0
                    _bw3_tol_hz = 5000.0
                    if any(abs(abs(d_off - p['offset_hz']) - _bw3) < _bw3_tol_hz
                           and 3.0 < (p.get('pmr_db', d_pmr) - d_pmr) < 15.0
                           for p in preambles):
                        if self.debug >= 1:
                            _ddt = (sample_pos / self.wb_fs) + d.get('preamble_t_s', 0.0)
                            print(f"         [BW3ALIAS] drop {d['freq_hz']/1e6:.4f}MHz "
                                  f"abst={_ddt:.2f}s pwr={d_pwr:.1f}dB "
                                  f"(3×BW spectral fold of stronger peak in batch)",
                                  flush=True)
                        continue
                    preambles.append({
                        'offset_hz':   float(d_off),
                        # Preamble position IN THE SAVED FILE, in nb samples:
                        # the file starts pre_hop_s BEFORE the window, and
                        # preamble_t_s is the gate's within-window time.
                        # Hardcoded 0 made the PASS-2 time-isolated resubmit
                        # below dead code for every gate save — the sf<=9
                        # hop0/hop1 same-capture recovery never ran (UC
                        # audit g).
                        'time_sample': int(max(
                            0.0, pre_hop_s + float(d.get('preamble_t_s', 0.0)))
                            * nb_fs),
                        'pmr_db':      float(d.get('bw_quality_db', 0.0)),
                        'peak_pwr_db': d_pwr,
                        'status':      'LOCK',
                        'preamble_t_s': float(d.get('preamble_t_s', 0.0)),
                        'forced_dechirp': bool(d.get('forced_dechirp')),  # carry through for the cooldown dedup
                        'patience_trial': bool(d.get('patience_trial')),
                        'nb_ix':       d.get('_nb_ix'),  # cropped-stream key (NB items)
                        'src_det':     d,   # the ORIGINATING detection — the
                                            # save must carry ITS sf/bw label,
                                            # not the anchor's (UC audit c)
                    })
                if not preambles:
                    continue   # finally balances task_done (see above)
                if self.debug >= 1:
                    print(
                        f"         RECENTRE {len(preambles)} gate "
                        f"detection(s) in this batch",
                        flush=True)

                # Iterate found preambles — each becomes a separate save.
                for p_idx, p in enumerate(preambles):
                    carrier_hz = p['offset_hz']
                    _recenter_status = p['status']
                    _recenter_pmr = p['pmr_db']
                    _preamble_sample = p['time_sample']
                    # The detection this preamble came from (1:1).  Anchor-
                    # only labelling gave every co-window frame the ANCHOR's
                    # sf/bw — chimera saves that are undecodable at their
                    # true preset (UC audit c).
                    _src = p.get('src_det') or anchor
                    if _recenter_status == 'NO_LOCK':
                        continue
                    if self.debug >= 1:
                        print(
                            f"         [P{p_idx}] {_recenter_status} "
                            f"pmr={_recenter_pmr:.1f}dB t="
                            f"{_preamble_sample / nb_fs * 1000:.0f}ms "
                            f"offset={carrier_hz/1e3:+.2f}kHz",
                            flush=True)

                    # ---- Cross-batch redundant-capture dedup (RF time) ----
                    # One packet is re-detected in every overlapping window that
                    # contains its preamble → ~4-5 captures per packet, and
                    # decoding all of them dominates processing time.  abs_t_s is
                    # the preamble's ABSOLUTE RF time (sample_pos/wb_fs + within-
                    # window preamble offset), IDENTICAL across those overlapping
                    # windows.  Keep up to 2 captures per (carrier ±BW/2, time
                    # ±0.35 s) cluster — KEEP-2 not keep-1: the single strongest-
                    # power capture is sometimes the least decodable (off-centre
                    # / preamble at window edge), so keeping a second gives the
                    # decoder a fallback (keep-1 lost 3/120 on MEDIUM_FAST).
                    # 0.35 s is far under the ~0.9 s hop0→hop1 gap so distinct
                    # hops are preserved.  RF time → immune to replay speed.
                    _cand_freq = anchor['freq_hz'] + carrier_hz
                    _abs_t_s = (sample_pos / self.wb_fs) + p.get('preamble_t_s', 0.0)
                    _ftol = max(anchor['bw'] * 0.5, 60000.0)
                    # Dedup time window must be < the hop0→hop1 relay gap (else
                    # it MERGES the two hops on one carrier and drops hop1) and
                    # > the abs_t jitter of one preamble's overlapping-window
                    # copies.  The relay gap scales with airtime (the relay must
                    # receive the full packet before rebroadcasting), so a fixed
                    # 0.35 s was fine for SF12 but FAR wider than SF7/500's fast
                    # relay (~100-200 ms) → it merged SF7 hop0+hop1 → ~3-4 hop1
                    # lost live.  Scale with ~24 symbols of airtime (well under
                    # the gap, well over the jitter); floor 0.03 s.
                    _dt_win = max(0.03, 24.0 * (2 ** sf) / bw)
                    # Fix #2 (forced-dechirp cooldown): a channelized channel's forced
                    # dechirp re-fires every window a packet spans (and on the payload),
                    # so its abs_t scatters past the airtime window above and the same
                    # packet gets captured many times.  Anchored to the channel (fix #1,
                    # one carrier), collapse those repeats with a wide per-channel cooldown
                    # — keeps the channel's earliest hop, drops the rest.
                    if p.get('forced_dechirp'):
                        _dt_win = DECHIRP_DEDUP_WIN_S
                    _match = 0
                    _skip = False
                    with self._saved_abs_t_lock:
                        # PER-LABEL keep-N (2026-07-08): the cluster key now
                        # includes (sf, bw).  Same-label re-detections of one
                        # frame across overlapping windows still collapse to
                        # keep-N exactly as before — but a CONFLICTING-label
                        # detection of the same frame gets its own slots.
                        # Ground truth (matrix, SF12/62.5k = the only 0/5
                        # combo): separate slow-scan assemblies emitted the
                        # same burst as SF10/31.25k (partial-burst evidence,
                        # sc 0.67) and SF12/62.5k (full evidence, sc 0.98);
                        # first-arriving mislabels consumed both (freq, time)
                        # slots and the correct-label save was REFUSED —
                        # whichever label arrived first won regardless of
                        # merit.  With per-label slots both are captured and
                        # the decoder's CRC arbitrates (mislabeled captures
                        # fail fast; the post-decode dedup absorbs the
                        # duplicate emit).  Bounded cost: one extra capture
                        # pair per label-conflict, which only slow-pass
                        # straddle presets exhibit.
                        # Patience-trial dets dedup by CARRIER ONLY: the
                        # rotating (sf,bw) hypothesis must not hand each rung
                        # its own keep-N slots (soundness review: one promoted
                        # carrier fanned out into N differently-labeled
                        # captures per packet).
                        # _src label, not the stale loop variable `d` (the
                        # WEAKEST det from the batch power-sort — UC audit a):
                        # the registry must match/register under the label
                        # the save actually carries.
                        # Option A extends the cluster key: besides exact
                        # (sf, bw), a same-SLOPE member collapses onto the
                        # cluster too (bw^2/2^sf identical) — so a down-alias
                        # (SF7/62.5k of a real SF9/125k, SF9/62.5k of SF11/125k)
                        # no longer opens its own keep-2 slots and its own
                        # never-CRCing decode jobs.  The registry tuple now
                        # carries bw_quality_db as a 5th field for the quality-
                        # ranked eviction below.  LORA_SLOPE_DEDUP=0 → exact
                        # per-(sf,bw) HEAD behaviour.
                        _lbl_blind = bool(_src.get('patience_trial'))
                        _cand_slope = _slope(_src['sf'], _src['bw'])
                        _cand_q = float(_src.get('bw_quality_db', 0.0))
                        # Option B: effective keep collapses 2→1 under decode
                        # backlog so a real beacon's 2nd fallback capture is not
                        # queued while the single slow worker is saturated.
                        _eff_keep = self._dedup_keep
                        if (self._keep1_under_load and self._dedup_keep > 1
                                and self._decoder is not None
                                and self._decoder.pending()
                                    > self._keep1_load_thresh):
                            _eff_keep = 1
                        _cluster_idx = []
                        _best_diff_q = -1e30   # best q of a same-slope,
                        #   DIFFERENT-(sf,bw) member already in this cluster
                        for _i, _e in enumerate(self._saved_abs_t):
                            _sf_hz, _st, _ssf, _sbw = _e[0], _e[1], _e[2], _e[3]
                            if (abs(_cand_freq - _sf_hz) < _ftol
                                    and abs(_abs_t_s - _st) < _dt_win
                                    and (_lbl_blind
                                         or (_ssf == _src['sf']
                                             and _sbw == _src['bw'])
                                         or (self._slope_dedup
                                             and _slope(_ssf, _sbw)
                                                 == _cand_slope))):
                                _match += 1
                                _cluster_idx.append(_i)
                                if (self._slope_dedup
                                        and (_ssf != _src['sf']
                                             or _sbw != _src['bw'])):
                                    _best_diff_q = max(_best_diff_q, _e[4])
                        # RECORD-TIME same-slope alias gate (the eviction below
                        # only rewrites the registry — it cannot un-create a
                        # decode job for an alias that already filled a free
                        # keep-N slot or beat the weakest one).  A same-slope
                        # member whose raw dechirp quality is MARGIN dB below a
                        # different-label sibling already in the cluster is the
                        # 3-6 dB-weaker down-alias (SF7/62.5k of a real
                        # SF9/125k) — skip RECORDING it entirely so it never
                        # becomes a decode job, regardless of slot availability.
                        # A genuinely-strongest same-slope member is never
                        # gated (it clears the margin), preserving a real weak
                        # SF7/62.5k that has no stronger sibling.
                        if (self._slope_dedup
                                and _best_diff_q - _cand_q > self._slope_margin_db):
                            _skip = True
                        elif _match < _eff_keep:
                            self._saved_abs_t.append(
                                (_cand_freq, _abs_t_s,
                                 _src['sf'], _src['bw'], _cand_q))
                            if len(self._saved_abs_t) > 4000:
                                _cut = _abs_t_s - 10.0
                                self._saved_abs_t = [
                                    e for e in self._saved_abs_t
                                    if e[1] > _cut]
                        elif self._slope_dedup and _cluster_idx:
                            # Cluster full: quality-ranked eviction.  If this
                            # det's bw_quality_db beats the weakest slot's, evict
                            # + replace that slot so the true higher-q member
                            # always wins; the 3-6 dB-weaker alias never
                            # displaces the real parent.  Registry stays bounded
                            # (replace, not add) — at most a few evictions until
                            # the highest-q member holds the slot.  REGRESSION
                            # GUARD: the SF12/62.5k-vs-SF10/31.25k GT pair is
                            # same-slope (~954); keeping the higher-raw-q member
                            # preserves the required decode of both texts.
                            _weak_i = min(
                                _cluster_idx,
                                key=lambda _j: self._saved_abs_t[_j][4])
                            if _cand_q > self._saved_abs_t[_weak_i][4]:
                                self._saved_abs_t[_weak_i] = (
                                    _cand_freq, _abs_t_s,
                                    _src['sf'], _src['bw'], _cand_q)
                            else:
                                _skip = True
                        else:
                            _skip = True
                    if _skip:
                        if self.debug >= 1:
                            print(f"         [RFDEDUP] skip "
                                  f"{_cand_freq / 1e6:.4f}MHz t={_abs_t_s:.2f}s",
                                  flush=True)
                        continue

                    # ---- GLOBAL IQ-image rejection ----
                    # Drop this capture if a MUCH stronger (>10 dB) carrier sits
                    # at its mirror frequency (2*center - f) at ~the same RF time.
                    # Compares powers from the cross-batch registry, so it catches
                    # images whose source real signal was detected in a different
                    # (overlapping) window than this one — which the per-window
                    # check in detect_preamble misses (~2/3 of images leaked).
                    _mirror_f = 2.0 * self.center_hz - _cand_freq
                    with self._img_lock:
                        _cand_pwr = max(
                            (c[1] for c in self._img_carriers
                             if abs(c[0] - _cand_freq) < 60000.0
                             and abs(c[2] - _abs_t_s) < 1.5), default=-99.0)
                        _mir_pwr = max(
                            (c[1] for c in self._img_carriers
                             if abs(c[0] - _mirror_f) < 60000.0
                             and abs(c[2] - _abs_t_s) < 1.5), default=-99.0)
                        # bwq companion metric (2026-07-09): slow-pass dets
                        # carry a FLOOR-CLAMPED placeholder power (pwr20 on
                        # both the real carrier and its image), so the 10 dB
                        # power test can never fire for them.  Dechirp
                        # quality separates the pair robustly instead — the
                        # conjugated image loses ~6 dB of dechirp coherence
                        # (measured g42: real 33.8-35.3 dB vs image
                        # 27.5-28.6 dB; same split on the 62.5k slice).
                        _cand_bwq = max(
                            (c[3] for c in self._img_carriers if len(c) > 3
                             and abs(c[0] - _cand_freq) < 60000.0
                             and abs(c[2] - _abs_t_s) < 1.5), default=-99.0)
                        _mir_bwq = max(
                            (c[3] for c in self._img_carriers if len(c) > 3
                             and abs(c[0] - _mirror_f) < 60000.0
                             and abs(c[2] - _abs_t_s) < 1.5), default=-99.0)
                    # CHAN-DECHIRP (forced_dechirp) detections are exempt: the
                    # carrier was independently confirmed by the despread
                    # matched filter at a KNOWN channel (learned/seeded/
                    # promoted), and their placeholder floor-level power/bwq
                    # make both mirror tests trivially (and wrongly) satisfied
                    # whenever any strong carrier sits near the mirror.
                    _forced_here = any(
                        _dd.get('forced_dechirp')
                        and not _dd.get('patience_trial')
                        and abs(_dd['freq_hz'] - _cand_freq) < 60e3
                        for _dd in detections)
                    if not _forced_here and (
                            _mir_pwr > _cand_pwr + 10.0
                            or (_mir_bwq > -99.0 and _cand_bwq > -99.0
                                and _mir_bwq > _cand_bwq + 4.5)):
                        if self.debug >= 1:
                            print(f"         [IMG-REJECT] {_cand_freq / 1e6:.4f}"
                                  f"MHz ({_cand_pwr:.0f}dB bwq={_cand_bwq:.0f}) "
                                  f"mirror of {_mirror_f / 1e6:.4f}MHz "
                                  f"({_mir_pwr:.0f}dB bwq={_mir_bwq:.0f})",
                                  flush=True)
                        continue

                    # Build a per-preamble synthetic detection dict so the
                    # rest of the save-worker (POSTDEDUP, filename, decoder
                    # submission) works unchanged.  From _src, NOT anchor:
                    # the filename's SF/BW label must be this detection's
                    # own (UC audit c).  freq is identical either way
                    # (carrier_hz was measured relative to the anchor).
                    d = dict(_src)
                    d['freq_hz'] = anchor['freq_hz'] + carrier_hz
                    off = d['freq_hz'] - self.center_hz
                    if _nb_job is not None:
                        # Already extracted by the cropper, centred on
                        # this detection's own carrier (preambles come
                        # 1:1 from gate detections, and the extraction
                        # centre below is exactly the detection freq).
                        nb = _nb_job['streams'].get(p.get('nb_ix'))
                        if nb is None or len(nb) == 0:
                            continue
                        nb = nb[:int((buf_dur_s + max_pkt_s) * nb_fs)]
                    else:
                        # Re-extract narrowband centred precisely on this
                        # preamble's carrier — reuses cached forward FFT.
                        nb, nb_fs = extract_narrowband_fft(
                            extended, self.wb_fs, off, export_bw,
                            fft_cache=_ext_fft_cache)

                    # ---- CROP-CENTER FIX (2026-07-21) ----
                    # Measure the packet's max-hold plateau ON the NB slice
                    # just built; if the crop center is >5 kHz off the
                    # plateau midpoint, recenter the EXPORTED bytes and add
                    # the correction to the FILENAME freq below.  Fail-safe
                    # contract: _nb_plateau_offset returns None (abstain)
                    # unless it positively identifies a bw-wide plateau, and
                    # _crop_corr stays 0.0 → byte-identical legacy capture.
                    # Detection state (d/_src freq_hz, DETECTED lines, dedup
                    # keys, channel learning) is never touched.
                    #
                    # BURST-GATED (2026-07-21 follow-up): the meter is
                    # restricted to THIS preamble's time window instead of
                    # the slice's first 3 s.  A time-blind max-hold picks
                    # the strongest bw-wide plateau anywhere in the slice —
                    # in dense traffic that is often a NEIGHBOR packet at a
                    # different carrier (stress-bed live leg: a truly-
                    # centered SF9 capture was "corrected" -33.5 kHz onto a
                    # neighbor's plateau and every decode then needed a
                    # 53 kHz rescue recenter), or a merged union that
                    # dilutes the midpoint (two SF11 captures measured
                    # -15.5k/-29.6k full-slice vs -36.3k/-39.4k
                    # burst-gated == the value every isolated SF11 capture
                    # measures).  Window = preamble-8 .. preamble+40
                    # symbols (label's own symbol time): covers preamble
                    # (16) + SFD (4.25) + header (8) + margin for the
                    # gate's preamble_t_s jitter (PASS-2 below trusts the
                    # same position to ~8 syms), while shrinking the
                    # collision cross-section ~10-20x vs 3 s.  Too-short
                    # windows (alias-label trials, truncated streams) fall
                    # below the meter's 4*2048-sample minimum → abstain,
                    # exactly the fail-safe contract above.
                    _crop_corr = 0.0
                    if _CROP_CENTER_FIX and len(nb):
                        _d_sym_n = int(round((2 ** d['sf']) * nb_fs
                                             / float(d['bw'])))
                        _w0 = max(0, _preamble_sample - 8 * _d_sym_n)
                        _w1 = min(len(nb),
                                  _preamble_sample + 40 * _d_sym_n)
                        _po = _nb_plateau_offset(nb[_w0:_w1], nb_fs,
                                                 float(d['bw']))
                        if _po is not None and abs(_po) > 5000.0:
                            if _nb_job is None:
                                # Direct path: re-extract at the corrected
                                # center — the cached forward FFT makes this
                                # one cheap extra inverse.
                                _nb2, _fs2 = extract_narrowband_fft(
                                    extended, self.wb_fs, off + _po,
                                    export_bw, fft_cache=_ext_fft_cache)
                                if len(_nb2):
                                    nb, nb_fs = _nb2, _fs2
                                    _crop_corr = float(_po)
                            else:
                                # Lazy path: the cropper's wideband parts are
                                # gone (streams only), so recenter the NB
                                # stream itself with a heterodyne mix —
                                # equivalent to a re-extract at off+corr
                                # except in the empty brick-wall band edges
                                # (the packet spans ±bw/2+corr, well inside
                                # the ±bw slice).  Detection unchanged.
                                _ph = np.arange(len(nb), dtype=np.float64)
                                nb = (nb * np.exp(
                                    (-2j * np.pi * _po / nb_fs) * _ph)
                                      ).astype(np.complex64)
                                _crop_corr = float(_po)
                            if self.debug >= 1 and _crop_corr != 0.0:
                                print(f"         [CROP-CENTER] "
                                      f"{d['freq_hz'] / 1e6:.4f}MHz "
                                      f"corr={_crop_corr / 1e3:+.1f}kHz "
                                      f"({'mix' if _nb_job is not None else 'reextract'})",
                                      flush=True)
                    _preamble_sym = _preamble_sample // sym_n if sym_n > 0 else 0
                    _relay_after_syms     = _preamble_sym + 149
                    _original_before_syms = max(0, _preamble_sym - 8)

                    # Post-CFO global dedup is DISABLED.  It used wall-clock
                    # time, which during offline replay (which can run slower
                    # than real-time) compresses the dedup window enough that
                    # a legitimate hop1 ~0.9 s after its hop0 lands in the
                    # same wall-clock TTL even when they're a half-second
                    # apart in audio time — and gets dropped because their
                    # TCXO-locked carriers share the 10 kHz bucket.
                    # Downstream (PacketID, hops_taken) dedup correctly
                    # suppresses true duplicate decodes without affecting
                    # legitimate near-co-channel relays.  The cost of letting
                    # one packet's adjacent-window duplicates through is one
                    # extra save_worker invocation per packet, which is
                    # absorbed by the 4-thread save_worker pool.
                    corrected_abs = self.center_hz + off  # used in filename

                    # Encode the true capture rate so the decoder reads it
                    # exactly instead of GUESSING from file size (the guess is
                    # ambiguous and mis-read lower-rate captures as higher).
                    # dt_str's millisecond digits are replaced by a global
                    # save sequence — same token shape, but two saves can
                    # never share a filename (same-ms finalizes of same-
                    # carrier packets used to, and the 4 parallel save
                    # workers then corrupted the file racing tofile).
                    # CROP-CENTER FIX: the filename freq is the decoder's
                    # starting-CFO truth for the file's DC — it must name the
                    # (possibly corrected) actual crop center.  _crop_corr is
                    # 0.0 on abstain/kill-switch → byte-identical legacy name.
                    _sq = next(self._fname_seq)
                    fname = (f"SF{d['sf']}_{fmt_bw(d['bw'])}"
                             f"_{(d['freq_hz'] + _crop_corr) / 1e6:.4f}MHz"
                             f"_{int(round(nb_fs / 1000))}ksps"
                             f"_pwr{int(round(d.get('peak_power_db', 0.0)))}"
                             f"_{dt_str[:-3]}{_sq % 1000:03d}.cf32")
                    fpath = os.path.join(self.export_dir, fname)
                    # Capacity guard: shed this capture if the undecoded backlog
                    # already fills the byte budget.  Only in decode mode — the
                    # files are ref-counted and unlinked, so their bytes release
                    # as decodes finish; export-only keeps files by design.
                    # Policy is DROP-NEWEST: once the backlog is full, arriving
                    # captures are shed regardless of strength (already-queued
                    # weaker ones are not evicted — they hold worker jobs that
                    # are hard to claw back).  Simple and safe; the [CAPTURE-
                    # DROP] log line makes the shedding visible.
                    _fsize = len(nb) * 8          # complex64 = 8 bytes/sample
                    if self._decoder is not None:
                        _admit, _drop, _pend = self._decoder._cap_try_reserve(
                            fpath, _fsize)
                        if not _admit:
                            if _drop == 1 or _drop % 25 == 0:
                                print(f"         → [CAPTURE-DROP] backlog "
                                      f"{_pend/1e6:.0f}MB ≥ budget "
                                      f"{self._decoder._cap_budget/1e6:.0f}MB — "
                                      f"shedding captures ({_drop} dropped; "
                                      f"decode can't drain fast enough)",
                                      flush=True)
                            continue
                    try:
                        nb.astype(np.complex64).tofile(fpath)
                    except Exception:
                        # Failed write (disk full / perms): release the budget
                        # reservation — no decode will ever _ref_dec it — and
                        # drop any partial file, then let the outer handler log.
                        if self._decoder is not None:
                            self._decoder._cap_release(fpath)
                        try:
                            os.unlink(fpath)
                        except OSError:
                            pass
                        raise
                    dur_ms = len(nb) / nb_fs * 1000
                    print(f"         → Saved {fname} "
                          f"({dur_ms:.0f}ms, {len(nb)} samples "
                          f"@ {nb_fs/1000:.0f}kHz) abst={_abs_t_s:.2f}s", flush=True)
                    if self._decoder:
                        # Phase-2 bucket key: (cand_freq, abs_t_s, ftol, ttol)
                        # — same tuple + tolerances the save-side RFDEDUP
                        # uses at lines 2216-2228, so cross-worker dedup
                        # matches save-side cluster definition exactly.
                        # ttol scales with airtime (24 symbols, floor 30 ms)
                        # so ACK at +50 ms from DM at SAME carrier stays in
                        # a SEPARATE bucket on SF7 (ttol=49 ms) — exactly
                        # the case a global 0.35 s ceiling would have merged.
                        _bkey = (_cand_freq, _abs_t_s, _ftol, _dt_win)
                        # TRIAL LABEL LADDER (2026-07-16, defect-2 companion):
                        # the decoder locks SF/BW from the capture FILENAME —
                        # it can relabel SF at the same BW (WRONG_SF retry)
                        # but can NEVER cross bandwidths, so an alias-labeled
                        # trial capture is undecodable no matter how wide the
                        # crop.  Submit the SAME (family-wide-cropped) bytes
                        # once per same-slope family label via hardlinks (no
                        # extra bytes on disk; each link has its own refcount/
                        # unlink lifecycle and capacity reservation).  Links
                        # are created BEFORE the primary submit so a fast
                        # decode+unlink of the primary cannot race them.
                        # Bounded: ladder families have at most 2 members, and
                        # only acquisition-internal patience trials take this
                        # path.  CRC still arbitrates identity; publish-on-
                        # decode-confirm is unchanged.
                        _fam_links = []
                        if _lbl_blind:
                            for _fsf, _fbw in patience_trial_family(
                                    _src['sf'], _src['bw']):
                                if (_fsf == _src['sf']
                                        and abs(_fbw - _src['bw']) < 1.0):
                                    continue          # primary label itself
                                _sq2 = next(self._fname_seq)
                                _fname2 = (
                                    f"SF{_fsf}_{fmt_bw(_fbw)}"
                                    f"_{(d['freq_hz'] + _crop_corr) / 1e6:.4f}MHz"
                                    f"_{int(round(nb_fs / 1000))}ksps"
                                    f"_pwr{int(round(d.get('peak_power_db', 0.0)))}"
                                    f"_{dt_str[:-3]}{_sq2 % 1000:03d}.cf32")
                                _fpath2 = os.path.join(self.export_dir,
                                                       _fname2)
                                _admit2, _, _ = self._decoder._cap_try_reserve(
                                    _fpath2, _fsize)
                                if not _admit2:
                                    continue
                                try:
                                    os.link(fpath, _fpath2)
                                except OSError:
                                    try:
                                        nb.astype(np.complex64).tofile(_fpath2)
                                    except Exception:
                                        self._decoder._cap_release(_fpath2)
                                        continue
                                _fam_links.append((_fpath2, _fname2))
                        # Pass 1 — full-file primary with SIC: SC-scans the
                        # buffer, decodes the strongest preamble, subtracts
                        # its reconstructed signal, and re-searches the
                        # residual.  Catches multi-packet files when SIC
                        # cancellation is clean (per-symbol amplitude OK).
                        self._decoder.submit(fpath, fname, bucket_key=_bkey,
                                             slow=bool(d.get('patience_trial')),
                                             lineage=d)
                        # Family-label decode jobs (trial captures only) —
                        # same bucket, slow tier: once any label in the
                        # bucket decodes, the phase-2 bucket dedup skips the
                        # rest, so the ladder costs nothing after a confirm.
                        for _fp2, _fn2 in _fam_links:
                            self._decoder.submit(_fp2, _fn2, bucket_key=_bkey,
                                                 slow=True, lineage=d)
                        # Pass 2 — TIME-ISOLATED on this preamble.  Useful at
                        # FAST SF where the hop1 relay arrives ~50-200 ms
                        # after hop0 and they share a single capture file;
                        # iso zeros everything outside [preamble-16,+149]
                        # syms, giving the decoder a clean window with only
                        # THIS preamble's packet in it.  At SLOW SF (≥10)
                        # hop airtime is multi-second so hop0/hop1 land in
                        # SEPARATE captures — PASS 2 just doubles the decode
                        # workload with no gain.  Skipping it on slow SFs is
                        # the biggest single latency win for SF11/12 live.
                        # Set LORA_PASS2_SUBMIT=1 to force PASS 2 on slow SF
                        # if a deployment hits same-window slow-SF concurrency.
                        _pass2_force = os.environ.get('LORA_PASS2_SUBMIT', '0') == '1'
                        _pass2_enable = sf <= 9 or _pass2_force
                        if _pass2_enable and _preamble_sym > 0 and _preamble_sample > 0:
                            _iso_after  = max(0, _preamble_sym - 16)
                            _iso_before = _preamble_sym + 149
                            self._decoder.submit(
                                fpath, fname,
                                relay_after_syms=_iso_after,
                                relay_before_syms=_iso_before,
                                bucket_key=_bkey,
                                slow=bool(d.get('patience_trial')),
                                lineage=d)
            except Exception as e:
                print(f"[RECORDER] save error: {e}", file=sys.stderr, flush=True)
            finally:
                if _spool_path is not None:
                    try:
                        del extended_full          # release the memmap
                        os.unlink(_spool_path)
                    except Exception:
                        pass
                # Release the batch's big arrays BEFORE blocking on the next
                # get(): an idle save worker otherwise pins the last
                # capture's wideband/narrowband buffers (hundreds of MB) in
                # dead locals for the whole quiet period (UC audit q) — x4
                # workers on a 4 GB Pi.
                item = extended_full = nb = None
                extended = _ext_fft_cache = None
                _all = _bparts = _tparts = _tp = _mm = None
                detections = _nb_job = _dets_sorted = preambles = None
                self._save_queue.task_done()

    def _extract_nb(self, iq, offset_hz, target_bw):
        return extract_narrowband_fft(iq, self.wb_fs, offset_hz, target_bw)

    def flush(self):
        """Wait for all pending saves to complete."""
        if self._crop_queue is not None:
            self._crop_queue.join()
        self._save_queue.join()

    def reset_prev(self):
        """No-op — prev_hop no longer used in recordings."""
        pass


class BackgroundDecoder:
    """Decode captured .cf32 files in a background thread.

    Uses a PERSISTENT worker subprocess to avoid re-paying Python startup
    + numpy import costs.  The worker stays alive between decodes.
    """

    def __init__(self, aes_key=None, no_key=False, verbose=False,
                 host_mem_ctx=None):
        self._verbose = verbose
        self._aes_key = aes_key
        self._no_key = no_key
        # HOST/CONFIG memory context for the principled worker-RAM reserve.
        # Keys: ring_bytes, detect_pool_bytes, win_n (see the RAM-cap block in
        # the CPU/RAM auto-scale below).  Defaults to {} so the reserve
        # degrades safely to just OS_slack when absent.
        self._host_mem_ctx = host_mem_ctx or {}
        self._queue = queue.Queue()
        self._lock = threading.Lock()
        # Structured packet log (JSONL) the web UI tails.  Each decoded/encrypted
        # [PKT] record from the workers is appended here with a receive timestamp.
        # Set LORA_PKT_LOG to '' to disable.
        self._pkt_log_path = os.environ.get('LORA_PKT_LOG', '/tmp/lora_packets.jsonl')
        self._pkt_log_lock = threading.Lock()

        # Opt-in raw decode-job log (LORA_DECODE_RAWLOG=path): appends every
        # job lifecycle event (SUBMIT, SKIP-*, JOB with the worker's full raw
        # output, REQUEUE-SLOW) so a capture's live outcome is always
        # attributable.  The compact stdout is a lossy display — it hides
        # mid-ladder results, drops [BUDGET] markers, and renders some
        # failures unlabeled — which made "decoded but hidden", "ran and
        # failed", and "job silently lost" indistinguishable post-hoc (the
        # 2026-07-07 phantom live-vs-offline gap cost a full day exactly
        # this way).  Default off; zero cost besides one branch per event.
        self._rawlog_path = os.environ.get('LORA_DECODE_RAWLOG', '')
        self._rawlog_lock = threading.Lock()

        # Post-decode dedup: suppress same (PacketID, hops_taken) within 60s.
        # Prevents the same relay hop from being reported twice when the Welch
        # PSD finds the chirp at two different instantaneous frequencies.
        self._packet_dedup = {}          # (pkt_id_str, hops_taken) -> last_seen_time
        self._packet_dedup_lock = threading.Lock()

        # Payload fingerprint dedup: suppress duplicate raw payloads within 30s.
        # Catches non-Meshtastic frames (LoRaWAN, MeshCore, CRC-only) that slip past
        # PacketID dedup because they lack a parseable packet ID.
        # (Algorithm ported from meshtastic-sniffer main.c fingerprint ring.)
        self._fp_dedup = {}              # fingerprint (int) -> last_seen_time
        self._fp_dedup_lock = threading.Lock()

        # Truthful decode counters (observability).  The compact STDOUT summary
        # interleaves secondary-pass "(no CRC/mesh parsed)" / "CRC: FAIL" lines
        # with real decodes, so counting DECRYPTED/[PKT] lines in stdout badly
        # under-reports actual decodes (this caused a false "decode is broken"
        # investigation).  These count what actually reaches the packet log —
        # the data path the web reads — so [STAT] can report an authoritative
        # number.  _decoded_total = decode events (incl. redundant captures of
        # one packet); _decoded_keys = distinct packets (by raw_hex/pktid/text).
        self._decoded_total = 0
        self._decoded_keys = set()
        self._decoded_lock = threading.Lock()
        # Carrier freqs (Hz) of decode-confirmed [PKT] records since the last
        # drain — the patience gate's FUTILITY feedback (a promotion whose
        # trial captures decode is real and must never be futility-retired).
        # Capped: if nothing drains (patience off / futility off) it cannot
        # grow.
        self._confirmed_freqs = []

        # Phase-2 chase optimization: cross-worker bucket dedup + in-flight
        # tracking with dispatch deferral.
        # PROBLEM: when KEEP-2 + PASS-2-iso both fire for one RF cluster, up
        # to 4 worker jobs queue up for the same frame.  The original
        # dispatch-time dedup checked only `_decoded_buckets` (admitted
        # emits) — but sibling jobs are pulled from the queue back-to-back
        # before the first worker has time to emit, so the check fires
        # ZERO times in practice (race: sibling B dispatches before sibling
        # A's emit lands).  Result: optimization didn't help Meshtastic.
        # FIX: add in-flight tracking.  When worker A starts processing
        # bucket X, mark X as in-flight; when sibling B's dispatch comes
        # up, wait briefly (poll every 100 ms up to 2 s) for A to either
        # admit or finish.  If A admits → B skips entirely.  If A finishes
        # without admitting → B dispatches normally.  Workers waiting on
        # bucket X don't block other workers from bucket Y (each worker
        # thread runs an independent dispatch loop), so total throughput
        # is preserved.  Stale in-flight entries (age > max_age) are auto-
        # GC'd so a crashed/stuck worker can't deadlock siblings forever.
        # Bucket tolerance matches save-side RFDEDUP per-frame ftol/dt_win,
        # so it's proven not to merge hop0/hop1.
        self._fname_bucket = {}          # fpath -> bucket_key tuple
        self._decoded_buckets = []       # list of bucket_key tuples (admitted)
        self._inflight_buckets = {}      # bucket_key -> dispatch_timestamp
        self._bucket_lock = threading.Lock()
        # Diagnostic counters — printed periodically via _diag_print
        self._diag_marks = 0
        self._diag_already_decoded_hits = 0
        self._diag_inflight_hits = 0
        self._diag_wait_skip = 0
        self._diag_wait_dispatch = 0

        # Pre-decode dedup: multi-res Welch finds the same LoRa packet at
        # 2-3 different center frequencies (peak candidates within the LoRa
        # bandwidth) AND in 2 consecutive overlap windows.  Each candidate
        # becomes a save file.  Decoding all of them wastes ~5-30s of chase
        # + sweep on the off-center captures while the canonical capture
        # already decoded the packet.  Filter at submit() before the slow
        # decoder ever sees the file.
        #
        # A save is a duplicate of a recently-submitted save if:
        #   |Δfreq| < 0.6 MHz  (within one LoRa-BW from each other)
        #   |Δtime| < 0.6 s    (same or adjacent gate fire)
        # Hop0 and hop1 are at different freqs (Meshtastic hops channels), so
        # this won't collapse a hop0/hop1 pair.  Two messages 6s apart on the
        # same freq are >0.6s apart, also preserved.
        self._submit_recent = []         # list of (freq_mhz, timestamp_s, fname)
        self._submit_dedup_lock = threading.Lock()
        # Honour the SAME keep-N as the save-side RFDEDUP.  The recorder
        # intentionally saves up to LORA_DEDUP_KEEP captures per (carrier, RF-
        # time) cluster so the decoder gets a FALLBACK copy when the first one
        # is off-centre / preamble-at-window-edge.  If submit() collapses those
        # to one (the original skip-after-first behaviour), keep-2 is silently
        # nullified and live decodes only the first-arriving copy — which is
        # sometimes the bad one (proven: SF7 live missed Test12/Test17 hop1
        # whose GOOD second copy decoded fine offline at the same budget).  So
        # allow up to _dedup_keep submissions per cluster.
        self._dedup_keep = int(os.environ.get('LORA_DEDUP_KEEP', '2'))

        # Per-fpath reference counter for capture-file cleanup.  A single
        # capture can be processed by up to four worker jobs (primary, primary-
        # slow-requeue, relay-after, relay-before) and we cannot delete the file
        # until ALL of them are done — but we also must NOT leak it indefinitely
        # (its tmpfs backing fills /dev/shm and OOM-kills the pipeline; see
        # issue #3).  Refcount: increment at every enqueue site (submit + slow
        # re-queue), decrement after each worker job, unlink when count hits 0.
        # Deterministic — no time-based reaper guessing safe intervals.
        self._refs = {}
        self._refs_lock = threading.Lock()

        # ---- Pending-capture byte budget (backpressure the unbounded decode
        # queues never provided) ----
        # Captures accumulate on the export FS until their decode finishes; if
        # they arrive faster than the workers drain them, an unbounded backlog
        # fills the backing store (tmpfs → OOM; disk → page-cache pressure).
        # LORA_CAPTURE_BUDGET_MB caps the total bytes of UNDECODED captures on
        # disk: over budget, a new capture is dropped at the recorder (the
        # weakest-quality ones are shed first because detections are saved
        # strongest-first) instead of growing the backlog without bound.
        # 0/unset = unbounded (legacy behaviour).
        try:
            self._cap_budget = int(os.environ.get('LORA_CAPTURE_BUDGET_MB', '0')) * 1024 * 1024
        except ValueError:
            self._cap_budget = 0
        self._cap_bytes = {}          # fpath -> file size (counted ONCE per file)
        self._pending_bytes = 0
        self._cap_lock = threading.Lock()
        self._cap_dropped = 0

        # Find decoder.py next to this script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self._decoder_script = os.path.join(script_dir, 'decoder.py')
        if not os.path.isfile(self._decoder_script):
            print("[DECODE] decoder.py not found — decoding disabled",
                  file=sys.stderr, flush=True)
            self._decoder_script = None

        # Worker POOL: K persistent decoder subprocesses, each driven by its
        # own manager thread, all pulling from the shared self._queue.  A single
        # serial decoder was the throughput bottleneck once the gate was made
        # fast (dec_q grew to ~220 on a 60-msg burst, ~11x slower than real
        # time).  Dedup (_packet_dedup / _fp_dedup / _submit_recent) is all
        # lock-protected and the unique-hop set is order-independent, so
        # parallel managers are safe.  Cap workers so the pool + gate + each
        # worker's numpy threads don't oversubscribe the CPU.
        self._active_count = 0          # decodes currently in flight (for pending())
        self._workers = []              # subprocess handles, for drain cleanup
        self._workers_lock = threading.Lock()
        # CPU-affinity isolation: on multi-core hosts, reserve the first 4
        # cores for the GATE process (main, stdin-reader, save-worker, detect
        # pool) and pin DECODE workers to the remaining cores.  Without this
        # isolation, Linux's CFS scheduler co-locates decode workers and
        # gate-reader on the same physical cores during burst (24-core host
        # observed 2026-06-01: 14 decode workers preempt the gate's stdin
        # reader → gate falls from 28 Msps to 19 Msps → USB ring buffer
        # overflows → samples dropped).  With explicit affinity partitioning,
        # decode workers cannot preempt the reader and the gate keeps full
        # 28 Msps regardless of worker count.  If the user has externally
        # restricted affinity (e.g. wrapped the gate in `taskset -c 0-3`),
        # respect that and let workers share — the user's taskset implies
        # they accept the trade-off of fewer worker cores.
        try:
            _affinity_now = os.sched_getaffinity(0)
            _total = os.cpu_count() or 4
            if len(_affinity_now) >= max(8, _total - 2):
                # Essentially unrestricted: reserve cores for gate, pin
                # workers to the rest.  Worker cores = affinity − first 4.
                _gate_reserve = set(sorted(_affinity_now)[:4])
                _worker_aff = _affinity_now - _gate_reserve
                self._worker_affinity = (_worker_aff if _worker_aff
                                          else _affinity_now)
            else:
                # User-restricted (e.g. taskset to a few cores): share.
                self._worker_affinity = None
        except (AttributeError, OSError):
            self._worker_affinity = None
        # Adaptive throttle: when the gate is actively dropping samples
        # (reader.drops growing between STAT intervals), the main loop sets
        # this event.  Workers then sleep briefly between jobs so memory
        # bandwidth frees up for the gate's stdin reader.  Self-clears when
        # the gate recovers.  Empirically (2026-06-01): SF10 BW250 30-msg
        # bursts on 24-cpu host saturated 14 workers, dropped 24.5% of the
        # 28 Msps stream.  6 static workers eliminate drops but regress SF12
        # latency.  Adaptive throttle keeps SF7/11/12 fast in the common case
        # AND backs off only when SF10's actual over-detection load arrives.
        self._gate_stress = threading.Event()
        # Pause duration when stressed — long enough to let the gate's reader
        # thread complete a USB transfer (~3 ms at 28 Msps × 32768 samples)
        # without starving steady-state SF12 throughput.
        self._stress_pause_s = 0.05
        # Fail-fast fast-tier budget while the gate is stressed (CPU seconds,
        # pre-SF-scaling — SF12 gets ~8.5 s vs the ~57 s a full 10 s budget
        # scales to).  Bailed decodes are requeued to the slow tier.
        self._stress_budget = float(os.environ.get('LORA_STRESS_BUDGET_S', '1.5'))
        # Decode-queue depth that triggers fail-fast budgets for FRESH jobs
        # (see the worker loop): deep queue = fresh packets take the fast
        # path, marginals defer to the idle slow tier.  0 disables.
        try:
            self._pressure_n = int(os.environ.get('LORA_DECODE_PRESSURE_N', '8'))
        except ValueError:
            self._pressure_n = 8
        # Pressure-path budget (flat CPU seconds across SF, like the stress
        # path).  A 4.0 s variant ("let clean SF11 pass first try") was
        # A/B-tested and LOST to 1.5 s (med latency 108/114 vs 78/109 s,
        # uniq tied): the requeue-slow path recovers bounced cleans promptly
        # and the tighter budget keeps the single worker freer.  1.5 =
        # identical to the proven v2 behavior (-20% med latency, +2.5
        # decodes vs no pressure trigger).
        try:
            self._pressure_budget = float(
                os.environ.get('LORA_DECODE_PRESSURE_BUDGET_S', '1.5'))
        except ValueError:
            self._pressure_budget = 1.5
        # Respect cgroup/taskset CPU affinity — `os.cpu_count()` returns the
        # SYSTEM count (e.g. 24) even when the process is pinned to 4 cores via
        # taskset / cgroup; on those hosts that would auto-scale to 16 decode
        # workers fighting for 4 cores, starving the gate.  Prefer the affinity
        # count when available (Linux-only) so the auto-scale matches what we
        # are actually allowed to run on.
        try:
            _ncpu = len(os.sched_getaffinity(0))
        except (AttributeError, OSError):
            _ncpu = os.cpu_count() or 4
        # Decode is the pipeline bottleneck (the gate runs ~real-time; decode of
        # the per-packet captures, esp. SF11/12 off-centre recovery, lags).
        # Each decode worker is single-threaded ≈ one core of decode throughput.
        # OFFLINE: use most cores (batch throughput).  LIVE: too many decode
        # workers STARVE the gate's reader/main threads of CPU, so the gate
        # falls below 28 Msps and the ring buffer overflows → SAMPLE DROPS
        # (measured: 16 workers → gate 28→22.6 Msps, ~13% samples dropped under
        # load).  In live the decode backlog is cheap (dec_q is just file paths;
        # captures are on disk and decode catches up during idle) — what must
        # NOT slip is the gate's realtime sample intake.  So cap workers via
        # LORA_DECODE_WORKERS (live sets it low, e.g. ncpu - detect - reserve).
        _env_nw = os.environ.get('LORA_DECODE_WORKERS')
        try:
            _env_nw_i = int(_env_nw) if _env_nw else 0
        except ValueError:
            _env_nw_i = 0
        # -1 / 0 / unset → auto-scale.  Plug-and-play on any core count:
        # leave the gate, detect-pool, and a small reserve enough CPU.
        # Workers are niced +10 (set in _lower_priority) so the OS preempts
        # them for the realtime-critical gate, which lets us oversubscribe
        # mildly on low-core hosts without dropping samples.  At LOW core
        # counts (≤6), the `_ncpu - 6` formula would clamp to the minimum
        # 2 workers and dec_q backs up at slow SFs; instead reserve a
        # proportional fraction so smaller hosts get more decode bandwidth.
        if _ncpu <= 4:
            _w_auto = max(2, _ncpu - 1)             # 4 cores → 3 workers
        elif _ncpu <= 8:
            _w_auto = max(2, _ncpu - 2)             # 8 cores → 6 workers
        else:
            _w_auto = max(2, min(16, _ncpu - 6))    # ≥12 cores → ncpu-6 (cap 16)
        # RAM-CAP (UC audit (s) re-exam 2026-07-12): the count above is
        # CPU-scaled with NO memory term, but decode workers are the DOMINANT
        # RAM consumer — each ratchets to ~the recycle bound (LORA_DECODE_
        # WORKER_RSS_MB, ~1.5-2 GB anon) and nothing else in the pipeline
        # comes close.  On a small-RAM many-core host (e.g. 4 GB / 8 vCPU)
        # the CPU scale picks 6 workers → ~12 GB → OOM.  Cap the AUTO count so
        # peak worker RSS fits system RAM, leaving a reserve for the gate/ring/
        # detect-pool + OS.  This is a pure no-loss trade: fewer workers = more
        # decode LATENCY (captures live on /dev/shm and catch up during idle),
        # never sample loss, and it also REDUCES gate starvation.  Big-RAM
        # hosts are unaffected (the CPU cap dominates).  An explicit
        # LORA_DECODE_WORKERS>0 always wins (operator knows the host).
        if _env_nw_i <= 0:
            # Each env knob parses with ITS OWN fallback so one typo'd value
            # (e.g. LORA_OS_SLACK_MB=1g) degrades to that knob's default with a
            # warning instead of silently disabling the ENTIRE RAM cap — which
            # would resurrect the exact small-host OOM this block prevents.
            def _envf(name, dflt):
                _v = os.environ.get(name)
                if not _v:
                    return dflt
                try:
                    return float(_v)
                except ValueError:
                    print(f"[DECODER] ignoring malformed {name}={_v!r} "
                          f"(using {dflt})", file=sys.stderr, flush=True)
                    return dflt
            try:
                with open('/proc/meminfo') as _mi:
                    _memtot_mb = next(int(l.split()[1]) for l in _mi
                                      if l.startswith('MemTotal')) / 1024.0
                _rss_lim = _envf('LORA_DECODE_WORKER_RSS_MB', 1500.0)
                # peak ≈ recycle bound + ratchet headroom above it
                _per_worker_mb = max(512.0, _rss_lim) + 512.0
                # PRINCIPLED RESERVE (was a fixed 2048 MB rate-blind guess that
                # was only right when the ring happened to be ~1 GB).  The
                # reserve is EVERYTHING that is NOT a decode worker, derived
                # from host+config values.  ring_bytes is LAZY (np.zeros) so it
                # is invisible in current RSS at decoder-init and MUST be added
                # explicitly.  An explicit LORA_DECODE_RAM_RESERVE_MB now
                # OVERRIDES the derivation (was merely its default constant).
                _reserve_mb = _envf('LORA_DECODE_RAM_RESERVE_MB', 0.0)
                # unset / 0 / malformed -> derive.  (To force a worker COUNT
                # use LORA_DECODE_WORKERS, which bypasses this block entirely.)
                if _reserve_mb <= 0:
                    _ctx = self._host_mem_ctx
                    _win_n = float(_ctx.get('win_n', 0) or 0)          # rate*window
                    _ring_b = float(_ctx.get('ring_bytes', 0) or 0)    # capped; 0 if file
                    _dpool_b = float(_ctx.get('detect_pool_bytes', 0) or 0)  # 0 if serial
                    # Gate working set: O(win_n) complex64 buffers (welch PSD,
                    # win_n-FFT plan scratch, max-hold, convert-ahead, one save
                    # window) — measured 7-9x win_n*8 at 10-20 Msps; 10x safe.
                    _gate_mult = _envf('LORA_GATE_WIN_MULT', 10.0)
                    _gate_b = _gate_mult * _win_n * 8.0
                    # The ONE constant: numpy/scipy/FFTW-plan base (~80 MB
                    # measured) + OS headroom + small save/crop margin.
                    _os_slack_mb = _envf('LORA_OS_SLACK_MB', 768.0)
                    _reserve_mb = ((_ring_b + _dpool_b + _gate_b) / (1024.0 * 1024.0)
                                   + _os_slack_mb)
                _w_ram = int(max(1, (_memtot_mb - _reserve_mb) / _per_worker_mb))
                if _w_ram < _w_auto:
                    print(f"[DECODER] RAM cap: {_memtot_mb:.0f}MB total, "
                          f"reserve {_reserve_mb:.0f}MB, ~{_per_worker_mb:.0f}MB/"
                          f"worker → {_w_ram} decode workers "
                          f"(cpu-scale wanted {_w_auto})",
                          file=sys.stderr, flush=True)
                    _w_auto = _w_ram
            except (OSError, StopIteration, ValueError):
                # /proc/meminfo unreadable/absent (non-Linux) — cap unavailable.
                print("[DECODER] RAM cap unavailable (no /proc/meminfo) — "
                      "using CPU scale only", file=sys.stderr, flush=True)
        self._n_workers = (_env_nw_i if _env_nw_i > 0 else _w_auto)
        # ---- Two-tier decode on the SAME workers (no oversubscription) ----
        # Each manager serves the FAST queue first; only when it's empty does it
        # pull from the SLOW queue.  A capture the fast pass BAILS on ("[BUDGET]
        # … exhausted" → ran out of realtime budget, may still hold an undecoded
        # packet) is re-queued to the slow queue and re-decoded with a big
        # per-job budget.  Because it's the same worker pool, total CPU is
        # unchanged (no extra processes to oversubscribe the gate) — the slow
        # re-decodes simply run in the idle gaps after each burst.  Captures live
        # on /dev/shm so nothing is lost; the rare budget-bound straggler just
        # arrives a few seconds later instead of being dropped → 100% of
        # captured packets without slowing the realtime path.
        self._slow_queue = queue.Queue()
        # CAPTURE-FIRST DECODE DEFERRAL (2026-07-06): under ring pressure
        # the decode workers' memory-bandwidth load starves the reader
        # (measured: 33 concurrent SF12 decodes -> 17.5 of 20 Msps
        # sustained -> CATCHUP skips every ~47 s -> no contiguous frame
        # windows). Captures are files already — decode is deferrable.
        # The gate feeds ring occupancy each window; high pressure routes
        # NEW jobs to the slow queue (drained by the existing idle-driven
        # release + a pressure-clear release). Saturation becomes packet
        # LATENCY instead of packet LOSS.
        self._pressure = 0.0
        self._defer_state = False
        # Slow-tier concurrency accounting.  Both release paths used to gate
        # on "_active_count == 0" (nothing decoding AT ALL) to keep at most
        # one heavy re-decode off a burst.  On any host whose worker pool is
        # kept busy by the fast tier — i.e. every low-core host under load —
        # that condition is essentially never true, so the slow queue NEVER
        # drained in-run and pressure-deferred packets were LOST at shutdown
        # instead of merely delayed (measured 2026-07-22: 192 jobs stranded
        # across 14 live presets, queue depth to 42, decode 30-58%).  The
        # intent was "one slow re-decode at a time", which is exactly what
        # this counter expresses — without coupling it to unrelated fast-tier
        # work.  Released jobs are tracked by path so the count survives the
        # requeue/relay paths.
        self._slow_inflight = 0
        self._slow_released_job = None
        # Instrumentation (validation-only): per-job identity counter so a
        # packet record can be tied back to the job that produced it (and thus
        # whether that job was deferred to the slow tier).  Its OWN lock — never
        # self._lock — so it cannot introduce a lock-ordering deadlock with the
        # scheduler lock that submit() paths may already hold.
        self._job_id_ctr = 0
        self._job_id_lock = threading.Lock()
        self._slow_budget = float(os.environ.get('LORA_SLOW_BUDGET_S', '45'))
        # Slow mop-up runs ONLY when capture is not active.  A big-budget
        # re-decode running DURING live capture holds workers and starves the
        # gate's 28 Msps intake → sample drops (measured).  The gate clears this
        # flag at shutdown/EOF so the slow tier drains the straggler queue then
        # (captures are on disk; for a continuously-running deployment, clear it
        # during genuinely idle stretches instead).
        self._capture_active = True
        self._threads = []
        for _ in range(self._n_workers):
            t = threading.Thread(target=self._manager, daemon=True)
            t.start()
            self._threads.append(t)
        # Legacy alias: some shutdown paths reference self._thread.
        self._thread = self._threads[0]

    def _start_worker(self, budget_s=None):
        """Launch persistent worker subprocess.  budget_s overrides the decode
        CPU-time budget (the slow tier passes a large value)."""
        if self._decoder_script is None:
            return None

        # Build the key argument for the worker
        key_setup = ""
        if self._no_key:
            key_setup = """
import decoder as _dec
_dec._mesh_no_key = True
_dec._mesh_aes_key = bytes(16)
"""
        elif self._aes_key:
            key_setup = f"""
import decoder as _dec
_dec._mesh_aes_key = bytes.fromhex('{self._aes_key.hex()}')[:16]
"""
        else:
            key_setup = """
import decoder as _dec
"""

        # Persistent worker: import once, then loop reading paths from stdin
        worker_code = f"""
import sys, os
sys.path.insert(0, {repr(os.path.dirname(self._decoder_script))})
{key_setup}
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, line_buffering=True)
sys.stderr = open(os.devnull, 'w')

while True:
    line = sys.stdin.readline()
    if not line:
        break
    toks = line.strip().split('\\t')
    fpath = toks[0]
    if not fpath:
        continue
    relay_after = relay_before = None
    if len(toks) > 1 and toks[1]:
        mode = toks[1]
        if mode.startswith('after:'):
            relay_after = int(mode[6:])
        elif mode.startswith('before:'):
            relay_before = int(mode[7:])
        elif mode.startswith('iso:'):
            # iso:A,B — time-window isolation: blank before A and after B.
            _a, _b = mode[4:].split(',')
            relay_after = int(_a)
            relay_before = int(_b)
    # toks[2] = optional per-job decode budget (seconds).  The SLOW tier passes
    # a big value so a fast-pass straggler gets fully recovered; empty = use the
    # worker's env default (fast tier).
    _bo = float(toks[2]) if len(toks) > 2 and toks[2] else None
    sys.stdout.write("__BEGIN__ " + os.path.basename(fpath) + "\\n")
    sys.stdout.flush()
    try:
        _dec.process_file(fpath, relay_after=relay_after, relay_before=relay_before, _budget_override=_bo)
    except Exception as e:
        sys.stdout.write("  ERROR: " + str(e) + "\\n")
    sys.stdout.write("__END__\\n")
    sys.stdout.flush()
"""
        try:
            # Give the decoder enough budget to finish both decode passes
            # (PASS 1 sweep + PASS 2 re-centre sweep, ~2 s each).  The default
            # 1.5 s bails before PASS 2, dropping marginal hops that need
            # re-centring (run_4: Test4/Test18 both hops).  The decoder runs
            # in a separate, mostly-idle process (the detector is the
            # pipeline bottleneck), so a larger budget costs no real-time.
            # Non-LoRa still fast-rejects: no preamble → instant bail before
            # any sweep, so the budget only applies to real LoRa captures.
            _env = dict(os.environ)
            _env.setdefault('LORA_DECODE_BUDGET_S', '12.0')
            if budget_s is not None:
                _env['LORA_DECODE_BUDGET_S'] = str(budget_s)
            # Single-threaded decode workers (1 thread each).  The decoder's
            # budget is now CPU-time (process_time) so it stays accurate under
            # heavy multi-process load — but that ONLY holds if each worker is
            # single-threaded (multi-threaded FFTs would inflate process_time
            # and exhaust the budget early).  Parallelism comes from the POOL
            # of K workers, not from threads within a worker, so capping at 1
            # thread costs nothing and makes the per-decode budget robust.
            for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                       'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
                _env[_v] = '1'
            proc = subprocess.Popen(
                [sys.executable, '-c', worker_code],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                env=_env,
                preexec_fn=self._lower_priority)
            # Pin worker to the non-gate cores so it cannot preempt the
            # gate's stdin-reader / detect-pool threads on cores 0-3.
            if self._worker_affinity:
                try:
                    os.sched_setaffinity(proc.pid, self._worker_affinity)
                except (AttributeError, OSError):
                    pass
            with self._workers_lock:
                self._workers.append(proc)
            return proc
        except Exception as e:
            print(f"[DECODE] Failed to start worker: {e}",
                  file=sys.stderr, flush=True)
            return None

    def _rawlog(self, tag, what, text=''):
        """Append one job-lifecycle event to the raw decode log (no-op unless
        LORA_DECODE_RAWLOG is set).  `text` (the worker's raw output) is
        framed so multi-line records stay parseable."""
        if not self._rawlog_path:
            return
        try:
            with self._rawlog_lock:
                with open(self._rawlog_path, 'a') as f:
                    f.write(f"@@@ {time.time():.3f} {tag} {what}\n")
                    if text:
                        f.write(text.rstrip('\n') + "\n@@@ END\n")
        except Exception:
            pass

    def _cap_try_reserve(self, fpath, size):
        """Reserve `size` bytes for a new capture against the pending-capture
        budget.  Returns (admitted, dropped_total, pending_bytes).  admitted=
        False → the caller must NOT write the capture (backlog full).  Budget 0
        → always admitted.  The reserved bytes are released in
        `_ref_dec_and_maybe_unlink` when the capture's decodes all finish."""
        if not self._cap_budget:
            return True, 0, 0
        with self._cap_lock:
            if self._pending_bytes + size > self._cap_budget:
                self._cap_dropped += 1
                return False, self._cap_dropped, self._pending_bytes
            self._pending_bytes += size
            self._cap_bytes[fpath] = size
            return True, 0, self._pending_bytes

    def _cap_release(self, fpath):
        """Release a reservation made by `_cap_try_reserve` WITHOUT a decode
        having run — the write failed (disk full, perms) so no _ref_dec will
        ever fire for it.  Without this, every failed write ratchets
        _pending_bytes upward until the budget reads permanently full and the
        guard silently drops ALL future captures — a worse failure than the
        one it guards against."""
        if not self._cap_budget:
            return
        with self._cap_lock:
            _sz = self._cap_bytes.pop(fpath, None)
            if _sz:
                self._pending_bytes -= _sz

    def _ref_inc(self, fpath):
        """Increment outstanding-job count for `fpath`.  Called on every
        enqueue (submit() + slow re-queue) so the file is held until every
        queued worker has finished with it."""
        with self._refs_lock:
            self._refs[fpath] = self._refs.get(fpath, 0) + 1

    def _ref_dec_and_maybe_unlink(self, fpath):
        """Decrement outstanding-job count and, if zero, unlink the capture
        file.  Called once per worker job completion (success, failure, or
        crash) from the manager thread's `finally` block, so the count is
        guaranteed to wind down even on exceptions."""
        with self._refs_lock:
            n = self._refs.get(fpath, 1) - 1
            if n > 0:
                self._refs[fpath] = n
                return
            self._refs.pop(fpath, None)
        # Release this file's reserved bytes from the capacity budget (whether or
        # not the file is actually unlinked below — its decode is done either way).
        if self._cap_budget:
            with self._cap_lock:
                _sz = self._cap_bytes.pop(fpath, None)
                if _sz:
                    self._pending_bytes -= _sz
        if os.environ.get('LORA_KEEP_IQ'):
            return          # debugging: keep capture files for autopsy
        try:
            os.unlink(fpath)
        except OSError:
            pass        # file already gone (manual delete / startup reap)

    def _bucket_already_decoded(self, fpath):
        """Phase-2 check: has any capture in the same RF bucket already
        produced a verified emit?  Linear-scan the decoded_buckets list
        using PER-FRAME tolerances stored at submit time so the cross-worker
        dedup matches the save-side RFDEDUP definition exactly (per-SF
        dt_win is critical: ACK arrives ~50 ms after a DM at the SAME
        carrier, and a global ceiling like 0.35 s would incorrectly merge
        ACK + DM into the same bucket and suppress the ACK capture)."""
        with self._bucket_lock:
            bkey = self._fname_bucket.get(fpath)
            if bkey is None:
                return False
            cand_freq, abs_t_s, _, _ = bkey
            for (df, dt, ftol, ttol) in self._decoded_buckets:
                if (abs(cand_freq - df) < ftol
                        and abs(abs_t_s - dt) < ttol):
                    return True
        return False

    def _mark_bucket_decoded(self, fname):
        """Phase-2 admit: a worker just produced a verified emit for this
        fname (admitted by _packet_dedup or _fp_dedup).  Look up the bucket
        we recorded at submit() time and add it to _decoded_buckets so
        sibling captures in the same cluster skip decode entirely."""
        with self._bucket_lock:
            for fpath, bkey in self._fname_bucket.items():
                if fpath.endswith(fname) or fname.endswith(os.path.basename(fpath)):
                    self._decoded_buckets.append(bkey)
                    if len(self._decoded_buckets) > 200:
                        self._decoded_buckets = self._decoded_buckets[-100:]
                    self._diag_marks += 1
                    return

    # Phase-2 in-flight tracking — closes the race window between sibling
    # dispatches.  Without this, sibling B's dispatch-time check sees the
    # bucket as "not admitted" because sibling A's worker hasn't finished
    # yet → both run in parallel → optimization is bypassed entirely (we
    # observed 0 bucket-skips on real Meshtastic burst traffic before this
    # was added).

    # Max age for an in-flight entry before it's considered stale and GC'd.
    # 5.0 s covers SF12 LONG_SLOW decodes (the slowest standard preset) with
    # margin.  If a worker has been holding a bucket longer than this it's
    # presumed crashed/wedged and sibling dispatches proceed normally.
    _INFLIGHT_MAX_AGE_S = 5.0

    def _is_bucket_inflight(self, fpath):
        """True if another worker is currently processing a sibling capture
        in the same RF bucket as `fpath`.  Auto-GCs entries older than
        _INFLIGHT_MAX_AGE_S so a crashed worker can't deadlock siblings."""
        with self._bucket_lock:
            bkey = self._fname_bucket.get(fpath)
            if bkey is None:
                return False
            cand_freq, abs_t_s, _, _ = bkey
            now = time.time()
            stale = []
            hit = False
            for key, dispatch_ts in self._inflight_buckets.items():
                if now - dispatch_ts > self._INFLIGHT_MAX_AGE_S:
                    stale.append(key)
                    continue
                df, dt, ftol, ttol = key
                if (abs(cand_freq - df) < ftol
                        and abs(abs_t_s - dt) < ttol):
                    hit = True
                    break
            for k in stale:
                self._inflight_buckets.pop(k, None)
            return hit

    def _mark_bucket_inflight(self, fpath):
        """Mark this fpath's bucket as in-flight (a worker is about to start
        decoding it).  No-op if fpath has no bucket key."""
        with self._bucket_lock:
            bkey = self._fname_bucket.get(fpath)
            if bkey is None:
                return
            self._inflight_buckets[bkey] = time.time()

    def _clear_bucket_inflight(self, fpath):
        """Worker is done with this fpath (success or fail) — release the
        in-flight entry so a future capture for the same bucket can dispatch
        without waiting.  No-op if fpath has no bucket key."""
        with self._bucket_lock:
            bkey = self._fname_bucket.get(fpath)
            if bkey is None:
                return
            self._inflight_buckets.pop(bkey, None)

    def set_pressure(self, frac):
        self._pressure = float(frac)
        _now = time.time()
        if frac > 0.5 and not self._defer_state:
            self._defer_state = True
            self._defer_t = _now
            print("         [DECODE-DEFER] ring pressure %.0f%% — new "
                  "decodes deferred to idle" % (frac * 100), flush=True)
        elif (frac < 0.20 and self._defer_state
              and _now - getattr(self, '_defer_t', 0) > 20.0):
            # min-dwell 20 s + deep-clear 20%: the first cut (50/30, no
            # dwell) toggled 72 times in one leg — running decodes can't
            # be recalled, so each release re-spiked pressure instantly
            self._defer_state = False
            print("         [DECODE-DEFER] pressure cleared — resuming",
                  flush=True)

    def maybe_release_pressure(self):
        """Pressure-clear release: one deferred job at a time while the
        ring is comfortable, without requiring full radio-idle."""
        if self._defer_state or self._pressure > 0.20:
            return
        if self._slow_queue.qsize() == 0 or not self._queue.empty():
            return
        self._release_one_slow()

    def submit(self, fpath, fname, relay_after_syms=None, relay_before_syms=None,
               bucket_key=None, slow=False, lineage=None):
        """Queue a capture file for background decoding.

        relay_after_syms:  blank first N BW-rate symbols → find hop AFTER primary.
        relay_before_syms: blank from symbol N onward   → find hop BEFORE primary.
        bucket_key:        (cand_freq_hz, abs_t_s) for Phase-2 cross-worker
                           bucket dedup; None disables the dedup for this job.
        lineage:           the source detection dict (Phase-1 audit lineage);
                           carries _lin_* fields stamped at detect emit / slow
                           consume.  None when audit is off.
        """
        if not self._decoder_script:
            return

        # Phase-2: remember which RF bucket this fpath belongs to so the
        # dispatch loop can later skip captures whose bucket already produced
        # a verified emit.  Cap the dict to avoid growth on stale fnames.
        if bucket_key is not None:
            with self._bucket_lock:
                self._fname_bucket[fpath] = bucket_key
                if len(self._fname_bucket) > 4000:
                    # Drop the oldest ~half by insertion order (dict preserves it).
                    _keep = list(self._fname_bucket.items())[-2000:]
                    self._fname_bucket = dict(_keep)

        # NO pre-decode dedup here.  It used to skip captures within
        # Δfreq<0.15 MHz AND Δt<0.6 s of a recent submission — but it keyed on
        # the filename's WALL-CLOCK timestamp, which cannot tell a same-packet
        # overlapping-window copy (true dedup target) from a relay hop1 ~0.5 s
        # after hop0 at the same carrier (a DISTINCT packet).  With the gate now
        # detecting that time-separated hop1 as its own packet, the 0.6 s window
        # merged hop0+hop1 and DROPPED hop1 — the exact miss we're fixing.  The
        # save-side RFDEDUP already deduplicates correctly using RF abs_t (a
        # tight ~0.03 s window that DOES separate hop0/hop1), so by the time we
        # get here the captures are already keep-N-limited per true RF cluster.
        # The post-decode _packet_dedup collapses any duplicate RESULTS, so
        # submitting them all is safe (no FP / no double-count).
        self._ref_inc(fpath)
        # slow=True (patience trials): a trial capture is a HYPOTHESIS — it
        # must never compete with real traffic for the fast decode workers
        # (measured: trial jobs reordered a knife-edge marginal decode's
        # worker assignment and flipped it).  The slow tier drains in idle
        # gaps, which is exactly the right priority for acquisition probes.
        _slow_tier = slow or self._defer_state
        self._rawlog('SUBMIT', f"{fname} relay={relay_after_syms},{relay_before_syms} "
                               f"tier={'slow' if _slow_tier else 'fast'}")
        # Instrumentation (validation-only): job-local metadata carried WITH the
        # job tuple (6th element) — NOT looked up later in the prunable
        # _fname_bucket.  capture_anchor_rf_s = the crop's candidate-preamble RF
        # time (bucket_key[1]); it anchors the capture, not a specific packet
        # (one crop can emit several [PKT]s), so it is labelled honestly and no
        # per-packet rf_event_s is derived (the decoder exposes no within-crop
        # offset).  The tuple's index 0 (fpath) and object identity are
        # unchanged, so the slow-tier reservation (_slow_take/_slow_job) is
        # unaffected.
        with self._job_id_lock:
            self._job_id_ctr += 1
            _jid = self._job_id_ctr
        # initial_slow is IMMUTABLE (the tier at submission); ever_deferred and
        # requeue_count accumulate — so a realtime job later requeued reads
        # initial_slow=False, ever_deferred=True.  float() the anchor so a numpy
        # scalar can't break json.dumps downstream.
        # Preserve ALL FOUR bucket values (freq, rf_s, freq_tol, time_tol) so the
        # offline scorer can associate repeated same-payload occurrences using
        # the detector's OWN recorded time tolerance (never exact float equality
        # on rf_s).  float() each so a numpy scalar can't break json.dumps.
        def _bf(_i):
            if (bucket_key is not None and len(bucket_key) > _i
                    and bucket_key[_i] is not None):
                return float(bucket_key[_i])
            return None
        _meta = {'job_id': _jid,
                 'submit_mono_ns': time.monotonic_ns(),
                 'capture_anchor_freq_hz': _bf(0),
                 'capture_anchor_rf_s': _bf(1),
                 'capture_anchor_freq_tol_hz': _bf(2),
                 'capture_anchor_time_tol_s': _bf(3),
                 'initial_slow': bool(_slow_tier),
                 'ever_deferred': bool(_slow_tier),
                 'requeue_count': 0}
        # Phase-1 lineage: allocate a LOGICAL work_id (reused on requeue) and
        # carry producer / window|scan / detector-side detection fields (kept
        # SEPARATE from the decoder-reported bw/sf/freq).  Emit a work_submitted
        # lifecycle event + ledger note so starved/failed work is observable.
        if _LID is not None:
            _lin = lineage if isinstance(lineage, dict) else {}
            _wid = _LID.next('work')
            _meta['work_id'] = _wid
            _meta['queue_attempt_seq'] = 0
            _meta['producer'] = _lin.get('_lin_producer')
            _meta['parent_detection_id'] = _lin.get('_lin_detection_id')
            # exact candidate linkage is Phase-2; Phase-1 carries only a HINT.
            _meta['parent_candidate_id'] = None
            _meta['candidate_hint_id'] = _lin.get('_lin_candidate_hint_id')
            _meta['candidate_bin_delta'] = _lin.get('_lin_candidate_bin_delta')
            _meta['window_id'] = _lin.get('_lin_window_id')
            _meta['scan_id'] = _lin.get('_lin_scan_id')
            _meta['candidate_link_status'] = _lin.get('_lin_candidate_link_status')
            # detector-side values (NEVER overwritten by decoder output)
            _meta['detector_freq_hz'] = (float(_lin['freq_hz'])
                                         if _lin.get('freq_hz') is not None else _bf(0))
            _meta['detector_sf'] = _lin.get('sf')
            _meta['detector_bw_hz'] = (float(_lin['bw'])
                                       if _lin.get('bw') is not None else None)
            _meta['detector_preamble_rf_s'] = _bf(1)
            if _WLED is not None:
                _WLED.note_submit(_wid)
            _run_event('work_submitted', run_id=_RUN_ID, work_id=_wid,
                       queue_attempt_seq=0, job_id=_jid, producer=_meta['producer'],
                       window_id=_meta['window_id'], scan_id=_meta['scan_id'],
                       parent_detection_id=_meta['parent_detection_id'],
                       parent_candidate_id=_meta['parent_candidate_id'],
                       queue_tier=('slow' if _slow_tier else 'fast'),
                       detector_sf=_meta['detector_sf'],
                       detector_bw_hz=_meta['detector_bw_hz'],
                       detector_freq_hz=_meta['detector_freq_hz'])
        if _slow_tier:
            _run_event('slow_enqueued', job_id=_jid, reason='initial_slow',
                       requeue_count=0)
        (self._slow_queue if _slow_tier else self._queue).put(
            (fpath, fname, relay_after_syms, relay_before_syms, None, _meta))

    def pending(self):
        with self._lock:
            return (self._queue.qsize() + self._slow_queue.qsize()
                    + self._active_count)

    def decoded_counts(self):
        """(total_decode_events, distinct_packets) reaching the packet log.
        Authoritative — unlike counting DECRYPTED/[PKT] lines in the compact
        stdout, which the secondary-pass diagnostics badly under-report."""
        with self._decoded_lock:
            return self._decoded_total, len(self._decoded_keys)

    def drain_confirmed_freqs(self):
        """Decode-confirmed carrier freqs (Hz) since the last call — the
        patience gate's futility feedback.  Cheap (one lock, usually empty)."""
        with self._decoded_lock:
            if not self._confirmed_freqs:
                return ()
            _o = self._confirmed_freqs
            self._confirmed_freqs = []
        return _o

    def maybe_release_straggler(self):
        """Release ONE deferred straggler for a big-budget re-decode IF the
        decode system is idle right now — fast queue empty AND nothing decoding.
        The GATE calls this on a no-signal window, so the trigger is the gate's
        own real-time idle state (no signal in the air + no decode backlog), not
        a guessed time threshold.  That makes it fully SELF-ADAPTING to traffic
        density: dense traffic → few no-signal windows / busy decoders → rarely
        fires (stragglers wait); sparse traffic → many idle windows → drains
        freely.  active_count==0 caps it to one slow re-decode at a time so it
        never piles onto a burst, and being driven by real idle it never starves
        the realtime gate.  No tunable, no polling thread."""
        if self._slow_queue.qsize() == 0:
            return
        if not self._queue.empty():
            return
        self._release_one_slow()

    def _release_one_slow(self):
        """Move ONE slow-tier job into the fast queue, at most one in flight.

        Callers have already established that no fast-tier work is waiting.
        The released job is consumed by the SAME fixed worker pool, so this
        adds no processes and cannot oversubscribe the gate; if every worker
        is busy it simply waits its turn in the fast queue instead of rotting
        in the slow one.

        Realtime guard: gate on LIVE ring occupancy, not on the _defer_state
        latch.  The latch exists to route NEW jobs to the slow tier and holds
        for a 20 s min-dwell; gating releases on it would re-starve the queue
        on exactly the busy hosts this fix targets.  Live pressure is the
        honest "is there room for heavy work right now" signal, and the
        _gate_stress throttle still pauses a running re-decode if the reader
        starts dropping.  File mode leaves _pressure at 0.0 → unchanged."""
        if self._pressure > 0.20:
            return
        with self._lock:
            if self._slow_inflight >= 1:
                return
        try:
            _job = self._slow_queue.get_nowait()
        except queue.Empty:
            return
        with self._lock:
            self._slow_inflight += 1
            self._slow_released_job = _job
        self._queue.put(_job)

    def _slow_take(self, job):
        """Claim the slow-tier reservation iff `job` IS the released slow job.

        Identity-based, NOT path-keyed: several submissions legitimately share
        one fpath (the PASS-2 time-isolated variant, relay probes, and the
        [BUDGET] requeue), so matching on path let a FAST-tier job retire the
        slow reservation while the slow job was still running — permitting a
        second heavy re-decode against the live gate."""
        with self._lock:
            if (self._slow_released_job is not None
                    and job is self._slow_released_job):
                self._slow_released_job = None
                return True
        return False

    def _slow_retire(self):
        """Release the in-flight slow reservation.

        MUST run on EVERY exit path of the dispatch-loop iteration that
        claimed it.  The early `continue`s (no decoder script, SKIP-BUCKET,
        SKIP-INFLIGHT-ADMIT, worker-start failure) are not corner cases —
        SKIP-BUCKET is the *designed* outcome for patience-trial families
        submitted under one bucket key — and leaking there pins
        _slow_inflight at 1, which turns both release paths into permanent
        no-ops and re-starves the slow queue unconditionally."""
        with self._lock:
            self._slow_inflight = max(0, self._slow_inflight - 1)

    def _slow_dispose(self, meta, slow_rel, outcome, executed, result_mono_ns,
                      parsed_records=None, logged_records=None):
        """Instrumentation (validation-only): the SINGLE exit-path disposition
        for a dispatched job.  Centralizes reservation retirement + the
        release/completion events so NO early-exit path can retire silently or
        recreate the coverage gap (jobs 57/61 in ref-pass1 hit an early exit
        whose bare _slow_retire() left slow_reserved without a matching
        slow_reservation_released).  Called at normal completion AND all four
        early exits.  `slow_rel` gates the release exactly as before; the
        completion event fires for any ever-deferred/initial-slow job with a
        terminal outcome (a real disposition time, never null)."""
        if slow_rel:
            self._slow_retire()
            _run_event('slow_reservation_released', job_id=meta.get('job_id'),
                       requeue_count=meta.get('requeue_count'))
        if meta.get('ever_deferred') or meta.get('initial_slow'):
            _kw = dict(job_id=meta.get('job_id'), outcome=outcome,
                       executed=executed, terminal=True,
                       requeue_count=meta.get('requeue_count'),
                       result_processed_mono_ns=result_mono_ns)
            if parsed_records is not None:
                _kw['parsed_records'] = parsed_records
            _run_event('slow_completed', **_kw)
        # Phase-1 work-lifecycle TERMINAL (centralized, exactly-once).  A job
        # that requeued THIS attempt is not terminal here (it terminalizes on its
        # later completion attempt).  Outcome taxonomy maps the dispatch _oc /
        # early-exit reason to a work outcome; 'completed_decode' is keyed on the
        # PARSED record count (parsed_records), never a substring.
        _wid = meta.get('work_id')
        if _WLED is not None and _wid is not None and not meta.get('_work_requeued'):
            _WORK_OUTCOME = {
                'packet': 'completed_decode', 'no_packet': 'completed_no_decode',
                'parse_error': 'failed_parse', 'subprocess_error': 'failed_worker',
                'manager_error': 'failed_worker', 'timeout': 'failed_timeout',
                'worker_start_failed': 'failed_worker',
                'skipped_bucket': 'cancelled_bucket_dedup',
                'skipped_inflight': 'cancelled_inflight',
                'no_decoder': 'cancelled_no_decoder'}
            # keyed on records LOGGED to the canonical sink (codex C1).  A
            # 'packet' (parsed>0) that logged ZERO records = telemetry lost
            # (failed_record_log), NOT completed_no_decode.
            if outcome == 'packet':
                if logged_records and logged_records > 0:
                    _wo = 'completed_decode'
                elif parsed_records and parsed_records > 0:
                    _wo = 'failed_record_log'
                else:
                    _wo = 'completed_no_decode'
            else:
                _wo = _WORK_OUTCOME.get(outcome, 'failed_' + str(outcome))
            if _WLED.note_terminal(_wid, _wo):
                _run_event('work_' + _wo, run_id=_RUN_ID, work_id=_wid,
                           queue_attempt_seq=meta.get('queue_attempt_seq'),
                           job_id=meta.get('job_id'), producer=meta.get('producer'),
                           parsed_records=parsed_records, logged_records=logged_records)

    def _slow_state_snapshot(self):
        """Instrumentation (validation-only): scheduler-lock-consistent snapshot
        of the deferred-tier state for the drain_end acceptance invariant."""
        with self._lock:
            return {'slow_inflight': self._slow_inflight,
                    'slow_reservation_present': self._slow_released_job is not None,
                    'slow_queue_size': self._slow_queue.qsize(),
                    'fast_queue_size': self._queue.qsize()}

    def drain_slow_into_fast(self):
        """Move the slow mop-up queue into the fast queue so the blocking
        workers re-decode the budget-bound stragglers (with their big per-job
        budget).  Called at shutdown/EOF — i.e. AFTER capture, so these heavy
        re-decodes never compete with the realtime gate."""
        while True:
            try:
                self._queue.put(self._slow_queue.get_nowait())
            except queue.Empty:
                break

    def drain(self, timeout=30.0):
        """Wait for pending decodes to finish (e.g. on shutdown).  Keeps folding
        the slow mop-up queue into the fast queue so stragglers re-queued during
        the drain itself (a backlogged fast job that bailed) are also recovered."""
        deadline = time.time() + timeout
        while self.pending() > 0 and time.time() < deadline:
            self.drain_slow_into_fast()
            time.sleep(0.2)
        # Close every pool worker's stdin so it exits cleanly; kill stragglers.
        with self._workers_lock:
            _workers = list(self._workers)
        for w in _workers:
            if w and w.poll() is None:
                try:
                    w.stdin.close()
                    w.wait(timeout=5)
                except Exception:
                    try:
                        w.kill()
                    except Exception:
                        pass

    @staticmethod
    def _lower_priority():
        """Run in child process before exec — lower scheduling priority."""
        try:
            os.nice(10)
        except OSError:
            pass

    def _manager(self):
        """Manager thread: drives its OWN persistent worker subprocess.  Serves
        the FAST queue first; only when it is empty does it pull from the SLOW
        queue (fast-pass stragglers re-decoded with a big per-job budget).  Same
        worker pool for both → no CPU oversubscription of the realtime gate."""
        worker = None
        while True:
            # Plain blocking get on the fast queue — NO polling loop (10 threads
            # waking on a timeout competed with the gate's realtime thread and
            # caused sample drops).  The slow mop-up queue is drained by moving
            # its items INTO this fast queue at shutdown (drain_slow_into_fast),
            # so these same blocking workers process the stragglers then, with
            # the big per-job budget carried in the job tuple.
            job = self._queue.get()
            # Adaptive throttle: if the gate is actively dropping samples,
            # delay this decode start to free memory bandwidth for the
            # gate's reader thread.  Self-clearing event — once the gate
            # recovers (main loop observes drops stable), it clears the
            # flag and decoders resume full speed.
            if self._gate_stress.is_set():
                time.sleep(self._stress_pause_s)
            fpath, fname, relay_after_syms, relay_before_syms, job_budget, _meta = job
            if _meta is None:
                _meta = {}
            # ATTEMPT-LOCAL requeue flag (codex): reset FALSE at every attempt
            # entry so a stale True from a prior attempt can never suppress this
            # attempt's terminalization (a requeue->early-exit/exception would
            # otherwise leave the work outstanding).  Set True only in the
            # requeue block below when THIS attempt actually requeues.
            _meta['_work_requeued'] = False
            # Named for what it is: when the manager DEQUEUED the item (not
            # necessarily when decode execution starts).
            _manager_dequeue_mono_ns = time.monotonic_ns()   # instrumentation
            # Claim the slow-tier reservation ONCE, here, by job identity; every
            # exit path below must retire it (see _slow_retire).
            _slow_rel = self._slow_take(job)
            if _slow_rel:
                _run_event('slow_reserved', job_id=_meta.get('job_id'),
                           requeue_count=_meta.get('requeue_count'))
            # Load-aware decode budget: when the gate is under pressure (ring
            # filling or dropping — same signal as the pause above), a
            # fast-tier decode runs with a FAIL-FAST budget instead of the
            # full recovery budget.  The full budget assumes "the decoder
            # process is mostly idle"; on a host where decode CPU competes
            # with the live gate (any low-core machine, or a busy band on a
            # big one) that assumption starves ingest — measured live on a
            # 4-core host: SF-scaled 28-57 s CPU per capture cascading into
            # ring loss.  A decode that bails on the small budget emits
            # [BUDGET] and is requeued to the SLOW tier below — decoded in
            # the next idle gap, so marginal packets are DEFERRED under
            # load, never lost.  Idle behavior is byte-identical.
            _stress_shrunk = False
            # DECODE-PRESSURE fail-fast (latency lever, 2026-07-21): the
            # gate-stress trigger below covers RING pressure only — when the
            # gate keeps up but the DECODE queue runs deep, queued jobs still
            # got the full idle-recovery budget, so 88-102 s marginal grinds
            # held the single low-core worker while fresh beacons waited
            # (measured live real-beacon queue latency: med ~100 s).  A deep
            # decode queue now triggers the same battle-tested fail-fast +
            # REQUEUE-SLOW path: fresh packets decode in seconds, marginal
            # stragglers are DEFERRED to idle gaps (big-budget slow tier),
            # never lost.  LORA_DECODE_PRESSURE_N jobs (0 disables).
            _dec_pressure = False
            if job_budget is None and self._pressure_n > 0:
                _dec_pressure = self.pending() > self._pressure_n
            _press_only = False
            if job_budget is None and (self._gate_stress.is_set()
                                       or _dec_pressure):
                _press_only = _dec_pressure and not self._gate_stress.is_set()
                # Flat CPU budget: process_file multiplies any budget by
                # 2^((sf-7)/2) (SF12 ≈ 5.7x — right for idle recovery, wrong
                # for fail-fast).  Pre-divide by the same factor so the
                # stressed budget is ~flat across SF: high-SF marginals bail
                # to the slow tier instead of holding cores under load.
                import re as _re_sf
                _m_sf = _re_sf.search(r'SF(\d+)_', fname)
                _sf_j = int(_m_sf.group(1)) if _m_sf else 7
                _flat = (self._pressure_budget if _press_only
                         else self._stress_budget)
                job_budget = _flat / (2.0 ** ((_sf_j - 7) * 0.5))
                _stress_shrunk = True
            if self._decoder_script is None:
                self._slow_dispose(_meta, _slow_rel, 'no_decoder',
                                   executed=False,
                                   result_mono_ns=time.monotonic_ns())
                self._queue.task_done()
                continue

            # Phase-2 chase optimization: skip dispatch if a sibling capture
            # in the same RF bucket has already produced a verified emit.
            # The same admission predicate as the post-decode dedup — only
            # fires AFTER the frame reached the GUI — so no real packet is
            # lost; we just avoid running a full _decode_attempt to produce
            # a record the post-decode dedup would discard anyway.  Skipping
            # at dispatch is the biggest possible win: the worker never
            # imports the IQ, never builds FFT plans, never allocates the
            # (n_sym, N) symbol matrices.
            if self._bucket_already_decoded(fpath):
                self._rawlog('SKIP-BUCKET', os.path.basename(fpath))
                self._ref_dec_and_maybe_unlink(fpath)
                self._slow_dispose(_meta, _slow_rel, 'skipped_bucket',
                                   executed=False,
                                   result_mono_ns=time.monotonic_ns())
                self._queue.task_done()
                if self._verbose:
                    print(f"         [BUCKET-DEDUP] skip {os.path.basename(fpath)} "
                          f"— sibling already emitted", file=sys.stderr, flush=True)
                continue

            # Phase-2 in-flight wait: if a sibling worker is CURRENTLY
            # processing the same bucket, wait briefly (up to 2 s, polling
            # every 100 ms) for it to either admit or finish.  Closes the
            # race window where sibling B's dispatch fires before sibling
            # A's emit lands.  Without this wait, the simple "already
            # decoded" check above returns False for B because A hasn't
            # finished yet, and we end up doing the full decode twice in
            # parallel.  Worker thread blocking here doesn't reduce overall
            # throughput because each worker thread runs an independent
            # dispatch loop — threads waiting on bucket X stay idle, while
            # threads pulling jobs from bucket Y keep running.  2 s is well
            # above typical decode time (SF7-SF11 finish in 0.5-1.5 s); the
            # in-flight entry also has a 5 s GC timeout so a crashed worker
            # can never deadlock siblings indefinitely.
            if self._is_bucket_inflight(fpath):
                _wait_deadline = time.time() + 2.0
                _skipped = False
                while time.time() < _wait_deadline:
                    time.sleep(0.1)
                    if self._bucket_already_decoded(fpath):
                        # Sibling admitted while we waited — skip cleanly.
                        self._rawlog('SKIP-INFLIGHT-ADMIT', os.path.basename(fpath))
                        self._ref_dec_and_maybe_unlink(fpath)
                        self._slow_dispose(_meta, _slow_rel, 'skipped_inflight',
                                           executed=False,
                                           result_mono_ns=time.monotonic_ns())
                        self._queue.task_done()
                        if self._verbose:
                            print(f"         [BUCKET-DEDUP] skip {os.path.basename(fpath)} "
                                  f"— sibling admitted during wait",
                                  file=sys.stderr, flush=True)
                        _skipped = True
                        break
                    if not self._is_bucket_inflight(fpath):
                        # Sibling finished without admitting — dispatch
                        # normally (we may have better luck at this offset).
                        break
                if _skipped:
                    continue

            # Mark this bucket in-flight BEFORE writing to the worker stdin
            # so siblings see the in-flight state immediately.
            self._mark_bucket_inflight(fpath)

            # Start this thread's worker if not running
            if worker is None or worker.poll() is not None:
                worker = self._start_worker()
                if worker is None:
                    self._slow_dispose(_meta, _slow_rel, 'worker_start_failed',
                                       executed=False,
                                       result_mono_ns=time.monotonic_ns())
                    self._queue.task_done()
                    continue

            with self._lock:
                self._active_count += 1

            t_start = time.time()
            # Instrumentation (validation-only): initialize ALL diagnostic state
            # BEFORE the job try-block so the finally's slow_completed can never
            # reference an undefined variable (which would mask the real
            # exception).  Each is overwritten on the normal path below.
            output = ''
            _result_processed_mono_ns = None
            timed_out = False
            worker_died = False
            _pkt_parsed_count = 0
            _pkt_logged_count = 0    # records actually written (codex C1)
            _job_exc = None
            try:
                # Send file path, search mode, and per-job budget to worker.
                # Format: "<fpath>\t<mode>\t<budget>\n"
                # mode: empty = primary; "after:N" = relay after; "before:N" = orig before.
                if relay_after_syms is not None and relay_before_syms is not None:
                    # Time-isolated decode: blank before relay_after AND after
                    # relay_before, so the decoder sees only the requested
                    # window — used to lock onto a specific preamble.
                    relay_field = f'iso:{relay_after_syms},{relay_before_syms}'
                elif relay_after_syms is not None:
                    relay_field = f'after:{relay_after_syms}'
                elif relay_before_syms is not None:
                    relay_field = f'before:{relay_before_syms}'
                else:
                    relay_field = ''
                _bud_tok = '' if job_budget is None else f'{job_budget:.1f}'
                worker.stdin.write(fpath + '\t' + relay_field + '\t' + _bud_tok + '\n')
                worker.stdin.flush()

                # Read output until __END__ delimiter.
                # Wall-clock timeout must scale with SF: the decoder's CPU-time
                # budget is LORA_DECODE_BUDGET_S * 2^((sf-7)/2) (SF11 = 4x,
                # SF12 = ~5.7x), and an off-centre capture's recovery passes
                # (±BW alias re-centre + mask-and-continue) can legitimately use
                # most of that budget in WALL time too.  A fixed 90 s killed slow
                # SF11/12 recoveries mid-flight (e.g. LONG_TURBO Test26 hop1: the
                # one capture that decodes it needs >90 s, so it was abandoned
                # and the hop went missing) even though the same capture decodes
                # fine standalone.  Scale the wall deadline off the same budget
                # with headroom; floor at 90 s so low-SF behaviour is unchanged.
                import re as _re_t
                _bud = float(os.environ.get('LORA_DECODE_BUDGET_S', '12.0') or 0)
                _sfm = _re_t.search(r'SF(\d+)_', fname)
                _sf_t = int(_sfm.group(1)) if _sfm else 7
                _scaled_bud = _bud * (2.0 ** ((_sf_t - 7) * 0.5)) if _bud > 0 else 60.0
                _wall = max(90.0, _scaled_bud * 1.5 + 30.0)
                output_lines = []
                deadline = time.time() + _wall
                timed_out = True
                worker_died = False
                import select as _select
                # Raw-fd line reader: a bare readline() blocks with no
                # deadline at all, so a worker that hangs WITHOUT printing
                # (the common OpenBLAS/fork wedge) stalled this manager
                # thread forever — the deadline was only ever checked
                # between lines (UC audit e).  select()+readline() is NOT
                # the fix (readline's buffer can hold lines the fd no
                # longer shows, faking a timeout on a healthy worker), so
                # read the fd raw into our own buffer and split lines.
                # worker.stdout is read NOWHERE else, so its text-layer
                # buffer stays empty and can't steal bytes from us.
                _rbuf = b''
                _wfd = worker.stdout.fileno()
                while time.time() < deadline:
                    _nl = _rbuf.find(b'\n')
                    if _nl < 0:
                        _rdy, _, _ = _select.select(
                            [_wfd], [], [],
                            min(1.0, max(0.05, deadline - time.time())))
                        if not _rdy:
                            continue
                        try:
                            _chunk = os.read(_wfd, 65536)
                        except OSError:
                            _chunk = b''
                        if not _chunk:
                            worker_died = True
                            worker = None
                            break
                        _rbuf += _chunk
                        continue
                    line = _rbuf[:_nl].decode('utf-8', 'replace').rstrip('\r')
                    _rbuf = _rbuf[_nl + 1:]
                    if line == '__END__':
                        timed_out = False
                        break
                    if line.startswith('__BEGIN__'):
                        continue
                    output_lines.append(line)

                if timed_out:
                    # Distinguish worker death (EOF mid-job) from a genuine
                    # wall-deadline overrun: both used to report the same
                    # "TIMEOUT (90s)" even when the worker crashed instantly
                    # or the deadline wasn't 90 s (UC audit e).
                    if worker_died:
                        output_lines.append(
                            'ERROR: DECODE WORKER DIED (EOF mid-job)')
                    else:
                        output_lines.append(
                            f'ERROR: DECODE TIMEOUT ({_wall:.0f}s wall)')
                    try:
                        if worker:
                            worker.kill()
                    except Exception:
                        pass
                    worker = None

                output = '\n'.join(output_lines)
                # Instrumentation (validation-only): ONE timestamp for when the
                # manager finished processing this job's result — defined for
                # EVERY job (packet, no-packet, error) so slow_completed below
                # can reference it regardless of outcome.
                _result_processed_mono_ns = time.monotonic_ns()
                elapsed = time.time() - t_start
                self._rawlog('JOB', f"{fname} mode={relay_field or '-'} "
                                    f"budget={job_budget} elapsed={elapsed:.1f}s "
                                    f"timed_out={timed_out}", output or '(empty)')

                # Structured packet log: append each [PKT] record (decoded or
                # encrypted-header-only) with a receive timestamp to the JSONL the
                # web UI tails.  The web dedups by (pktid,hop); here we just append.
                if '[PKT]' in output:
                    import json as _json
                    _now = time.time()
                    _out = []
                    for _ln in output.split('\n'):
                        if _ln.startswith('[PKT] '):
                            try:
                                _r = _json.loads(_ln[6:])
                            except Exception:
                                continue
                            _pkt_parsed_count += 1   # instrumentation (parsed OK)
                            # Authoritative decode count (data-path truth, not the
                            # noisy compact stdout).  Distinct-packet key prefers a
                            # stable identity: Meshtastic PacketID+hop, else the raw
                            # payload hex, else the decoded text.
                            _uk = (_r.get('pkt_id') or _r.get('packet_id')
                                   or _r.get('raw_hex') or _r.get('text'))
                            with self._decoded_lock:
                                self._decoded_total += 1
                                if _uk is not None:
                                    self._decoded_keys.add((_r.get('proto'), _uk))
                                try:
                                    _cfm = float(_r.get('freq_mhz') or 0.0)
                                except (TypeError, ValueError):
                                    _cfm = 0.0
                                if _cfm > 0.0:
                                    self._confirmed_freqs.append(_cfm * 1e6)
                                    if len(self._confirmed_freqs) > 64:
                                        del self._confirmed_freqs[:-64]
                            if self._pkt_log_path:
                                _r['ts'] = _now
                                # Instrumentation (validation-only): timing +
                                # provenance so pre/post-input-EOF can be
                                # separated offline and a deferred job proven to
                                # complete during input.  capture_anchor_rf_s is
                                # the crop's candidate-preamble RF time (NOT a
                                # per-packet event time); rf_event_s is null
                                # because the decoder exposes no within-crop
                                # offset — labelled honestly via rf_time_basis.
                                _r['run_id'] = _RUN_ID
                                _r['result_processed_mono_ns'] = _result_processed_mono_ns
                                _r['job_submit_mono_ns'] = _meta.get('submit_mono_ns')
                                _r['manager_dequeue_mono_ns'] = _manager_dequeue_mono_ns
                                _r['decode_job_id'] = _meta.get('job_id')
                                _r['initial_slow'] = _meta.get('initial_slow')
                                _r['ever_deferred'] = _meta.get('ever_deferred')
                                _r['requeue_count'] = _meta.get('requeue_count')
                                # Phase-1 lineage (audit on): work_id + producer +
                                # detector-side fields, kept SEPARATE from the
                                # decoder-reported bw/sf/freq already in _r.
                                if _meta.get('work_id') is not None:
                                    _r['work_id'] = _meta.get('work_id')
                                    _r['queue_attempt_seq'] = _meta.get('queue_attempt_seq')
                                    _r['producer'] = _meta.get('producer')
                                    _r['parent_detection_id'] = _meta.get('parent_detection_id')
                                    _r['parent_candidate_id'] = _meta.get('parent_candidate_id')
                                    _r['candidate_hint_id'] = _meta.get('candidate_hint_id')
                                    _r['candidate_bin_delta'] = _meta.get('candidate_bin_delta')
                                    _r['window_id'] = _meta.get('window_id')
                                    _r['scan_id'] = _meta.get('scan_id')
                                    _r['candidate_link_status'] = _meta.get('candidate_link_status')
                                    _r['detector_freq_hz'] = _meta.get('detector_freq_hz')
                                    _r['detector_sf'] = _meta.get('detector_sf')
                                    _r['detector_bw_hz'] = _meta.get('detector_bw_hz')
                                    _r['detector_preamble_rf_s'] = _meta.get('detector_preamble_rf_s')
                                _r['decode_capture_fname'] = fname
                                _anchor = _meta.get('capture_anchor_rf_s')
                                _r['capture_anchor_freq_hz'] = _meta.get('capture_anchor_freq_hz')
                                _r['capture_anchor_rf_s'] = _anchor
                                _r['capture_anchor_freq_tol_hz'] = _meta.get('capture_anchor_freq_tol_hz')
                                _r['capture_anchor_time_tol_s'] = _meta.get('capture_anchor_time_tol_s')
                                _r['rf_time_basis'] = (
                                    'detector_candidate_preamble'
                                    if _anchor is not None else None)
                                _r['rf_event_s'] = None
                                _out.append(_json.dumps(_r, separators=(',', ':')))
                    if _out and self._pkt_log_path:
                        try:
                            with self._pkt_log_lock:
                                with open(self._pkt_log_path, 'a') as _pf:
                                    _pf.write('\n'.join(_out) + '\n')
                            # count only AFTER a successful write (codex C1):
                            # 'completed_decode' means a record was LOGGED, not
                            # merely parsed.
                            _pkt_logged_count += len(_out)
                        except Exception:
                            pass

                # Fast-pass straggler recovery: if a PRIMARY fast decode BAILED
                # on its realtime budget it may still hold an undecoded packet
                # (marginal CFO/SNR needing the rescue, or a 2nd packet).
                # Re-queue it to the SLOW tier with a big budget; the same
                # workers run it in the post-burst idle gap.  post-decode
                # _packet_dedup suppresses anything already emitted → no
                # double-count.  Only PRIMARY (not relay/iso) jobs, only from
                # the fast tier (is_slow guards against infinite re-queue).
                _will_requeue = ((job_budget is None or _stress_shrunk) and output
                                 and '[BUDGET]' in output
                                 and relay_after_syms is None
                                 and relay_before_syms is None)
                if _will_requeue:
                    self._ref_inc(fpath)
                    self._rawlog('REQUEUE-SLOW', fname)
                    # Carry the job-local meta forward.  initial_slow is
                    # IMMUTABLE (this was a fast-tier job that BAILED); mark
                    # ever_deferred and bump requeue_count.  Same job_id so the
                    # eventual deferred completion is traceable to the original.
                    _rq_meta = dict(_meta)
                    _rq_meta['ever_deferred'] = True
                    _rq_meta['requeue_count'] = _rq_meta.get('requeue_count', 0) + 1
                    # This attempt requeued -> mark it so the completion dispose
                    # below does NOT terminalize (the re-tiered job terminalizes
                    # on its later completion attempt).  Flag lives on THIS
                    # attempt's meta only (the forwarded _rq_meta copy is reset
                    # False at its own attempt entry).
                    _meta['_work_requeued'] = True
                    # Phase-1 lineage: same LOGICAL work_id, incremented attempt.
                    if _rq_meta.get('work_id') is not None:
                        _rq_meta['queue_attempt_seq'] = _rq_meta.get('queue_attempt_seq', 0) + 1
                        _run_event('work_retiered', run_id=_RUN_ID,
                                   work_id=_rq_meta['work_id'],
                                   queue_attempt_seq=_rq_meta['queue_attempt_seq'],
                                   job_id=_rq_meta.get('job_id'), reason='budget_requeue')
                    _run_event('slow_enqueued', job_id=_rq_meta.get('job_id'),
                               reason='budget_requeue',
                               requeue_count=_rq_meta['requeue_count'])
                    self._slow_queue.put((fpath, fname, None, None,
                                          self._slow_budget, _rq_meta))
                # (Work-lifecycle terminal accounting is centralized in
                # _slow_dispose; the attempt-local _work_requeued flag was reset
                # False at attempt entry and set True above iff this attempt
                # requeued.)

                if self._verbose:
                    if output and output.strip():
                        text = ''.join(
                            '         \u2502 ' + ln + '\n'
                            for ln in output.rstrip().split('\n'))
                        sys.stdout.write(text)
                        sys.stdout.flush()
                else:
                    # Secondary passes get a suffix so _compact suppresses failures
                    # and labels successes with the appropriate tag.
                    if relay_after_syms is not None:
                        _display_fname = fname + '[relay]'
                    elif relay_before_syms is not None:
                        _display_fname = fname + '[pre]'
                    else:
                        _display_fname = fname
                    text = self._compact(output or '', _display_fname, elapsed)
                    if text:
                        sys.stdout.write(text)
                        # Latency markers: wall-clock emit time per decoded packet
                        # (correlate pktid with the send log to get send->decoded
                        # latency).  Cheap; only fires on a real emitted packet.
                        import re as _re_lat
                        for _lm in _re_lat.finditer(
                                r'PacketID: (0x[0-9a-f]+)\s+Flags: 0x[0-9a-f]+ '
                                r'\(hop_limit=\d+ hop_start=\d+ hops_taken=(\d+)\)',
                                text):
                            sys.stdout.write(
                                f"         [LAT] {time.time():.3f} "
                                f"{_lm.group(1)} hop{_lm.group(2)}\n")
                        sys.stdout.flush()

            except Exception as e:
                _job_exc = repr(e)   # instrumentation: real failure, not no_packet
                self._rawlog('JOB-ERROR', f"{fname} exc={e!r}")
                sys.stdout.write(
                    f"         [DECODE ERROR] {fname}: {e}\n")
                sys.stdout.flush()
                if worker:
                    try:
                        worker.kill()
                    except Exception:
                        pass
                    worker = None
            finally:
                # RSS-bound worker recycle: decode workers ratchet to ~2 GB
                # anon each (numpy scratch never returned to the OS) and
                # nothing ever recycled them — the measured OOM driver on
                # 24/7 hosts (UC audit q).  A graceful stdin-close between
                # jobs costs one lazy respawn (~1-2 s, already the crash
                # path) only when the bound is actually exceeded.
                if worker is not None and worker.poll() is None:
                    try:
                        with open(f'/proc/{worker.pid}/statm') as _sm:
                            _wrss_mb = (int(_sm.read().split()[1])
                                        * (os.sysconf('SC_PAGE_SIZE') // 1024)
                                        / 1024.0)
                        _rss_lim = float(os.environ.get(
                            'LORA_DECODE_WORKER_RSS_MB', '1500') or 0)
                        if _rss_lim > 0 and _wrss_mb > _rss_lim:
                            print(f"[DECODER] worker rss {_wrss_mb:.0f}MB > "
                                  f"{_rss_lim:.0f}MB — recycling",
                                  file=sys.stderr, flush=True)
                            try:
                                worker.stdin.close()
                                worker.wait(timeout=5)
                            except Exception:
                                try:
                                    worker.kill()
                                except Exception:
                                    pass
                            worker = None
                    except Exception:
                        pass
                with self._lock:
                    self._active_count -= 1
                # Instrumentation (validation-only): NORMAL completion goes
                # through the SAME _slow_dispose helper as the four early exits
                # (centralized so no path can retire without logging).  Outcome
                # is a specific taxonomy (not everything-is-no_packet), and
                # 'packet' is keyed on the PARSED record count, not a substring.
                if _job_exc is not None:
                    _oc = 'manager_error'      # exception in the manager
                elif worker_died:
                    _oc = 'subprocess_error'   # worker EOF mid-job
                elif timed_out:
                    _oc = 'timeout'            # wall-deadline overrun
                elif _pkt_parsed_count > 0:
                    _oc = 'packet'             # >=1 parsed [PKT] record
                elif '[PKT]' in output:
                    _oc = 'parse_error'        # [PKT] present, none parsed
                else:
                    _oc = 'no_packet'          # clean run, produced nothing
                # work outcome is keyed on records actually LOGGED to the
                # canonical sink (codex C1) — NO parsed fallback.  When no
                # pkt-log sink is configured, _pkt_logged_count stays 0 and a
                # parsed-but-unlogged 'packet' becomes failed_record_log
                # (telemetry lost) rather than completed_no_decode.
                self._slow_dispose(_meta, _slow_rel, _oc, executed=True,
                                   result_mono_ns=(_result_processed_mono_ns
                                                   or time.monotonic_ns()),
                                   parsed_records=_pkt_parsed_count,
                                   logged_records=_pkt_logged_count)
                self._queue.task_done()
                # Phase-2: clear the in-flight bucket entry so waiting
                # sibling workers can proceed (either they'll see the bucket
                # as admitted and skip, or they'll dispatch normally if the
                # admit didn't happen).  ALWAYS run, even on exception /
                # timeout / worker crash — otherwise siblings deadlock until
                # the 5 s GC timeout.
                self._clear_bucket_inflight(fpath)
                # Reference count for capture-file cleanup.  ALWAYS run, even
                # on exception / timeout / worker crash — those decrement paths
                # are exactly when leaks would otherwise accumulate.
                self._ref_dec_and_maybe_unlink(fpath)

    @staticmethod
    def _payload_fingerprint(payload_bytes):
        """64-bit XOR-fold fingerprint with rotate-left-1 per 8-byte word.
        Two copies of the same payload differing by ≤14 bit errors will have
        Hamming distance ≤14 between their fingerprints (with high probability).
        (Algorithm from meshtastic-sniffer main.c dedup_fingerprint.)"""
        h = 0
        i = 0
        while i + 8 <= len(payload_bytes):
            w = int.from_bytes(payload_bytes[i:i + 8], 'little')
            h ^= w
            h = ((h << 1) | (h >> 63)) & 0xFFFFFFFFFFFFFFFF
            i += 8
        if i < len(payload_bytes):
            h ^= int.from_bytes(payload_bytes[i:], 'little')
        return h

    @staticmethod
    def _hamming64(a, b):
        """Hamming distance between two 64-bit integers."""
        return bin(a ^ b).count('1')

    def _compact(self, output, fname, elapsed=0):
        """Extract key decode results from full process_file output.

        When the decoder's chained-mask produces MULTIPLE decode results
        from the same file (e.g. hop0 + hop1 sharing one carrier within a
        single capture), the output contains several `--- Meshtastic Packet
        ---` blocks separated by `--- Retry #N ---` markers.  We split the
        output at those markers and recurse so each result emits its own
        record (different hops_taken values are distinct legitimate decodes
        and must each appear in the log)."""
        _is_relay = fname.endswith('[relay]')
        _is_pre   = fname.endswith('[pre]')
        _is_secondary = _is_relay or _is_pre

        # Multi-result path: if the output contains '--- Retry' markers,
        # split into sections (one per attempt) and emit each section's
        # result independently.  Pass attempts that only contain DIAG
        # output (no CRC line, no mesh block) are silently dropped by the
        # single-section path below.
        if '--- Retry' in output:
            # Split BEFORE each retry marker so each section is one attempt.
            import re as _re2
            sections = _re2.split(r'(?=\n  --- Retry #\d+ ---)', output)
            results = []
            for sec in sections:
                if sec.strip():
                    r = self._compact_single(sec, fname, elapsed,
                                              _is_relay, _is_pre, _is_secondary)
                    if r:
                        results.append(r)
            return ''.join(results)

        return self._compact_single(output, fname, elapsed,
                                     _is_relay, _is_pre, _is_secondary)

    def _compact_single(self, output, fname, elapsed,
                         _is_relay, _is_pre, _is_secondary):
        """Parse one decoder section (single decode attempt)."""
        lines = output.split('\n')
        crc_line = None
        mesh_lines = []
        diag_lines = []
        error_lines = []
        in_result = False
        in_mesh = False
        in_retry = False  # stop tracking errors after retry starts
        error = None
        hdr_lines = []  # header bins for diagnostics
        preamble_line = None
        header_summary = None
        no_preamble_lock = False  # set only when pass-1 attempt-0 found NO preamble
        _saw_sweep = False        # True once a CRC sweep has run (preamble was found)

        protocol_line = None
        payload_hex_line = None
        header_guess_lines = []
        lorawan_lines = []
        meshcore_lines = []
        in_lorawan = False
        in_meshcore = False

        for line in lines:
            s = line.strip()
            # Once a retry starts, stop tracking errors from the primary attempt
            if '--- Retry' in s:
                in_retry = True
            if s.startswith('sweep:') or s.startswith('DIAG sweep'):
                _saw_sweep = True  # preamble was found at least once
            # Only flag "no preamble lock" if NO sweep ever ran (preamble never found).
            # If sweeps ran but CRC failed, this is a bit-error failure, not a
            # missing-preamble failure — a different and more informative diagnosis.
            if 'count PMR>8dB=0' in s and 'count PMR>6dB=0' in s and not _saw_sweep:
                no_preamble_lock = True
            if s.startswith('ERROR:') or 'ERROR:' in s:
                if not in_retry:
                    error = 'ERROR'
                error_lines.append(s)
            if s.startswith('DIAG '):
                diag_lines.append(s)
            if s.startswith('sweep:'):
                diag_lines.append('DIAG ' + s)
            # Detect error/failure conditions (only from primary attempt)
            if not in_retry:
                for err_key in ('NO PREAMBLE', 'NO SFD', 'NO CFO',
                               'NOT ENOUGH HEADER SYMBOLS', 'NOT ENOUGH DATA'):
                    if s.startswith(err_key):
                        error = err_key
                if '*** HEADER CHECKSUM FAILED ***' in s:
                    error = 'HEADER CHECKSUM FAILED'
                if '*** INVALID PAYLOAD LENGTH ***' in s:
                    error = 'INVALID PAYLOAD LENGTH'
            # Capture diagnostic lines
            if s.startswith('Preamble bin:'):
                preamble_line = s
            if s.startswith('hdr['):
                hdr_lines.append(s)
            if s.startswith('Header:'):
                header_summary = s
            if s.startswith('net_off:'):
                if preamble_line:
                    preamble_line += '  ' + s
            # CRC line in RESULT section
            if '=== RESULT ===' in s:
                in_result = True
                continue
            if in_result and s.startswith('CRC:'):
                crc_line = s
                in_result = False
            # Protocol line (falls between RESULT and packet block)
            if s.startswith('Protocol:'):
                protocol_line = s
            # Raw payload hex + header guess (unknown protocol, CRC OK)
            if s.startswith('Payload (') and 'bytes):' in s:
                payload_hex_line = s
            if s.startswith('[Header guess]'):
                header_guess_lines.append(s)
            # LoRaWAN frame block
            if '--- LoRaWAN Frame ---' in s:
                in_lorawan = True
                continue
            if in_lorawan:
                if s == '' or s.startswith('---') or s.startswith('==='):
                    in_lorawan = False
                elif s:
                    lorawan_lines.append(s)
            # MeshCore packet block
            if '--- MeshCore Packet ---' in s:
                in_meshcore = True
                continue
            if in_meshcore:
                if s == '' or s.startswith('---') or s.startswith('==='):
                    in_meshcore = False
                elif s:
                    meshcore_lines.append(s)
            # Meshtastic packet block
            if '--- Meshtastic Packet ---' in s:
                in_mesh = True
                continue
            if in_mesh:
                if s == '' or s.startswith('==='):
                    in_mesh = False
                elif s:
                    mesh_lines.append(s)

        # Decode evidence beyond the Meshtastic block / RESULT CRC line.
        # A MeshCore (or LoRaWAN) frame that decodes MID-LADDER (e.g. attempt 0
        # wins via try-both) emits its packet block + [PKT] inline and the
        # ladder then continues into retries, so the job's output has NO
        # '=== RESULT ===' section and NO '--- Meshtastic Packet ---' block.
        # The old predicates below only looked at crc_line/mesh_lines and
        # rendered such jobs as "(no CRC/mesh parsed)" — hiding a VERIFIED
        # decode from the terminal (the [PKT] record still reached the packet
        # log, which is why the log and the compact stdout disagreed: the
        # 2026-07-02 counting trap, re-discovered 2026-07-07 after it burned
        # a full day of leg benches showing a phantom live-vs-offline gap).
        # Gate on verification evidence: the packet blocks also print for
        # structural-only 'candidate' frames (garbage sharing the 0x12 sync),
        # which must keep rendering as failures.
        _decoded_blocks = (bool(meshcore_lines or lorawan_lines)
                           and ('[DECRYPTED' in output
                                or '"confidence":"verified"' in output))

        # Secondary pass: suppress all error/failure output — finding nothing is normal.
        _crc_bad = (not crc_line
                    or 'FAIL' in (crc_line or '')
                    or 'not present' in (crc_line or ''))
        if _is_secondary and _crc_bad and not mesh_lines and not _decoded_blocks:
            return ''

        # If a later attempt produced a real CRC result, prefer it over an
        # early-attempt error (e.g. HEADER CHECKSUM FAILED from attempt 0
        # shouldn't mask a CRC FAIL from attempt 3 that got further).
        if error and not crc_line and not mesh_lines and not _decoded_blocks:
            parts = [f"         [DECODE {elapsed:.0f}s] {fname}: {error}"]
            for el in error_lines[:10]:
                parts.append(f"           {el}")
            for dl in diag_lines[:25]:
                parts.append(f"           {dl}")
            if error == 'HEADER CHECKSUM FAILED':
                if preamble_line:
                    parts.append(f"           {preamble_line}")
                for hl in hdr_lines:
                    parts.append(f"           {hl}")
                if header_summary:
                    parts.append(f"           {header_summary}")
            return '\n'.join(parts) + '\n'

        if not crc_line and not mesh_lines and not _decoded_blocks:
            if _is_secondary:
                return ''  # secondary pass found nothing — expected, stay quiet
            # Decoder produced output but we didn't recognize it; surface a hint.
            nonempty = [ln.strip() for ln in lines if ln.strip()]
            if nonempty:
                parts = [f"         [DECODE {elapsed:.0f}s] {fname}: (no CRC/mesh parsed)"]
                for ln in nonempty[:8]:
                    parts.append(f"           {ln}")
                return '\n'.join(parts) + '\n'
            return ''

        # Post-decode PacketID dedup: same (PacketID, hops_taken) within 60s
        # is a duplicate Welch-peak detection of the same physical transmission.
        # Different hops_taken values are NOT suppressed (original vs relay).
        import re as _re
        _pkt_id = None
        _hops_taken = None
        for ml in mesh_lines:
            _m = _re.search(r'PacketID:\s+(0x[0-9a-fA-F]+)', ml)
            if _m:
                _pkt_id = _m.group(1).lower()
            _m = _re.search(r'hops_taken=(\d+)', ml)
            if _m:
                _hops_taken = int(_m.group(1))
        if _pkt_id is not None and _hops_taken is not None and 'FAIL' not in (crc_line or ''):
            _pkey = (_pkt_id, _hops_taken)
            _now_pkt = time.time()
            with self._packet_dedup_lock:
                stale = [k for k, t in self._packet_dedup.items()
                         if _now_pkt - t > 60.0]
                for k in stale:
                    del self._packet_dedup[k]
                if _pkey in self._packet_dedup:
                    _tag = ('RELAY' if _is_relay else 'PRE' if _is_pre else 'DECODE')
                    return (f"         [{_tag} {elapsed:.0f}s] "
                            f"[DUP] {_pkt_id} hop{_hops_taken} — "
                            f"same hop already decoded recently\n")
                self._packet_dedup[_pkey] = _now_pkt
                # Phase-2: tell the dispatch loop that this RF bucket is
                # covered so sibling captures (KEEP-2 second save + PASS-2-iso
                # resubmit) skip their decode at dispatch time.
                self._mark_bucket_decoded(fname)

        # Payload fingerprint dedup: suppress same raw payload within 30s.
        # Targets non-Meshtastic frames (LoRaWAN, MeshCore, CRC-only) that have no
        # packet ID, and can appear twice from overlapping Welch windows or band-edge effects.
        # MUST NOT run when we already have a Meshtastic PacketID — that path is handled
        # by (PacketID, hops_taken) dedup above, which correctly distinguishes hop0 from
        # hop1.  Hop0 and hop1 carry nearly-identical payloads (only the flags byte
        # differs), so the FP-DEDUP Hamming-cluster check would falsely suppress every
        # legitimate relay copy.
        _crc_ok_for_fp = bool(crc_line and 'OK' in crc_line
                              and 'FAIL' not in crc_line)
        # Phase-2 admit-gap fix (2026-07-08): a mid-ladder verified decode
        # (MeshCore via try-both, etc.) has NO RESULT-block crc_line, so it
        # never reached this path — its bucket was never admitted and every
        # sibling capture decoded redundantly (measured: 0 real-carrier
        # SKIP-BUCKET events across whole SF12 MeshCore legs while SF11
        # Meshtastic siblings deduped fine).  A verified decrypt/MAC is at
        # least as strong an identity as a bare CRC-16 pass, so let it
        # drive the same fingerprint dedup + bucket admit.  Fingerprint the
        # verified [PKT] record's raw_hex — canonical across siblings —
        # NOT the section's "Payload hex:" line (that prints the pre-
        # try-both drift-on bytes, which differ per capture).
        # _decoded_blocks requires verification evidence, so candidate-only
        # frames still never admit.
        _fp_hex = None
        if _crc_ok_for_fp and payload_hex_line:
            _m_fp = _re.search(r'Payload \(\d+ bytes\): ([0-9a-fA-F]+)',
                               payload_hex_line)
            _fp_hex = _m_fp.group(1) if _m_fp else None
        elif _decoded_blocks:
            _m_fp = _re.search(r'"raw_hex":"([0-9a-fA-F]+)"', output)
            _fp_hex = _m_fp.group(1) if _m_fp else None
        if _fp_hex and _pkt_id is None:
            try:
                _raw_fp = bytes.fromhex(_fp_hex)
                if len(_raw_fp) >= 4:
                    _fp = BackgroundDecoder._payload_fingerprint(_raw_fp)
                    _now_fp = time.time()
                    _FP_WINDOW = 30.0
                    _FP_THRESH = 14
                    with self._fp_dedup_lock:
                        # Prune stale entries
                        for _k in [k for k, t in self._fp_dedup.items()
                                   if _now_fp - t > _FP_WINDOW]:
                            del self._fp_dedup[_k]
                        # Check for a matching cluster (same payload ± bit errors)
                        if any(BackgroundDecoder._hamming64(_fp, _k) <= _FP_THRESH
                               for _k in self._fp_dedup):
                            _tag_fp = ('RELAY' if _is_relay else 'PRE' if _is_pre
                                       else 'DECODE')
                            return (f"         [{_tag_fp} {elapsed:.0f}s] "
                                    f"[FP-DEDUP] duplicate payload suppressed\n")
                        if len(self._fp_dedup) < 512:
                            self._fp_dedup[_fp] = _now_fp
                            # Phase-2: same as for _packet_dedup admit —
                            # bucket is now covered; sibling captures
                            # skip decode at dispatch.
                            self._mark_bucket_decoded(fname)
            except Exception:
                pass

        _decode_tag = ('RELAY' if _is_relay else 'PRE' if _is_pre else 'DECODE')
        parts = []
        if not crc_line and _decoded_blocks:
            # Mid-ladder decode (no RESULT section in this attempt): label the
            # block so the terminal shows the frame instead of "(no CRC/mesh
            # parsed)".  Verified status is carried by the [DECRYPTED]/packet
            # lines that follow.
            parts.append(f"         [{_decode_tag} {elapsed:.0f}s] "
                         f"{'MeshCore' if meshcore_lines else 'LoRaWAN'} "
                         f"packet decoded (mid-ladder)")
        if crc_line:
            suffix = ' (no preamble lock — preamble absent or corrupted in capture)' if (no_preamble_lock and 'FAIL' in crc_line) else ''
            parts.append(f"         [{_decode_tag} {elapsed:.0f}s] {crc_line}{suffix}")
            if protocol_line:
                parts.append(f"         {protocol_line}")
            if 'FAIL' in (crc_line or ''):
                for dl in diag_lines[:25]:
                    parts.append(f"         {dl}")
        if payload_hex_line:
            parts.append(f"         {payload_hex_line}")
        for ml in header_guess_lines:
            parts.append(f"         {ml}")
        for ml in lorawan_lines:
            parts.append(f"         {ml}")
        for ml in meshcore_lines:
            parts.append(f"         {ml}")
        for ml in mesh_lines:
            parts.append(f"         {ml}")
        return '\n'.join(parts) + '\n'


def fmt_bw(bw_hz):
    if bw_hz >= 1000:
        v = bw_hz / 1000
        return f"{v:.2f}k" if v != int(v) else f"{int(v)}k"
    return str(bw_hz)


class AutoGainGovernor:
    """Software AGC that maximises weak-signal reach.

    Sensitivity on the HackRF is quantisation-limited well past the shipped
    manual defaults: measured on this pipeline, gain 60 has ~12 dB more SNR
    margin than gain 32 (noise floor rises only ~10 dB for ~28 dB more gain).
    So the policy is asymmetric — climb whenever the band is clean, back off
    only when clipping is SUSTAINED:

      - CLIPPED window = the same subsampled >0.5 % ADC-clip test [SAT] uses.
      - Back off (-STEP dB) when >DOWN_FRAC of the last DOWN_WIN windows
        clipped.  A nearby beacon at ~10 % duty never reaches that, so burst
        clipping keeps the high-sensitivity gain — clipped LoRa bursts still
        detect AND decode (measured: 2/2 beacons through 33-97 % clip), and
        the [SAT] spur gate handles their splatter.
      - Climb (+STEP dB) when the last UP_WIN windows are all clip-free and
        the previous window had no detections (never mid-packet).
      - A climb that gets reverted by overload within UP_WIN windows sets a
        ceiling at the reverted gain for CEIL_TTL_WIN windows (no flapping
        against a persistent strong source).
      - Manual override: a ctl-file write without our 'agc' marker (the web
        UI's live gain edit) disables the governor for the rest of the run.

    Actuation is a read-modify-write of the LORA_SDR_CTL json that soapy_rx
    already polls (0.5 s) for seamless live setGain — center_hz is preserved.
    """
    STEP = 4.0
    DOWN_FRAC = 0.25
    DOWN_WIN = 60
    UP_WIN = 120
    COOL_WIN = 30
    CEIL_TTL_WIN = 1800          # ~30 min of 1 s windows
    # SEVERE-clip fast backoff (item 9): a SINGLE window clipping this
    # fraction of samples is unambiguous overload — back off immediately,
    # bypassing the sustained-duty DOWN_WIN logic and the climb cooldown.
    # The duty threshold (25% over 60 windows) was tuned for a continuous
    # strong carrier; BURSTY close-node traffic clips 43-50% in individual
    # windows but never reaches that duty, so the governor ratcheted
    # 44->60 dB into ever-worse clipping (diagnosed live 2026-07-09).  A 44%
    # window is not the tolerable sensitivity-clip the duty gate allows.
    SEVERE_FRAC = 0.12
    SEVERE_PACE = 2              # windows to wait between severe backoffs so
                                 # the gain change applies + is re-measured
                                 # before stepping again (no over-correction)

    def __init__(self, ctl_path, start_gain, gmin, gmax):
        self.ctl_path = ctl_path
        self.gain = float(start_gain)
        self.gmin, self.gmax = float(gmin), float(gmax)
        # Window constants are env-overridable for testing (defaults are the
        # production behavior; a live climb test at 120 windows/step takes
        # ~15 min, at 15 it takes ~2 min).
        self.UP_WIN = int(os.environ.get('LORA_AGC_UP_WIN', self.UP_WIN))
        self.DOWN_WIN = int(os.environ.get('LORA_AGC_DOWN_WIN', self.DOWN_WIN))
        self.COOL_WIN = int(os.environ.get('LORA_AGC_COOL_WIN', self.COOL_WIN))
        self.SEVERE_FRAC = float(os.environ.get('LORA_AGC_SEVERE_FRAC',
                                                self.SEVERE_FRAC))
        self.enabled = True
        self.win = 0
        self.cool = 0
        self.severe_cool = 0
        self.hist = collections.deque(maxlen=self.UP_WIN)
        self.ceiling = None
        self.ceil_expire = 0
        self.busy_wait = 0

    def external_ctl(self, ctl_dict):
        """Called by the main loop whenever the ctl file changes on disk.
        A gain present WITHOUT our agc marker is a manual user edit."""
        if not self.enabled:
            return
        if ctl_dict.get('gain') is not None and not ctl_dict.get('agc'):
            self.enabled = False
            print(f"  [AGC] manual gain override ({ctl_dict['gain']} dB) — "
                  f"governor off for this run", file=sys.stderr, flush=True)

    def _write(self):
        cur = {}
        try:
            with open(self.ctl_path) as f:
                cur = json.load(f)
        except (OSError, ValueError):
            pass
        cur['gain'] = self.gain
        cur['agc'] = True
        tmp = self.ctl_path + '.tmp'
        try:
            with open(tmp, 'w') as f:
                json.dump(cur, f)
            os.replace(tmp, self.ctl_path)   # atomic — soapy_rx never sees a partial file
        except OSError as e:
            print(f"  [AGC] ctl write failed: {e}", file=sys.stderr, flush=True)

    def observe(self, clipped, busy, clip_frac=None):
        """Feed one 1-s window; may step the gain. busy = detections in the
        previous window (blocks climbing, never blocks backing off).
        clip_frac = fraction of samples clipped this window (for the severe
        fast-backoff); None keeps the old boolean-only behaviour."""
        if not self.enabled:
            return
        self.win += 1
        self.hist.append(bool(clipped))
        if self.severe_cool > 0:
            self.severe_cool -= 1
        # SEVERE-clip fast backoff: one unambiguously-overloaded window steps
        # the gain down NOW, bypassing the duty logic AND the climb cooldown.
        # Paced by SEVERE_PACE windows so the applied gain change is re-measured
        # before the next step (no over-correction); the ceiling set here stops
        # the climb logic from ratcheting straight back up.
        if (clip_frac is not None and clip_frac >= self.SEVERE_FRAC
                and self.severe_cool == 0
                and self.gain - self.STEP >= self.gmin):
            old = self.gain
            self.gain -= self.STEP
            self.ceiling = old
            self.ceil_expire = self.win + self.CEIL_TTL_WIN
            self.severe_cool = self.SEVERE_PACE
            self.cool = self.COOL_WIN          # also block climbing for a bit
            print(f"  [AGC] gain {old:.0f}->{self.gain:.0f} dB (SEVERE clip "
                  f"{clip_frac*100:.0f}% in ONE window — fast backoff; "
                  f"ceiling {self.ceiling:.0f})", file=sys.stderr, flush=True)
            self._write()
            self.hist.clear()
            return
        if self.cool > 0:
            self.cool -= 1
            return
        if self.ceiling is not None and self.win >= self.ceil_expire:
            self.ceiling = None

        recent = list(self.hist)[-self.DOWN_WIN:]
        if len(recent) >= self.DOWN_WIN and \
                sum(recent) > self.DOWN_FRAC * len(recent) and \
                self.gain - self.STEP >= self.gmin:
            frac = sum(recent) / len(recent)
            old = self.gain
            self.gain -= self.STEP
            # Gain `old` just proved too high — don't return to it until the
            # ceiling expires, whether or not a climb put us there.  Without
            # this, a periodic strong source that appears long after the climb
            # causes a slow climb/back-off oscillation (observed live).
            self.ceiling = old
            self.ceil_expire = self.win + self.CEIL_TTL_WIN
            print(f"  [AGC] gain {old:.0f}->{self.gain:.0f} dB "
                  f"(sustained clip {frac*100:.0f}% of last {len(recent)} windows; "
                  f"ceiling {self.ceiling:.0f} for ~{self.CEIL_TTL_WIN//60} min)",
                  file=sys.stderr, flush=True)
            self._write()
            self.hist.clear()
            self.cool = self.COOL_WIN
            return

        _up = min(self.gain + self.STEP, self.gmax)   # partial last step lands on gmax
        _climb_ready = (len(self.hist) == self.UP_WIN
                        and not any(self.hist)
                        and _up > self.gain + 0.01
                        and (self.ceiling is None or _up < self.ceiling - 0.01))
        if _climb_ready and busy:
            # Don't starve on a band with continuous (weak, non-clipping)
            # traffic: after COOL_WIN windows of climb-ready-but-busy, step
            # anyway — the band has proven clip-free for a full UP_WIN.
            self.busy_wait += 1
            if self.busy_wait < self.COOL_WIN:
                return
        if _climb_ready:
            self.busy_wait = 0
            old = self.gain
            self.gain = _up
            print(f"  [AGC] gain {old:.0f}->{self.gain:.0f} dB "
                  f"(clip-free {self.UP_WIN} windows — climbing for sensitivity)",
                  file=sys.stderr, flush=True)
            self._write()
            self.hist.clear()
            self.cool = self.COOL_WIN


def main():
    p = argparse.ArgumentParser(description='LoRa Schmidl-Cox Detector')
    p.add_argument('-r', '--rate', type=int, default=40_000_000)
    p.add_argument('-b', '--bandwidth', type=int, default=28_000_000)
    p.add_argument('-c', '--center', type=float, default=915.0,
                   help='Center frequency in MHz (e.g. 915.0). Values > 1e5 '
                        'are treated as Hz and converted — passing Hz here '
                        'used to silently inflate every printed/logged '
                        'absolute frequency and capture filename a '
                        'million-fold (internally harmless: all DSP is '
                        'center-relative, which is how it went unnoticed).')
    p.add_argument('-t', '--format', default='sc16', choices=['sc8', 'sc16'])
    p.add_argument('-f', '--file', default=None)
    p.add_argument('--window', type=float, default=1.0)
    # Default matches the shipped lora.toml [detect] overlap (0.5).  It was
    # 0.1 here for years while production ran 0.5 via the web config — a
    # silent divergence: manual/CLI runs had an ~18% structural DEAD ZONE
    # (measured: an SF11/125k preamble starting in the last ~160 ms of a
    # 0.9 s hop fits in NEITHER window — the 16-symbol edge-skip needs the
    # span fully inside one window, and 0.1 overlap < the 262 ms span).
    # A 30-position straddle sweep detects 30/30 at 0.5 vs 26/30 at 0.1.
    p.add_argument('--overlap', type=float, default=0.5)
    # 0.55 matches the web tune default — the CLI sat at 0.7 for years
    # (same class of CLI/production divergence as the overlap default).
    p.add_argument('--threshold', type=float, default=0.55,
                   help='Schmidl-Cox preamble score threshold [0..1]. '
                        'During preamble all same-bin chirps score → 1.0; '
                        'noise scores near 0.  Default 0.7 works well in practice.')
    p.add_argument('--energy-threshold', type=float, default=5.0,
                   help='dB above PSD median to qualify as an energy peak. '
                        'With the n_avg=50 gate Welch, SF7/500k margin is ~6dB, '
                        'SF12 spans the whole buffer with much higher SNR.  '
                        'The SC + dechirp stages filter any non-LoRa peaks, so '
                        'this can stay sensitive without false detections.')
    p.add_argument('--spur-reject', type=float, default=SPUR_REJECT_DB,
                   help='Suppress detections >N dB below strongest')
    p.add_argument('--dc-notch', type=float, default=0.0,
                   help='Notch ±N MHz around center freq to suppress LO phase noise. '
                        'Set to 0.5-1.0 if signal is far from center. '
                        'Keep at 0 if signal is near center.')
    p.add_argument('--spur-notch', default='',
                   help='Blank known hardware spurs in the Stage-1 PSD. '
                        'Comma-separated list of absolute frequencies in MHz '
                        '(e.g. 920.0,923.5) or freq:half_width_MHz pairs '
                        '(e.g. 920.0:0.5,923.5:0.3). Default half-width: 0.5 MHz. '
                        'Example: --spur-notch 920.0 suppresses the bladeRF1 '
                        'LO harmonic at 920 MHz when tuned to 915 MHz.')
    p.add_argument('--channels', default='',
                   help='Channelizer mode (beta): ADDITIVELY run a dechirp matched-'
                        'filter on these learned LoRa channels every window — far more '
                        'sensitive than the energy gate, catching weak/far packets it '
                        'and Schmidl-Cox miss.  Rides ON TOP of the normal wideband '
                        'scan (no masking).  Comma-separated center_MHz:bandwidth_kHz'
                        '[:sf] tokens, e.g. 906.875:250,910.525:250:7.  Empty = off.')
    p.add_argument('--export-iq', default=None, metavar='DIR',
                   help='Export detected signal IQ to DIR as .cf32 files')
    p.add_argument('--decode', action='store_true', default=None,
                   help='Decode captured packets (default: on when --export-iq is set)')
    p.add_argument('--no-decode', action='store_true',
                   help='Disable packet decoding')
    p.add_argument('--decode-verbose', action='store_true',
                   help='Show full decode output (default: compact)')
    p.add_argument('-k', '--key', default=None,
                   help='AES key: base64, 32-char hex, or NOKEY (default: meshtastic)')
    p.add_argument('-d', '--debug', type=int, default=0)
    p.add_argument('--detect-workers', type=int, default=0,
                   help='Detection worker PROCESSES. 0 = inline serial (original '
                        'path, fine offline). N>0 fans each gate window out to a '
                        'pool of N single-threaded detect processes (true '
                        'parallelism, no GIL) for sustained full-rate live. '
                        '-1 = AUTO-SCALE from cpu_count (plug-and-play: ~ncpu/4, '
                        'capped 2..8) — use this so it adapts to any machine.')
    p.add_argument('--buf-seconds', type=float, default=45.0,
                   help='Ring buffer size in seconds for live input (default: 6.0). '
                        'Larger = fewer skips but more RAM (28Msps sc16: ~112MB/sec)')
    p.add_argument('--cooldown', type=float, default=0.0,
                   help='Minimum cooldown floor in seconds for duplicate suppression '
                        '(default: 0 = auto only). Cooldown scales automatically with SF/BW: '
                        'sym_time * 148.25, floored at 2×hop. Pass a value to '
                        'enforce a minimum. Uses a 20 kHz frequency bucket so relays with '
                        'different crystal offsets are treated as distinct signals.')
    p.add_argument('--config', default=None,
                   help='Load radio/detect/decode defaults from a lora.toml so you '
                        "don't pass long flag strings (explicit CLI flags still "
                        'override). Opt-in: only applied when --config is given.')
    p.add_argument('--agc', choices=['off', 'on'], default='off',
                   help='Software auto-gain for weak-signal reach. Climbs gain '
                        'while the band is clip-free, backs off only on SUSTAINED '
                        'ADC clipping (>25%% of recent windows) — burst clipping '
                        'from a nearby node keeps the high-sensitivity gain '
                        '(clipped LoRa still decodes). Needs LORA_SDR_CTL (the '
                        'live-gain file soapy_rx polls); live input only.')
    p.add_argument('--agc-start', type=float, default=None,
                   help='Gain (dB) the SDR was launched with — the AGC steps '
                        'relative to this. Required for --agc on.')
    p.add_argument('--agc-min', type=float, default=8.0,
                   help='AGC lower gain bound (dB).')
    p.add_argument('--agc-max', type=float, default=62.0,
                   help='AGC upper gain bound (dB).')
    # Pre-parse just to find --config, then use the config values as the argparse
    # DEFAULTS — anything NOT passed on the CLI comes from the config, anything
    # passed overrides it.  Opt-in so existing flag-based invocations (and the
    # validated harness) are unaffected.
    _pre, _ = p.parse_known_args()
    if _pre.config is not None:
        try:
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            from lora_config import load_config as _load_cfg
            _cfg = _load_cfg(_pre.config)
            _r, _d, _dec = _cfg['radio'], _cfg['detect'], _cfg['decode']
            _k = _dec.get('key')
            p.set_defaults(
                rate=int(_r['rate_hz']), bandwidth=int(_r['bandwidth_hz']),
                center=float(_r['center_mhz']), format=_r['format'],
                threshold=float(_d['threshold']),
                energy_threshold=float(_d['energy_threshold']),
                overlap=float(_d['overlap']),
                detect_workers=int(_d['detect_workers']),
                buf_seconds=float(_d['buf_seconds']),
                export_iq=_dec['export_dir'],
                key=(None if _k in (None, 'default') else _k))
            # env-driven knobs (setdefault → an explicit env var still wins)
            os.environ.setdefault('LORA_COMMIT_LAG', str(_d.get('commit_lag', 4)))
            os.environ.setdefault('LORA_DECODE_BUDGET_S', str(_dec.get('budget_s', 10.0)))
            # Only export workers if user pinned a positive count.  Otherwise
            # leave it unset so the auto-scale path (~ncpu-6, clamped 2..16) runs.
            _cfg_w = int(_dec.get('workers', -1) or -1)
            if _cfg_w > 0:
                os.environ.setdefault('LORA_DECODE_WORKERS', str(_cfg_w))
            os.environ.setdefault('LORA_PKT_LOG',
                                  _dec.get('packet_log', '/tmp/lora_packets.jsonl'))
            print(f"Loaded config {_cfg.get('_path')}", file=sys.stderr, flush=True)
        except Exception as _e:
            print(f"WARNING: --config failed ({_e}); using defaults/flags",
                  file=sys.stderr, flush=True)
    a = p.parse_args()
    # Unit robustness: -c is MHz by contract (the web pipeline passes 915.0),
    # but Hz has been passed by hand often enough to cause real analysis
    # mistakes downstream. Normalize: nothing on Earth monitors LoRa below
    # 100 kHz-as-MHz, so >1e5 can only mean Hz.
    if a.center > 1e5:
        a.center /= 1e6

    # Parse --spur-notch: "920.0,923.5:0.3" → [(920e6, 0.5e6), (923.5e6, 0.3e6)]
    _spur_notch_hz = []
    if a.spur_notch:
        for tok in a.spur_notch.split(','):
            tok = tok.strip()
            if not tok:
                continue
            if ':' in tok:
                freq_s, hw_s = tok.split(':', 1)
                _spur_notch_hz.append((float(freq_s) * 1e6, float(hw_s) * 1e6))
            else:
                _spur_notch_hz.append((float(tok) * 1e6, 0.5e6))  # default ±0.5 MHz

    # Channelizer (beta): the dechirp matched-filter runs on the channelizer's learned
    # channels (--channels).  ADDITIVE — these ride ON TOP of the full wideband energy
    # gate (no mask, no narrowing), so the gate keeps discovering everything else while
    # the dechirp tracks known channels at negative SNR.  Each is (center_hz, bw_hz, sf).
    _chan_sf = _parse_chan_tokens(a.channels)

    # Channels to run the dechirp matched-filter on (the channelizer's learned channels,
    # capped).  Each gets a forced per-window dechirp scan so weak/far packets the energy
    # gate misses still get despread and detected.  Each is (center_hz, bw_hz, sf).
    _dechirp_src = _chan_sf[:DECHIRP_MAX_CHANS]
    if _dechirp_src:
        print('[GATE] channelizer dechirp matched-filter on %d/%d channel(s)' % (
            len(_dechirp_src), len(_chan_sf)), file=sys.stderr, flush=True)

    # Patience-gate state (see constants at top).  _pat_hits: per-bin
    # exponentially-decayed exceedance counts on the max-hold PSD; _pat_seen:
    # the matching effective window count (for duty-cycle math); _pat_promoted:
    # carrier_bin -> (expire_wc, center_hz) of active promotions.
    _pat_hits = np.zeros(4096, np.float32)
    # Per-BIN effective observation count (not a scalar): masked bins (strong
    # peaks / recent detections) do not tick, and _pat_note_detection zeroes
    # seen alongside hits — otherwise duty = hits/seen is systematically
    # UNDERESTIMATED for bins whose clocks keep resetting, and spur-field
    # junk ducks any duty cap (g31: 7 junk promotions reappeared this way).
    _pat_seen = np.zeros(4096, np.float32)
    _pat_promoted = {}
    _pat_det_bins = {}   # bin -> last wc a REAL detection published there (any path
                         # incl. slow-pass); patience must never pursue carriers
                         # that are being detected — mask + expire around them.
    # Futility retire state (see PATIENCE_FUTILE_M constants): per-promotion
    # energetic-window counts with zero decode confirms, the set of promotions
    # a decode confirm has vouched for (futility-exempt until TTL), and the
    # per-bin re-promotion cooldown gate written by a futility demote.
    _pat_futile = {}      # promoted bin -> futile energy-present window count
    _pat_confirmed = set()  # promoted bins with >=1 decode confirm
    _pat_cooldown = {}    # bin -> wc when re-promotion is allowed again
    def _pat_note_detection(_freq_hz):
        """A REAL detection published at _freq_hz: zero accumulated hits and
        expire any promotion within ±1 MHz — patience only pursues carriers
        nothing else is detecting (this covers the slow-pass, which the
        strong-peak accumulation mask cannot see)."""
        if not PATIENCE_ON:
            return
        _b = 2048 + int(round((_freq_hz - _live_center_mhz * 1e6)
                              / (a.rate / 4096.0)))
        if not (0 <= _b < 4096):
            return
        _pat_det_bins[_b] = wc
        _pat_hits[max(0, _b - 410):min(4096, _b + 411)] = 0.0
        _pat_seen[max(0, _b - 410):min(4096, _b + 411)] = 0.0
        for _pb in [p for p in _pat_promoted if abs(p - _b) <= 410]:
            print('[PATIENCE] retired %.4fMHz — carrier is being detected '
                  'normally' % (_pat_promoted[_pb][1] / 1e6),
                  file=sys.stderr, flush=True)
            del _pat_promoted[_pb]
            _pat_futile.pop(_pb, None)
            _pat_confirmed.discard(_pb)
        # A real detection also clears any futility cooldown nearby — the
        # carrier is demonstrably real, so re-promotion must not be gated.
        for _qb in [q for q in _pat_cooldown if abs(q - _b) <= 410]:
            del _pat_cooldown[_qb]

    if PATIENCE_ON:
        print('[GATE] patience gate: margin %+.1f dB, promote at %.0f hits '
              '(~%d-win horizon), duty<%d%%, cap %d, ttl %d win, '
              'futile-retire %s'
              % (PATIENCE_MARGIN_DB, PATIENCE_MIN_HITS, PATIENCE_HORIZON_WIN,
                 int(PATIENCE_DUTY_MAX * 100), PATIENCE_CAP, PATIENCE_TTL_WIN,
                 ('at %d energetic win (cooldown %d)'
                  % (PATIENCE_FUTILE_M, PATIENCE_FUTILE_COOLDOWN))
                 if PATIENCE_FUTILE_M > 0 else 'off'),
              file=sys.stderr, flush=True)

    # [PRESCREEN] outcome-learned persistent-junk state (see constants at
    # top).  _jk_hits: per-bin decayed gate-peak hit counts (spread ±2 so
    # window-to-window centroid jitter doesn't split a spur's duty across
    # neighbours); _jk_seen: matching decayed window count (scalar — hits and
    # seen share one recursion so a 100%-duty bin reads duty≈1 at any age,
    # and the seen>=0.8*horizon gate is what enforces the minutes-long
    # observation); _jk_elev/_jk_wid: EMA elevation (dB over window floor) /
    # contour width per bin while unlearned — frozen into the entry at learn
    # time as the breakthrough reference; _jk_junk: bin -> [elev_ref,
    # width_ref, last_fire_wc]; _jk_det_bins: bin -> last wc a REAL detection
    # published there (any path — the learn veto).
    _jk_hits = np.zeros(4096, np.float32)
    _jk_seen = 0.0
    _jk_elev = np.zeros(4096, np.float32)
    _jk_wid = np.zeros(4096, np.float32)
    _jk_junk = {}
    _jk_det_bins = {}
    _jk_deaf = 0                 # VERIFY: prescreen-rejected peak turned into
                                 # a full-path detection (must stay 0)
    _jk_verify_recent = __import__('collections').deque()
    _jk_sup_peaks = 0            # peaks suppressed (or would-be, in VERIFY)
    _jk_sup_windows = 0          # windows fully quieted by suppression

    def _jk_note_detection(_freq_hz):
        """A REAL detection published at _freq_hz (any path: inline, pooled
        commit, slow-pass): record the bin, zero accumulated junk-hits around
        it and UNLEARN any junk entry near it — a detecting carrier is never
        junk.  VERIFY mode: match the detection against recently
        would-suppressed bins; any hit is a deafness event."""
        nonlocal _jk_deaf
        if not PRESCREEN_ON:
            return
        _b = 2048 + int(round((_freq_hz - _live_center_mhz * 1e6)
                              / (a.rate / 4096.0)))
        if not (0 <= _b < 4096):
            return
        _jk_det_bins[_b] = wc
        _jk_hits[max(0, _b - 15):min(4096, _b + 16)] = 0.0
        for _jb in [j for j in _jk_junk if abs(j - _b) <= 15]:
            print('[PRESCREEN] unlearned %.4fMHz — real detection at the bin'
                  % ((_live_center_mhz * 1e6
                      + (_jb - 2048) * (a.rate / 4096.0)) / 1e6),
                  file=sys.stderr, flush=True)
            del _jk_junk[_jb]
        if PRESCREEN_VERIFY:
            while _jk_verify_recent and wc - _jk_verify_recent[0][0] > 40:
                _jk_verify_recent.popleft()
            # width-adaptive match: wide-BW gate centroids err up to ±bw/2
            # (g500 lesson), so a fixed ±3-bin match would MISS real
            # deafness on wide presets — the instrument must over-report,
            # never under-report.
            for _vwc, _vbins in _jk_verify_recent:
                if any(abs(_b - _vb) <= max(3, _vhw + 2)
                       for _vb, _vhw in _vbins):
                    _jk_deaf += 1
                    print('[PRESCREEN-DEAF] detection at %.4fMHz matches a '
                          'bin the prescreen would have suppressed (wc %d) — '
                          'deafness counter now %d'
                          % (_freq_hz / 1e6, _vwc, _jk_deaf),
                          file=sys.stderr, flush=True)
                    break

    if PRESCREEN_ON:
        print('[GATE] prescreen (junk learner)%s: duty>=%d%% over ~%d-win '
              'horizon, probe every %d, margin %+.1f dB, gone %d win, cap %d'
              % (' [VERIFY — nothing suppressed]' if PRESCREEN_VERIFY else '',
                 int(PRESCREEN_DUTY * 100), PRESCREEN_LEARN_WIN,
                 PRESCREEN_PROBE_EVERY, PRESCREEN_MARGIN_DB,
                 PRESCREEN_GONE_WIN, PRESCREEN_CAP),
              file=sys.stderr, flush=True)

    # Build the multiprocess detect pool FIRST — before any reader/recorder/
    # decoder THREAD starts.  Forking a multithreaded process inherits the
    # other threads' held locks and the workers deadlock (the live gate hung
    # exactly here); here the process is still single-threaded, so fork is safe.
    # AUTO-SCALE detect workers from the machine's core count when asked (-1).
    # Plug-and-play: ~1/4 of the cores (leaving the rest for the gate, the niced
    # decode pool, and the OS), clamped to 2..8.  Validated value was 6 on a
    # 24-core box (24//4=6).  A box too weak to sustain the rate is caught at
    # runtime by the keep-up monitor (it warns + recommends a lower --rate)
    # rather than silently dropping samples.
    _hybrid = False
    if a.detect_workers is not None and a.detect_workers < 0:
        # Use affinity-aware count (see comment near the decode worker block).
        try:
            _ncpu_auto = len(os.sched_getaffinity(0))
        except (AttributeError, OSError):
            _ncpu_auto = os.cpu_count() or 4
        # Pool only when it can genuinely win: ncpu//4 workers, and if that
        # is <2 run SERIAL instead.  The old max(2, ...) floor forced a
        # 2-worker pool onto small hosts, where the pool's IPC/shm overhead
        # costs more than it parallelizes — measured live on a 4-core host:
        # 2 pooled workers pinned 2 cores at ~90% constant overhead and
        # throttled intake to ~6.3 Msps, while serial detection freed both
        # cores (quiet-window main-loop time dropped to ~0).  Hosts with
        # >=8 cores are unchanged (24-core validated value stays 6).
        _w = min(8, _ncpu_auto // 4)
        if _w < 2 and _ncpu_auto >= 4:
            # HYBRID DISPATCH (UC audit B, measured): both earlier verdicts
            # were right for their regime — a QUIET band pays the pool's
            # constant overhead for nothing (serial wins), but on a BUSY
            # band a 2-worker pool on the SAME 4 cores sustains 8.2 Msps
            # vs 3.8-4.4 serial (pipelining hides detect latency).  So on
            # 4-7-core hosts: build the small pool, then route PER WINDOW
            # in the main loop — inline while quiet, pool when candidate
            # peaks pile up or the ring backs up.  LORA_HYBRID=0 restores
            # pure serial; 2-3-core hosts stay serial (no room to win).
            if str(os.environ.get('LORA_HYBRID', '1')).strip().lower() not in (
                    '0', 'false', 'no', 'off'):
                _w = 2
                _hybrid = True
        a.detect_workers = _w if _w >= 2 else 0
        print(f"Detect workers: AUTO = "
              f"{a.detect_workers if a.detect_workers else 'serial'} "
              f"(cpu_count={_ncpu_auto}"
              f"{', hybrid dispatch' if _hybrid else ''})", flush=True)

    _pool = None
    _orig_detect_workers = a.detect_workers   # request, before serial-fallback
    if a.detect_workers and a.detect_workers > 0:
        from detect_pool import DetectPool
        _win_n = int(a.rate * a.window)
        # +8 not +4: slot exhaustion made dispatch spin in _commit_oldest
        # during busy sweep phases (detect phase spiking to 787 ms avg ->
        # ring wraps every ~2 min).  Hybrid small hosts get +4: quiet
        # windows bypass the pool entirely there, so the deep slot cushion
        # is unnecessary and shm is the scarce resource on a 4 GB Pi.
        # RAM co-sizing (codex OOM review 2026-07-23): each slot is a real
        # win_n*8-byte shm array (160 MB @20 Msps).  On a small-RAM host the
        # deep +8 cushion (10 slots = 1.6 GB) is memory the joint plan can't
        # spare — under permanent overload measured slot concurrency stays ~6-7
        # anyway, so cap the cushion to +4 (=6 slots, recovers ~640 MB for the
        # ring / headroom) when RAM is tight.  Big-RAM hosts keep +8.
        try:
            with open('/proc/meminfo') as _mi0:
                _memtot_slots = next(int(l.split()[1]) for l in _mi0
                                     if l.startswith('MemTotal:')) * 1024
        except Exception:
            _memtot_slots = 16 << 30
        _slot_extra = 4 if (_hybrid or _memtot_slots < int(12e9)) else 8
        _n_slots = a.detect_workers + _slot_extra
        if _hybrid:
            # RAM gate: the slots are real anonymous shm.  If they would
            # eat >30% of MemAvailable, hybrid loses to OOM risk — fall
            # back to serial (the previous small-host behaviour).
            _shm_need = _n_slots * _win_n * 8
            try:
                with open('/proc/meminfo') as _mi:
                    _avail_kb = next(int(l.split()[1]) for l in _mi
                                     if l.startswith('MemAvailable:'))
            except Exception:
                _avail_kb = 0
            if _avail_kb and _shm_need > _avail_kb * 1024 * 0.30:
                print(f"Detect pool: hybrid disabled — {_n_slots} slots need "
                      f"{_shm_need/1e9:.1f}GB shm vs "
                      f"{_avail_kb/1e6:.1f}GB available (running serial)",
                      flush=True)
                _hybrid = False
                a.detect_workers = 0
        if a.detect_workers and a.detect_workers > 0:
            _pool = DetectPool(
                n_workers=a.detect_workers, n_slots=_n_slots,
                win_n=_win_n,
                params=dict(wb_fs=a.rate, wb_bw=a.bandwidth, center=a.center,
                            sc_threshold=a.threshold, ethresh=a.energy_threshold,
                            dc_notch=a.dc_notch, spur_notch=_spur_notch_hz or None,
                            dechirp_chans=([(_c - a.center * 1e6, _s, _b)
                                            for (_c, _b, _s) in _dechirp_src] or None)))
            print(f"Detect pool: {a.detect_workers} worker processes "
                  f"({_n_slots} slots{', hybrid' if _hybrid else ''})",
                  flush=True)

    # ---- UNIFIED STARTUP MEMORY PLAN (codex OOM review 2026-07-23) ----
    # Ring + detect-pool slots + per-process private baselines + crop/decode
    # transients + OS headroom are ONE budget.  The old sizers were INDEPENDENT:
    # the ring took "40% of MemAvailable" (StreamBuffer.__init__) while the pool
    # separately reserved its slots — so on an 8 GB Pi at 20 Msps they summed
    # PAST RAM (ring 3.6 + slots 1.6 + 3 python baselines + decode ~2 + OS) and
    # OOM'd even with every queue bounded (measured: --no-decode plateaus ~5.4GB,
    # --decode OOMs).  Size the RING from what's LEFT after everything that
    # coexists, so the aggregate working set fits with margin.  Live path only
    # (file mode uses IQReader, no ring).  Big-RAM hosts are unaffected — the
    # plan only ever LOWERS buf_seconds (the requested value stays the ceiling).
    if a.file is None:
        _ring_bps = 4 if a.format == 'sc16' else 2      # ring stores RAW int16/int8
        _slot_bytes = (_n_slots * _win_n * 8) if (_pool is not None) else 0
        try:
            with open('/proc/meminfo') as _mi:
                _avail_b = next(int(l.split()[1]) for l in _mi
                                if l.startswith('MemAvailable:')) * 1024
        except Exception:
            _avail_b = 8 << 30
        _will_decode = ((a.decode if a.decode is not None
                         else (a.export_iq is not None)) and not a.no_decode)
        # Reserves calibrated to MEASURED Pi peaks (2026-07-23): at 20 Msps the
        # MAIN process private heap alone reached ~2.5 GB with decode on (numpy/
        # scipy/pyFFTW + gate arrays + the crop/save/decode pipeline), on top of
        # ring + shm slots.  Each figure carries margin over the observed value
        # so the aggregate leaves ~1 GB of MemAvailable headroom, not zero.
        _os_reserve = int(1.5e9)                        # OS + page cache (~1 GB obs + margin)
        _main_working = int(3.0e9 if _will_decode       # main heap: ~2.5 GB obs + 0.5 margin
                            else 1.5e9)                 # (no-decode main is far lighter)
        _worker_b = (a.detect_workers or 0) * int(0.5e9)  # per detect-worker private + margin
        # Explicit spike headroom NEVER given to the ring: the steady-state
        # reserves above under-predict transient peaks (decode crop copies +
        # pyFFTW scratch).  Hold ~1.2 GB back so the MemAvailable lower envelope
        # stays ~1 GB+ (codex margin guidance — a full-budget ring dipped to
        # ~370 MB, too thin; MALLOC_ARENA_MAX=2 now caps the fragmentation creep).
        _spike_margin = int(1.2e9)
        _ring_budget_b = max(0, _avail_b - _os_reserve - _slot_bytes
                             - _main_working - _worker_b - _spike_margin)
        _plan_bufs = _ring_budget_b / float(a.rate * _ring_bps)
        # Feasibility target: a ring shorter than ~8 s can't bridge slow-preset
        # overload bursts.  If the SAFE budget can't afford it, that's a signal
        # the host is over-configured for this rate — warn, but NEVER allocate
        # past the safety budget (floor 3 s just avoids a degenerate ring).
        if _plan_bufs < 8.0:
            print(f"[MEMPLAN] ring budget only {_plan_bufs:.1f}s (<8s target) on "
                  f"{_avail_b/1e9:.1f}GB after slots {_slot_bytes/1e9:.1f}GB + main "
                  f"{_main_working/1e9:.1f}GB + workers {_worker_b/1e9:.1f}GB + OS "
                  f"1.5GB — host tight for {a.rate/1e6:.0f} Msps this config",
                  file=sys.stderr)
        _plan_bufs = max(3.0, _plan_bufs)
        if _plan_bufs < a.buf_seconds:
            print(f"[MEMPLAN] ring {a.buf_seconds:.0f}s -> {_plan_bufs:.1f}s "
                  f"(joint budget: {_avail_b/1e9:.1f}GB avail - OS 1.5 - slots "
                  f"{_slot_bytes/1e9:.1f} - main {_main_working/1e9:.1f} - workers "
                  f"{_worker_b/1e9:.1f} = {_ring_budget_b/1e9:.1f}GB ring)",
                  file=sys.stderr)
            a.buf_seconds = _plan_bufs

    fp = open(a.file, 'rb') if a.file else sys.stdin.buffer

    # Use StreamBuffer for pipe/stdin (keeps pipe drained during detection),
    # plain IQReader for file input (no blocking issue)
    is_live = (a.file is None)
    if is_live:
        reader = StreamBuffer(fp, a.format, a.rate, buf_seconds=a.buf_seconds)
        buf_mb = int(a.rate * a.buf_seconds * (4 if a.format == 'sc16' else 2) / 1e6)
        print(f"Stream: ring buffer {a.buf_seconds:.1f}s ({buf_mb} MB) — "
              f"pipe never blocks", file=sys.stderr)
    else:
        reader = IQReader(fp, a.format)

    win_n = int(a.rate * a.window)
    hop_n = int(win_n * (1.0 - a.overlap))
    # Hop-read ownership (see _PREALLOC rule): at overlap > 0 (production)
    # every hop read is fully consumed before the next read — the slide
    # copies it into `buf` (or np.concatenate makes an owned array when a
    # carry-tail exists) — so the reader may hand out its reused buffer.
    # At overlap <= 0, hop_n >= win_n makes `buf = iq[-win_n:]` retain a
    # VIEW of the returned array across iterations: reads must be owned.
    _hop_owned = hop_n >= win_n

    print(f"=== LoRa Schmidl-Cox Detector ===", file=sys.stderr)
    print(f"Rate: {a.rate / 1e6:.1f}Msps  BW: {a.bandwidth / 1e6:.1f}MHz  "
          f"Center: {a.center:.3f}MHz", file=sys.stderr)
    print(f"Window: {a.window:.2f}s  Hop: {hop_n / a.rate:.2f}s", file=sys.stderr)
    print(f"Max peaks: {MAX_ENERGY_PEAKS}  "
          f"SC threshold: {a.threshold:.2f}  "
          f"Spur reject: {a.spur_reject:.0f}dB  "
          f"DC notch: {'±' + str(a.dc_notch) + 'MHz' if a.dc_notch > 0 else 'off'}",
          file=sys.stderr)
    print(f"Strategy: energy scan → 1Msps extraction → multi-lag Schmidl-Cox → dechirp confirm",
          file=sys.stderr)
    print(f"Listening...\n", file=sys.stderr)

    buf = np.zeros(win_n, dtype=np.complex64)
    # ---- LAZY INT16 QUIET WINDOW (see _LAZY at module top) ----
    # Enabled for serial detect and hybrid dispatch (the low-core hosts the
    # quiet-window win targets).  Pooled NON-hybrid hosts keep the eager
    # path: they dispatch EVERY window to a pool slot, so nothing is quiet,
    # and the reader's convert-ahead overlap is worth more there.
    # While _lz is active, `buf` is None on quiet windows and is bound to
    # _lz.materialize() (a persistent reused buffer, same retention rules
    # as the old in-place-slid window) whenever the peak list is non-empty.
    _lz = None
    if _LAZY and (_pool is None or _hybrid):
        _lz = _LazyWindow(win_n, a.format == 'sc16')
        buf = None
        print(f"Lazy quiet window: ON (raw "
              f"{'int16' if a.format == 'sc16' else 'int8'} slide, "
              f"materialize on gate peaks"
              f"{', VERIFY' if _LAZY_VERIFY else ''})", file=sys.stderr)
    pre_hop = None   # wideband data immediately before buf — for recording lookback
    from collections import deque as _deque
    _slow_halves = _deque(maxlen=8)   # last 8 hops = contiguous 4 s for the slow pass
    _slow_tick = 0
    _slow_seeds = {}
    _slow_backoff = {}   # bucket -> current fruitless-scan cooldown (windows)
    _slow_fruitless = {} # bucket -> consecutive fruitless scans this streak
    _slow_exact = {}     # bucket -> freshest exact peak bin (for seeding)
    _slow_busy = [False] # background slow-scan in flight (drop-if-busy)
    # Rate-bound for slow-scan LAUNCHES on low-core hosts (UC audit D,
    # measured: 20 scans x 1.05 s in 39 busy windows = 21% of total runtime
    # on a 4-core taskset).  On <8 cores enforce a minimum tick spacing
    # between scans and skip launches while the ring is under pressure —
    # deferred triggers stay in _slow_pending and refire while the
    # persistent-peak streak lasts, so a real slow preamble (>=1 s, streak
    # across many ticks) still gets scanned within its 4 s assembly window.
    # >=8-core hosts are unchanged (0 = no bound).
    try:
        _ncpu_gate = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        _ncpu_gate = os.cpu_count() or 8
    try:
        _SLOW_MIN_TICKS = int(os.environ.get(
            'LORA_SLOW_MIN_TICKS', '4' if _ncpu_gate < 8 else '0') or 0)
    except ValueError:
        _SLOW_MIN_TICKS = 0
    _slow_last_fire = -10**9
    _slow_pending = {}   # bucket -> (exact_bin, expiry_tick): triggers that
                         # fired while a scan was in flight, deferred instead
                         # of dropped (the drop-if-busy race lost ~2/29 real
                         # SF12 frames per leg whose trigger ticks all landed
                         # in another bucket's busy shadow — typically their
                         # own IQ-image's scan)
    # maxsize=1 backstop: single-flight (below) keeps _slow_busy TRUE until a
    # result is drained AND processed, so at most one result is ever pending —
    # but the bound guarantees the ~640 MB _iql assemblies can never pile up
    # unboundedly (the dominant 20 Msps OOM driver: the old unbounded queue +
    # busy-cleared-at-scan-complete let a new scan launch while the prior 640 MB
    # result sat undrained, growing RSS ~200 MB/s).  A hit-less scan drops _iql
    # entirely (puts None), so only scans that actually detect retain the array.
    _slow_results = queue.Queue(maxsize=1)   # (dets_slow, _iq_long|None, reg_tot_s, trig, scan_id)
    _slow_iql_dropped = [0]          # telemetry: hit-less scans whose 640MB _iql was freed
    _slow_max_qdepth = [0]           # telemetry: peak _slow_results depth observed
    # Tail samples consumed last iter (for save).  Prepended to this iter's iq
    # so the audio timeline stays continuous — without this, every save creates
    # a `tail_n` audio gap between adjacent Welch windows and packets that land
    # in that gap are never detected.
    _carry_tail = None
    buf_pos = tot_s = tot_d = wc = 0
    _sat_max = 0.0   # worst ADC-clip fraction since the last [STAT] (surfaced as a warning)
    # Saturation subsample: target a FIXED ~256k-sample statistical population
    # regardless of rate (host/rate-agnostic — the old hardcoded 32x scanned
    # ~625k strided samples across the whole 160MB window every hop; under the
    # detect-worker memory-bus contention that cache-hostile gather measured
    # ~67ms, far above its ~16ms idle cost).  256k gives a clip-fraction
    # standard error <0.01% — negligible vs the 0.5% engage threshold — while
    # touching ~2.4x fewer cache lines.  Floor of 32 keeps low-rate behaviour.
    _sat_stride = max(32, win_n // 262144)
    # Worst-case preamble duration across all Meshtastic presets:
    #   SF12/62.5k = 16×4096/62500 = 1.049s  →  not capturable within hop anyway
    #   SF12/125k  = 16×4096/125000 = 0.524s  ← needs most lookback in-window
    #   SF11/125k  = 16×2048/125000 = 0.262s
    #   SF11/250k  = 0.131s, SF7/500k = 0.004s
    # Cap pre_hop at MAX_PRE_HOP_S: 0.6s covers SF12/125k preamble with ~15%
    # margin.  Save file grows from ~1.4s to ~1.9s at 20Msps but decode succeeds.
    _MAX_PRE_HOP_S = 0.6
    _max_pre_hop_n = min(hop_n, int(_MAX_PRE_HOP_S * a.rate))
    # ---- Short-sweep slice cache (UC audit E, bit-exact) ----
    # At production hop (0.5 s = exactly 10 short-sweep steps) the low
    # slices of each window are the SAME ABSOLUTE SAMPLES as the high
    # slices of the previous window, so their Welch PSDs are byte-identical
    # and recomputed for nothing.  Cache the PRISTINE (un-notched) welch
    # output keyed by ABSOLUTE sample position: a hit ⟺ identical samples ⟺
    # identical PSD, so notch+find_peaks run fresh on a COPY every window
    # and the emitted peaks are bit-exact.  Keying by absolute position
    # makes every reset/skip/carry-tail path a natural cache MISS (never a
    # wrong hit), so no explicit invalidation is needed.  LORA_SLICE_CACHE=0
    # disables; LORA_SLICE_CACHE_VERIFY=1 asserts each hit against a fresh
    # welch.
    _slice_cache = {}          # abs_start_sample -> [psd, psd_max|None, notch_sig, peaks]
    _slice_cache_on = os.environ.get('LORA_SLICE_CACHE', '1') != '0'
    # Reuse the post-processed per-slice PEAK list too (not just the PSD) on a
    # cache hit with unchanged notch signature — skips copy+notch+find_peaks,
    # the sweep's dominant cost.  A/B kill-switch (default on); disable to prove
    # detection-equivalence vs the recompute-every-window path.
    _slice_peak_cache_on = os.environ.get('LORA_SLICE_PEAK_CACHE', '1') != '0'
    _slice_cache_verify = os.environ.get('LORA_SLICE_CACHE_VERIFY') == '1'
    _slice_cache_hits = 0
    _slice_cache_miss = 0
    # ---- pre_hop reference reuse (UC audit E, bit-exact) ----
    # pre_hop is buf[hop_n-_max_pre_hop_n:hop_n] of the pre-slide window —
    # at production overlap 0.5 those are the SAME ABSOLUTE SAMPLES already
    # owned by an earlier iteration's _hop_own (kept alive in _slow_halves
    # and fed as immutable refs to feed_tail — verified zero-copy).  Match
    # by ABSOLUTE position and reuse a view of that array instead of a fresh
    # 80 MB/window copy; ANY skip/carry-tail/reset fails the match and falls
    # back to the copy, so it is correctness-automatic.  Safe because every
    # pre_hop / _hop_own consumer is strictly read-only (recorder parts are
    # concatenated, never mutated; feed_tail holds refs).  LORA_PREHOP_REF=0
    # disables; LORA_PREHOP_VERIFY=1 asserts each reuse against a fresh copy.
    from collections import deque as _deque_ph
    _prehop_hist = _deque_ph(maxlen=4)   # (abs_start, hop_array) newest-last
    _prehop_ref_on = os.environ.get('LORA_PREHOP_REF', '1') != '0'
    _prehop_verify = os.environ.get('LORA_PREHOP_VERIFY') == '1'
    _prehop_ref_hits = 0
    _prehop_copies = 0
    tot_skip = 0
    _warned_slow = False   # keep-up monitor: warn once if the gate can't sustain rate
    t_start = time.time()
    center_hz = a.center * 1e6
    # Live center frequency: poll the SDR control file (the same file soapy_rx uses
    # to retune the radio) so a center change relabels detections WITHOUT a restart.
    # _live_center_mhz is read by the closures/loop below; the integer-bin transition
    # blip during a manual change lasts < 1 window and self-corrects.
    _live_center_mhz = a.center
    _ctl_path = os.environ.get('LORA_SDR_CTL') or None
    try:
        _ctl_mtime = os.stat(_ctl_path).st_mtime if _ctl_path else None
    except OSError:
        _ctl_mtime = None
    _last_ctl_poll = 0.0

    _agc = None
    if a.agc == 'on':
        if a.file is not None:
            print("AGC: ignored (file input)", file=sys.stderr)
        elif not _ctl_path:
            print("AGC: disabled — LORA_SDR_CTL not set (no live-gain path "
                  "to the SDR)", file=sys.stderr)
        elif a.agc_start is None:
            print("AGC: disabled — --agc-start (launch gain) not given",
                  file=sys.stderr)
        else:
            _agc = AutoGainGovernor(_ctl_path, a.agc_start,
                                    a.agc_min, a.agc_max)
            print(f"AGC: on — start {a.agc_start:.0f} dB, range "
                  f"[{a.agc_min:.0f}, {a.agc_max:.0f}], climb after "
                  f"{_agc.UP_WIN} clean windows, back off at "
                  f">{_agc.DOWN_FRAC*100:.0f}% clip duty",
                  file=sys.stderr)

    recorder = None
    if a.export_iq:
        recorder = SignalRecorder(
            a.export_iq, a.rate, center_hz, hop_n, debug=a.debug)
        print(f"Export: {a.export_iq}/ (window+pre-read tail, fs=8×BW)",
              file=sys.stderr)

    decoder = None
    do_decode = a.decode if a.decode is not None else (a.export_iq is not None)
    if do_decode and not a.no_decode:
        # Parse AES key
        import base64
        aes_key = None
        no_key = False
        if a.key:
            k = a.key
            if k.upper() in ('NOKEY', 'HAM', 'NONE', '0'):
                no_key = True
            elif len(k) == 32 and all(c in '0123456789abcdefABCDEF' for c in k):
                aes_key = bytes.fromhex(k)
            else:
                try:
                    kb = base64.b64decode(k)
                    if len(kb) in (16, 32):
                        aes_key = kb[:16]
                    else:
                        print(f"WARNING: Invalid key ({len(kb)} bytes). Using default.",
                              file=sys.stderr)
                except Exception:
                    print("WARNING: Could not decode key. Using default.",
                          file=sys.stderr)

        # Host/config memory context for the principled decode-worker RAM
        # reserve (see BackgroundDecoder RAM-cap block).  All inputs are in
        # scope: a.rate, a.buf_seconds, a.format, a.window, a.detect_workers
        # (finalized above), is_live.
        _bps_ring = 4 if a.format == 'sc16' else 2
        _ring_bytes = 0
        if is_live:                                   # file replay has no ring
            _ring_bytes = int(a.rate * a.buf_seconds * _bps_ring)
            try:                                      # mirror StreamBuffer 40% self-cap
                with open('/proc/meminfo') as _mi:
                    _avail_kb = next(int(l.split()[1]) for l in _mi
                                     if l.startswith('MemAvailable'))
                _ring_bytes = int(min(_ring_bytes, _avail_kb * 1024 * 0.40))
            except (OSError, StopIteration, ValueError):
                pass
        _win_n_ctx = int(a.rate * a.window)
        _dpool_bytes = 0
        if a.detect_workers and a.detect_workers > 0:
            _dpool_bytes = int((a.detect_workers + 8) * _win_n_ctx * 8)
        decoder = BackgroundDecoder(
            aes_key=aes_key, no_key=no_key, verbose=a.decode_verbose,
            host_mem_ctx=dict(ring_bytes=_ring_bytes,
                              detect_pool_bytes=_dpool_bytes,
                              win_n=_win_n_ctx))
        mode = "verbose" if a.decode_verbose else "compact"
        key_info = "NOKEY" if no_key else ("custom" if aes_key else "default")
        print(f"Decode: {mode} (key={key_info})", file=sys.stderr)

    if recorder and decoder:
        recorder.set_decoder(decoder)

    # Instrumentation (validation-only): one RUNMETA line with the RESOLVED
    # EFFECTIVE runtime values (post-derivation, not argv) under the shared
    # run_id — reconciles the runner manifest with what the detector actually
    # ran.  Emitted here (after pool/reader/decoder exist) so effective decode
    # workers and the real ring size are known.
    try:
        _eff_ring_n = getattr(reader, '_ring_n', None)
    except Exception:
        _eff_ring_n = None
    # effective detect workers = ACTUAL live pool process count (not the argparse
    # request, which may differ after the serial-fallback derivation).
    try:
        _eff_detw = len(_pool._workers) if _pool is not None else 0
    except Exception:
        _eff_detw = (0 if _pool is None else None)
    _run_event('runmeta', wb_fs=a.rate, fmt=a.format, is_live=is_live,
               requested_detect_workers=_orig_detect_workers,
               effective_detect_workers=_eff_detw,
               effective_decode_workers=(decoder._n_workers
                                         if decoder is not None else 0),
               ring_n=_eff_ring_n, buf_seconds=a.buf_seconds,
               overlap=a.overlap, t_start_wall=t_start,
               slow_no_ring_defer=_SLOW_NO_RING_DEFER)

    # Multiprocess detection pipeline state (the pool itself was built earlier,
    # before threads started, so its fork is safe).  With --detect-workers N>0
    # each gate window is fanned out to N single-threaded detect processes
    # (identical results) via shared memory and committed (print +
    # recorder.update) in window order with a lag.  Each capture's forward
    # "tail" is reconstructed from the next in-flight window (windows overlap
    # 50%), so no post-detect ring read is needed.
    _inflight = __import__('collections').deque()
    # Candidate-lifecycle audit (validation-only, gated off).  4096-pt gate FFT.
    _cand_mgr = _CandAuditMgr(a.rate, 4096) if _CAND_AUDIT else None
    _caw = None   # current window's audit object (None when audit off / not yet built)
    _seq = 0
    # Commit lag = how many windows stay in flight before the oldest is committed
    # (printed + saved + decoded).  At hop≈0.5 s this lag IS the airtime→result
    # latency floor (default 8 → ~4 s).  Env-overridable to probe lower latency:
    # a steady smaller fixed depth (still 1 commit/iter, no clustering) commits
    # sooner.  Floor is set by (a) results being ready when committed and (b) not
    # pulling save/decode contention into a burst.
    # Base commit lag (min in-flight depth before the oldest is committed).  This
    # is the CONTENTION floor — committing pulls save+decode earlier, and below
    # ~4 the memory-bound save/decode steals from the 28 Msps gate → drops
    # (measured: _LAG=3 dropped, =4 held, for the densest preset SF7).  The
    # ACTUAL commit also waits for enough trailing windows to cover the detected
    # packet's tail (see _tail_windows_needed) — so high-SF packets, whose tails
    # span many windows, auto-defer to a higher effective lag while low-SF
    # packets commit at this floor for minimum latency.  Universal across SF/BW:
    # no per-preset tuning.  Env-overridable for experiments.
    _LAG = int(os.environ.get('LORA_COMMIT_LAG', 4)) if _pool is not None else 0
    # Samples one trailing window contributes to a tail — must match the slice
    # _commit_oldest uses (slot[:win_n][hop_n:] = win_n-hop_n samples).
    _tail_seg_n = max(1, win_n - hop_n)

    def _tail_windows_needed(dets):
        """Trailing in-flight windows required to fully cover the tail of the
        longest packet in `dets` (0 if no detections).  Mirrors the tail span
        used in _commit_oldest so the commit never fires before the capture's
        forward samples exist — which is what makes a low _LAG safe for EVERY SF
        (SF12's multi-second packet needs several trailing windows; SF7 needs 1)."""
        if not dets:
            return 0
        _a0 = max(dets, key=lambda x: x.get('peak_power_db', 0.0))
        # patience_trial dets size for the same-slope family's slowest
        # symbol time (see patience_cap_params) — an alias-sized tail
        # truncates the true payload.  Non-trial: identical to 2**sf/bw.
        _need = int((16 + 4.25 + 8 + 120) * patience_cap_params(_a0)[1] * a.rate)
        return -(-_need // _tail_seg_n)   # ceil

    def _commit_oldest():
        """Pop the oldest in-flight window, wait for its detection, print and
        hand it to the recorder.  Tail = the next window's forward overlap."""
        nonlocal tot_d
        it = _inflight.popleft()
        _t_res0 = time.time()
        dets0 = _pool.result(it['seq'])
        _caw_c = it.get('cand_audit')
        if _caw_c is not None:
            _caw_c.emit(dets0, it.get('buf_len', win_n), 'completed',
                        time.monotonic_ns())
        _res_wait = time.time() - _t_res0
        if _res_wait > 0.2 and os.environ.get('LORA_SLOW_DEBUG'):
            print(f"[COMMIT-WAIT] {_res_wait:.2f}s on seq {it['seq']}",
                  file=sys.stderr, flush=True)
        _elapsed = time.time() - t_start
        for d in dets0:
            if d.get('patience_trial'):
                continue   # acquisition-internal: publish only on decode confirm
            _pat_note_detection(d['freq_hz'])
            _jk_note_detection(d['freq_hz'])
            tot_d += 1
            _abst = it['tot_s'] / a.rate + d.get('preamble_t_s', 0.0)
            _bwq = (f" bwq={d['bw_quality_db']:.0f}dB abst={_abst:.2f}s"
                    if a.debug >= 1 else "")
            print(f"[{_elapsed:6.1f}s] DETECTED freq={d['freq_mhz']:.4f}MHz "
                  f"SF={d['sf']} BW={fmt_bw(d['bw'])} sc={d['detect_conf']:.2f} "
                  f"pwr={d['peak_power_db']:.1f}dB{_bwq}", flush=True)
        if recorder:
            buf0 = _pool.slot_array(it['slot'])[:win_n]
            _tparts = []
            _need_tail_n = 0
            if _inflight and dets0:
                # Tail must cover the FULL packet AFTER the preamble.  A single
                # next-window overlap (~0.5 s) is enough for short-SF packets,
                # but long ones run past it and lose their payload: SF11/125
                # packets are ~1.4 s and SF12/125 ~2.9 s, so a 2 s capture
                # truncates the payload and the decoder yields only the header
                # plus a handful of nibbles (LONG_MODERATE was 38/120 for this
                # reason).  Gather the forward-overlap (hop_n:) of as many
                # subsequent in-flight windows as needed to span max_pkt_s for
                # the detected SF/BW.  Short-SF presets need only 1 window, so
                # their behaviour is unchanged.  Copies (slots get reused).
                _a0 = max(dets0, key=lambda x: x.get('peak_power_db', 0.0))
                _sym_t0 = (2 ** _a0['sf']) / _a0['bw']
                _need_tail_n = int((16 + 4.25 + 8 + 120) * _sym_t0 * a.rate)
                _acc = 0
                for _itn in _inflight:
                    _seg = _pool.slot_array(_itn['slot'])[:win_n][hop_n:]
                    _tparts.append(_seg.copy())
                    _acc += len(_seg)
                    if _acc >= _need_tail_n:
                        break
            if recorder._nb_mode:
                # NB-pending pool commit (UC audit q): crop base+tail straight
                # to NB via the same cropper the serial live path uses — the
                # save queue holds KB-scale NB, not the ~610 MB wideband array
                # recorder.update() would queue.  The tail is fully known now
                # (trailing windows in flight), so it is fed and finalized with
                # no deferral.
                recorder.update_deferred(dets0, buf0, it['tot_s'],
                                         pre_hop=it['pre_hop'],
                                         need_tail_n=_need_tail_n,
                                         tail_parts=_tparts)
            else:
                # Legacy wideband path (LORA_NB_PENDING=0, or NB setup
                # degenerate): eager concat, save worker FFT-crops.
                tail0 = np.concatenate(_tparts) if _tparts else None
                recorder.update(dets0, buf0, it['tot_s'], tail=tail0,
                                pre_hop=it['pre_hop'])
        _pool.release_slot(it['slot'])

    try:
      _prof = {'read': 0.0, 'slide': 0.0, 'welch': 0.0, 'notch': 0.0,
               'sat': 0.0, 'detect': 0.0, 'tail': 0.0, 'recorder': 0.0,
               'slow': 0.0, 'feed': 0.0, 'catchup': 0.0, 'n': 0,
               'rd_io': 0.0, 'rd_slide': 0.0,   # split of 'read': IO/convert vs window copy
               'sweep': 0.0, 'sweep_welch': 0.0, 'sweep_post': 0.0}  # split of 'notch'
      # PROF: parallel per-stage THREAD-CPU time (wall >> cpu => blocking, not
      # compute) — same keys as _prof, stamped at the same boundaries.
      _proft = {k: 0.0 for k in _prof}
      _tt_step = time.thread_time()
      _t_iter_start = time.time()
      # Waterfall: opt-in via LORA_PSD_FILE (web sets a tmpfs path).  Throttled,
      # reuses _psd_gate below → no extra FFT.  A sibling '<file>.off' marker (set
      # by the web's persistent toggle) suppresses emission live, so the gate spends
      # zero cycles on PSD frames when the user has the waterfall disabled.
      _psd_file = os.environ.get('LORA_PSD_FILE')
      _psd_off = (_psd_file + '.off') if _psd_file else None
      _psd_fps = float(os.environ.get('LORA_PSD_FPS', '10') or 10)
      _psd_last = 0.0
      # Max-hold detection (LORA_MAXHOLD, default ON): on top of the mean welch,
      # also take the per-bin MAX over the welch segments (free — same FFTs) and
      # merge its peaks.  The sparse welch catches a short packet in only a few
      # segments; the mean dilutes it ~/n_segs (up to ~18 dB), max-hold keeps the
      # catching segment's full power → ~+4-5 dB sensitivity (SF8-12), 0 false
      # peaks on noise.  Additive: the validated mean path is untouched; this only
      # appends extra candidates (SC+dechirp+CRC still confirm each).  Set
      # LORA_MAXHOLD=0 to disable on a very weak host (fewer candidates to confirm).
      _maxhold = str(os.environ.get('LORA_MAXHOLD', '1')).strip().lower() in ('1', 'true', 'yes', 'on')
      # Keep-up-driven peak throttle state (see the throttle block in the loop).
      # HI bounds the merged multi-pass candidate list on any host; LO is the
      # floor under sustained pressure (the strongest 3 are always processed).
      try:
          _PEAK_CAP_HI = max(3, int(os.environ.get('LORA_PEAK_CAP', '24')))
      except ValueError:
          _PEAK_CAP_HI = 24
      _PEAK_CAP_LO = 3
      _peak_cap = _PEAK_CAP_HI
      _cap_calm = 0
      _ring_n_total = max(1, int(a.rate * a.buf_seconds))
      # Proactive budget clamp state ([GATE-BUDGET] block in the loop; live
      # only).  EWMAs (alpha 0.3) of the MEASURED per-peak detect cost and the
      # fixed non-detect per-window overhead, both fed from the existing PROF
      # counters — no new timing.  'margin' is the fraction of the hop budget
      # the detect phase may spend; LORA_GATE_BUDGET=0 disables the clamp
      # entirely (exact legacy behavior).  pp==0.0 means "no measurement yet"
      # (cap stays at _PEAK_CAP_HI until the first busy window lands).
      _bgt = {'pp': 0.0, 'ovh': 0.0, 'ovh_cum': 0.0,
              'cap_last': _PEAK_CAP_HI,
              'margin': _psenv('LORA_GATE_BUDGET', 0.85, float, lo=0.0)}
      # Non-finite / absurd margins would crash the int() in the clamp
      # (LORA_GATE_BUDGET=inf -> OverflowError mid-flight); _psenv's lo-check
      # passes inf, so guard here.  Pure comparisons: NaN fails both sides,
      # inf fails the upper bound — no math import needed at this scope.
      if not (0.0 <= _bgt['margin'] <= 10.0):
          print(f"[GATE] ignoring out-of-range LORA_GATE_BUDGET="
                f"{_bgt['margin']} (using 0.85)", file=sys.stderr, flush=True)
          _bgt['margin'] = 0.85
      # Floor: shed-weakest-first keeps the strongest peaks (= the real
      # traffic).  2 by default; 1 fits hosts whose per-peak cost x2 exceeds
      # the hop even at floor (measured Pi: 2x171ms + 223ms ovh > 500ms hop
      # -> a slow deficit still builds -> one CATCHUP skip per few min).
      _BUDGET_CAP_FLOOR = int(_psenv('LORA_GATE_BUDGET_FLOOR', 2, int, lo=1))
      _TRIAL_DEFER = os.environ.get('LORA_TRIAL_DEFER', '1') != '0'
      _hop_budget_s = hop_n / float(a.rate)

      def _budget_track(_dt, _npk):
          # [GATE-BUDGET] EWMA maintenance, called right after the detect
          # phase.  Per-peak cost: this window's detect-phase elapsed over the
          # peaks it actually processed (pooled windows measure dispatch +
          # backpressure commits — small while workers keep up, ≈ the true
          # per-window worker cost once the pool saturates, which is exactly
          # the deficit regime the clamp exists for).  Overhead: per-window
          # delta of the cumulative PROF welch+gate/sweep+sat counters.
          # 'read' is deliberately EXCLUDED: in live mode it is dominated by
          # the reader's blocking wait-for-samples (≈ the whole hop on a host
          # that keeps up — pacing, not compute; measured 444-486 ms live vs
          # 9-10 ms file-mode for the same work), and including it pinned the
          # cap at floor on healthy hosts.  Its true convert cost (~10 ms) is
          # conservatively absorbed by the margin.  A negative delta means the
          # [STAT] block just reset the counters — resync and skip that sample.
          _cum = _prof['welch'] + _prof['notch'] + _prof['sat']
          _d_ovh = _cum - _bgt['ovh_cum']
          _bgt['ovh_cum'] = _cum
          if _d_ovh >= 0.0:
              _bgt['ovh'] = (_d_ovh if _bgt['ovh'] == 0.0
                             else 0.3 * _d_ovh + 0.7 * _bgt['ovh'])
          if _npk >= 1:
              _pp = _dt / _npk
              _bgt['pp'] = (_pp if _bgt['pp'] == 0.0
                            else 0.3 * _pp + 0.7 * _bgt['pp'])
      # Hybrid routing threshold + telemetry (UC audit B): windows with more
      # candidate peaks than this go through the pool; fewer run inline.
      try:
          _HYB_PEAKS = max(0, int(os.environ.get('LORA_HYBRID_PEAKS', '2')))
      except ValueError:
          _HYB_PEAKS = 2
      _hyb_inline = 0
      _hyb_pooled = 0

      def _notch_psd(_pp):
          if a.dc_notch > 0:
              _fr = a.rate / 4096; _db = max(1, int(round(a.dc_notch * 1e6 / _fr))); _cc = 4096 // 2
              _pp[max(0, _cc - _db):min(4096, _cc + _db + 1)] = np.median(_pp)
          if _spur_notch_hz:
              _fr2 = a.rate / 4096
              for _sf, _hw in _spur_notch_hz:
                  _sb = 4096 // 2 + int(round((_sf - _live_center_mhz * 1e6) / _fr2))
                  _nb = max(1, int(round(_hw / _fr2)))
                  _pp[max(0, _sb - _nb):min(4096, _sb + _nb + 1)] = np.median(_pp)
      while True:
        _t_step = time.time()
        _tt_step = time.thread_time()
        dets = []   # per-iteration default: the detect-pool path commits its
                    # detections asynchronously and never binds `dets` in this
                    # scope — the slow-pass block below reads it either way
                    # (crashed the pool path with UnboundLocalError before)
        # ---- Live SDR control: pick up a center-freq change (~2x/sec) ----
        if _ctl_path and _t_step - _last_ctl_poll > 0.5:
            _last_ctl_poll = _t_step
            try:
                _mt = os.stat(_ctl_path).st_mtime
                if _mt != _ctl_mtime:
                    _ctl_mtime = _mt
                    with open(_ctl_path) as _cf:
                        _cc = json.load(_cf)
                    if _agc is not None:
                        _agc.external_ctl(_cc)
                    if _cc.get('center_hz') is not None:
                        _nc = float(_cc['center_hz']) / 1e6
                        if _nc != _live_center_mhz:
                            _live_center_mhz = _nc
                            # The recorder reads center_hz live to compute each
                            # capture's wideband crop offset (off = freq_hz -
                            # center_hz).  It MUST track the retune or it crops the
                            # wrong slice and every decode fails after a move.
                            if recorder is not None:
                                recorder.center_hz = _live_center_mhz * 1e6
                            sys.stderr.write('detector: live center -> %.4f MHz\n'
                                             % _live_center_mhz); sys.stderr.flush()
                            # Patience accumulator is indexed by ABSOLUTE PSD
                            # bin — a retune shifts every carrier's bin, so
                            # stale counts would alias onto wrong frequencies
                            # and promoted-cluster bookkeeping would zero the
                            # wrong bins.  Reset; carriers re-earn quickly.
                            _pat_hits[:] = 0.0
                            _pat_seen[:] = 0.0
                            _pat_promoted.clear()
                            _pat_futile.clear()
                            _pat_confirmed.clear()
                            _pat_cooldown.clear()
                            # [PRESCREEN] junk state is bin-indexed too: a
                            # retune would remap every learned spur onto an
                            # arbitrary NEW frequency and suppress real
                            # signals there.  Hard reset (permissive: junk
                            # re-earns over the full horizon, nothing is
                            # ever suppressed at a wrong frequency).
                            if PRESCREEN_ON:
                                _jk_hits[:] = 0.0
                                _jk_elev[:] = 0.0
                                _jk_wid[:] = 0.0
                                _jk_seen = 0.0
                                _jk_junk.clear()
                                _jk_det_bins.clear()
                                _jk_verify_recent.clear()
                    # LIVE CHANNEL APPLY (channel-acquisition phase B): the web
                    # writes the channelizer's fed set here so a changed set
                    # (new learn, seed edit, enable toggle) takes effect WITHOUT
                    # a pipeline restart (the old path dropped in-flight frames
                    # on every re-channelize).  The ctl file's mtime changes on
                    # every AGC gain write too, so compare before announcing.
                    if _cc.get('channels') is not None:
                        _new_ch = _parse_chan_tokens(_cc['channels'])
                        if ([(c[0], c[2]) for c in _new_ch]
                                != [(c[0], c[2]) for c in _chan_sf]):
                            _chan_sf = _new_ch
                            _dechirp_src = _chan_sf[:DECHIRP_MAX_CHANS]
                            print('[GATE] channelizer live-apply: dechirp '
                                  'matched-filter on %d/%d channel(s)'
                                  % (len(_dechirp_src), len(_chan_sf)),
                                  file=sys.stderr, flush=True)
            except (OSError, ValueError):
                pass
        # ---- Read hop_n samples ----
        # If last iter consumed tail samples (for the save), prepend them
        # here and read fewer fresh samples — keeps the audio timeline
        # contiguous and prevents 38ms-per-iter cumulative gap.
        carry_n = len(_carry_tail) if _carry_tail is not None else 0
        if _lz is not None:
            carry_n //= 2      # lazy carries are RAW interleaved I,Q elems
        fresh_n = max(0, hop_n - carry_n)
        if is_live:
            if fresh_n > 0:
                _t_io0 = time.time()
                result = (reader.read_raw(fresh_n) if _lz is not None
                          else reader.read(fresh_n, owned=_hop_owned))
                _prof['rd_io'] += time.time() - _t_io0
                if result[0] is None:
                    break
                iq_fresh, skipped = result
            else:
                iq_fresh, skipped = (_lz.empty_raw() if _lz is not None
                                     else np.zeros(0, dtype=np.complex64)), 0

            if skipped > 0:
                # Temporal discontinuity — reset state
                tot_skip += skipped
                skip_s = skipped / a.rate
                elapsed = time.time() - t_start
                print(f"[{elapsed:6.1f}s] SKIP {skipped/1e6:.1f}M samples "
                      f"({skip_s:.2f}s) — detection took too long, "
                      f"ring buffer wrapped", file=sys.stderr)
                if recorder:
                    recorder.break_pending(f"(ring skip {skip_s:.2f}s)")
                if is_live and not _warned_slow:
                    _warned_slow = True
                    print(f"[{elapsed:6.1f}s] *** KEEP-UP WARNING: this machine "
                          f"cannot sustain {a.rate/1e6:.0f} Msps — samples are "
                          f"being DROPPED and packets WILL be missed.  Reduce the "
                          f"sample rate/bandwidth (-r and -b, e.g. -r {int(a.rate/2e6)}000000 "
                          f"-b {int(a.bandwidth/2e6)}000000) and/or raise "
                          f"--buf-seconds, or run on a faster host. ***",
                          file=sys.stderr, flush=True)
                # Reset the sliding window — fill entirely from scratch
                buf_pos = 0
                if _lz is not None:
                    _lz.reset()
                    buf = None
                else:
                    buf = np.zeros(win_n, dtype=np.complex64)
                pre_hop = None
                _prehop_hist.clear()   # abs continuity broken by the rebuild
                _slice_cache.clear()
                _carry_tail = None
                carry_n = 0
                if recorder:
                    recorder.reset_prev()
        else:
            if fresh_n > 0:
                iq_fresh = (reader.read_raw(fresh_n) if _lz is not None
                            else reader.read(fresh_n, owned=_hop_owned))
                if iq_fresh is None:
                    break
            else:
                iq_fresh = (_lz.empty_raw() if _lz is not None
                            else np.zeros(0, dtype=np.complex64))

        if _carry_tail is not None and len(_carry_tail) > 0:
            iq = np.concatenate([_carry_tail, iq_fresh]) if len(iq_fresh) > 0 else _carry_tail
        else:
            iq = iq_fresh
        _carry_tail = None

        # In lazy mode `iq` is RAW interleaved elems — 2 per IQ sample.
        _n_iq = (len(iq) // 2) if _lz is not None else len(iq)
        tot_s += _n_iq
        if _n_iq >= win_n:
            pre_hop = None   # full replacement — no valid lookback
            if _lz is not None:
                # set_full COPIES, so the reader's reused raw buffer is safe
                _lz.set_full(iq)
                buf = None
            else:
                # RETAINS A VIEW of iq across iterations — safe because iq is
                # always OWNED here: a concatenate result, an owned=True tail
                # read (_carry_tail), or an owned=_hop_owned hop read (this
                # branch needs len(iq_fresh) >= win_n, i.e. hop_n >= win_n).
                buf = iq[-win_n:]
        else:
            sh = win_n - _n_iq
            # Save the data about to slide off: it immediately precedes the new
            # window and is contiguous with it.  Used as recording lookback so
            # the decoder always sees the full preamble even when it started
            # before the detection window.
            if recorder:
                # Save the most recent _max_pre_hop_n samples from the portion
                # about to leave the window.  These are contiguous with the new
                # window: buf[hop_n-_max_pre_hop_n:hop_n] = [T-0.4s, T-0.1s).
                # Cap at 0.4s: enough lookback for SF11/125k preamble (0.26s)
                # while keeping save files to ~1.4s instead of 1.9s, cutting
                # decode queue wait roughly in half.
                #
                # The OLD (pre-slide) buf spans abs [tot_s-len(iq)-win_n,
                # tot_s-len(iq)); pre_hop's abs range is [_ph0, _ph1).
                _ph1 = tot_s - _n_iq - win_n + hop_n
                _ph0 = _ph1 - _max_pre_hop_n
                _ph_ref = None
                if _prehop_ref_on:
                    # Reuse an earlier _hop_own if one FULLY CONTAINS the
                    # pre_hop abs range (identical samples by construction).
                    for _has, _harr in _prehop_hist:
                        if _has <= _ph0 and _has + len(_harr) >= _ph1:
                            _ph_ref = _harr[_ph0 - _has:_ph1 - _has]
                            break
                if _ph_ref is not None:
                    if _prehop_verify:
                        _fresh = (buf[hop_n - _max_pre_hop_n:hop_n]
                                  if _lz is None else
                                  _lz.seg(hop_n - _max_pre_hop_n,
                                          _max_pre_hop_n))
                        if not np.array_equal(_ph_ref, _fresh):
                            print("[PREHOP-REF] MISMATCH — reference is NOT "
                                  "bit-exact!", file=sys.stderr, flush=True)
                    pre_hop = _ph_ref          # zero-copy view of owned array
                    _prehop_ref_hits += 1
                else:
                    # lazy: seg_owned converts JUST this pre-slide segment
                    # (fresh private allocation — same retention as .copy())
                    pre_hop = (buf[hop_n - _max_pre_hop_n:hop_n].copy()
                               if _lz is None else
                               _lz.seg_owned(hop_n - _max_pre_hop_n,
                                             _max_pre_hop_n))
                    _prehop_copies += 1
            _t_sl0 = time.time()
            if _lz is not None:
                _lz.slide(iq)      # 40 MB raw slide vs 320 MB c64 traffic
                buf = None         # nothing may touch buf until materialized
            else:
                buf[:sh] = buf[len(iq):]
                buf[sh:] = iq
            _prof['rd_slide'] += time.time() - _t_sl0
        buf_pos += _n_iq
        if buf_pos < win_n:
            continue
        _prof['read'] += time.time() - _t_step
        _proft['read'] += time.thread_time() - _tt_step
        _t_step = time.time()
        _tt_step = time.thread_time()

        # ---- Quick energy scan to decide if tail pre-read needed ----
        # compute PSD once here; pass it to detect_preamble to avoid recomputation.
        #
        # Welch averaging trade-off: more segments → smoother noise floor estimate,
        # but more dilution of short signals.  For a 10ms SF7/500k preamble in a
        # 1s buffer, signal-to-noise-floor ratio is
        #     SNR_psd ≈ 10·log10(1 + SNR_per_seg · N_sig / N_total)
        # n_avg=200 (step=5ms): signal in 2/200 → 8.2dB peak.  Threshold 7 → 1.2dB
        # margin — too tight; noise variance or environment shifts cause misses.
        # n_avg=50  (step=20ms): signal in 1/50  → 11.3dB peak.  Threshold 5 →
        # 6.3dB margin.  Noise-floor std rises from ~0.3dB to ~0.6dB (1/√n_segs),
        # still well below the 5dB threshold so false-positives stay negligible.
        # Higher SFs (preamble ≥ 100ms) span many segments under either setting,
        # so this change only HELPS short-preamble signals; long ones are
        # unaffected.
        if _maxhold:
            _psd_gate, _psd_gmax = (
                _lz.welch(0, win_n, n_avg=50, also_max=True)
                if _lz is not None else
                welch_psd(buf, nfft=4096, n_avg=50, also_max=True))
            _notch_psd(_psd_gate)
        else:
            _psd_gate, _psd_gmax = (
                _lz.welch(0, win_n, n_avg=50) if _lz is not None
                else welch_psd(buf, nfft=4096, n_avg=50)), None
            _notch_psd(_psd_gate)
        if a.dc_notch > 0:
            _fres = a.rate / 4096
            _dc_bins = max(1, int(round(a.dc_notch * 1e6 / _fres)))
            _dc_c = 4096 // 2
            _psd_gate[max(0, _dc_c - _dc_bins):min(4096, _dc_c + _dc_bins + 1)] = np.median(_psd_gate)
        if _spur_notch_hz:
            _fres2 = a.rate / 4096
            for _sf, _hw in _spur_notch_hz:
                _sb = 4096 // 2 + int(round((_sf - _live_center_mhz * 1e6) / _fres2))
                _nb = max(1, int(round(_hw / _fres2)))
                _psd_gate[max(0, _sb - _nb):min(4096, _sb + _nb + 1)] = np.median(_psd_gate)
        _prof['welch'] += time.time() - _t_step
        _proft['welch'] += time.thread_time() - _tt_step
        _t_step = time.time()
        _tt_step = time.thread_time()
        if _psd_file and (time.time() - _psd_last) >= (1.0 / _psd_fps):
            _psd_last = time.time()
            if not (_psd_off and os.path.exists(_psd_off)):
                _emit_psd_frame(_psd_file, _psd_gate)
        # Width floor: drop narrow CW/LO/clipping spikes (a few bins) BEFORE they reach
        # the expensive per-peak extract+SC+dechirp — the main source of high-gain
        # overload.  A LoRa channel is a wide flat hump; a spur is a sharp spike.  The
        # 1s + max-hold passes previously appended peaks with NO width check (only the
        # short-window sweep at `_w >= 30` filtered).  Carrier-aware (bins/Hz scales with
        # sample rate) and CAPPED so it can never exceed ~0.5x the narrowest configurable
        # LoRa BW (31.25 kHz) — so no real channel is dropped at ANY sample rate, with
        # margin.  At normal wideband rates (≤~16 Msps) this is 4 bins, killing the 1-3
        # bin CW/LO/clipping spikes; at higher rates it backs off (20 Msps -> 3 bins,
        # 61 Msps -> no-op) so a narrow channel spanning few bins is never at risk.
        _min_w = min(4, max(1, int(round(0.5 * 31250.0 / (a.rate / 4096.0)))))
        # Candidate audit: begin THIS window (every processed window gets exactly
        # one audit object, emitted later at commit/inline/drain).
        _caw = (_cand_mgr.new_window(tot_s, _live_center_mhz * 1e6)
                if _cand_mgr is not None else None)
        _gate_peaks = [_p for _p in find_peaks(_psd_gate, thresh_db=a.energy_threshold, fres_hz=a.rate / 4096.0)
                       if _p[1] >= _min_w]
        if _psd_gmax is not None:
            _notch_psd(_psd_gmax)
            for _mh in find_peaks(_psd_gmax, thresh_db=a.energy_threshold, fres_hz=a.rate / 4096.0):
                if _mh[1] >= _min_w and not any(abs(_mh[0] - _g[0]) < 15 for _g in _gate_peaks):
                    _gate_peaks.append(_mh)

        # ------------------------------------------------------------------
        # PSD DIAGNOSTIC (validation-only, gated OFF; additive — no behavior
        # change).  For windows whose RF interval contains a target time
        # (LORA_PSD_DIAG_TIMES, comma rf-seconds — the hop-0 controls AND the
        # hop-1 packets), dump the pre-rejection gate inputs around the carrier
        # band so the n_raw=0 question (true sub-threshold energy vs a peak
        # prominence/width/merge rejection) can be answered offline.  Emits mean
        # _psd_gate + max-hold _psd_gmax stats, the RAW find_peaks output in-band
        # (BEFORE the _min_w / merge filters -> shows the first-failed predicate),
        # baseline floor, threshold, and window RMS/clip.
        if _PSD_DIAG and _psd_diag_times:
            _nfft = len(_psd_gate)
            _win_len = win_n if _lz is not None else len(buf)
            _rf0 = tot_s / a.rate
            _rf1 = (tot_s + _win_len) / a.rate
            _hit = [_tt for _tt in _psd_diag_times if _rf0 <= _tt < _rf1]
            if _hit:
                import math as _m
                _fres = a.rate / _nfft
                _ctr = _nfft // 2                       # center bin from len(psd), not 2048
                _cbin = int(round((_psd_diag_freq_hz - _live_center_mhz * 1e6)
                                  / _fres)) + _ctr
                _blo, _bhi = max(0, _cbin - _psd_diag_halfbins), min(_nfft, _cbin + _psd_diag_halfbins)

                def _jnum(x):                            # JSON-safe: NaN/Inf -> None
                    try:
                        xf = float(x)
                        return xf if _m.isfinite(xf) else None
                    except Exception:
                        return None

                def _arr(a2):
                    # FULL precision (codex): rounding to 2dp could move a bin
                    # across nf+6 / nf+4 under the strict '>' predicate.  6dp is
                    # far below any dB granularity that matters yet keeps volume small.
                    return [None if not _m.isfinite(float(v)) else round(float(v), 6)
                            for v in a2]

                _nf = float(np.median(_psd_gate))        # THE find_peaks baseline
                _mean_band = _psd_gate[_blo:_bhi]
                _bmax = float(np.max(_mean_band)) if len(_mean_band) else None
                _barg = (int(np.argmax(_mean_band)) + _blo) if len(_mean_band) else None
                _cval = float(_psd_gate[_cbin]) if 0 <= _cbin < _nfft else None
                _gmax_band = None
                if _psd_gmax is not None:
                    _gb = _psd_gmax[_blo:_bhi]
                    _gmax_band = {'max_db': _jnum(np.max(_gb)) if len(_gb) else None,
                                  'argbin': (int(np.argmax(_gb)) + _blo) if len(_gb) else None,
                                  'baseline_db': _jnum(np.median(_psd_gmax)),
                                  'band_db': _arr(_gb)}
                # RAW find_peaks in-band, BEFORE _min_w / merge (first-failed):
                _raw_mean_pk = [[int(p[0]), int(p[1]), _jnum(p[2])]
                                for p in find_peaks(_psd_gate, thresh_db=a.energy_threshold,
                                                    fres_hz=_fres) if _blo <= p[0] < _bhi]
                _raw_max_pk = ([[int(p[0]), int(p[1]), _jnum(p[2])]
                                for p in find_peaks(_psd_gmax, thresh_db=a.energy_threshold,
                                                    fres_hz=_fres) if _blo <= p[0] < _bhi]
                               if _psd_gmax is not None else [])
                _adm_pk = [[int(g[0]), int(g[1]), _jnum(g[2])]
                           for g in _gate_peaks if _blo <= g[0] < _bhi]
                # notch state intersecting the band
                _dc_halfbins = int(round((a.dc_notch or 0.0) * 1e6 / _fres))
                _spur_ranges = []
                for (_sf_hz, _sh_hz) in (_spur_notch_hz or []):
                    _sb = int(round((_sf_hz - _live_center_mhz * 1e6) / _fres)) + _ctr
                    _shw = int(round((_sh_hz or 0.0) / _fres))
                    if _sb + _shw >= _blo and _sb - _shw < _bhi:
                        _spur_ranges.append([_sb - _shw, _sb + _shw])
                _carrier_notched = (abs(_cbin - _ctr) <= _dc_halfbins
                                    or any(lo <= _cbin < hi for lo, hi in _spur_ranges))
                _rms = None; _clipf = None
                try:
                    _bb = buf[:win_n] if _lz is None else None
                    if _bb is not None and len(_bb):
                        _rms = _jnum(np.sqrt(np.mean(_bb.real.astype(np.float64) ** 2
                                                     + _bb.imag.astype(np.float64) ** 2)))
                        _clipf = _jnum(np.mean((np.abs(_bb.real) > 0.98)
                                               | (np.abs(_bb.imag) > 0.98)))
                except Exception:
                    pass
                _step = max(_nfft, _win_len // 50)
                _n_segs = max(1, min(50, (_win_len - _nfft) // _step + 1))
                _run_event('psd_diag',
                           window_id=(_caw.awid if _caw is not None else None),
                           rf_start_s=_rf0, rf_end_s=_rf1,
                           sample_start=int(tot_s), sample_end=int(tot_s + _win_len),
                           target_times=_hit,
                           target_offsets_s=[round(_tt - _rf0, 4) for _tt in _hit],
                           carrier_freq_hz=_psd_diag_freq_hz, carrier_bin=_cbin,
                           center_bin=_ctr, band_start_bin=_blo, band_end_bin=_bhi,
                           fres_hz=_fres, nfft=_nfft, welch_n_avg=50,
                           welch_step_samples=_step, welch_n_segs=_n_segs,
                           welch_also_max=bool(_maxhold),
                           # find_peaks production predicates (offline reconstructs
                           # the first-failed predicate from the band arrays + these)
                           energy_threshold_db=a.energy_threshold, min_bins=3,
                           base_contour_db=4.0, base_min_bins=4,
                           wide_run_bins=(int(round(700e3 / _fres))),
                           min_width_bins=_min_w,
                           gate_baseline_db=_nf,
                           mean_psd_band_max_db=_jnum(_bmax),
                           mean_psd_band_argbin=_barg,
                           carrier_value_db=_jnum(_cval),
                           band_max_minus_baseline_db=(_jnum(_bmax - _nf)
                                                       if _bmax is not None else None),
                           carrier_value_minus_baseline_db=(_jnum(_cval - _nf)
                                                            if _cval is not None else None),
                           mean_psd_band_db=_arr(_mean_band),
                           maxhold_band=_gmax_band,
                           raw_meanpsd_peaks_inband=_raw_mean_pk,
                           raw_maxhold_peaks_inband=_raw_max_pk,
                           admitted_gate_peaks_inband=_adm_pk,
                           dc_notch_halfbins=_dc_halfbins,
                           spur_notch_band_ranges=_spur_ranges,
                           carrier_notched=bool(_carrier_notched),
                           # search-range facts: the energy gate's find_peaks runs
                           # over the FULL _psd_gate (no bandwidth-edge restriction;
                           # a.bandwidth only bounds per-peak extraction), so the
                           # active search range is [0, nfft) minus notches.
                           search_bin_range=[0, _nfft],
                           carrier_in_search_range=bool(0 <= _cbin < _nfft
                                                        and not _carrier_notched),
                           band_bins_excluded_from_search=(
                               ([[max(_blo, _ctr - _dc_halfbins),
                                  min(_bhi, _ctr + _dc_halfbins)]]
                                if abs(_cbin - _ctr) <= _dc_halfbins
                                or (_ctr - _dc_halfbins < _bhi
                                    and _ctr + _dc_halfbins > _blo) else [])
                               + _spur_ranges),
                           window_rms=_rms, clip_fraction=_clipf,
                           maxhold_on=bool(_maxhold))


        # ---- Multi-resolution Welch: short-window sweep ----
        # The 1s Welch above dilutes a 25ms SF7/500k burst by 40x, so its
        # per-bin peak above median can drop below the 5dB threshold when
        # other persistent (CW-like) signals contest the top peak slots.
        # Sweep 100ms slices at 50% overlap so short bursts dominate their
        # slice; merge any new carrier candidates (not already in the 1s peak
        # list) into _gate_peaks. SF/BW-agnostic: long preambles already
        # dominate the 1s pass, so the short pass only adds peaks for short
        # bursts that the 1s pass missed.  Cost: ~20 Welches with n_avg=10
        # at nfft=4096 ≈ 10ms per main-loop iteration.
        _t_sweep0 = time.time()
        _SHORT_WIN_S = 0.100
        _SHORT_OVERLAP = 0.5
        _SHORT_N_AVG = 10
        _short_n = int(_SHORT_WIN_S * a.rate)
        _short_step = max(1, int(_short_n * (1 - _SHORT_OVERLAP)))
        _short_peaks_all = []
        _n_short_slices = max(0, ((win_n if _lz is not None else len(buf))
                                  - _short_n) // _short_step + 1)
        # Post-processing (notch + find_peaks) on a slice's PRISTINE cached PSD
        # is deterministic given the notch parameters + energy threshold — so
        # for a cache-HIT slice whose notch signature is unchanged, the emitted
        # peak list is bit-identical and can be reused, skipping the per-slice
        # copy+notch+2×find_peaks (measured ~165ms/iter, the sweep's dominant
        # cost — the Welch itself is already 50%-overlap cached).  The signature
        # invalidates the peak cache (NOT the PSD cache) on any live retune /
        # spur-notch / dc-notch / threshold change.  The window-specific dedup
        # against _gate_peaks stays per-window, so behaviour is unchanged.
        # EXACT center (no rounding — _notch_psd and emitted absolute freqs
        # consume the unrounded value, so two distinct centers must not share a
        # key).  _maxhold is startup-fixed but included for free safety.
        _notch_sig = (_live_center_mhz, a.dc_notch, a.energy_threshold,
                      bool(_maxhold), tuple(_spur_notch_hz) if _spur_notch_hz else ())
        for _si in range(_n_short_slices):
            _ss = _si * _short_step
            _seg = buf[_ss:_ss + _short_n] if _lz is None else None
            # Absolute sample index of this slice's first sample (buf[0] is
            # at tot_s - win_n).  The cache key — identical across windows
            # for the same physical samples.
            _abs_start = tot_s - win_n + _ss
            _cs = _slice_cache.get(_abs_start) if _slice_cache_on else None
            _t_sw0 = time.time()
            if _cs is None:
                if _lz is not None:
                    # lazy: only this slice's ~10 Welch segments convert
                    if _maxhold:
                        _pw, _pw_max = _lz.welch(_ss, _short_n,
                                                 n_avg=_SHORT_N_AVG,
                                                 also_max=True)
                    else:
                        _pw, _pw_max = _lz.welch(_ss, _short_n,
                                                 n_avg=_SHORT_N_AVG), None
                elif _maxhold:
                    _pw, _pw_max = welch_psd(_seg, nfft=4096,
                                             n_avg=_SHORT_N_AVG, also_max=True)
                else:
                    _pw, _pw_max = welch_psd(_seg, nfft=4096,
                                             n_avg=_SHORT_N_AVG), None
                # Mutable [pristine_psd, pristine_psd_max, peak_notch_sig, peaks]
                # — PSD stays pristine (every use copies first); the last two
                # slots memoize the post-processed peak list per notch signature.
                _cs = [_pw, _pw_max, None, None]
                if _slice_cache_on:
                    _slice_cache[_abs_start] = _cs
                _slice_cache_miss += 1
            else:
                _slice_cache_hits += 1
                if _slice_cache_verify:
                    # Recompute fresh and assert the cache is bit-exact.
                    if _lz is not None:
                        if _maxhold:
                            _vw, _vw_max = _lz.welch(_ss, _short_n,
                                                     n_avg=_SHORT_N_AVG,
                                                     also_max=True)
                        else:
                            _vw, _vw_max = _lz.welch(_ss, _short_n,
                                                     n_avg=_SHORT_N_AVG), None
                    elif _maxhold:
                        _vw, _vw_max = welch_psd(_seg, nfft=4096,
                                                 n_avg=_SHORT_N_AVG,
                                                 also_max=True)
                    else:
                        _vw, _vw_max = welch_psd(_seg, nfft=4096,
                                                 n_avg=_SHORT_N_AVG), None
                    _bad = (not np.array_equal(_vw, _cs[0])) or (
                        (_cs[1] is None) != (_vw_max is None)) or (
                        _vw_max is not None
                        and not np.array_equal(_vw_max, _cs[1]))
                    if _bad:
                        print(f"[SLICE-CACHE] MISMATCH at abs={_abs_start} "
                              f"— cache is NOT bit-exact!", file=sys.stderr,
                              flush=True)
            _prof['sweep_welch'] += time.time() - _t_sw0
            _t_sp0 = time.time()
            # Peak-result reuse: on a slice whose pristine PSD is cached AND
            # whose notch signature matches the memoized one, the post-processed
            # peak list is bit-identical — reuse it and skip copy+notch+peaks.
            if (_slice_peak_cache_on and _cs[2] == _notch_sig
                    and _cs[3] is not None and not _slice_cache_verify):
                _short_peaks_all.extend(_cs[3])
                _prof['sweep_post'] += time.time() - _t_sp0
                continue
            _peak_reuse_ref = (_cs[3] if (_cs[2] == _notch_sig
                                          and _cs[3] is not None) else None)
            # Copy before notching so the cached array stays pristine.
            _slice_peaks = []
            _p_short = _cs[0].copy()
            _p_short_max = _cs[1].copy() if _cs[1] is not None else None
            _notch_psd(_p_short)
            if a.dc_notch > 0:
                _p_short[max(0, _dc_c - _dc_bins):min(4096, _dc_c + _dc_bins + 1)] = np.median(_p_short)
            if _spur_notch_hz:
                _fres_short = a.rate / 4096
                for _sf, _hw in _spur_notch_hz:
                    _sb_s = 4096 // 2 + int(round((_sf - _live_center_mhz * 1e6) / _fres_short))
                    _nb_s = max(1, int(round(_hw / _fres_short)))
                    _p_short[max(0, _sb_s - _nb_s):min(4096, _sb_s + _nb_s + 1)] = np.median(_p_short)
            # Width filter: a LoRa SF7/500k chirp spans ≥73 bins (500 kHz at
            # 6.84 kHz/bin) when partially captured by a short Welch segment;
            # narrowband CW interferers cluster in ≤10 bins.  Requiring a peak
            # width ≥ 30 bins (≈200 kHz) keeps LoRa-shaped peaks and rejects
            # the CW spurs that would otherwise burn detect_preamble compute.
            # Lower BWs (125 k, 62.5 k) produce narrower runs but they also
            # have long enough preambles to surface in the 1s pass already,
            # so the short-window contribution is only needed for ≥250 k.
            for _bin, _w, _db, *_pkrest in find_peaks(_p_short, thresh_db=a.energy_threshold, fres_hz=a.rate / 4096.0):
                if _w >= 30:
                    _slice_peaks.append((_bin, _w, _db,
                                         _pkrest[0] if _pkrest else None))
            if _p_short_max is not None:
                _notch_psd(_p_short_max)
                for _bin, _w, _db, *_pkrest in find_peaks(_p_short_max, thresh_db=a.energy_threshold, fres_hz=a.rate / 4096.0):
                    if _w >= 30:
                        _slice_peaks.append((_bin, _w, _db,
                                             _pkrest[0] if _pkrest else None))
            if _slice_cache_verify and _peak_reuse_ref is not None:
                # The reuse shortcut WOULD have returned _peak_reuse_ref; assert
                # the freshly recomputed peaks match it bit-for-bit (invalidation
                # test: exercise with LORA_SLICE_CACHE_VERIFY=1 while changing
                # center / threshold / dc-notch / spur list live).
                if tuple(_slice_peaks) != _peak_reuse_ref:
                    print(f"[SLICE-PEAK-CACHE] MISMATCH at abs={_abs_start} "
                          f"— peak reuse would NOT be exact!", file=sys.stderr,
                          flush=True)
            if _slice_cache_on:              # memoize for the next window's reuse
                _cs[2] = _notch_sig
                _cs[3] = tuple(_slice_peaks)  # immutable — never aliased/mutated
            _short_peaks_all.extend(_slice_peaks)
            _prof['sweep_post'] += time.time() - _t_sp0
        # Dedup by bin proximity (±15 bins ≈ ±100kHz) — narrower than 30 bins
        # so adjacent-channel hop0/hop1 pairs (commonly ±175 kHz apart) are
        # not collapsed into a single candidate. Sort by power desc so the
        # strongest survives within a cluster.
        _DEDUP_BINS = 15
        _short_peaks_all.sort(key=lambda p: -p[2])
        for _sp in _short_peaks_all:
            if not any(abs(_sp[0] - _mp[0]) < _DEDUP_BINS for _mp in _gate_peaks):
                _gate_peaks.append(_sp)
        _prof['sweep'] += time.time() - _t_sweep0
        # Prune the slice cache to the current window's span: only the
        # previous window's slices are ever reused (positions advance by one
        # hop each window), so anything older than this window's oldest
        # sample can never hit again.  Bounds the cache to <=_n_short_slices
        # entries (~19) regardless of run length.
        if _slice_cache_on and _slice_cache:
            _cut = tot_s - win_n
            for _k in [_k for _k in _slice_cache if _k < _cut]:
                del _slice_cache[_k]

        # ---- Patience gate: accumulate per-bin exceedances ----
        # A beacon too weak for the energy gate still concentrates max-hold
        # energy at its carrier bin (a slow chirp sweeps ~one 4.9 kHz bin per
        # short Welch segment), at the SAME bin every transmission.  Noise
        # exceedances scatter over 4096 bins; a real repeating source piles
        # them onto one bin cluster.  dB-domain, floor-relative (median), so
        # AGC gain moves don't poison the statistic.
        # SPLATTER GUARD (soundness review 2026-07-09): mask ±1 MHz around
        # every REAL gate peak this window before accumulating — patience is
        # for carriers the gate CANNOT see, and a strong signal's adjacent-
        # channel splatter otherwise accumulates at its own bins and promotes
        # (measured: a 10-beacon SF11/500k recording promoted its splatter at
        # -530 kHz and phantom-detected it under rotating trial labels).
        # Runs AFTER the full real-peak assembly (1s + max-hold + short-sweep
        # merge), BEFORE any forced/synthetic peaks are appended.
        # ---- Backlog-gated trial deferral ([TRIAL-DEFER]) ----
        # Forced dechirp trials (patience promotions + channelizer channels)
        # are budget-EXEMPT by design — and measured on-Pi they are what
        # busts realtime during dense traffic even with the [GATE-BUDGET]
        # cap at floor (p90 detect 2450 ms at cap 2 with promotions active
        # vs ~565 ms without → the remaining CATCHUP ring-skips).  Patience/
        # channelizer forcing is a SENSITIVITY feature for a host that is
        # keeping up; while a genuine backlog exists (>= one hop of unread
        # samples, same test as the budget clamp) the forced trials pause —
        # real gate peaks keep full service — and resume the moment the
        # deficit clears.  Futility counting pauses too (no trial ran = no
        # evidence).  LORA_TRIAL_DEFER=0 disables (legacy always-force).
        _win_backlogged = (is_live and _TRIAL_DEFER
                           and reader.available() >= hop_n)
        if _win_backlogged != _bgt.get('defer_last', False):
            print('[TRIAL-DEFER] backlog — forced trials paused'
                  if _win_backlogged else
                  '[TRIAL-DEFER] backlog cleared — forced trials resume',
                  file=sys.stderr, flush=True)
            _bgt['defer_last'] = _win_backlogged
        if PATIENCE_ON:
            _pp_pat = _psd_gmax if _psd_gmax is not None else _psd_gate
            _pat_decay = 1.0 - 1.0 / PATIENCE_HORIZON_WIN
            _pat_hits *= _pat_decay
            _pat_seen *= _pat_decay
            _pat_obs = np.ones(4096, np.float32)
            _pat_med = float(np.median(_pp_pat))
            _exc_pat = _pp_pat > (_pat_med + PATIENCE_MARGIN_DB)
            # STRONG peaks only (elevation >= 15 dB — the codebase's
            # established strong-peak bar): splatter implies a strong
            # parent.  Masking around EVERY peak (incl. transient noise
            # peaks at 6-8 dB) starved genuinely-invisible weak carriers
            # below the promotion bar (measured on the 0 dB A/B).
            for _rpk in _gate_peaks:
                if float(_rpk[2]) - _pat_med < 15.0:
                    continue
                _rb = int(_rpk[0])
                _exc_pat[max(0, _rb - 205):min(4096, _rb + 206)] = False
                _pat_obs[max(0, _rb - 205):min(4096, _rb + 206)] = 0.0
            # ...and ±2 MHz around anything REALLY detected in the last ~5 min
            # (any path, incl. slow-pass): far-node acquisition targets are
            # ISOLATED carriers; everything inside an active carrier's spur
            # field is the gate's business, not patience's (g31's junk sat at
            # 1.1-1.4 MHz — just outside the old ±1 MHz radius).
            for _db_, _dwc in list(_pat_det_bins.items()):
                if wc - _dwc > 600:
                    del _pat_det_bins[_db_]
                    continue
                _exc_pat[max(0, _db_ - 410):min(4096, _db_ + 411)] = False
                _pat_obs[max(0, _db_ - 410):min(4096, _db_ + 411)] = 0.0
            _pat_hits += _exc_pat
            _pat_seen += _pat_obs

            # ---- FUTILITY RETIRE (see PATIENCE_FUTILE_M constants) ----
            # A promotion whose bin repeatedly shows energy while its forced
            # dechirp confirms NOTHING is a birdie/spur (the fs/4 flicker
            # burned ~2.6 s/window for 10 min/TTL-cycle on a Pi).  Count
            # energy-present windows only — beacon-gap windows never tick, so
            # a sparse real beacon keeps full patience between packets.
            if PATIENCE_FUTILE_M > 0:
                # Decode confirms (trial captures decode on the slow tier and
                # land asynchronously): vouch for nearby promotions forever
                # (until TTL) and lift any nearby cooldown.
                if decoder is not None:
                    for _cfhz in decoder.drain_confirmed_freqs():
                        _cfb = 2048 + int(round(
                            (_cfhz - _live_center_mhz * 1e6)
                            / (a.rate / 4096.0)))
                        for _pb in [p for p in _pat_promoted
                                    if abs(p - _cfb) <= 410]:
                            if _pb not in _pat_confirmed:
                                _pat_confirmed.add(_pb)
                                _pat_futile.pop(_pb, None)
                                print('[PATIENCE] confirmed %.4fMHz by '
                                      'decode — futility exempt'
                                      % (_pat_promoted[_pb][1] / 1e6),
                                      file=sys.stderr, flush=True)
                        for _qb in [q for q in _pat_cooldown
                                    if abs(q - _cfb) <= 410]:
                            del _pat_cooldown[_qb]
                # Futile-trial count: raw max-hold exceedance at the promoted
                # bin (±1 for centroid jitter) — the same statistic that
                # earned the promotion, deliberately UNmasked (a nearby real
                # detection retires the promotion via _pat_note_detection
                # before masking could matter).  Deferred windows (backlog —
                # see _win_backlogged below) don't count: no trial ran, so
                # the window is not evidence of futility.
                _thr_fut = _pat_med + PATIENCE_MARGIN_DB
                for _pb in list(_pat_promoted):
                    if _win_backlogged:
                        break
                    if _pb in _pat_confirmed:
                        continue
                    if float(_pp_pat[max(0, _pb - 1):_pb + 2].max()) <= _thr_fut:
                        continue        # energy-absent window: does not count
                    _pat_futile[_pb] = _pat_futile.get(_pb, 0) + 1
                    if _pat_futile[_pb] < PATIENCE_FUTILE_M:
                        continue
                    print('[PATIENCE] futility-retired %.4fMHz — %d '
                          'energetic windows, zero confirms; cooldown %d win'
                          % (_pat_promoted[_pb][1] / 1e6, _pat_futile[_pb],
                             PATIENCE_FUTILE_COOLDOWN),
                          file=sys.stderr, flush=True)
                    _pat_cooldown[_pb] = wc + PATIENCE_FUTILE_COOLDOWN
                    _pat_hits[max(0, _pb - 4):_pb + 5] = 0.0
                    _pat_seen[max(0, _pb - 4):_pb + 5] = 0.0
                    del _pat_promoted[_pb]
                    del _pat_futile[_pb]

        # ---- [PRESCREEN] outcome-learned persistent-junk suppression ----
        # Sits AFTER patience accumulation (its masks see the untouched peak
        # list) and BEFORE the throttle + channelizer/patience forced-peak
        # appends (forced peaks are never suppressed; shedding junk here also
        # stops junk from eating throttle cap slots that would otherwise
        # push out weak REAL peaks).  If every peak in a window is learned
        # junk, _gate_peaks empties and the window takes the lazy quiet path
        # below — no materialization, no chunk-FFT, no per-peak work.
        # Candidate audit: snapshot the ENERGY candidates at their fullest —
        # AFTER the short-peak merge, BEFORE junk-suppression / shed / dedup.
        if _caw is not None:
            _caw.snapshot_raw(_gate_peaks)
        _jk_sup_this_win = False     # suppressed junk fired here (AGC busy)
        if PRESCREEN_ON:
            _jk_decay = 1.0 - 1.0 / PRESCREEN_LEARN_WIN
            _jk_seen = _jk_seen * _jk_decay + 1.0
            _jk_hits *= _jk_decay
            _jk_floor = float(np.median(_psd_gate))
            for _pk in _gate_peaks:
                _jb0 = int(_pk[0])
                if not (0 <= _jb0 < 4096):
                    continue
                _jk_hits[max(0, _jb0 - 2):min(4096, _jb0 + 3)] += 1.0
                _je = float(_pk[2]) - _jk_floor
                _jw = float(_pk[1])
                if _jk_elev[_jb0] > 0.0:
                    _jk_elev[_jb0] += 0.2 * (_je - _jk_elev[_jb0])
                    _jk_wid[_jb0] += 0.2 * (_jw - _jk_wid[_jb0])
                else:
                    _jk_elev[_jb0] = _je
                    _jk_wid[_jb0] = _jw
            # fast unlearn: spur vanished (its bin stopped firing) — drop the
            # entry so a future real signal there is never met by stale junk
            for _jb0 in [j for j, r in _jk_junk.items()
                         if wc - r[2] > PRESCREEN_GONE_WIN]:
                print('[PRESCREEN] unlearned %.4fMHz — bin stopped firing'
                      % ((_live_center_mhz * 1e6
                          + (_jb0 - 2048) * (a.rate / 4096.0)) / 1e6),
                      file=sys.stderr, flush=True)
                del _jk_junk[_jb0]
            # learn scan (cheap, every 8 windows): a bin at >=duty for the
            # full horizon with no detection anywhere near it is a spur
            if ((wc & 7) == 0 and len(_jk_junk) < PRESCREEN_CAP
                    and _jk_seen >= 0.8 * PRESCREEN_LEARN_WIN):
                for _db0 in [k for k, v in _jk_det_bins.items()
                             if wc - v > 600]:
                    del _jk_det_bins[_db0]
                _jcand = np.where(_jk_hits >= PRESCREEN_DUTY * _jk_seen)[0]
                _fres_jk = a.rate / 4096.0
                _cen_jk = _live_center_mhz * 1e6
                for _cb in _jcand[np.argsort(-_jk_hits[_jcand])]:
                    if len(_jk_junk) >= PRESCREEN_CAP:
                        break
                    _cb = int(_cb)
                    if any(abs(_cb - _jb0) <= 4 for _jb0 in _jk_junk):
                        continue            # already-learned cluster
                    if any(abs(_cb - _db0) <= 205 and wc - _dwc0 <= 600
                           for _db0, _dwc0 in _jk_det_bins.items()):
                        continue            # real traffic within ±1 MHz
                    if any(abs((_cc0 - _cen_jk) / _fres_jk + 2048 - _cb)
                           < max(3.0, _bw0 / _fres_jk) + 4
                           for _cc0, _bw0, _sf0 in _dechirp_src):
                        continue            # inside a channelizer channel
                    if any(abs(_cb - _pb0) <= 12 for _pb0 in _pat_promoted):
                        continue            # patience-promoted carrier
                    _fe = float(_jk_elev[max(0, _cb - 2):
                                         min(4096, _cb + 3)].max())
                    _fw = float(_jk_wid[max(0, _cb - 2):
                                        min(4096, _cb + 3)].max())
                    _jk_junk[_cb] = [_fe, _fw, wc]
                    print('[PRESCREEN] learned junk %.4fMHz (elev %.1f dB, '
                          'w %.0f, duty %d%%, %d/%d slots) — suppressing; '
                          'probe every %d win'
                          % ((_cen_jk + (_cb - 2048) * _fres_jk) / 1e6, _fe,
                             _fw,
                             int(100 * _jk_hits[_cb] / max(_jk_seen, 1e-9)),
                             len(_jk_junk), PRESCREEN_CAP,
                             PRESCREEN_PROBE_EVERY),
                          file=sys.stderr, flush=True)
            # apply: suppress peaks on learned junk bins — except on probe
            # windows (aligned so most junk windows go fully quiet) and for
            # breakthrough peaks (stronger/wider than the learned spur)
            if _jk_junk:
                _jk_probe = (wc % PRESCREEN_PROBE_EVERY) == 0
                _jk_kept = []
                _jk_sup_bins = []
                for _pk in _gate_peaks:
                    _jb0 = int(_pk[0])
                    _jrec = None
                    for _jb1, _jr1 in _jk_junk.items():
                        if abs(_jb0 - _jb1) <= 3:
                            _jrec = _jr1
                            break
                    if _jrec is None:
                        _jk_kept.append(_pk)
                        continue
                    _jrec[2] = wc            # junk bin still firing
                    if _jk_probe:
                        _jk_kept.append(_pk)     # scheduled full-path probe
                        continue
                    if (float(_pk[2]) - _jk_floor
                            > _jrec[0] + PRESCREEN_MARGIN_DB):
                        _jk_kept.append(_pk)     # breakthrough: stronger
                        continue                 # than the learned spur
                    if float(_pk[1]) > 2.0 * max(_jrec[1], 1.0) + 2.0:
                        _jk_kept.append(_pk)     # much wider than the spur
                        continue
                    # (bin, half-width) — VERIFY's deafness match is
                    # width-adaptive so wide-BW centroid error can't hide
                    # a real suppression casualty
                    _jk_sup_bins.append(
                        (_jb0, int(round(float(_pk[1]) / 2.0)) + 1))
                if _jk_sup_bins:
                    _jk_sup_this_win = True
                    _jk_sup_peaks += len(_jk_sup_bins)
                    if PRESCREEN_VERIFY:
                        # record only — nothing suppressed in VERIFY mode
                        _jk_verify_recent.append((wc, tuple(_jk_sup_bins)))
                        while (_jk_verify_recent
                               and wc - _jk_verify_recent[0][0] > 40):
                            _jk_verify_recent.popleft()
                    else:
                        if _caw is not None:
                            _kept_ids = {id(_x) for _x in _jk_kept}
                            _caw.drop([_p for _p in _gate_peaks
                                       if id(_p) not in _kept_ids], 'junk_suppressed')
                        _gate_peaks = _jk_kept
                        if not _gate_peaks:
                            _jk_sup_windows += 1

        # ---- Keep-up-driven peak throttle ----
        # The energy gate's job is bounding COMPUTE, not judging what's real —
        # SC + dechirp reject noise (measured: zero false positives even at
        # energy_threshold 5).  So instead of a conservative fixed threshold
        # that silently drops weak-but-real signals, run sensitive and clamp
        # the number of candidate peaks per window ONLY when the host is
        # measurably falling behind.  The pressure signal is ring occupancy —
        # the direct precursor of the CATCHUP sample-drop path (which fires at
        # 80% fill): throttle at >50% fill, recover when calm (<20% for 8
        # consecutive windows).  A healthy host never throttles and keeps full
        # sensitivity; a weak host in a busy band sheds the WEAKEST candidates
        # first instead of dropping raw samples.  Truncation is always logged —
        # a silent cap reads as "band was quiet" when it wasn't.  Applied
        # BEFORE the channelizer's forced peaks so learned channels are never
        # shed.  _PEAK_CAP_HI also bounds the merged multi-pass peak list on
        # any host (the per-pass find_peaks cap of MAX_ENERGY_PEAKS never
        # bounded the merged 1s + max-hold + short-sweep total).
        # ---- Proactive per-window compute budget ([GATE-BUDGET]) ----
        # The ring>50% backstop below is REACTIVE: it only trips after tens of
        # seconds of backlog have already accumulated, halves the cap, then
        # relaxes (+1 per 8 calm windows) while the deficit rebuilds — measured
        # on-Pi (4 Msps live) this oscillated through wholesale CATCHUP ring
        # skips of ~35-41 s each that discarded real beacons unseen.  Clamp
        # PROACTIVELY instead: from the EWMAs above, size the peak list so a
        # window's EXPECTED detect cost fits inside the hop budget with margin
        # — the deficit never accumulates in the first place.  Host-agnostic
        # by construction: on a fast host the per-peak EWMA is tiny, the
        # computed cap lands >= _PEAK_CAP_HI, and behavior is unchanged.
        # Floor 2 (not _PEAK_CAP_LO): shed-weakest-first means the strongest
        # peaks — the real traffic — are always processed.  The ring backstop
        # is kept UNCHANGED underneath as a second line of defense; the
        # effective cap is the min of both.  FILE mode (is_live False) never
        # touches any of this.
        _cap_budget = _PEAK_CAP_HI
        _avail = reader.available() if is_live else 0
        if (is_live and _bgt['margin'] > 0.0 and _bgt['pp'] > 0.0
                and _avail >= hop_n):
            # Engage only on GENUINE backlog (>= one hop of unread samples):
            # a keeping-up host is structurally exempt — its reader never
            # runs ahead — so the clamp cannot degrade healthy hosts (the
            # verify pass measured a real decode lost to a false clamp when
            # this condition was absent).  A deficit host banks a hop within
            # ~0.5 s and engages ~40x sooner than the ring>50% backstop.
            _bud_s = _hop_budget_s * _bgt['margin'] - _bgt['ovh']
            _cap_new = min(_PEAK_CAP_HI,
                           max(_BUDGET_CAP_FLOOR, int(_bud_s / _bgt['pp'])))
            # Hysteresis: the int() truncation wobbles +-1 on noisy per-peak
            # EWMAs (measured 34-41 cap transitions per 240 s leg without it);
            # adopt only >=2 moves.  The ring backstop handles emergencies.
            if abs(_cap_new - _bgt['cap_last']) >= 2:
                print(f"[GATE-BUDGET] per-peak {_bgt['pp']*1000:.0f}ms "
                      f"ovh {_bgt['ovh']*1000:.0f}ms "
                      f"hop {_hop_budget_s*1000:.0f}ms — "
                      f"budget cap -> {_cap_new}",
                      file=sys.stderr, flush=True)
                _bgt['cap_last'] = _cap_new
            _cap_budget = _bgt['cap_last']
        if is_live:
            _ring_frac = _avail / float(_ring_n_total)
            if _ring_frac > 0.5:
                if _peak_cap > _PEAK_CAP_LO:
                    _peak_cap = max(_PEAK_CAP_LO, _peak_cap // 2)
                    print(f"[GATE-THROTTLE] ring {int(_ring_frac*100)}% full — "
                          f"peak cap -> {_peak_cap}", file=sys.stderr, flush=True)
                _cap_calm = 0
            elif _ring_frac < 0.2:
                _cap_calm += 1
                if _cap_calm >= 8 and _peak_cap < _PEAK_CAP_HI:
                    _peak_cap += 1
                    _cap_calm = 0
        _cap_eff = min(_peak_cap, _cap_budget)
        if len(_gate_peaks) > _cap_eff:
            _gate_peaks.sort(key=lambda p: -p[2])
            if _caw is not None:
                # binding source recorded, no arbitrary tie-break
                _bind = ('both' if _peak_cap == _cap_budget
                         else 'peak_cap' if _cap_eff == _peak_cap else 'budget')
                _caw.drop(_gate_peaks[_cap_eff:], 'shed_' + _bind)
            _n_shed = len(_gate_peaks) - _cap_eff
            _gate_peaks = _gate_peaks[:_cap_eff]
            print(f"[GATE-THROTTLE] shed {_n_shed} weakest candidate peak(s) "
                  f"(cap {_cap_eff})", file=sys.stderr, flush=True)

        # Channelizer dechirp matched-filter: FORCE a candidate at each channelizer
        # channel every window — even with no energy peak — so the sensitive dechirp
        # stage always runs on it.  This bypasses the +Ndb energy gate (which discards
        # weak/far signals that would despread fine), giving dedicated-narrowband-
        # receiver sensitivity.  No false-positive risk: dechirp + CRC reject empty
        # windows; cost bounded by the cap.  ADDITIVE — non-channel peaks untouched.
        if _dechirp_src and not _win_backlogged:
            _fres_gate = a.rate / 4096.0
            _cen_hz = _live_center_mhz * 1e6
            for _cc, _bw_hz, _csf in _dechirp_src:
                _cbin = 2048 + int(round((_cc - _cen_hz) / _fres_gate))
                if 0 <= _cbin < 4096 and not any(abs(_pk[0] - _cbin) < 15 for _pk in _gate_peaks):
                    _wbins = max(3, int(round(_bw_hz / _fres_gate)))
                    # carry the channel's REAL PSD value (not a placeholder) so the
                    # spur-reject / power ordering downstream treat it correctly
                    _gate_peaks.append((_cbin, _wbins, float(_psd_gate[_cbin]), float(np.median(_psd_gate))))   # forced; dechirp decides
                    if _caw is not None:
                        _caw.add(_gate_peaks[-1], 'forced', hyp_sf=_csf, hyp_bw=_bw_hz)
            # Fix #3 (per-channel compute): a strong signal's sidelobes each fall within
            # ±bw of a channel and would EACH fire the expensive forced dechirp + decode.
            # Keep only the peak NEAREST each channel centre (the carrier); drop the
            # in-channel sidelobes so the dechirp runs ONCE per channel.  Peaks OUTSIDE
            # the learned channels (wideband discovery) are untouched.
            for _cc, _bw_hz, _csf in _dechirp_src:
                _cbin = 2048 + int(round((_cc - _cen_hz) / _fres_gate))
                _w = max(1, int(round(_bw_hz / _fres_gate)))
                _in = [_pk for _pk in _gate_peaks if abs(_pk[0] - _cbin) < _w]
                if len(_in) > 1:
                    _keep = min(_in, key=lambda _pk: abs(_pk[0] - _cbin))
                    _drop = {id(_pk) for _pk in _in if _pk is not _keep}
                    if _caw is not None:
                        _caw.drop([_pk for _pk in _in if _pk is not _keep], 'dedup_sidelobe')
                    _gate_peaks = [_pk for _pk in _gate_peaks if id(_pk) not in _drop]

        # ---- Patience gate: promote + force ----
        # Promotion scan (cheap, every PATIENCE_SCAN_EVERY windows): pick the
        # highest-hit bins that cleared the bar, veto near-DC / CW-duty /
        # already-channelized carriers, and register a promotion with a TTL.
        # A promotion is ONLY a carrier — the forced peak below hands it to
        # the same matched-SC + dechirp pipeline the channelizer uses, which
        # identifies SF/BW from the signal itself and rejects non-LoRa within
        # seconds of windows (bounded CPU, additive, nothing masked).
        if PATIENCE_ON and wc % PATIENCE_SCAN_EVERY == 0:
            _fres_pat = a.rate / 4096.0
            _cen_pat = _live_center_mhz * 1e6
            for _pb in [b for b, (e, _h) in _pat_promoted.items() if wc >= e]:
                print('[PATIENCE] expired %.4fMHz — no confirmed acquisition '
                      'within TTL; re-earning' % (_pat_promoted[_pb][1] / 1e6),
                      file=sys.stderr, flush=True)
                _pat_hits[max(0, _pb - 4):_pb + 5] = 0.0
                del _pat_promoted[_pb]
                _pat_futile.pop(_pb, None)
                _pat_confirmed.discard(_pb)
            for _qb in [q for q, e in _pat_cooldown.items() if wc >= e]:
                del _pat_cooldown[_qb]      # futility cooldown served
            if len(_pat_promoted) < PATIENCE_CAP:
                _pcand = np.where(_pat_hits >= PATIENCE_MIN_HITS)[0]
                for _cb in _pcand[np.argsort(-_pat_hits[_pcand])]:
                    if len(_pat_promoted) >= PATIENCE_CAP:
                        break
                    _cb = int(_cb)
                    if abs(_cb - 2048) < 25:
                        continue                    # DC/LO skirt (±122 kHz —
                                                    # ±6 missed skirt bins at
                                                    # +11/-24 on two60)
                    if any(abs(_cb - _pb0) < 12 for _pb0 in _pat_promoted):
                        continue                    # already promoted (cluster)
                    if any(abs(_cb - _qb0) <= 12 for _qb0 in _pat_cooldown):
                        continue                    # futility cooldown: this
                                                    # bin burned a full futile
                                                    # cycle with zero confirms
                    if any(abs(_cb - _db0) <= 205 and wc - _dwc0 <= 600
                           for _db0, _dwc0 in _pat_det_bins.items()):
                        continue                    # being detected normally
                    if float(_pat_seen[_cb]) < 45.0:
                        continue    # < ~22 s of CLEAN observation: a bin whose
                                    # clock keeps resetting (spur field) never
                                    # earns promotion on a fresh hit burst
                    _duty = float(_pat_hits[_cb]) / max(float(_pat_seen[_cb]), 1.0)
                    if _duty > PATIENCE_DUTY_MAX:
                        continue                    # CW/continuous — not a beacon
                    if any(abs((_cc0 - _cen_pat) / _fres_pat + 2048 - _cb)
                           < max(3.0, _bw0 / _fres_pat)
                           for _cc0, _bw0, _sf0 in _dechirp_src):
                        continue                    # a channel already covers it
                    _lo = max(0, _cb - 4); _hi = min(4096, _cb + 5)
                    _wv = _pat_hits[_lo:_hi].astype(np.float64)
                    _cbin_f = float((_wv * np.arange(_lo, _hi)).sum()
                                    / max(_wv.sum(), 1e-9))
                    _phz = _cen_pat + (_cbin_f - 2048.0) * _fres_pat
                    _pat_promoted[_cb] = (wc + PATIENCE_TTL_WIN, _phz)
                    _pat_futile.pop(_cb, None)      # fresh futility clock
                    _pat_confirmed.discard(_cb)
                    print('[PATIENCE] promoted %.4fMHz (hits %.1f, duty %d%%, '
                          '%d/%d slots) — forcing dechirp until confirmed or TTL'
                          % (_phz / 1e6, float(_pat_hits[_cb]),
                             int(_duty * 100), len(_pat_promoted), PATIENCE_CAP),
                          file=sys.stderr, flush=True)
        # Forced peak per active promotion, every window (mirrors the
        # channelizer force above; 125 kHz width placeholder — the per-peak
        # pipeline measures the real width/preset itself).
        if _pat_promoted and not _win_backlogged:
            _fres_pat = a.rate / 4096.0
            _cen_pat = _live_center_mhz * 1e6
            for _pb, (_pexp, _phz) in _pat_promoted.items():
                _cbin = 2048 + int(round((_phz - _cen_pat) / _fres_pat))
                if (0 <= _cbin < 4096
                        and not any(abs(_pk[0] - _cbin) < 15 for _pk in _gate_peaks)):
                    _gate_peaks.append((_cbin,
                                        max(3, int(round(125e3 / _fres_pat))),
                                        float(_psd_gate[_cbin]),
                                        float(np.median(_psd_gate))))
                    if _caw is not None:
                        # patience hypothesis (sf,bw) is set downstream by the
                        # rotating trial list, not here → None at candidate stage
                        _caw.add(_gate_peaks[-1], 'patience')

        # Per-window dechirp channel list: the channelizer's fed set PLUS a
        # rotating (sf, bw) trial pair per patience promotion.  This is what
        # routes a promoted carrier into the CHAN-DECHIRP despread path (the
        # sensitive one) — without it the promotion's forced peak runs only
        # the plain SC chain and the promotion is pointless below ~+2 dB.
        # Rebuilt every window; handed to BOTH detect paths (pool dispatch
        # carries it per-task so live channel changes reach the workers too).
        _win_chans = [(_c - _live_center_mhz * 1e6, _s, _b)
                      for (_c, _b, _s) in _dechirp_src]
        if _pat_promoted and not _win_backlogged:
            _cen_pat = _live_center_mhz * 1e6
            for _pi, (_pb, (_pexp, _phz)) in enumerate(_pat_promoted.items()):
                for _k in range(PATIENCE_TRIALS_PER_WIN):
                    _tsf, _tbw = PATIENCE_TRIALS[
                        (wc * PATIENCE_TRIALS_PER_WIN + _pi + _k)
                        % len(PATIENCE_TRIALS)]
                    _win_chans.append((_phz - _cen_pat, _tsf, _tbw, True))

        # Try to release ONE deferred straggler for a big-budget re-decode.  Drive
        # this off the decode system's ACTUAL idle state — maybe_release_straggler
        # self-gates on (slow queue non-empty AND fast queue empty AND nothing
        # decoding) — NOT off the gate's per-window peak count.  Peak count was a
        # broken proxy for "idle": at high gain find_peaks returns spur peaks every
        # window, so the old `if not has_energy` guard was never true and the slow
        # queue never drained (observed: 4 SF11 stragglers stuck indefinitely while
        # the gate sat with no real traffic).  Realtime stays protected by the
        # _gate_stress throttle, which pauses the heavy re-decode if the gate starts
        # dropping samples; this call itself is just a few cheap checks.
        if decoder is not None:
            decoder.maybe_release_straggler()
        if decoder and is_live and (wc & 3) == 0:
            # every 4th window: available() takes the ring lock — per-
            # window polling added measurable reader contention
            _press = reader.available() / max(1, int(a.rate * a.buf_seconds))
            decoder.set_pressure(_press)
            decoder.maybe_release_pressure()
        _prof['notch'] += time.time() - _t_step
        _proft['notch'] += time.thread_time() - _tt_step
        _t_step = time.time()
        _tt_step = time.thread_time()

        # NB: the tail read is deferred to AFTER detect_preamble so we never
        # silently consume IQ samples we won't use.  The old code read up to
        # 3 × win_n (= 3 s) of post-window samples whenever the gate fired,
        # then discarded them whole if detect_preamble returned no detections
        # (or if `tail_skipped` was non-zero).  Any LoRa packet that landed
        # entirely inside that discarded tail vanished without a log entry,
        # which is why the user saw "no DETECTED / no Saved within ±5 s of
        # the expected packet" cases.
        pre_tail = None

        # ---- Saturation check ----
        # When the TX is nearby and gain is high, the ADC clips.  Clipping
        # creates harmonic/IM distortion that looks like extra signals to the
        # Welch PSD and triggers duplicate detections.  Detect saturation by
        # looking at what fraction of samples are near full-scale.  If > 0.5 %
        # of samples are clipped, tighten spur rejection so artefacts far
        # below the dominant peak are suppressed before the SC stage.
        # The clip threshold is format-dependent: SC16 is normalised by 2048
        # (full-scale ≈ 16.0), SC8 by 128 (full-scale ≈ 1.0).
        _norm_scale = 2048.0 if a.format == 'sc16' else 128.0
        # 90 % of ADC full-scale per format.  sc16 arrives shifted to the full
        # int16 range (SoapyHackRF <<8), so full-scale is 32767; sc8 is raw
        # int8, so full-scale is 127.  Using 32767 for sc8 put the threshold
        # ~230× above reachable amplitude and [SAT] could never fire.
        _fullscale = 32767.0 if a.format == 'sc16' else 127.0
        _clip_thresh = 0.9 * (_fullscale / _norm_scale)
        # Saturation check: just need to know if a meaningful fraction of
        # samples is clipping.  Computing np.percentile on 28 M complex samples
        # was ~250 ms per iteration — over a quarter of the hop budget.  We
        # subsample (rate-derived _sat_stride, ~256k samples — see init) and
        # test in the SQUARED-magnitude domain (re²+im²) so we skip the per-
        # sample sqrt that np.abs() does; the clip threshold is squared to
        # match.  Peak amplitude (debug/AGC display only) is the sqrt of the
        # max, one scalar op.  lazy: strided() converts just the subsample.
        _sub = _lz.strided(_sat_stride) if _lz is not None else buf[::_sat_stride]
        _sr = _sub.real
        _si = _sub.imag
        _mag2_sub = _sr * _sr + _si * _si
        _clip2 = _clip_thresh * _clip_thresh
        _peak_amp = float(np.sqrt(_mag2_sub.max())) if len(_mag2_sub) else 0.0
        _sat_frac = float(np.count_nonzero(_mag2_sub > _clip2)) / max(1, len(_mag2_sub))
        # Boundary refinement (codex review): the coarse stride trades precision
        # for speed and a deterministic stride can ALIAS with periodic clipping.
        # Whenever the coarse estimate shows ANY hint of clipping (>0.1%, well
        # below the 0.5% engage threshold and far above the ~0.01% noise floor),
        # re-measure with the dense stride-32 grid so the engage DECISION uses
        # the accurate fraction.  No clipping (the overwhelmingly common case) →
        # coarse ~0 → skip → full speed; only genuinely-clipping windows pay the
        # dense recompute.
        if _sat_stride > 32 and _sat_frac > 0.001:
            _sub2 = _lz.strided(32) if _lz is not None else buf[::32]
            _m2r = _sub2.real
            _m2i = _sub2.imag
            _mag2_d = _m2r * _m2r + _m2i * _m2i
            _sat_frac = float(np.count_nonzero(_mag2_d > _clip2)) / max(1, len(_mag2_d))
            if len(_mag2_d):
                _peak_amp = float(np.sqrt(_mag2_d.max()))
        if _sat_frac > _sat_max:
            _sat_max = _sat_frac
        _spur_db  = a.spur_reject
        if _sat_frac > 0.005:   # > 0.5 % of samples clipping
            # Saturation creates harmonic distortion that looks like extra
            # peaks, so engage real spur rejection — but FLAT at 35 dB.
            # The old clip-scaled ramp (15 dB base + up to 20 dB with clip
            # rate) made LIGHT clipping the most aggressive rejector: a
            # local node kissing the rail (0.5-2 % clip) silently ate real
            # co-window signals 15-35 dB down that are trivially decodable,
            # while heavy clipping (>7 %) kept them.  Measured live (gain-60
            # capture, 39 clipped windows incl. natural 0.6-1.6 % light
            # clip): 35 dB admits ZERO false detections from real front-end
            # IMD/harmonics — SC + dechirp reject them — and recovers the
            # weak co-window signals the ramp dropped.  When NOT saturating,
            # spur rejection stays at the (very high) default so legitimate
            # weak LoRa peaks aren't dropped just because a stronger one is
            # present in the same window.
            _spur_db = 35.0
            if a.debug >= 1:
                print(f"  [SAT] peak={_peak_amp:.3f}  clipped={_sat_frac*100:.2f}%  "
                      f"spur_reject={a.spur_reject:.0f}→{_spur_db:.0f}dB",
                      file=sys.stderr, flush=True)

        if _agc is not None:
            # busy = the gate sees candidate peaks this window — a climb could
            # glitch an in-flight packet, so only climb on a truly idle band.
            # Backing off under sustained clip is never blocked by busy.
            # busy: OR in prescreen-suppressed junk — suppression must not
            # change what the AGC sees (a spur-only band previously blocked
            # gain climbs; keep that behavior identical with prescreen on).
            _agc.observe(_sat_frac > 0.005,
                         busy=bool(_gate_peaks) or _jk_sup_this_win,
                         clip_frac=_sat_frac)

        _prof['sat'] += time.time() - _t_step
        _proft['sat'] += time.thread_time() - _tt_step
        _t_step = time.time()
        _tt_step = time.thread_time()
        # ---- Detect ----
        # SC buffer = buf only (no pre_hop concat).  Using buf (1s / 28M samples)
        # with a shared FFT cache (chunk=65536) costs ~228ms one-time + ~8ms/peak,
        # well under the ~1273ms hop budget even with 5 simultaneous peaks.
        # pre_hop is still passed to recorder.update for full-preamble capture.
        # SF12/BW31.5kHz preambles (1040ms) that straddle the window boundary may
        # occasionally be missed; all other SF/BW combinations fit entirely within
        # the 1s window (longest: SF12/BW125k = 262ms preamble).
        wc += 1; tw = time.time()
        # [GATE-BUDGET] the list is FINAL here (cap, channelizer forced peaks
        # and patience placeholders all applied above) — this is the count the
        # detect phase actually pays for.
        _n_pk_det = len(_gate_peaks)
        # --- Hybrid per-window routing (UC audit B) ---
        # Quiet window on a small host: skip the pool round-trip (slot
        # memcpy + worker wakeup + lagged commit) and detect inline — with
        # few peaks that costs ~the fixed sweep only.  Inline is ONLY taken
        # once the in-flight queue is drained: pooled commits rebuild their
        # packet tails from the TRAILING in-flight windows, so an inline
        # window in the middle of that chain would hole the tail.  Steady
        # state keeps depth == _LAG, so quiet windows drain ready commits
        # (no-detection commits are cheap) until empty, then route inline;
        # any window with real peak pressure or a filling ring goes back
        # through the pool.
        _route_inline = False
        if _pool is not None and _hybrid:
            if (len(_gate_peaks) <= _HYB_PEAKS
                    and (not is_live
                         or reader.available() / float(_ring_n_total) < 0.3)):
                _ncd = 0
                while (_inflight and _ncd < 6
                       and _pool.ready(_inflight[0]['seq'])):
                    _pd = _pool.peek(_inflight[0]['seq'])
                    if _pd and len(_inflight) - 1 < _tail_windows_needed(_pd):
                        break   # tail still needs trailing windows — stay pooled
                    _commit_oldest()
                    _ncd += 1
                _route_inline = not _inflight
            if _route_inline:
                _hyb_inline += 1
            else:
                _hyb_pooled += 1
        if _pool is not None and not _route_inline:
            # --- Multiprocess detect: dispatch this window, commit lagged ---
            # The producer never blocks on detection; workers run it in
            # parallel.  The capture's forward tail is reconstructed in
            # _commit_oldest from the next in-flight window's overlap, so we
            # do NOT read a tail from the ring here.
            while _pool.n_free() == 0:
                _commit_oldest()
            _slot = _pool.acquire_slot()
            if _lz is not None and buf is None:
                # MATERIALIZATION TRIGGER (pool): every dispatched window
                # needs the full c64 window in its slot — commits rebuild
                # pending packet tails from TRAILING in-flight windows, so
                # even a no-peak window's slot data can be read later.
                buf = _lz.materialize()
            _pool.slot_array(_slot)[:len(buf)] = buf
            _pool.dispatch(_slot, _seq, len(buf), _psd_gate, _gate_peaks,
                           spur_db=_spur_db, center=_live_center_mhz,
                           dechirp_chans=(_win_chans or None))
            if _caw is not None:
                _caw.set_admitted(_gate_peaks, time.monotonic_ns(), pool_seq=_seq)
            _inflight.append({'seq': _seq, 'slot': _slot,
                              'pre_hop': pre_hop, 'tot_s': tot_s,
                              'cand_audit': _caw, 'buf_len': len(buf)})
            _seq += 1
            # Tail-aware commit: once depth exceeds the base lag, commit the
            # oldest ONLY when its detection is ready AND enough trailing windows
            # exist to cover its packet's tail.  Low-SF (tail≈1 window) commits
            # at the base lag → minimum latency; high-SF (tail spans many
            # windows) defers until its tail is in flight → complete capture.
            # Works for EVERY SF/BW with no per-preset tuning.  Capped at 2
            # commits/iter so a deferral release can't stall the realtime read
            # (the multi-commit stall that dropped samples in earlier attempts).
            # The slot-full forced commit above is the backpressure valve.
            _nc = 0
            while len(_inflight) > _LAG and _nc < 2:
                _old = _inflight[0]
                if not _pool.ready(_old['seq']):
                    break   # detection not done yet — don't block the reader
                if len(_inflight) - 1 < _tail_windows_needed(_pool.peek(_old['seq'])):
                    break   # tail not fully in flight yet — defer (high SF)
                _commit_oldest()
                _nc += 1
            elapsed = time.time() - t_start
            dt = time.time() - tw
            _prof['detect'] += dt
            if is_live and _bgt['margin'] > 0.0:
                _budget_track(dt, _n_pk_det)
            _t_step = time.time()
        else:
            if _lz is not None and not _gate_peaks:
                # LAZY quiet window: with an empty cached-peaks list
                # detect_preamble returns [] BEFORE it ever touches iq
                # (`if not peaks: return []` precedes the FFT cache and the
                # dechirp_chans use) — skip the call so the raw window never
                # materializes.  The peak list here is FINAL: real 1s/max-
                # hold peaks, short-sweep merges, channelizer forced peaks
                # and patience placeholders were all appended above, so any
                # of them firing lands in _gate_peaks and takes the else.
                dets = []
            else:
                if _lz is not None and buf is None:
                    # MATERIALIZATION TRIGGER (inline detect): peaks exist
                    buf = _lz.materialize()
                dets = detect_preamble(buf, a.rate, a.bandwidth,
                                       _live_center_mhz,
                                       sc_threshold=a.threshold,
                                       ethresh=a.energy_threshold,
                                       spur_db=_spur_db,
                                       dc_notch_mhz=a.dc_notch,
                                       spur_notch_hz=_spur_notch_hz or None,
                                       debug=a.debug,
                                       cached_psd=_psd_gate,
                                       cached_peaks=_gate_peaks,
                                       dechirp_chans=(_win_chans or None))
            dt = time.time() - tw; elapsed = time.time() - t_start
            _prof['detect'] += dt
            if is_live and _bgt['margin'] > 0.0:
                _budget_track(dt, _n_pk_det)
            _t_step = time.time()
            # Candidate audit (INLINE/serial path): synchronous — emit now with
            # this window's dets (pool_seq stays None).  Covers the lazy quiet
            # window too (empty _gate_peaks → dets=[] → n_raw=0 record).
            if _caw is not None:
                _caw.set_admitted(_gate_peaks, time.monotonic_ns())
                _caw.emit(dets, len(buf) if buf is not None else win_n,
                          'inline', time.monotonic_ns())

            # Cross-window duplicate suppression is intentionally removed.
            # Any frequency-bucket dedup at this stage cannot distinguish the
            # original transmission from a relay on the same channel (co-located
            # nodes differ by only 1-3 kHz, well under any bucket that reliably
            # catches same-signal repeats from overlapping windows).  The
            # post-decode PacketID dedup in BackgroundDecoder._compact() handles
            # same-packet duplicates correctly while allowing different hops_taken
            # values to pass through as distinct packets.

            for d in dets:
                if d.get('patience_trial'):
                    continue   # acquisition-internal: publish only on decode confirm
                _pat_note_detection(d['freq_hz'])
                _jk_note_detection(d['freq_hz'])
                tot_d += 1
                bwq = f" bwq={d['bw_quality_db']:.0f}dB" if a.debug >= 1 else ""
                print(f"[{elapsed:6.1f}s] DETECTED freq={d['freq_mhz']:.4f}MHz "
                      f"SF={d['sf']} BW={fmt_bw(d['bw'])} "
                      f"sc={d['detect_conf']:.2f} "
                      f"pwr={d['peak_power_db']:.1f}dB"
                      f"{bwq}", flush=True)

            # Read the post-window tail ONLY when there is something to save.
            # The tail provides additional IQ for the recorder to extract the
            # full LoRa frame when the preamble landed near the end of `buf`.
            if recorder and is_live and dets:
                # DEFERRED TAIL (2026-07-05): the airtime-sized tail is no
                # longer read BLOCKING here (2.4 s stall per SF11/125k
                # detection — the dominant live-loop stall; see CATCHUP
                # analysis in FOLLOWUP). The recorder holds the job and
                # feed_tail() below completes it from the hops the loop
                # processes anyway — identical samples, zero stall, and
                # capture length no longer depends on pipeline speed.
                _max_pkt_s = max(
                    148.25 * patience_cap_params(d)[1] for d in dets)
                # Cap = min(airtime, 24*win_n) (UC audit p).  Was
                # min(airtime, 3*win_n, ring/2) — a stale blocking-read-era
                # bound that truncated every frame past ~4.5 s (SF12/62.5k =
                # 9.7 s, SF12/125k = 4.9 s) to a decoded header + cut payload.
                # The deferred tail accumulates from LOOP HOPS (refs), not ring
                # reads, so the ring/2 bound never applied; the slow-pass path
                # already raised to 24*win_n for exactly this reason.  NB-mode
                # crops each hop to KB as it arrives, so the longer pending job
                # is tiny (and still bounded by the 3-job cap).
                _max_tail_n = min(int(_max_pkt_s * a.rate), 24 * win_n)
                if os.environ.get('LORA_NO_DEFER'):
                    # owned: the tail is retained as _carry_tail across the
                    # next read (and can become `buf` via iq[-win_n:])
                    if _lz is not None:
                        # lazy: read the tail RAW (owned — retained as the
                        # raw carry) and convert once for the recorder
                        tail_data, tail_skipped = reader.read_raw(
                            _max_tail_n, owned=True)
                        if tail_data is not None:
                            pre_tail = _lz.convert_owned(tail_data)
                            tot_s += len(pre_tail)
                            _carry_tail = tail_data
                    else:
                        tail_data, tail_skipped = reader.read(_max_tail_n,
                                                              owned=True)
                        if tail_data is not None:
                            pre_tail = tail_data
                            tot_s += len(pre_tail)
                            _carry_tail = pre_tail
                    if tail_skipped:
                        tot_skip += tail_skipped
                else:
                    recorder.update_deferred(dets, buf, tot_s,
                                             pre_hop=pre_hop,
                                             need_tail_n=_max_tail_n)
                    dets = []   # consumed: skip the immediate update below
            elif recorder and not is_live and dets:
                # File mode: size the tail to the actual frame duration.  Kept
                # at 3*win_n — this is the OFFLINE blocking-read path whose
                # _carry_tail continuity mechanism can't absorb a multi-second
                # tail (verified: a 24*win_n read fragments the capture).  The
                # deferred LIVE/POOL paths (the (p) targets, production) use
                # feed_tail refs, not this read, and DO get the 24*win_n cap.
                _max_pkt_s = max(
                    148.25 * patience_cap_params(d)[1] for d in dets)
                _max_tail_n = min(int(_max_pkt_s * a.rate), 3 * win_n)
                if _max_tail_n > 0:
                    # owned: retained as _carry_tail across the next read
                    # (and can become `buf` via iq[-win_n:])
                    if _lz is not None:
                        # lazy: RAW tail (owned by construction — a fresh
                        # frombuffer view) + one conversion for the recorder
                        tail_data = reader.read_raw(_max_tail_n)
                        if tail_data is not None:
                            pre_tail = _lz.convert_owned(tail_data)
                            tot_s += len(pre_tail)
                            _carry_tail = tail_data
                    else:
                        tail_data = reader.read(_max_tail_n, owned=True)
                        if tail_data is not None:
                            pre_tail = tail_data
                            tot_s += len(pre_tail)
                            _carry_tail = pre_tail

            _prof['tail'] += time.time() - _t_step
            _t_step = time.time()
            if recorder:
                recorder.update(dets, buf, tot_s, tail=pre_tail, pre_hop=pre_hop)
            _prof['recorder'] += time.time() - _t_step
        _t_step = time.time()

        # ---- LONG-WINDOW SLOW-PRESET PASS (31.25 kHz + 41.67 kHz) ----
        # Their preambles (0.5-1.6 s at SF11/12) are longer than one
        # analysis window and can NEVER fully sit inside `buf` — the
        # mirror of the short-window pass's fast-burst problem. A slow
        # preamble's signature in the 1 s domain: a PERSISTENT NARROW
        # peak across >=2 consecutive windows that the standard pass did
        # not detect. Only that trigger pays for the 3 s scan (assembled
        # from the last 6 hops), and only the triggering peaks are
        # processed (cached_peaks seeds — no long-buffer PSD). Idle cost
        # ~zero; ambient junk flickers and never accumulates the streak.
        _hop_own = (_lz.seg_owned(max(0, win_n - hop_n),
                                  min(hop_n, win_n))
                    if _lz is not None else
                    buf[-hop_n:].copy())
                                         # ONE owned copy per hop, shared
                                         # by the slow assembly AND every
                                         # pending tail (feed_tail holds
                                         # REFS — a view of the sliding
                                         # buf would mutate under them,
                                         # which zeroed straddler decodes
                                         # on the first zero-copy attempt).
                                         # Lazy: seg_owned is a private
                                         # allocation with identical bits
                                         # (conversion is exact), whether
                                         # converted fresh (quiet) or
                                         # copied from the materialization.
        _slow_halves.append(_hop_own)
        # Record for pre_hop reuse: this hop covers abs [tot_s-hop_n, tot_s).
        # It is an immutable owned array (same one _slow_halves/feed_tail
        # hold), so a later window's pre_hop can view it instead of copying.
        _prehop_hist.append((tot_s - hop_n, _hop_own))
        if recorder:
            _t_feed0 = time.time()
            recorder._dbg_tot_s = tot_s
            recorder.feed_tail(_hop_own)
            _prof['feed'] += time.time() - _t_feed0
        _t_slowblk0 = time.time()
        _slow_tick += 1
        _det_bins = set()
        for d in dets:
            _det_bins.add(int(round((d['freq_hz'] - _live_center_mhz * 1e6)
                                    / (a.rate / 4096.0))) + 2048)
        _floor_gate_db = float(np.median(_psd_gate))
        _new_seeds = {}
        for _p in _gate_peaks:
            # PEAK-RELATIVE narrowness: the gate's contour width is floor-
            # relative and useless for strong signals (a 55 dB same-room
            # SF12/31.25k beacon spans 1.1 MHz at floor+6 via its phase-
            # noise skirt — it was dismissed as 'wide' while its -41 dB IQ
            # image seeded instead). A narrow signal is narrow at
            # PEAK-12 dB at any strength: 31.25/41.67k span 7-9 bins
            # there, 125 kHz spans ~26. Bound: 16 bins (78 kHz).
            _b0 = int(_p[0])
            if not (2 <= _b0 <= 4093):
                continue
            _pkdb = float(_psd_gate[_b0-2:_b0+3].max())
            _wthr = max(_pkdb - 12.0, _floor_gate_db + 6.0)
            _lo_w = _b0
            while _lo_w > 0 and _psd_gate[_lo_w-1] > _wthr:
                _lo_w -= 1
            _hi_w = _b0
            while _hi_w < 4095 and _psd_gate[_hi_w+1] > _wthr:
                _hi_w += 1
            if _hi_w - _lo_w + 1 > 16:
                continue                      # not narrow at peak-12
            if _p[2] - _floor_gate_db < 10.0:
                continue                      # ambient-junk strength — slow
                                              # preambles at usable SNR are
                                              # >=10 dB elevated (ambient
                                              # flicker hugs the threshold
                                              # and fired the scan every
                                              # tick before this gate)
            _b = int(_p[0])
            if any(abs(_b - _db) < 8 for _db in _det_bins):
                continue                      # already detected/handled
            # KEY BY 8-BIN BUCKET, not raw bin: power centroids jitter
            # +/-1-2 bins window-to-window, so raw-bin keys minted fresh
            # entries every window — streaks rebuilt in 3 windows and the
            # cooldown/backoff NEVER engaged (measured: 304 scans in
            # 10 min, before AND after the backoff patch — the giveaway).
            _bk = _b // 8
            _slow_exact[_bk] = _b             # freshest exact bin: SEEDS use
                                              # this, never the bucket center
                                              # (center is up to 4 bins =
                                              # 19.5 kHz off — outside a
                                              # 31.25k matched crop entirely;
                                              # cost g31 its captures)
            _prev = _slow_seeds.get(_bk, 0)
            if _prev < 0:
                _new_seeds[_bk] = _prev + 1   # cooling down
                continue
            _new_seeds[_bk] = _prev + 1
        # COOLDOWNS SURVIVE ABSENCE: rebuilt-from-current-peaks state let
        # flickery junk (hovering at the 10 dB seed bar) drop out for one
        # window, VANISH its negative cooldown entry, and re-trigger 3
        # windows later — 282-304 scans/10 min through TWO prior fixes
        # (keys were never the leak; state persistence was).
        # POSITIVE streaks DECAY by 1 on absence instead of evaporating
        # (2026-07-07): a GATE-THROTTLE shed can drop the real carrier's
        # peak for ONE window mid-preamble; full evaporation then reset
        # the streak to 0 with only ~3-4 seeding ticks in an SF12/41.67k
        # preamble — the frame's trigger never fired (set10 counter 14,
        # sheds observed in-window).  Decay keeps a one-window dropout
        # recoverable (3 -> 2 -> present -> 3 -> trigger) while junk that
        # flickers every other window oscillates 1-2 and never reaches
        # the >=3 trigger bar — same FP cadence as full evaporation.
        # Two absent windows still zero the streak (the streak-break
        # reset below then clears punishment history one window later
        # than before — same between-frames behaviour).
        for _bk_c, _c_c in _slow_seeds.items():
            if _bk_c in _new_seeds:
                continue
            if _c_c < 0:
                _new_seeds[_bk_c] = _c_c + 1
            elif _c_c > 1:
                _new_seeds[_bk_c] = _c_c - 1
        # STREAK BREAK resets punishment history: a real bucket that got
        # one badly-timed fruitless scan starts fresh at its next burst
        # (junk never breaks streak, so its history survives)
        for _bk_c in list(_slow_backoff):
            if _bk_c not in _new_seeds:
                _slow_backoff.pop(_bk_c, None)
                _slow_fruitless.pop(_bk_c, None)
        _slow_seeds = {_b: _c for _b, _c in _new_seeds.items()}
        # >=3 consecutive windows: even the shortest straddler preamble
        # (SF11/31.25k, 1.2 s) plus its packet body spans 3+ windows
        _trig = [_slow_exact.get(_bk, _bk * 8 + 4)
                 for _bk, _c in _slow_seeds.items() if _c >= 3]
        # DEFER-NOT-DROP (2026-07-07): a trigger that lands while another
        # scan is in flight used to be silently discarded; with only ~3-4
        # seeding ticks per SF12/41.67k preamble (and streaks decaying on
        # absence) the busy shadow of a NEIGHBOR bucket's scan — typically
        # the frame's own IQ image, which triggers off the same beacon —
        # could consume every trigger opportunity for the REAL carrier
        # (set10 counter 17: image detected, real carrier never scanned).
        # Stash busy-blocked triggers and fire them on the next free tick:
        # the 4 s assembly still contains the full 1.8 s preamble after a
        # few 0.5 s hops, so a deferred scan sees the same signal.  Expiry
        # 4 ticks keeps a stale trigger from scanning an assembly whose
        # preamble has already scrolled out.
        _slow_pending = {k: v for k, v in _slow_pending.items()
                         if v[1] >= _slow_tick}
        _trig_bks = {int(_b) // 8 for _b in _trig}
        _fire = list(_trig) + [v[0] for k, v in _slow_pending.items()
                               if k not in _trig_bks]
        if os.environ.get('LORA_SLOW_DEBUG'):
            print(f"[SLOW] tick={_slow_tick} halves={len(_slow_halves)} "
                  f"seeds={_slow_seeds} trig={_trig} "
                  f"pending={_slow_pending}", file=sys.stderr)
        # Low-core rate-bound (UC audit D): defer the launch when the last
        # scan was < _SLOW_MIN_TICKS ago or the ring says the gate is behind.
        # Deferrals keep _slow_pending intact so the trigger refires.
        _ring_frac = (reader.available() / float(_ring_n_total)) if is_live else 0.0
        _ring_pressure = bool(is_live and _ring_frac > 0.3)
        if (_cand_mgr is not None and _fire and len(_slow_halves) == 8
                and _ring_frac > _cand_mgr.slow_max_ring_frac):
            _cand_mgr.slow_max_ring_frac = _ring_frac
        _slow_rate_ok = True
        if _fire and _SLOW_MIN_TICKS and len(_slow_halves) == 8:
            if _slow_tick - _slow_last_fire < _SLOW_MIN_TICKS:
                _slow_rate_ok = False
            elif _ring_pressure and not _SLOW_NO_RING_DEFER:
                # ABLATION: when _SLOW_NO_RING_DEFER, ring pressure does NOT
                # defer the scan (busy/rate gates untouched).
                _slow_rate_ok = False
            if not _slow_rate_ok and os.environ.get('LORA_SLOW_DEBUG'):
                print(f"[SLOW] rate-bound defer tick={_slow_tick} "
                      f"(last={_slow_last_fire})", file=sys.stderr)
        # Slow-scan SUPPRESSION accounting.  actual _sup_ring counts only an
        # ACTUAL ring-caused deferral (false under the ablation); would_ring
        # counts whenever ring pressure was present (regardless of ablation) so
        # the ablation's coverage effect is attributable.
        if (_cand_mgr is not None and _fire and len(_slow_halves) == 8
                and (not (not _slow_busy[0] and _slow_rate_ok) or _ring_pressure)):
            _sup_busy = bool(_slow_busy[0])
            _sup_rate = bool(_SLOW_MIN_TICKS
                             and _slow_tick - _slow_last_fire < _SLOW_MIN_TICKS)
            _sup_ring = bool(_ring_pressure and not _SLOW_NO_RING_DEFER)
            _cand_mgr.note_slow_suppressed(_sup_busy, _sup_rate, _sup_ring,
                                           would_ring=_ring_pressure)
        if (_fire and len(_slow_halves) == 8 and not _slow_busy[0]
                and _slow_rate_ok and not _NO_SLOW):
            _slow_last_fire = _slow_tick
            _slow_pending.clear()
            # parts refs only — the 640 MB concat happens IN THE THREAD
            # (it was 92 ms median on the loop, second stall driver)
            _halves_bg = list(_slow_halves)
            _trig_bg = list(_fire)
            _seedpk_bg = [(float(_b), 13, 20.0, None) for _b in _trig_bg]
            _regpos_bg = tot_s
            _slow_busy[0] = True
            # Phase-1 slow-scan lifecycle: allocate scan_id, record STARTED
            # BEFORE launch (so a stuck scan is visible), with the ABSOLUTE
            # sample interval + ring positions for offline coverage derivation.
            _scan_id = _LID.next('scan') if _LID is not None else None
            if _scan_id is not None:
                if _cand_mgr is not None:
                    _cand_mgr.note_slow_started()
                _scan_len = sum(len(_h) for _h in _halves_bg)
                _run_event('slow_scan_started', run_id=_RUN_ID, scan_id=_scan_id,
                           seeds=len(_trig_bg), producer='slow_scan',
                           scan_end_sample=int(tot_s),
                           scan_start_sample=int(tot_s) - int(_scan_len),
                           rf_end_s=tot_s / a.rate,
                           rf_start_s=(tot_s - _scan_len) / a.rate,
                           ring_available=(reader.available() if is_live else None),
                           ring_total=_ring_n_total)

            def _slow_scan_bg(_hv=_halves_bg, _spk=_seedpk_bg,
                              _tg=_trig_bg, _rp=_regpos_bg, _sid=_scan_id):
                # BACKGROUND slow scan (0.5-0.6 s of FFT work): ran on the
                # main loop before and, stacked with p90 read+dispatch,
                # pushed iterations past the 500 ms hop budget -> ring
                # wraps. numpy FFTs release the GIL; drop-if-busy bounds
                # cost to one scan in flight.
                _t0 = time.time()
                try:
                    # concat INSIDE the guard, busy-flag cleared in finally:
                    # a MemoryError in the 640 MB concat (or anything else
                    # unexpected) used to leak _slow_busy=True forever,
                    # silently disabling the slow scan — i.e. all 31.25k /
                    # 41.67k detection — for the rest of the run (UC audit).
                    _iql = np.concatenate(_hv)   # off-loop
                    try:
                        _ds = detect_preamble(
                            _iql, a.rate, a.bandwidth, _live_center_mhz,
                            sc_threshold=a.threshold,
                            ethresh=a.energy_threshold,
                            spur_db=_spur_db, dc_notch_mhz=a.dc_notch,
                            spur_notch_hz=_spur_notch_hz or None,
                            debug=a.debug,
                            cached_peaks=_spk, only_bws={31250, 41667})
                    except Exception:
                        _ds = []
                    if os.environ.get('LORA_SLOW_DEBUG'):
                        print(f"[SLOW-SCAN] {time.time() - _t0:.2f}s "
                              f"seeds={len(_tg)} hits={len(_ds or [])} (bg)",
                              file=sys.stderr, flush=True)
                    if _sid is not None:
                        _run_event('slow_scan_completed', run_id=_RUN_ID,
                                   scan_id=_sid, duration_s=time.time() - _t0,
                                   hits=len(_ds or []), producer='slow_scan')
                    # Attack the payload itself (codex): a hit-less scan (the
                    # common case) has no consumer use for _iql — the consumer
                    # only touches _iq_long inside `if dets_slow:` — so drop the
                    # ~640 MB immediately instead of parking it in the queue.
                    if _ds:
                        _slow_results.put((_ds, _iql, _rp, _tg, _sid))
                    else:
                        _slow_results.put(([], None, _rp, _tg, _sid))
                        _iql = None
                        _slow_iql_dropped[0] += 1
                    # NOTE: _slow_busy stays TRUE here — single-flight now spans
                    # produce→drain→process (cleared in the consumer), so a new
                    # 640 MB scan cannot launch while this result is unconsumed.
                except Exception as _se:
                    print(f"[SLOW-SCAN] bg scan failed: {_se!r}",
                          file=sys.stderr, flush=True)
                    if _sid is not None:
                        if _cand_mgr is not None:
                            _cand_mgr.note_slow_failed()
                        _run_event('slow_scan_failed', run_id=_RUN_ID,
                                   scan_id=_sid, error=repr(_se), producer='slow_scan')
                    _slow_busy[0] = False   # failed → no result to drain; re-arm

            if is_live:
                threading.Thread(target=_slow_scan_bg, daemon=True).start()
            else:
                _slow_scan_bg()   # file mode: no realtime deadline, and the
                                  # loop outruns a background thread (stale
                                  # assemblies, dropped triggers)
        elif _trig:
            # Busy (or assembly still filling): defer this tick's fresh
            # triggers instead of dropping them.  A re-trigger refreshes
            # the pending entry (newest exact bin + expiry).
            for _b in _trig:
                _slow_pending[int(_b) // 8] = (_b, _slow_tick + 4)

        # consume finished background scans (bookkeeping on main thread)
        _slow_max_qdepth[0] = max(_slow_max_qdepth[0], _slow_results.qsize())
        while not _slow_results.empty():
            dets_slow, _iq_long, _reg_tot_s, _trig_done, _scan_done = _slow_results.get()
            # Phase-1 lineage: the slow scan owns NO _CandWin — stamp producer +
            # scan_id + a fresh detection_id onto each slow detection so its
            # decode record self-identifies (candidate_id is null: no
            # materialized energy candidate).  Detection logging is
            # candidate-independent (every dets_slow entry gets an id).
            if _cand_mgr is not None:
                _cand_mgr.note_slow_result(len(dets_slow))
                _n_consumed = 0
                for _d in dets_slow:
                    if isinstance(_d, dict):
                        _d['_lin_detection_id'] = (_LID.next('detection')
                                                   if _LID is not None else None)
                        _d['_lin_producer'] = 'slow_scan'
                        _d['_lin_scan_id'] = _scan_done
                        _d['_lin_window_id'] = None
                        _d['_lin_candidate_hint_id'] = None
                        _d['_lin_candidate_bin_delta'] = None
                        _d['_lin_candidate_matches_within_8bins'] = 0
                        _d['_lin_candidate_link_status'] = 'not_applicable'
                        _n_consumed += 1
                # hits-produced (slow_scan_completed) vs detections consumed here
                # must reconcile offline; emit the consumed count keyed by scan.
                _run_event('slow_scan_consumed', run_id=_RUN_ID,
                           scan_id=_scan_done, n_consumed=_n_consumed)
            _hit_bins = set()
            _hit_cool = {}       # hit bin -> post-hit cooldown (ticks)
            for d in dets_slow:
                _hb = int(round((d['freq_hz'] - _live_center_mhz * 1e6)
                                / (a.rate / 4096.0))) + 2048
                _hit_bins.add(_hb)
                # AIRTIME-SCALED HIT COOLDOWN (2026-07-07): the flat -4
                # (~2 s) re-armed the bucket MID-PACKET; the packet's own
                # tail then re-built the streak, the re-scan found no
                # preamble (long gone), and the fruitless ESCALATION
                # (fr=3 -> -8 -> -16 -> -32) punished the REAL carrier's
                # bucket into a hole its NEXT frame arrived inside of
                # (set12 live autopsy: bucket 141 at seeds=-6 as the new
                # preamble landed; image bucket 370 at streak 6 won the
                # trigger — image-only detection, frame lost, ~2/29 per
                # leg).  Sleep through the tail instead: half the max-PL
                # airtime in 0.5 s hop-ticks (full beacon tails are ~45%
                # of max-PL; a genuinely full-length packet's remaining
                # tail costs <=2 fruitless scans, which the fr<=2 grace
                # absorbs at -1 each).  A same-carrier follow-up frame
                # (relay hop, ACK) can't start before the current frame
                # ends, so the half-airtime sleep never eats one.
                _cool = int(np.ceil(0.5 * 148.25 * (2 ** d['sf'])
                                    / d['bw'] / 0.5))
                _hit_cool[_hb] = max(4, min(64, _cool))
            for _b in _trig_done:
                _bk = _b // 8
                _hits = [_hb for _hb in _hit_bins if abs(_b - _hb) < 8]
                if _hits:
                    _slow_seeds[_bk] = -max(_hit_cool[_hb] for _hb in _hits)
                    _slow_backoff[_bk] = 4
                    _slow_fruitless[_bk] = 0
                elif os.environ.get('LORA_NO_BACKOFF'):
                    _slow_seeds[_bk] = -4
                else:
                    _fr = _slow_fruitless.get(_bk, 0) + 1
                    _slow_fruitless[_bk] = _fr
                    if _fr <= 2:
                        # FALSE-EARLY TRIGGER GRACE (2026-07-06): the frame's
                        # own early preamble builds the trigger streak, but
                        # the first scan often fires while the 4 s assembly
                        # holds only ~0.5 s of frame -> fruitless.  The old
                        # escalation then backed the bin off longer than the
                        # REMAINING preamble (seeds only accumulate on
                        # preamble-like windows; payload never re-triggers)
                        # -> the real bin died for its own frame and the
                        # un-backed IQ-image bucket won the trigger instead
                        # (or nobody did).  Seed-state autopsy: bucket 141
                        # streak 1..6 pre-frame, fruitless at fr=2 -> -8
                        # ticks, image bucket 370 triggers, image-only
                        # detection; 4-5 of 29 frames lost per leg.  Two
                        # mild retries give the assembly +0.5 s of preamble
                        # each; real escalation starts at strike three, so
                        # persistent junk spurs cost at most two extra
                        # background scans per episode.
                        _slow_seeds[_bk] = -1
                    else:
                        _bo = min(256, _slow_backoff.get(_bk, 4) * 2)
                        _slow_backoff[_bk] = _bo
                        _slow_seeds[_bk] = -_bo
            if dets_slow:
                for d in dets_slow:
                    _pat_note_detection(d['freq_hz'])
                    _jk_note_detection(d['freq_hz'])
                    tot_d += 1
                    _abst = ((_reg_tot_s - len(_iq_long)) / a.rate
                             + d.get('preamble_t_s', 0.0))
                    print(f"[{time.time() - t_start:6.1f}s] DETECTED "
                          f"freq={d['freq_mhz']:.4f}MHz SF={d['sf']} "
                          f"BW={fmt_bw(d['bw'])} sc={d['detect_conf']:.2f} "
                          f"pwr={d['peak_power_db']:.1f}dB "
                          f"abst={_abst:.2f}s [slow-pass]",
                          flush=True)
                if recorder:
                    _slow_pkt_s = max(
                        (148.25 * (2 ** d['sf']) / d['bw']) for d in dets_slow)
                    # 24-window cap, not 12: SF12/31.25k frames run up to
                    # ~19.4 s and the 12 s cap GUARANTEED truncation (the
                    # soak's zero-decode wall for every >=9 s-airtime
                    # preset). Deferred tails accumulate from loop hops —
                    # NOT ring reads — so the old ring/2 bound does not
                    # apply to the live path.
                    _slow_tail_n = min(int(_slow_pkt_s * a.rate),
                                       24 * win_n)
                    # DEFERRED IN BOTH MODES: the file-mode blocking read
                    # CONSUMED stream the loop would otherwise process —
                    # every long-frame tail literally ate the next
                    # beacon's windows (12-window tails halved straddler
                    # detections; 24 made it worse). feed_tail() runs
                    # unconditionally below, so pending jobs complete
                    # from the hops in either mode.
                    # ASYNC-SCAN GAP FILL: hops that streamed while the
                    # background scan ran are still in _slow_halves —
                    # hand them to the job so the base abuts the first
                    # feed_tail hop (they were silently MISSING before:
                    # 0.5-1.0 s hole at every live slow capture's seam).
                    _gap_n = tot_s - _reg_tot_s
                    _gap_parts = None
                    _reg_pos = _reg_tot_s
                    if _gap_n > 0 and hop_n > 0:
                        _k = int(_gap_n) // hop_n
                        if (_k * hop_n == _gap_n
                                and 0 < _k <= len(_slow_halves)):
                            _gap_parts = list(_slow_halves)[-_k:]
                            _reg_pos = tot_s
                            _slow_tail_n = max(0, _slow_tail_n
                                               - _k * hop_n)
                        else:
                            # catchup/discontinuity between snapshot and
                            # registration — cannot fill seamlessly; the
                            # tail would splice. Register with the base
                            # only and let it truncate.
                            print(f"         [SLOW-GAP] unfillable gap "
                                  f"{_gap_n} samples — truncated base "
                                  f"capture", flush=True)
                            _slow_tail_n = 0
                    recorder.update_deferred(dets_slow, _iq_long,
                                             _reg_pos, pre_hop=None,
                                             need_tail_n=_slow_tail_n,
                                             owned_buf=True,
                                             gap_parts=_gap_parts)
            # Single-flight release: this result is fully drained AND processed
            # (its _iq_long is now cropped/owned by the recorder), so re-arm the
            # slow scan.  Cleared here — NOT at scan completion — so a new 640 MB
            # scan cannot start while a prior result is still pending/processing.
            _iq_long = None            # drop our local ref to the ~640 MB array
            _slow_busy[0] = False
        _prof['slow'] += time.time() - _t_slowblk0
        _t_step = time.time()

        if os.environ.get('LORA_MEMDBG') and (wc % 4) == 0:
            try:
                with open('/proc/self/status') as _st:
                    _rss = next(int(l.split()[1]) for l in _st
                                if l.startswith('VmRSS')) // 1024
            except Exception:
                _rss = -1
            _cq = recorder._crop_queue.qsize() if (recorder and getattr(recorder, '_crop_queue', None)) else -1
            _sq = recorder._save_queue.qsize() if recorder else -1
            _pd = len(recorder._pending) if recorder else -1
            _cod = getattr(recorder, '_crop_overflow_drops', -1) if recorder else -1
            _dq = decoder.pending() if decoder else -1
            print(f"[MEMDBG] rss={_rss}MB inflight={len(_inflight)} "
                  f"crop_q={_cq} save_q={_sq} pending={_pd} crop_drop={_cod} "
                  f"decoder_pending={_dq} ring_avail={reader.available() if is_live else 0}",
                  file=sys.stderr, flush=True)

        # ---- Skip-ahead ONLY if the ring buffer is near wrap ----
        # The old "skip when >2 windows behind" threshold was discarding
        # real LoRa packets during heavy back-to-back tests: the main loop
        # gets a few ms slower per iteration during long save-queue bursts,
        # accumulates ~2s of backlog, then voluntarily threw it away.  The
        # StreamBuffer's ring is sized for buf_seconds (default 6 s) so we
        # have plenty of headroom; only catch up when we are within ~1 s of
        # the ring actually overwriting unread samples (which would cause
        # silent loss anyway).  Until then we'd rather process slower than
        # drop packets.
        if is_live:
            avail = reader.available()
            _ring_n = int(a.rate * a.buf_seconds)
            # Trigger when used > 80 % of ring capacity (e.g. >4.8 s on a 6 s
            # ring), and only then drop back to 1 window from latest.
            if avail > int(0.8 * _ring_n):
                skipped_ahead = reader.skip_to_latest(win_n)
                if skipped_ahead > 0:
                    if _cand_mgr is not None:
                        # half-open skipped RF sample range [start, end): these
                        # samples are never processed → the scorer counts oracle
                        # occurrences here as input-coverage losses (primary).
                        _cand_mgr.note_skip(tot_s, tot_s + skipped_ahead)
                    tot_skip += skipped_ahead
                    skip_s = skipped_ahead / a.rate
                    print(f"[{elapsed:6.1f}s] CATCHUP skip {skipped_ahead/1e6:.1f}M "
                          f"samples ({skip_s:.2f}s) — ring buffer near wrap",
                          file=sys.stderr)
                    if recorder:
                        recorder.break_pending(
                            f"(catchup {skip_s:.2f}s)")
                    buf_pos = 0
                    if _lz is not None:
                        _lz.reset()
                        buf = None
                    else:
                        buf = np.zeros(win_n, dtype=np.complex64)
                    pre_hop = None  # stale after skip — reset for clean SC state
                    _prehop_hist.clear()   # abs continuity broken by catchup
                    _slice_cache.clear()
                    if recorder:
                        recorder.reset_prev()

        _prof['catchup'] += time.time() - _t_step
        _prof['n'] += 1
        # Keep-up monitor (runs regardless of debug level): the gate must process
        # at the input rate.  msps = samples processed / elapsed; if it sits below
        # the input rate after warmup, this host can't sustain the bandwidth and
        # samples will be dropped — warn ONCE with an actionable recommendation
        # instead of silently missing packets.
        if (is_live and not _warned_slow and wc % 10 == 0 and elapsed > 20.0):
            _msps_now = tot_s / elapsed / 1e6
            if _msps_now < 0.95 * (a.rate / 1e6):
                _warned_slow = True
                print(f"[{elapsed:6.1f}s] *** KEEP-UP WARNING: gate sustaining only "
                      f"{_msps_now:.1f} Msps vs {a.rate/1e6:.0f} Msps input — this "
                      f"host is too slow for the full bandwidth and WILL drop "
                      f"packets.  Reduce -r/-b (e.g. -r {int(a.rate/2e6)}000000 "
                      f"-b {int(a.bandwidth/2e6)}000000), raise --buf-seconds, "
                      f"lower --detect-workers contention, or use a faster host. ***",
                      file=sys.stderr, flush=True)
        # Adaptive worker throttle: every 5 windows (~2.5 s) check whether
        # the gate is under pressure.  Two signals, either sets _gate_stress:
        #   - reader.drops grew (ring already overwrote — the late signal), or
        #   - ring >50 % full (PROACTIVE: the consumer is falling behind but
        #     nothing is lost yet — the same fill signal the peak-cap
        #     governor uses).  Reacting here, before loss, lets the decode
        #     workers back off (pause + fail-fast budget) in time to save
        #     the ring instead of after it wraps.
        # Self-clearing when drops stop growing AND the ring has drained.
        # Per-window checking made things worse (event thrashing on
        # drop-counter jitter).
        if (is_live and wc % 5 == 0 and recorder
                and getattr(recorder, '_decoder', None)):
            _drops_now = reader.drops
            _dec = recorder._decoder
            _prev_d = getattr(_dec, '_prev_drops_obs', 0)
            _rfrac = reader.available() / float(_ring_n_total)
            if _drops_now > _prev_d or _rfrac > 0.5:
                _dec._gate_stress.set()
            elif _rfrac < 0.2:
                _dec._gate_stress.clear()
            _dec._prev_drops_obs = _drops_now
        if a.debug >= 1 and wc % 10 == 0:
            msps = tot_s / elapsed / 1e6 if elapsed > 0 else 0
            # save_queue: wideband batches waiting to be recentred + extracted.
            # dec_queue:  per-preamble files waiting on the decoder subprocess.
            # active:     1 if the decoder is currently inside process_file, else 0.
            # Pipeline is truly idle iff save_q == 0 AND dec_q == 0 AND active == 0.
            save_q = recorder._save_queue.qsize() if recorder else 0
            dec_q = recorder._decoder.pending() if (
                recorder and getattr(recorder, '_decoder', None)) else 0
            active = 0
            _dec_u = _dec_t = 0
            if recorder and getattr(recorder, '_decoder', None):
                with recorder._decoder._lock:
                    active = recorder._decoder._active_count
                _dec_t, _dec_u = recorder._decoder.decoded_counts()
            drops = reader.drops if is_live else 0
            # rss= : main-process resident memory, for OOM forensics (issue #6
            # follow-up: a 24/7 host was OOM-killed with 4.5 GB ANON in one
            # python3 — this pins WHICH run grows and how fast).  One 5-second
            # /proc read; the web health parser extracts fields by name, so
            # the insertion is format-safe.
            try:
                with open('/proc/self/statm') as _sm:
                    _rss_mb = int(_sm.read().split()[1]) * (os.sysconf('SC_PAGE_SIZE') // 1024) / 1024.0
                # Detect-pool workers hold the big per-process buffers, so
                # main-only rss= under-reports real footprint several-fold
                # on OOM-prone hosts (UC audit) — fold their RSS in too.
                if _pool is not None:
                    for _wp_ in getattr(_pool, '_workers', []):
                        try:
                            with open(f'/proc/{_wp_.pid}/statm') as _sm:
                                _rss_mb += (int(_sm.read().split()[1])
                                            * (os.sysconf('SC_PAGE_SIZE')
                                               // 1024) / 1024.0)
                        except Exception:
                            pass
            except Exception:
                _rss_mb = 0.0
            print(f"[STAT] {elapsed:.1f}s | {msps:.1f}Msps | win={wc} det={tot_d} "
                  f"decoded={_dec_u}u/{_dec_t} "
                  f"save_q={save_q} dec_q={dec_q} active={active} "
                  f"pipe={dt * 1000:.0f}ms"
                  + (f" rss={_rss_mb:.0f}M" if _rss_mb else "")
                  + (f" hyb={_hyb_inline}i/{_hyb_pooled}p"
                     if _hybrid and _pool is not None else "")
                  + (f" drops={drops/1e6:.1f}M" if drops else "")
                  + (f" clip={_sat_max*100:.1f}%" if _sat_max > 0.005 else "")
                  + (f" gain={_agc.gain:.0f}" if _agc is not None and _agc.enabled
                     else ""),
                  file=sys.stderr)
            _sat_max = 0.0   # reset the clip tracker for the next [STAT] interval
            if _prof['n'] > 0:
                n = _prof['n']
                print(f"  PROF (avg ms over {n} wins): "
                      f"read={_prof['read']/n*1000:.0f} "
                      f"welch={_prof['welch']/n*1000:.0f} "
                      f"gate+sweep={_prof['notch']/n*1000:.0f} "
                      f"sat={_prof['sat']/n*1000:.0f} "
                      f"detect={_prof['detect']/n*1000:.0f} "
                      f"tail={_prof['tail']/n*1000:.0f} "
                      f"recorder={_prof['recorder']/n*1000:.0f} "
                      f"slow={_prof['slow']/n*1000:.0f} "
                      f"feed={_prof['feed']/n*1000:.0f}"
                      + (f" catchup={_prof['catchup']/n*1000:.0f}"
                         if _prof['catchup'] else "")
                      + (f" slc={_slice_cache_hits}/"
                         f"{_slice_cache_hits + _slice_cache_miss}"
                         if _slice_cache_on else "")
                      + (f" phref={_prehop_ref_hits}/"
                         f"{_prehop_ref_hits + _prehop_copies}"
                         if _prehop_ref_on and (_prehop_ref_hits
                                                or _prehop_copies) else "")
                      + (f" jk={_jk_sup_peaks}sup/{len(_jk_junk)}bin"
                         f"/{_jk_sup_windows}qw"
                         + (f"/DEAF={_jk_deaf}" if PRESCREEN_VERIFY else "")
                         if PRESCREEN_ON and (_jk_sup_peaks or _jk_junk)
                         else ""),
                      file=sys.stderr)
                for k in _prof: _prof[k] = 0
                _prof['n'] = 0
                _slice_cache_hits = _slice_cache_miss = 0
                _prehop_ref_hits = _prehop_copies = 0

    except KeyboardInterrupt:
        pass

    # Drain any in-flight multiprocess-detect windows (commit them to the
    # recorder), then shut the detect pool down.
    if _pool is not None:
        while _inflight:
            _commit_oldest()
        _pool.close()

    # Candidate audit: all pool windows committed above (each emitted its record);
    # inline windows emitted during the loop.  Final drained summary — its
    # audit_valid requires windows_begun==windows_emitted AND outstanding==0 AND
    # zero invariant failures.  outstanding = any audit still attached to an
    # in-flight item (0 after the drain above).
    if _cand_mgr is not None:
        _outstanding = sum(1 for _it in _inflight if _it.get('cand_audit'))
        _cand_mgr.summary(_outstanding)

    # SOURCE SUMMARY (validation-only, gated): emitted ONCE here — after the
    # source/window loop exits, BEFORE the decoder drain — so source consumption
    # is settled while downstream work hasn't obscured the source boundary.  The
    # atomic StreamBuffer snapshot + window cursor let the offline B0 validator
    # close the ingress invariant (bytes_read_total==capture_bytes,
    # partial_bytes_discarded==0, ring_dropped==0, dequeued+residual==ingress).
    # window_sample_cursor (tot_s) is reported SEPARATELY (not as
    # 'samples_processed') because non-production branches can double-count carry.
    if _CAND_AUDIT and is_live:
        _src = reader.source_snapshot()
        _src_end = ('clean_eof' if (_src['saw_clean_eof'] and not _src['reader_error'])
                    else (_src['reader_error'] or 'not_clean_eof'))
        _run_event('source_summary', schema='source_summary/1',
                   source_end_reason=_src_end,
                   window_sample_cursor=int(tot_s),
                   samples_skipped=int(tot_skip),
                   **{k: v for k, v in _src.items() if k != 'saw_clean_eof'})

    # Drain pending saves first (they submit to decoder), then drain decoder
    if recorder:
        pending_saves = recorder._save_queue.qsize()
        if pending_saves > 0:
            print(f"\nWaiting for {pending_saves} pending save(s)...",
                  file=sys.stderr)
        recorder.flush()

    if recorder:
        try:
            recorder.flush_pending(wait=True)   # deferred tails: finalize
                                       # (and crop) before the decoder
                                       # drain so stragglers decode
        except Exception:
            pass
    # Instrumentation (validation-only): the live loop has exited; everything
    # from here is post-input-EOF drain.  Mark the boundary so offline scoring
    # can separate ordinary pipeline tail from the explicit drain_slow_into_fast.
    _run_event('live_loop_end',
               pending=(decoder.pending() if decoder else 0))
    if decoder:
        # Capture has ended (EOF) — move the deferred straggler queue into the
        # fast queue so the workers re-decode them now (big budget), without
        # having competed with the realtime gate during capture.
        _run_event('slow_drain_begin', pending=decoder.pending())
        decoder.drain_slow_into_fast()
        if decoder.pending() > 0:
            print(f"\nWaiting for {decoder.pending()} pending decode(s)...",
                  file=sys.stderr)
            decoder.drain(timeout=3600.0)
        # Acceptance invariant: at drain_end, pending=0 AND slow_inflight=0 AND
        # no reservation present AND both queues empty (read scheduler-lock-
        # consistent).
        _run_event('drain_end', pending=decoder.pending(),
                   **decoder._slow_state_snapshot())

    # Authoritative work/lineage reconciliation — AFTER the decoder drain, so
    # in-flight work is not mistaken for outstanding (codex ordering req).
    # Residual slow-scan results at shutdown are DISCARDED, exactly as audit-off
    # execution discards them (the consume loop lived in the live loop).  We do
    # NOT consume/decode them (that would change behavior and fake the
    # invariant); we wait (bounded) for any in-flight scan to land, then COUNT
    # the residuals as discarded_shutdown.  started == consumed + failed +
    # discarded is then an honest invariant.
    if _cand_mgr is not None:
        try:
            _tw = time.time()
            while _slow_busy[0] and time.time() - _tw < 3.0:
                time.sleep(0.02)
            _cand_mgr.note_slow_discarded(_slow_results.qsize())
        except Exception:
            pass
        _outstanding_final = sum(1 for _it in _inflight if _it.get('cand_audit'))
        _cand_mgr.lineage_summary(_outstanding_final)

    total_drops = reader.drops if is_live else 0
    _dec_summary = ''
    if decoder:
        _dt, _du = decoder.decoded_counts()
        _dec_summary = f" (decode=on: {_du} distinct packet(s), {_dt} decode events)"
    print(f"\nDone: {time.time() - t_start:.1f}s, {tot_s} samples, {tot_d} detections"
          + _dec_summary
          + (f" skipped={tot_skip/1e6:.1f}M" if tot_skip else ""),
          file=sys.stderr)
    if _PP_PROF_ON:
        _e, _s, _d = _PP_PROF['extract'], _PP_PROF['sc'], _PP_PROF['dechirp']
        _tot = _e + _s + _d
        print("[PP-PROF] per-peak cost breakdown (channelizer amortizes EXTRACT):\n"
              "  extract(crop+IFFT) = %.1fs (%.0f%%, %d calls)  <- channelizer target\n"
              "  schmidl-cox        = %.1fs (%.0f%%, %d calls)  <- per-signal, stays\n"
              "  dechirp            = %.1fs (%.0f%%, %d calls)  <- per-signal, stays\n"
              "  => max channelizer speedup if extract->0: %.2fx"
              % (_e, 100 * _e / max(1e-9, _tot), _PP_PROF['n_extract'],
                 _s, 100 * _s / max(1e-9, _tot), _PP_PROF['n_sc'],
                 _d, 100 * _d / max(1e-9, _tot), _PP_PROF['n_dechirp'],
                 _tot / max(1e-9, _s + _d)),
              file=sys.stderr)
        print("[PP-PROF] extraction memo potential: %d calls, %d distinct (F,bin,n) "
              "=> %.0f%% are EXACT DUPLICATES (bit-identical memo would eliminate)"
              % (_EXT_CALLS[0], len(_EXT_KEYS),
                 100.0 * (1 - len(_EXT_KEYS) / max(1, _EXT_CALLS[0]))),
              file=sys.stderr)
    if os.environ.get('LORA_PROF') and _prof.get('n'):
        _pt = time.time() - t_start
        _pn = _prof['n']
        _stages = ('read', 'slide', 'welch', 'notch', 'sat',
                   'detect', 'tail', 'recorder', 'slow', 'feed', 'catchup')
        print("[PROF] iters=%d wall=%.1fs  per-iter(ms):" % (_pn, _pt)
              + "".join(" %s=%.1f" % (k, _prof[k] / _pn * 1000.0) for k in _stages)
              + ("  sum_acct=%.1f" % (sum(_prof[k] for k in _stages) / _pn * 1000.0)),
              file=sys.stderr)
        # THREAD-CPU ms per iter (only stamped for read/welch/notch/sat) — where
        # wall >> cpu, the stage is BLOCKING (pacing/backpressure), not compute.
        print("[PROF] cpu-ms/iter:"
              + "".join(" %s(w=%.0f,c=%.0f)" % (
                  k, _prof[k] / _pn * 1000.0, _proft[k] / _pn * 1000.0)
                  for k in ('read', 'welch', 'notch', 'sat')),
              file=sys.stderr)
        print("[PROF] read-split: rd_io(convert/fastpath)=%.0fms rd_slide(window-copy)=%.0fms"
              % (_prof['rd_io'] / _pn * 1000.0, _prof['rd_slide'] / _pn * 1000.0),
              file=sys.stderr)
        _sc_tot = _slice_cache_hits + _slice_cache_miss
        print("[PROF] slice-cache: hits=%d miss=%d hit_rate=%.0f%% (%.1f slices/iter)"
              % (_slice_cache_hits, _slice_cache_miss,
                 100.0 * _slice_cache_hits / max(1, _sc_tot), _sc_tot / _pn),
              file=sys.stderr)
        print("[PROF] notch-split: sweep=%.0fms (welch=%.0f post=%.0f) rest=%.0fms"
              % (_prof['sweep'] / _pn * 1000.0, _prof['sweep_welch'] / _pn * 1000.0,
                 _prof['sweep_post'] / _pn * 1000.0,
                 (_prof['notch'] - _prof['sweep']) / _pn * 1000.0),
              file=sys.stderr)
        print("[PROF] slow-scan: max_qdepth=%d iql_dropped(hitless)=%d "
              "(qdepth must stay <=1 with the single-flight fix)"
              % (_slow_max_qdepth[0], _slow_iql_dropped[0]),
              file=sys.stderr)
        if _lz is not None:
            print("[PROF] lazy: converted=%.2fM samples/iter (%.1fx the %dM hop; "
                  "%.2f full-window materializations/iter of %dM)"
                  % (_lz._conv_samples / _pn / 1e6,
                     _lz._conv_samples / _pn / max(1, hop_n),
                     hop_n // 1_000_000,
                     _lz._mat_count / _pn, win_n // 1_000_000),
                  file=sys.stderr)
    if PRESCREEN_ON and (PRESCREEN_VERIFY or _jk_sup_peaks or _jk_junk):
        print(f"[PRESCREEN] summary: {_jk_sup_peaks} peak(s) "
              + ('would be ' if PRESCREEN_VERIFY else '')
              + f"suppressed, {_jk_sup_windows} window(s) fully quieted, "
              f"{len(_jk_junk)} junk bin(s) held at EOF"
              + (f", DEAFNESS={_jk_deaf}" if PRESCREEN_VERIFY else ""),
              file=sys.stderr, flush=True)
    if a.file: fp.close()

    # Instrumentation (validation-only): a reader-thread failure means the input
    # was corrupted/truncated — the run drained and printed Done, but its result
    # set is NOT trustworthy.  Exit NONZERO so the runner's detector_exit_code
    # independently marks the run invalid (redundant with the input_terminated
    # event / missing clean input_eof).
    if getattr(reader, '_reader_error', None):
        print(f"[FATAL] reader error invalidates run: {reader._reader_error}",
              file=sys.stderr, flush=True)
        sys.exit(3)


if __name__ == '__main__':
    main()
