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

import sys, os, time, argparse, numpy as np
import threading, queue, io, subprocess, fcntl

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
        from pyfftw.interfaces.scipy_fft import fft as _fft, ifft as _ifft
        _FFT_WORKERS = 1

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


class IQReader:
    def __init__(self, fp, fmt='sc16'):
        self.fp, self.sc16 = fp, (fmt == 'sc16')
        self.bps = 4 if self.sc16 else 2
    def read(self, n):
        raw = self.fp.read(n * self.bps)
        if len(raw) < n * self.bps: return None
        if self.sc16:
            s = np.frombuffer(raw, dtype=np.int16)
            return (s[0::2] + 1j * s[1::2]).astype(np.complex64) / 2048.0
        b = np.frombuffer(raw, dtype=np.int8)
        return (b[0::2] + 1j * b[1::2]).astype(np.complex64) / 128.0


class StreamBuffer:
    """Background-threaded IQ reader with ring buffer.

    Keeps the pipe/stdin permanently drained so bladeRF never drops samples.
    Stores raw int16/int8 pairs in a pre-allocated ring — conversion to
    complex64 happens on read() to halve memory (480 MB for 3s at 40 Msps).

    When the consumer can't keep up, the ring overwrites the oldest data.
    read() detects this and skips the read pointer forward, returning a
    `skipped` count so the caller can reset any stateful context.
    """

    def __init__(self, fp, fmt='sc16', rate=40_000_000, buf_seconds=3.0):
        self.sc16 = (fmt == 'sc16')
        self.bps = 4 if self.sc16 else 2
        self.rate = rate
        self._fp = fp

        ring_n = int(rate * buf_seconds)
        dtype = np.int16 if self.sc16 else np.int8
        self._ring = np.zeros(ring_n * 2, dtype=dtype)  # ×2 for I + Q
        self._ring_n = ring_n                             # capacity in IQ samples
        self._ring_elems = ring_n * 2                     # capacity in array elements

        self._wp = 0          # write position (absolute, in IQ samples)
        self._rp = 0          # read position (absolute, in IQ samples)
        self._lock = threading.Lock()
        self._eof = False
        self._total_drops = 0

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
            except (OSError, ValueError):
                self._eof = True
                break
            if not raw or len(raw) < self.bps:
                self._eof = True
                break
            n_samp = len(raw) // self.bps

            if self.sc16:
                data = np.frombuffer(raw[:n_samp * self.bps], dtype=np.int16)
            else:
                data = np.frombuffer(raw[:n_samp * self.bps], dtype=np.int8)

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

    def read(self, n):
        """Read n IQ samples as complex64.

        Returns (data, skipped) where:
          data:    np.complex64 array of length n, or None on EOF
          skipped: number of samples that were lost (ring overwrite)

        Each call returns temporally contiguous data. If samples were lost,
        `skipped` > 0 indicates a temporal gap since the previous read.
        """
        while True:
            with self._lock:
                avail = self._wp - self._rp
                if avail >= n:
                    skipped = 0
                    # Check if write pointer lapped us
                    if avail > self._ring_n:
                        lost = avail - self._ring_n
                        self._rp += lost
                        self._total_drops += lost
                        skipped = lost
                        avail = self._ring_n

                    # Extract from ring
                    start = (self._rp * 2) % self._ring_elems
                    n_elem = n * 2
                    end = start + n_elem
                    if end <= self._ring_elems:
                        raw = self._ring[start:end].copy()
                    else:
                        split = self._ring_elems - start
                        raw = np.concatenate([
                            self._ring[start:],
                            self._ring[:n_elem - split]
                        ])

                    self._rp += n

                    # Convert to complex64.  The interleaved int16/int8 I/Q
                    # buffer becomes complex64 via a single astype pass + a
                    # zero-copy reinterpret cast: [I0,Q0,I1,Q1,...] float32
                    # viewed as complex64 is [I0+jQ0, ...].  This is ~2.4×
                    # faster than the strided (raw[0::2]+1j*raw[1::2]) form
                    # (354→145 ms per 28 M samples) and bit-identical — a real
                    # win on the per-hop read budget at 28 Msps live.
                    scale = 2048.0 if self.sc16 else 128.0
                    raw_f = raw.astype(np.float32)
                    raw_f /= scale
                    return raw_f.view(np.complex64), skipped

                if self._eof:
                    return None, 0
            time.sleep(0.005)

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


def welch_psd(iq, nfft=4096, n_avg=64, also_max=False):
    n = len(iq)
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
        segs[i] = iq[s:s + nfft]
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
               wide_run_bins=100, min_sep_bins=20):
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

    Returns: list of (center_bin, width_bins, peak_db) tuples, sorted by
    peak_db descending, truncated to `max_peaks`.
    """
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
    for s, e in zip(starts.tolist(), ends.tolist()):
        _emit_run_peaks(psd_db, s, e, min_bins, wide_run_bins, min_sep_bins, peaks)
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


def _emit_run_peaks(psd_db, start, end, min_bins, wide_run_bins, min_sep_bins, peaks):
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
        peaks.append((start + _power_centroid(seg, 0, len(seg)), width, peak_db))
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
        peaks.append((start + _power_centroid(seg, 0, len(seg)), width, peak_db))
        return
    a, b = sorted((rel_seed, rel2_seed))
    dip = a + int(np.argmin(seg[a:b + 1]))   # split at the trough between them
    peaks.append((start + _power_centroid(seg, 0, dip), width, peak_db))
    peaks.append((start + _power_centroid(seg, dip, len(seg)), width, peak_db2))


# ---- Narrowband extraction ----

def extract_nb_fft_multi_bw(iq, wb_fs, offset_hz, target_bws, chunk=65536,
                            fft_cache=None):
    """
    Batched FFT-crop: forward FFTs computed ONCE, then crop at each BW.
    Returns list of (nb_iq, nb_fs) per BW. Not phase-coherent — fine for CNN.

    fft_cache: precomputed list of fftshift(fft(chunk)) arrays.  When provided,
    the forward FFT pass is skipped entirely — pass the same cache to every peak
    in a multi-peak window to avoid recomputing the same 28 M-sample FFT N times.
    """
    if fft_cache is None:
        n_chunks = max(1, len(iq) // chunk)
        fft_cache = []
        for c in range(n_chunks):
            s = c * chunk
            if s + chunk > len(iq): break
            fft_cache.append(np.fft.fftshift(_fft(iq[s:s + chunk])))

    cb = chunk // 2 + int(round(offset_hz * chunk / wb_fs))
    results = []
    for tbw in target_bws:
        bw_bins = max(4, int(round(tbw * chunk / wb_fs)))
        lo, hi = cb - bw_bins, cb + bw_bins
        cw = hi - lo
        parts = []
        for F in fft_cache:
            cropped = np.zeros(cw, dtype=np.complex64)
            lo_c, hi_c = max(0, lo), min(chunk, hi)
            dst = max(0, -lo)
            cropped[dst:dst + hi_c - lo_c] = F[lo_c:hi_c]
            parts.append(_ifft(np.fft.ifftshift(cropped)).astype(np.complex64))
        if parts:
            results.append((np.concatenate(parts), cw * wb_fs / chunk))
        else:
            results.append((np.array([], dtype=np.complex64), tbw * 2))
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

    # FFT and shift so DC is in the middle (reuse if caller is iterating
    # multiple offsets over the same wideband buffer).  Pad to a fast FFT
    # length to avoid prime-factor slowdowns; the cached F's length defines
    # N for all crop math so cached and fresh calls stay consistent.
    if fft_cache is not None and fft_cache:
        F = fft_cache[0]
        N = len(F)
    else:
        N = _next_fast_len(N0)
        if N != N0:
            iqp = np.empty(N, dtype=np.complex64)
            iqp[:N0] = iq
            iqp[N0:] = 0
            F = np.fft.fftshift(_fft(iqp))
        else:
            F = np.fft.fftshift(_fft(iq))
        if fft_cache is not None:
            fft_cache.append(F)

    # Number of output bins = N * (target_fs / wb_fs)
    n_out = int(round(N * target_fs / wb_fs))
    if n_out < 2:
        return np.array([], dtype=np.complex64), target_fs

    # Find the center bin corresponding to offset_hz
    freq_per_bin = wb_fs / N
    center_bin = N // 2 + int(round(offset_hz / freq_per_bin))

    # Crop n_out bins centered on the signal
    half = n_out // 2
    lo = center_bin - half
    hi = lo + n_out

    # Handle edge cases
    cropped = np.zeros(n_out, dtype=np.complex128)
    src_lo = max(0, lo)
    src_hi = min(N, hi)
    dst_lo = src_lo - lo
    dst_hi = dst_lo + (src_hi - src_lo)
    cropped[dst_lo:dst_hi] = F[src_lo:src_hi]

    # IFFT back to time domain — signal is now centered at DC
    result = _ifft(np.fft.ifftshift(cropped)).astype(np.complex64)

    # Scale to preserve amplitude (fewer bins → need to scale)
    result *= (n_out / N)

    return result, float(target_fs)



# ---- Dechirp ----

def generate_downchirp(sf, bw, fs):
    N = 2 ** sf
    osf = int(round(fs / bw))
    sps = N * osf
    t = np.arange(sps, dtype=np.float64) / fs
    Ts = N / bw
    phase = 2.0 * np.pi * (-bw / 2.0 * t + bw / (2.0 * Ts) * t * t)
    return np.exp(-1j * phase).astype(np.complex64)


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
        dechirped = dechirped.reshape(n_total, N, osf).mean(axis=2)
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


def dechirp_peak_quality(iq, sf, bw):
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
        # Proper decimation: average osf consecutive samples → N samples/symbol
        dechirped = dechirped.reshape(n_total_syms, N, osf).mean(axis=2)
    spectra = np.abs(_fft(dechirped, axis=1)) ** 2
    peaks = np.max(spectra, axis=1)
    means = np.mean(spectra, axis=1)
    all_ratios = np.where(means > 0, peaks / means, 0)

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
    4096:  [(8, 62500),  (9, 125000), (10, 250000)],
    # SF12/500 omitted here for the same architectural reason as SF11/500: it
    # would share lag 8192 with SF10/125 + SF11/250, and the resolver may
    # mis-vote on 2-symbol harmonics of those fundamentals.  Re-add only after
    # the resolver is hardened against same-lag fundamental-vs-harmonic confusion.
    8192:  [(9, 62500),  (10, 125000), (11, 250000)],
    # SF12/250 omitted on the same harmonic-confusion grounds (would share
    # lag 16384 with SF11/125).
    16384: [(10, 62500), (11, 125000)],
    32768: [(11, 62500), (12, 125000)],
    65536: [(12, 62500)],
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


def schmidl_cox_score(iq_1m, lag, n_sym=8, _curve=None):
    """Schmidl-Cox score for one lag: argmax of the SC sliding-mean curve.
    Returns (score, preamble_sample_idx) where score ∈ [0, 1].
    `_curve` reuses a precomputed curve from _schmidl_cox_curve (gate caching)."""
    mean_s = _curve if _curve is not None else _schmidl_cox_curve(iq_1m, lag, n_sym)
    if mean_s is None:
        return 0.0, 0
    best_idx = int(np.argmax(mean_s))
    return float(mean_s[best_idx]), best_idx * lag


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


def _resolve_sf_bw_ambig(iq_1m, candidates, fft_cache=None):
    """Pick best (sf, bw) from ambiguous Schmidl-Cox candidates via dechirp quality.
    fft_cache: reuses a single forward FFT across candidates with different BWs
    (same buffer, different narrowband extraction).  ~25 % per-call speedup when
    a bucket has 3+ candidates."""
    best_sf, best_bw, best_q = candidates[0][0], candidates[0][1], -99.0
    for sf, bw in candidates:
        nb, _ = extract_narrowband_fft(iq_1m, 1_000_000, 0.0, bw, fft_cache=fft_cache)
        if len(nb) < (2 ** sf) * 4:
            continue
        q, _ = dechirp_peak_quality(nb, sf, bw)
        if q > best_q:
            best_q, best_sf, best_bw = q, sf, bw
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
                nb_use = nb[_p:]
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
                    cached_psd=None, cached_peaks=None):
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
        peaks = find_peaks(psd, thresh_db=ethresh)
    if not peaks:
        return []

    if len(peaks) > 1:
        max_pwr = max(p[2] for p in peaks)
        peaks = [p for p in peaks if p[2] >= max_pwr - spur_db]

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
            _ffts = np.fft.fftshift(_ffts, axes=1)
            _nb_fft_cache = [_ffts[i] for i in range(_nc)]
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
        cb, bw_bins, pwr = peak
        off_hz = (cb - nfft_c / 2) * fres
        _ld = []

        iq_1m_parts = extract_nb_fft_multi_bw(
            iq, wb_fs, off_hz, [500_000], chunk=_NB_CHUNK,
            fft_cache=_nb_fft_cache)
        if not iq_1m_parts:
            return _ld
        iq_1m, _rate1m = iq_1m_parts[0]

        # Minimum length: 3 × worst-case symbol duration (SF12/125k at 1Msps = 32768)
        if len(iq_1m) < 32768 * 3:
            return _ld

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
        sc = {}
        sc_cache = {}
        for _lag in _ALL_LAGS_1M:
            if _lag <= 8192:
                _la = _lag
            else:
                # Long lags (SF11/12): the extract rate is ~999756 Hz (not exactly
                # 1 Msps), so the true symbol length is nominal_lag*(rate/1e6).
                # Use that SINGLE scaled lag — one SC call, as cheap as the
                # original integer scan.  (An earlier ±2-neighbourhood search did
                # 5× SC per long lag on EVERY peak and spiked the gate's per-window
                # detect to ~700 ms under live burst load → the gate fell below
                # 28 Msps and dropped samples.)  Single scaled SC still recovers
                # SF12: sc=0.997 at the scaled lag 32760 vs 0.091 at 32768.
                _la = int(round(_lag * _rate1m / 1e6)) if _rate1m else _lag
            _curve = _schmidl_cox_curve(iq_1m, _la, n_sym=8)
            sc_cache[_lag] = (_la, _curve)
            sc[_lag] = schmidl_cox_score(iq_1m, _la, n_sym=8, _curve=_curve)

        hit_lags = [(lag, sc[lag][0], sc[lag][1])
                    for lag in _ALL_LAGS_1M if sc[lag][0] >= sc_threshold]

        if debug >= 2:
            best_lag = max(_ALL_LAGS_1M, key=lambda l: sc[l][0])
            print(f"  Peak {off_hz / 1e6:+.3f}MHz pwr={pwr:.0f}dB  "
                  f"SC best lag={best_lag} score={sc[best_lag][0]:.3f}  "
                  f"hits={len(hit_lags)}",
                  file=sys.stderr)

        if not hit_lags:
            return _ld

        hit_lags.sort(key=lambda x: x[1], reverse=True)

        seen = set()
        _wide = bool(os.environ.get('LORA_SCAN_FULL'))
        _ffc = []   # one forward-FFT cache shared by all candidate extractions
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
                candidates = _SC_LAGS_1M[lag]
                if len(candidates) == 1:
                    sf, bw = candidates[0]
                else:
                    sf, bw, _ = _resolve_sf_bw_ambig(iq_1m, candidates, fft_cache=_ffc)

            # Find ALL time-separated preamble plateaus at this lag — so a
            # SECOND packet that shares this carrier + SF/BW (a relay hop1
            # ~0.5 s later, or simply another packet shortly after the first)
            # gets its OWN detection instead of being hidden behind the
            # strongest plateau.  The single-argmax SC saw only one of them, so
            # the later packet was never detected → never captured → lost (the
            # last live-miss root).  Reuse the SC curve already computed for this
            # lag in the scan (cached) — same math, no recompute.
            _lag_used, _curve = sc_cache[lag]
            _positions = schmidl_cox_peaks(iq_1m, _lag_used, n_sym=8,
                                           thr=sc_threshold, max_peaks=6,
                                           _curve=_curve)
            if not _positions:
                _positions = [(sc_score_best, _pos_best)]

            # Narrowband extraction depends only on bw (not the plateau
            # position), so compute it ONCE per hit lag instead of per plateau.
            nb, _nbfs = extract_narrowband_fft(iq_1m, 1_000_000, 0.0, bw)

            for sc_score, _pos in _positions:
                # Merge the SAME packet re-found at another lag (≈20 ms time
                # bucket); genuinely distinct packets — including close
                # back-to-back ones >20 ms apart — keep separate detections.
                _tb = int((_pos / _rate1m) / 0.02) if _rate1m else int(_pos)
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
                if 0 < _pos_nb < len(nb) - N_sf * 4:
                    nb_al = nb[_pos_nb:]
                else:
                    nb_al = nb
                if len(nb_al) >= N_sf * 4:
                    bw_quality, peak_bin = dechirp_peak_quality(nb_al, sf, bw)
                else:
                    bw_quality, peak_bin = -99.0, 0

                # Dechirp quality is the reliable LoRa indicator.  SC alone can
                # false-positive on any narrowband/CW signal.  Require both SC
                # and dechirp confirmation (also rejects wrong-SF lag aliases
                # and any spurious extra SC plateau, so multi-peak adds no FP).
                if bw_quality < DECHIRP_MIN_DB:
                    if debug >= 2:
                        print(f"    Rejected SF{sf}/BW{bw/1000:.0f}k: "
                              f"sc={sc_score:.2f} bwq={bw_quality:.1f}dB "
                              f"(need bwq>={DECHIRP_MIN_DB:.0f}dB)",
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
                if int(_pos) + 16 * _sym1m > len(iq_1m) or int(_pos) < 0:
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

                _ld.append({
                    'freq_hz':       freq_hz,
                    'freq_mhz':      freq_hz / 1e6,
                    'sf':            sf,
                    'bw':            bw,
                    'bw_cnn':        bw,        # compatibility field
                    'detect_conf':   float(sc_score),
                    'sf_conf':       float(sc_score),
                    'bw_quality_db': bw_quality,
                    'peak_power_db': pwr,
                    # Preamble start time WITHIN this window (seconds), combined
                    # downstream with the window sample position for an absolute
                    # RF timestamp (identical across overlapping windows that
                    # re-detect the same packet) — used to dedup captures.
                    'preamble_t_s':  (float(_pos) / _rate1m) if _rate1m else 0.0,
                })
                seen.add((sf, bw, _tb))
        return _ld

    if len(peaks) <= 1 or os.environ.get('LORA_DETECT_SERIAL'):
        # Serial per-peak — used when a higher level already parallelises
        # (e.g. a pool of single-threaded detect worker processes), so the
        # thread pool here would only oversubscribe.
        for _pk in peaks:
            dets.extend(_process_one_peak(_pk))
    else:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=min(len(peaks), 8)) as _ex:
            for _r in _ex.map(_process_one_peak, peaks):
                dets.extend(_r)

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
        # captures live in /dev/shm only as long as the decoder refcount is
        # non-zero; if the prior process was killed (OOM, Ctrl-C, etc.) those
        # files have no live refcount and are dead weight on tmpfs.  At a fresh
        # start nothing is in flight, so any pre-existing file is by definition
        # orphaned and safe to unlink.
        try:
            for _f in os.listdir(export_dir):
                _p = os.path.join(export_dir, _f)
                if os.path.isfile(_p):
                    try:
                        os.unlink(_p)
                    except OSError:
                        pass
        except OSError:
            pass

        self._decoder = None
        # BOUNDED queue: each item carries a full wideband gate window (~300 MB
        # at 28 Msps).  An unbounded queue OOM-killed the process on a 60-msg
        # run when the save-worker fell behind (queue hit 129 → ~40 GB).  With
        # a small maxsize the detector blocks (backpressure) instead of growing
        # memory without bound.  For offline file replay this only slows the
        # detector (no data lost); for live it caps memory at maxsize×window.
        self._save_queue = queue.Queue(maxsize=8)
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

    def set_decoder(self, decoder):
        self._decoder = decoder

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
                    self._save_queue.task_done()
                    continue

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
                            _batch_t0 + float(_d.get('preamble_t_s', 0.0))))
                    if len(self._img_carriers) > 4000:
                        _cut = _batch_t0 - 3.0
                        self._img_carriers = [
                            c for c in self._img_carriers if c[2] > _cut]

                sym_time = (2 ** sf) / bw
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
                extended = extended_full[:max_record_n]
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
                export_bw = min(int(bw * _exp_dec / 2), self.wb_fs // 2)

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
                nb_fs = float(export_bw * 2)
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
                _dets_sorted = sorted(detections,
                                      key=lambda d: -d.get('peak_power_db', 0.0))
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
                        'time_sample': 0,
                        'pmr_db':      float(d.get('bw_quality_db', 0.0)),
                        'peak_pwr_db': d_pwr,
                        'status':      'LOCK',
                        'preamble_t_s': float(d.get('preamble_t_s', 0.0)),
                    })
                if not preambles:
                    self._save_queue.task_done()
                    continue
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
                    _match = 0
                    with self._saved_abs_t_lock:
                        for _sf_hz, _st in self._saved_abs_t:
                            if (abs(_cand_freq - _sf_hz) < _ftol
                                    and abs(_abs_t_s - _st) < _dt_win):
                                _match += 1
                        if _match < self._dedup_keep:
                            self._saved_abs_t.append((_cand_freq, _abs_t_s))
                            if len(self._saved_abs_t) > 4000:
                                _cut = _abs_t_s - 10.0
                                self._saved_abs_t = [
                                    (f, t) for f, t in self._saved_abs_t
                                    if t > _cut]
                    if _match >= self._dedup_keep:
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
                    if _mir_pwr > _cand_pwr + 10.0:
                        if self.debug >= 1:
                            print(f"         [IMG-REJECT] {_cand_freq / 1e6:.4f}"
                                  f"MHz ({_cand_pwr:.0f}dB) mirror of "
                                  f"{_mirror_f / 1e6:.4f}MHz ({_mir_pwr:.0f}dB)",
                                  flush=True)
                        continue

                    # Build a per-preamble synthetic detection dict so the
                    # rest of the save-worker (POSTDEDUP, filename, decoder
                    # submission) works unchanged.
                    d = dict(anchor)
                    d['freq_hz'] = anchor['freq_hz'] + carrier_hz
                    off = d['freq_hz'] - self.center_hz
                    # Re-extract narrowband centred precisely on this
                    # preamble's carrier — reuses cached forward FFT.
                    nb, nb_fs = extract_narrowband_fft(
                        extended, self.wb_fs, off, export_bw,
                        fft_cache=_ext_fft_cache)
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
                    fname = (f"SF{d['sf']}_{fmt_bw(d['bw'])}"
                             f"_{d['freq_hz'] / 1e6:.4f}MHz"
                             f"_{int(round(nb_fs / 1000))}ksps"
                             f"_pwr{int(round(d.get('peak_power_db', 0.0)))}"
                             f"_{dt_str}.cf32")
                    fpath = os.path.join(self.export_dir, fname)
                    nb.astype(np.complex64).tofile(fpath)
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
                        # Pass 1 — full-file primary with SIC: SC-scans the
                        # buffer, decodes the strongest preamble, subtracts
                        # its reconstructed signal, and re-searches the
                        # residual.  Catches multi-packet files when SIC
                        # cancellation is clean (per-symbol amplitude OK).
                        self._decoder.submit(fpath, fname, bucket_key=_bkey)
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
                                bucket_key=_bkey)
            except Exception as e:
                print(f"[RECORDER] save error: {e}", file=sys.stderr, flush=True)
            finally:
                self._save_queue.task_done()

    def _extract_nb(self, iq, offset_hz, target_bw):
        return extract_narrowband_fft(iq, self.wb_fs, offset_hz, target_bw)

    def flush(self):
        """Wait for all pending saves to complete."""
        self._save_queue.join()

    def reset_prev(self):
        """No-op — prev_hop no longer used in recordings."""
        pass


class BackgroundDecoder:
    """Decode captured .cf32 files in a background thread.

    Uses a PERSISTENT worker subprocess to avoid re-paying Python startup
    + numpy import costs.  The worker stays alive between decodes.
    """

    def __init__(self, aes_key=None, no_key=False, verbose=False):
        self._verbose = verbose
        self._aes_key = aes_key
        self._no_key = no_key
        self._queue = queue.Queue()
        self._lock = threading.Lock()
        # Structured packet log (JSONL) the web UI tails.  Each decoded/encrypted
        # [PKT] record from the workers is appended here with a receive timestamp.
        # Set LORA_PKT_LOG to '' to disable.
        self._pkt_log_path = os.environ.get('LORA_PKT_LOG', '/tmp/lora_packets.jsonl')
        self._pkt_log_lock = threading.Lock()

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

    def submit(self, fpath, fname, relay_after_syms=None, relay_before_syms=None,
               bucket_key=None):
        """Queue a capture file for background decoding.

        relay_after_syms:  blank first N BW-rate symbols → find hop AFTER primary.
        relay_before_syms: blank from symbol N onward   → find hop BEFORE primary.
        bucket_key:        (cand_freq_hz, abs_t_s) for Phase-2 cross-worker
                           bucket dedup; None disables the dedup for this job.
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
        self._queue.put((fpath, fname, relay_after_syms, relay_before_syms, None))

    def pending(self):
        with self._lock:
            return (self._queue.qsize() + self._slow_queue.qsize()
                    + self._active_count)

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
        with self._lock:
            if self._active_count != 0:
                return
        try:
            self._queue.put(self._slow_queue.get_nowait())
        except queue.Empty:
            pass

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
            fpath, fname, relay_after_syms, relay_before_syms, job_budget = job
            if self._decoder_script is None:
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
                self._ref_dec_and_maybe_unlink(fpath)
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
                        self._ref_dec_and_maybe_unlink(fpath)
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
                    self._queue.task_done()
                    continue

            with self._lock:
                self._active_count += 1

            t_start = time.time()
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
                while time.time() < deadline:
                    line = worker.stdout.readline()
                    if not line:
                        worker = None
                        break
                    line = line.rstrip('\n')
                    if line == '__END__':
                        timed_out = False
                        break
                    if line.startswith('__BEGIN__'):
                        continue
                    output_lines.append(line)

                if timed_out:
                    # Worker didn't finish within deadline; kill to avoid desync.
                    output_lines.append('ERROR: DECODE TIMEOUT (90s)')
                    try:
                        if worker:
                            worker.kill()
                    except Exception:
                        pass
                    worker = None

                output = '\n'.join(output_lines)
                elapsed = time.time() - t_start

                # Structured packet log: append each [PKT] record (decoded or
                # encrypted-header-only) with a receive timestamp to the JSONL the
                # web UI tails.  The web dedups by (pktid,hop); here we just append.
                if self._pkt_log_path and '[PKT]' in output:
                    import json as _json
                    _now = time.time()
                    _out = []
                    for _ln in output.split('\n'):
                        if _ln.startswith('[PKT] '):
                            try:
                                _r = _json.loads(_ln[6:])
                                _r['ts'] = _now
                                _out.append(_json.dumps(_r, separators=(',', ':')))
                            except Exception:
                                pass
                    if _out:
                        try:
                            with self._pkt_log_lock:
                                with open(self._pkt_log_path, 'a') as _pf:
                                    _pf.write('\n'.join(_out) + '\n')
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
                if (job_budget is None and output and '[BUDGET]' in output
                        and relay_after_syms is None
                        and relay_before_syms is None):
                    self._ref_inc(fpath)
                    self._slow_queue.put((fpath, fname, None, None,
                                          self._slow_budget))

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
                with self._lock:
                    self._active_count -= 1
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

        # Secondary pass: suppress all error/failure output — finding nothing is normal.
        _crc_bad = (not crc_line
                    or 'FAIL' in (crc_line or '')
                    or 'not present' in (crc_line or ''))
        if _is_secondary and _crc_bad and not mesh_lines:
            return ''

        # If a later attempt produced a real CRC result, prefer it over an
        # early-attempt error (e.g. HEADER CHECKSUM FAILED from attempt 0
        # shouldn't mask a CRC FAIL from attempt 3 that got further).
        if error and not crc_line and not mesh_lines:
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

        if not crc_line and not mesh_lines:
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
        _crc_ok_for_fp = crc_line and 'OK' in crc_line and 'FAIL' not in crc_line
        if _crc_ok_for_fp and payload_hex_line and _pkt_id is None:
            _m_fp = _re.search(r'Payload \(\d+ bytes\): ([0-9a-fA-F]+)', payload_hex_line)
            if _m_fp:
                try:
                    _raw_fp = bytes.fromhex(_m_fp.group(1))
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


def main():
    p = argparse.ArgumentParser(description='LoRa Schmidl-Cox Detector')
    p.add_argument('-r', '--rate', type=int, default=40_000_000)
    p.add_argument('-b', '--bandwidth', type=int, default=28_000_000)
    p.add_argument('-c', '--center', type=float, default=915.0)
    p.add_argument('-t', '--format', default='sc16', choices=['sc8', 'sc16'])
    p.add_argument('-f', '--file', default=None)
    p.add_argument('--window', type=float, default=1.0)
    p.add_argument('--overlap', type=float, default=0.1)
    p.add_argument('--threshold', type=float, default=0.7,
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
    p.add_argument('--buf-seconds', type=float, default=6.0,
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
    if a.detect_workers is not None and a.detect_workers < 0:
        # Use affinity-aware count (see comment near the decode worker block).
        try:
            _ncpu_auto = len(os.sched_getaffinity(0))
        except (AttributeError, OSError):
            _ncpu_auto = os.cpu_count() or 4
        a.detect_workers = max(2, min(8, _ncpu_auto // 4))
        print(f"Detect workers: AUTO = {a.detect_workers} (cpu_count={_ncpu_auto})",
              flush=True)

    _pool = None
    if a.detect_workers and a.detect_workers > 0:
        from detect_pool import DetectPool
        _win_n = int(a.rate * a.window)
        _pool = DetectPool(
            n_workers=a.detect_workers, n_slots=a.detect_workers + 4,
            win_n=_win_n,
            params=dict(wb_fs=a.rate, wb_bw=a.bandwidth, center=a.center,
                        sc_threshold=a.threshold, ethresh=a.energy_threshold,
                        dc_notch=a.dc_notch, spur_notch=_spur_notch_hz or None))
        print(f"Detect pool: {a.detect_workers} worker processes "
              f"({a.detect_workers + 4} slots)", flush=True)

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
    pre_hop = None   # wideband data immediately before buf — for recording lookback
    # Tail samples consumed last iter (for save).  Prepended to this iter's iq
    # so the audio timeline stays continuous — without this, every save creates
    # a `tail_n` audio gap between adjacent Welch windows and packets that land
    # in that gap are never detected.
    _carry_tail = None
    buf_pos = tot_s = tot_d = wc = 0
    # Worst-case preamble duration across all Meshtastic presets:
    #   SF12/62.5k = 16×4096/62500 = 1.049s  →  not capturable within hop anyway
    #   SF12/125k  = 16×4096/125000 = 0.524s  ← needs most lookback in-window
    #   SF11/125k  = 16×2048/125000 = 0.262s
    #   SF11/250k  = 0.131s, SF7/500k = 0.004s
    # Cap pre_hop at MAX_PRE_HOP_S: 0.6s covers SF12/125k preamble with ~15%
    # margin.  Save file grows from ~1.4s to ~1.9s at 20Msps but decode succeeds.
    _MAX_PRE_HOP_S = 0.6
    _max_pre_hop_n = min(hop_n, int(_MAX_PRE_HOP_S * a.rate))
    tot_skip = 0
    _warned_slow = False   # keep-up monitor: warn once if the gate can't sustain rate
    t_start = time.time()
    center_hz = a.center * 1e6

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

        decoder = BackgroundDecoder(
            aes_key=aes_key, no_key=no_key, verbose=a.decode_verbose)
        mode = "verbose" if a.decode_verbose else "compact"
        key_info = "NOKEY" if no_key else ("custom" if aes_key else "default")
        print(f"Decode: {mode} (key={key_info})", file=sys.stderr)

    if recorder and decoder:
        recorder.set_decoder(decoder)

    # Multiprocess detection pipeline state (the pool itself was built earlier,
    # before threads started, so its fork is safe).  With --detect-workers N>0
    # each gate window is fanned out to N single-threaded detect processes
    # (identical results) via shared memory and committed (print +
    # recorder.update) in window order with a lag.  Each capture's forward
    # "tail" is reconstructed from the next in-flight window (windows overlap
    # 50%), so no post-detect ring read is needed.
    _inflight = __import__('collections').deque()
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
        _need = int((16 + 4.25 + 8 + 120) * ((2 ** _a0['sf']) / _a0['bw']) * a.rate)
        return -(-_need // _tail_seg_n)   # ceil

    def _commit_oldest():
        """Pop the oldest in-flight window, wait for its detection, print and
        hand it to the recorder.  Tail = the next window's forward overlap."""
        nonlocal tot_d
        it = _inflight.popleft()
        dets0 = _pool.result(it['seq'])
        _elapsed = time.time() - t_start
        for d in dets0:
            tot_d += 1
            _abst = it['tot_s'] / a.rate + d.get('preamble_t_s', 0.0)
            _bwq = (f" bwq={d['bw_quality_db']:.0f}dB abst={_abst:.2f}s"
                    if a.debug >= 1 else "")
            print(f"[{_elapsed:6.1f}s] DETECTED freq={d['freq_mhz']:.4f}MHz "
                  f"SF={d['sf']} BW={fmt_bw(d['bw'])} sc={d['detect_conf']:.2f} "
                  f"pwr={d['peak_power_db']:.1f}dB{_bwq}", flush=True)
        if recorder:
            buf0 = _pool.slot_array(it['slot'])[:win_n]
            tail0 = None
            if _inflight and dets0:
                # Tail must cover the FULL packet AFTER the preamble.  A single
                # next-window overlap (~0.5 s) is enough for short-SF packets,
                # but long ones run past it and lose their payload: SF11/125
                # packets are ~1.4 s and SF12/125 ~2.9 s, so a 2 s capture
                # truncates the payload and the decoder yields only the header
                # plus a handful of nibbles (LONG_MODERATE was 38/120 for this
                # reason).  Concatenate the forward-overlap (hop_n:) of as many
                # subsequent in-flight windows as needed to span max_pkt_s for
                # the detected SF/BW.  Short-SF presets need only 1 window, so
                # their behaviour is unchanged.
                _a0 = max(dets0, key=lambda x: x.get('peak_power_db', 0.0))
                _sym_t0 = (2 ** _a0['sf']) / _a0['bw']
                _need_tail_n = int((16 + 4.25 + 8 + 120) * _sym_t0 * a.rate)
                _tparts = []
                _acc = 0
                for _itn in _inflight:
                    _seg = _pool.slot_array(_itn['slot'])[:win_n][hop_n:]
                    _tparts.append(_seg.copy())
                    _acc += len(_seg)
                    if _acc >= _need_tail_n:
                        break
                if _tparts:
                    tail0 = np.concatenate(_tparts)
            recorder.update(dets0, buf0, it['tot_s'], tail=tail0,
                            pre_hop=it['pre_hop'])
        _pool.release_slot(it['slot'])

    try:
      _prof = {'read': 0.0, 'slide': 0.0, 'welch': 0.0, 'notch': 0.0,
               'sat': 0.0, 'detect': 0.0, 'tail': 0.0, 'recorder': 0.0,
               'catchup': 0.0, 'n': 0}
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

      def _notch_psd(_pp):
          if a.dc_notch > 0:
              _fr = a.rate / 4096; _db = max(1, int(round(a.dc_notch * 1e6 / _fr))); _cc = 4096 // 2
              _pp[max(0, _cc - _db):min(4096, _cc + _db + 1)] = np.median(_pp)
          if _spur_notch_hz:
              _fr2 = a.rate / 4096
              for _sf, _hw in _spur_notch_hz:
                  _sb = 4096 // 2 + int(round((_sf - a.center * 1e6) / _fr2))
                  _nb = max(1, int(round(_hw / _fr2)))
                  _pp[max(0, _sb - _nb):min(4096, _sb + _nb + 1)] = np.median(_pp)
      while True:
        _t_step = time.time()
        # ---- Read hop_n samples ----
        # If last iter consumed tail samples (for the save), prepend them
        # here and read fewer fresh samples — keeps the audio timeline
        # contiguous and prevents 38ms-per-iter cumulative gap.
        carry_n = len(_carry_tail) if _carry_tail is not None else 0
        fresh_n = max(0, hop_n - carry_n)
        if is_live:
            if fresh_n > 0:
                result = reader.read(fresh_n)
                if result[0] is None:
                    break
                iq_fresh, skipped = result
            else:
                iq_fresh, skipped = np.zeros(0, dtype=np.complex64), 0

            if skipped > 0:
                # Temporal discontinuity — reset state
                tot_skip += skipped
                skip_s = skipped / a.rate
                elapsed = time.time() - t_start
                print(f"[{elapsed:6.1f}s] SKIP {skipped/1e6:.1f}M samples "
                      f"({skip_s:.2f}s) — detection took too long, "
                      f"ring buffer wrapped", file=sys.stderr)
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
                buf = np.zeros(win_n, dtype=np.complex64)
                pre_hop = None
                _carry_tail = None
                carry_n = 0
                if recorder:
                    recorder.reset_prev()
        else:
            if fresh_n > 0:
                iq_fresh = reader.read(fresh_n)
                if iq_fresh is None:
                    break
            else:
                iq_fresh = np.zeros(0, dtype=np.complex64)

        if _carry_tail is not None and len(_carry_tail) > 0:
            iq = np.concatenate([_carry_tail, iq_fresh]) if len(iq_fresh) > 0 else _carry_tail
        else:
            iq = iq_fresh
        _carry_tail = None

        tot_s += len(iq)
        if len(iq) >= win_n:
            pre_hop = None   # full replacement — no valid lookback
            buf = iq[-win_n:]
        else:
            sh = win_n - len(iq)
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
                pre_hop = buf[hop_n - _max_pre_hop_n:hop_n].copy()
            buf[:sh] = buf[len(iq):]
            buf[sh:] = iq
        buf_pos += len(iq)
        if buf_pos < win_n:
            continue
        _prof['read'] += time.time() - _t_step
        _t_step = time.time()

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
            _psd_gate, _psd_gmax = welch_psd(buf, nfft=4096, n_avg=50, also_max=True)
        else:
            _psd_gate, _psd_gmax = welch_psd(buf, nfft=4096, n_avg=50), None
        if a.dc_notch > 0:
            _fres = a.rate / 4096
            _dc_bins = max(1, int(round(a.dc_notch * 1e6 / _fres)))
            _dc_c = 4096 // 2
            _psd_gate[max(0, _dc_c - _dc_bins):min(4096, _dc_c + _dc_bins + 1)] = np.median(_psd_gate)
        if _spur_notch_hz:
            _fres2 = a.rate / 4096
            for _sf, _hw in _spur_notch_hz:
                _sb = 4096 // 2 + int(round((_sf - a.center * 1e6) / _fres2))
                _nb = max(1, int(round(_hw / _fres2)))
                _psd_gate[max(0, _sb - _nb):min(4096, _sb + _nb + 1)] = np.median(_psd_gate)
        _prof['welch'] += time.time() - _t_step
        _t_step = time.time()
        if _psd_file and (time.time() - _psd_last) >= (1.0 / _psd_fps):
            _psd_last = time.time()
            if not (_psd_off and os.path.exists(_psd_off)):
                _emit_psd_frame(_psd_file, _psd_gate)
        _gate_peaks = find_peaks(_psd_gate, thresh_db=a.energy_threshold)
        if _psd_gmax is not None:
            _notch_psd(_psd_gmax)
            for _mh in find_peaks(_psd_gmax, thresh_db=a.energy_threshold):
                if not any(abs(_mh[0] - _g[0]) < 15 for _g in _gate_peaks):
                    _gate_peaks.append(_mh)

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
        _SHORT_WIN_S = 0.100
        _SHORT_OVERLAP = 0.5
        _SHORT_N_AVG = 10
        _short_n = int(_SHORT_WIN_S * a.rate)
        _short_step = max(1, int(_short_n * (1 - _SHORT_OVERLAP)))
        _short_peaks_all = []
        _n_short_slices = max(0, (len(buf) - _short_n) // _short_step + 1)
        for _si in range(_n_short_slices):
            _ss = _si * _short_step
            _seg = buf[_ss:_ss + _short_n]
            if _maxhold:
                _p_short, _p_short_max = welch_psd(_seg, nfft=4096, n_avg=_SHORT_N_AVG, also_max=True)
            else:
                _p_short, _p_short_max = welch_psd(_seg, nfft=4096, n_avg=_SHORT_N_AVG), None
            if a.dc_notch > 0:
                _p_short[max(0, _dc_c - _dc_bins):min(4096, _dc_c + _dc_bins + 1)] = np.median(_p_short)
            if _spur_notch_hz:
                _fres_short = a.rate / 4096
                for _sf, _hw in _spur_notch_hz:
                    _sb_s = 4096 // 2 + int(round((_sf - a.center * 1e6) / _fres_short))
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
            for _bin, _w, _db in find_peaks(_p_short, thresh_db=a.energy_threshold):
                if _w >= 30:
                    _short_peaks_all.append((_bin, _w, _db))
            if _p_short_max is not None:
                _notch_psd(_p_short_max)
                for _bin, _w, _db in find_peaks(_p_short_max, thresh_db=a.energy_threshold):
                    if _w >= 30:
                        _short_peaks_all.append((_bin, _w, _db))
        # Dedup by bin proximity (±15 bins ≈ ±100kHz) — narrower than 30 bins
        # so adjacent-channel hop0/hop1 pairs (commonly ±175 kHz apart) are
        # not collapsed into a single candidate. Sort by power desc so the
        # strongest survives within a cluster.
        _DEDUP_BINS = 15
        _short_peaks_all.sort(key=lambda p: -p[2])
        for _sp in _short_peaks_all:
            if not any(abs(_sp[0] - _mp[0]) < _DEDUP_BINS for _mp in _gate_peaks):
                _gate_peaks.append(_sp)

        has_energy = len(_gate_peaks) > 0
        # No signal in this window → the gate just did light work and is caught
        # up.  Use that real-time idle moment to let the decoder re-decode ONE
        # deferred straggler (big budget).  Driven by the gate's own idle state,
        # so it self-adapts to traffic density with no idle-time threshold.
        if not has_energy and decoder is not None:
            decoder.maybe_release_straggler()
        _prof['notch'] += time.time() - _t_step
        _t_step = time.time()

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
        _clip_thresh = 0.9 * (32767.0 / _norm_scale)   # 90 % of ADC full-scale
        # Saturation check: just need to know if a meaningful fraction of
        # samples is clipping.  Computing np.percentile on 28 M complex samples
        # was ~250 ms per iteration — over a quarter of the hop budget.  We
        # subsample by 32× (give us ~875 k samples, plenty for a statistical
        # estimate) and use np.abs() on the subsample only.  Total < 8 ms.
        _sub = buf[::32]
        _abs_sub = np.abs(_sub)
        _peak_amp = float(_abs_sub.max())
        _sat_frac = float(np.count_nonzero(_abs_sub > _clip_thresh)) / len(_abs_sub)
        _spur_db  = a.spur_reject
        if _sat_frac > 0.005:   # > 0.5 % of samples clipping
            # Saturation creates harmonic distortion that looks like extra
            # peaks.  Engage real spur rejection (15 dB base + up to 20 dB
            # extra scaled with clip rate, so 5 % clip → 35 dB total).  When
            # NOT saturating, spur rejection stays at the (very high) default
            # so legitimate weak LoRa peaks aren't dropped just because a
            # stronger one is present in the same window.
            _spur_db = 15.0 + min(20.0, _sat_frac * 300.0)
            if a.debug >= 1:
                print(f"  [SAT] peak={_peak_amp:.3f}  clipped={_sat_frac*100:.2f}%  "
                      f"spur_reject={a.spur_reject:.0f}→{_spur_db:.0f}dB",
                      file=sys.stderr, flush=True)

        _prof['sat'] += time.time() - _t_step
        _t_step = time.time()
        # ---- Detect ----
        # SC buffer = buf only (no pre_hop concat).  Using buf (1s / 28M samples)
        # with a shared FFT cache (chunk=65536) costs ~228ms one-time + ~8ms/peak,
        # well under the ~1273ms hop budget even with 5 simultaneous peaks.
        # pre_hop is still passed to recorder.update for full-preamble capture.
        # SF12/BW31.5kHz preambles (1040ms) that straddle the window boundary may
        # occasionally be missed; all other SF/BW combinations fit entirely within
        # the 1s window (longest: SF12/BW125k = 262ms preamble).
        wc += 1; tw = time.time()
        if _pool is not None:
            # --- Multiprocess detect: dispatch this window, commit lagged ---
            # The producer never blocks on detection; workers run it in
            # parallel.  The capture's forward tail is reconstructed in
            # _commit_oldest from the next in-flight window's overlap, so we
            # do NOT read a tail from the ring here.
            while _pool.n_free() == 0:
                _commit_oldest()
            _slot = _pool.acquire_slot()
            _pool.slot_array(_slot)[:len(buf)] = buf
            _pool.dispatch(_slot, _seq, len(buf), _psd_gate, _gate_peaks,
                           spur_db=_spur_db)
            _inflight.append({'seq': _seq, 'slot': _slot,
                              'pre_hop': pre_hop, 'tot_s': tot_s})
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
            _t_step = time.time()
        else:
            dets = detect_preamble(buf, a.rate, a.bandwidth, a.center,
                                   sc_threshold=a.threshold,
                                   ethresh=a.energy_threshold,
                                   spur_db=_spur_db, dc_notch_mhz=a.dc_notch,
                                   spur_notch_hz=_spur_notch_hz or None,
                                   debug=a.debug,
                                   cached_psd=_psd_gate,
                                   cached_peaks=_gate_peaks)
            dt = time.time() - tw; elapsed = time.time() - t_start
            _prof['detect'] += dt
            _t_step = time.time()

            # Cross-window duplicate suppression is intentionally removed.
            # Any frequency-bucket dedup at this stage cannot distinguish the
            # original transmission from a relay on the same channel (co-located
            # nodes differ by only 1-3 kHz, well under any bucket that reliably
            # catches same-signal repeats from overlapping windows).  The
            # post-decode PacketID dedup in BackgroundDecoder._compact() handles
            # same-packet duplicates correctly while allowing different hops_taken
            # values to pass through as distinct packets.

            for d in dets:
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
                # 50ms minimum wait for samples to settle in the ring buffer,
                # capped at 100ms total.
                _min_tail_n = int(0.05 * a.rate)
                _t_wait = time.time()
                while reader.available() < _min_tail_n:
                    if time.time() - _t_wait > 0.10:
                        break
                    time.sleep(0.005)
                _max_pkt_s = max(
                    (148.25 * (2 ** d['sf']) / d['bw']) for d in dets)
                _max_tail_n = min(int(_max_pkt_s * a.rate), 3 * win_n)
                avail = reader.available()
                tail_n = min(avail, _max_tail_n)
                if tail_n > 0:
                    tail_data, tail_skipped = reader.read(tail_n)
                    if tail_data is not None:
                        pre_tail = tail_data
                        tot_s += len(pre_tail)
                        _carry_tail = pre_tail
                    if tail_skipped > 0:
                        tot_skip += tail_skipped
                        skip_s = tail_skipped / a.rate
                        print(f"[{elapsed:6.1f}s] TAIL-SKIP {tail_skipped/1e6:.1f}M "
                              f"samples ({skip_s:.2f}s) during tail read",
                              file=sys.stderr)
            elif recorder and not is_live and dets:
                # File mode: size the tail to the actual frame duration.
                _max_pkt_s = max(
                    (148.25 * (2 ** d['sf']) / d['bw']) for d in dets)
                _max_tail_n = min(int(_max_pkt_s * a.rate), 3 * win_n)
                if _max_tail_n > 0:
                    tail_data = reader.read(_max_tail_n)
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
                    tot_skip += skipped_ahead
                    skip_s = skipped_ahead / a.rate
                    print(f"[{elapsed:6.1f}s] CATCHUP skip {skipped_ahead/1e6:.1f}M "
                          f"samples ({skip_s:.2f}s) — ring buffer near wrap",
                          file=sys.stderr)
                    buf_pos = 0
                    buf = np.zeros(win_n, dtype=np.complex64)
                    pre_hop = None  # stale after skip — reset for clean SC state
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
        # reader.drops grew.  If yes, set _gate_stress → workers sleep
        # briefly before next decode → DRAM bandwidth eases → gate recovers.
        # Self-clearing when drops stop growing.  Per-window checking made
        # things worse (event thrashing on drop-counter jitter).
        if (is_live and wc % 5 == 0 and recorder
                and getattr(recorder, '_decoder', None)):
            _drops_now = reader.drops
            _dec = recorder._decoder
            _prev_d = getattr(_dec, '_prev_drops_obs', 0)
            if _drops_now > _prev_d:
                _dec._gate_stress.set()
            else:
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
            if recorder and getattr(recorder, '_decoder', None):
                with recorder._decoder._lock:
                    active = recorder._decoder._active_count
            drops = reader.drops if is_live else 0
            print(f"[STAT] {elapsed:.1f}s | {msps:.1f}Msps | win={wc} det={tot_d} "
                  f"save_q={save_q} dec_q={dec_q} active={active} "
                  f"pipe={dt * 1000:.0f}ms"
                  + (f" drops={drops/1e6:.1f}M" if drops else ""),
                  file=sys.stderr)
            if _prof['n'] > 0:
                n = _prof['n']
                print(f"  PROF (avg ms over {n} wins): "
                      f"read={_prof['read']/n*1000:.0f} "
                      f"welch={_prof['welch']/n*1000:.0f} "
                      f"notch+peaks={_prof['notch']/n*1000:.0f} "
                      f"sat={_prof['sat']/n*1000:.0f} "
                      f"detect={_prof['detect']/n*1000:.0f} "
                      f"tail={_prof['tail']/n*1000:.0f} "
                      f"recorder={_prof['recorder']/n*1000:.0f}",
                      file=sys.stderr)
                for k in _prof: _prof[k] = 0
                _prof['n'] = 0

    except KeyboardInterrupt:
        pass

    # Drain any in-flight multiprocess-detect windows (commit them to the
    # recorder), then shut the detect pool down.
    if _pool is not None:
        while _inflight:
            _commit_oldest()
        _pool.close()

    # Drain pending saves first (they submit to decoder), then drain decoder
    if recorder:
        pending_saves = recorder._save_queue.qsize()
        if pending_saves > 0:
            print(f"\nWaiting for {pending_saves} pending save(s)...",
                  file=sys.stderr)
        recorder.flush()

    if decoder:
        # Capture has ended (EOF) — move the deferred straggler queue into the
        # fast queue so the workers re-decode them now (big budget), without
        # having competed with the realtime gate during capture.
        decoder.drain_slow_into_fast()
        if decoder.pending() > 0:
            print(f"\nWaiting for {decoder.pending()} pending decode(s)...",
                  file=sys.stderr)
            decoder.drain(timeout=3600.0)

    total_drops = reader.drops if is_live else 0
    print(f"\nDone: {time.time() - t_start:.1f}s, {tot_s} samples, {tot_d} detections"
          + (f" (decode={'on' if decoder else 'off'})" if decoder else "")
          + (f" skipped={tot_skip/1e6:.1f}M" if tot_skip else ""),
          file=sys.stderr)
    if a.file: fp.close()


if __name__ == '__main__':
    main()