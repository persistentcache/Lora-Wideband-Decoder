"""
lora_synth.py — Synthetic LoRa signal generator for ML training data.

Generates LoRa chirps with domain randomization across all Meshtastic configs:
  SF 7-12, BW 31.25k/62.5k/125k/250k/500k
  Impairments: CFO, multipath, phase noise, AWGN
"""

import numpy as np
from dataclasses import dataclass
from config import SF_LIST, BW_LIST, nb_sample_rate, symbol_duration


@dataclass
class LoRaParams:
    sf: int = 11
    bw: int = 250000
    fs: int = 500000       # narrowband sample rate (2×BW)
    sync_word: int = 0x2B  # Meshtastic default
    n_preamble: int = 16   # Meshtastic uses 16 symbols


def generate_upchirp(sf, bw, fs):
    """Generate one base up-chirp symbol."""
    N = 2 ** sf
    osf = int(round(fs / bw))
    sps = N * osf
    t = np.arange(sps, dtype=np.float64) / fs
    Ts = N / bw
    phase = 2.0 * np.pi * (-bw / 2.0 * t + bw / (2.0 * Ts) * t * t)
    return np.exp(1j * phase).astype(np.complex64)


def generate_downchirp(sf, bw, fs):
    return np.conj(generate_upchirp(sf, bw, fs))


def shift_chirp(base, symbol, N):
    osf = len(base) // N
    return np.roll(base, symbol * osf)


def generate_lora_preamble(params):
    """
    Generate LoRa preamble + sync word + SFD only. No data symbols.

    This is the detection target: 16 up-chirps + 2 sync chirps + 2.25 down-chirps.
    Identical for every packet at a given SF/BW/sync_word. The CNN learns to
    recognize this deterministic pattern rather than noisy random payload data.
    """
    sf, bw, fs, N = params.sf, params.bw, params.fs, 2 ** params.sf
    up = generate_upchirp(sf, bw, fs)
    dn = generate_downchirp(sf, bw, fs)

    syms = []
    # Preamble: 16 identical unmodulated up-chirps (parallel diagonal lines)
    for _ in range(params.n_preamble):
        syms.append(up.copy())

    # Sync word: 2 frequency-shifted up-chirps (shifted diagonal lines)
    scale = 1 << (sf - 5) if sf >= 5 else 1
    sw_hi = (params.sync_word >> 4) & 0xF
    sw_lo = params.sync_word & 0xF
    syms.append(shift_chirp(up, (sw_hi * scale) % N, N))
    syms.append(shift_chirp(up, (sw_lo * scale) % N, N))

    # SFD: 2.25 down-chirps (diagonal lines in opposite direction)
    syms.append(dn.copy())
    syms.append(dn.copy())
    syms.append(dn[:len(dn) // 4].copy())

    return np.concatenate(syms)


# ---- Impairments ----

def apply_cfo(sig, cfo_hz, fs):
    t = np.arange(len(sig), dtype=np.float64) / fs
    return sig * np.exp(1j * 2 * np.pi * cfo_hz * t).astype(np.complex64)


def apply_multipath(sig, n_taps=3, max_delay=5, rng=None):
    if rng is None: rng = np.random.default_rng()
    out = sig.copy()
    for _ in range(n_taps - 1):
        d = rng.integers(1, max_delay + 1)
        a = rng.uniform(0.1, 0.5) * np.exp(1j * rng.uniform(0, 2 * np.pi))
        r = np.zeros_like(sig)
        r[d:] = sig[:-d] * a
        out += r
    return out


def apply_phase_noise(sig, std_deg=2.0, rng=None):
    if rng is None: rng = np.random.default_rng()
    pw = np.cumsum(rng.normal(0, np.radians(std_deg), len(sig)))
    return sig * np.exp(1j * pw).astype(np.complex64)


def add_noise(sig, snr_db, rng=None):
    if rng is None: rng = np.random.default_rng()
    sp = np.mean(np.abs(sig) ** 2)
    np_ = sp / (10 ** (snr_db / 10))
    n = np.sqrt(np_ / 2) * (rng.standard_normal(len(sig)) + 1j * rng.standard_normal(len(sig)))
    return sig + n.astype(np.complex64)


# ---- Spectrogram ----

def compute_spectrogram(iq, nfft=256, hop=128):
    """Vectorized STFT → normalized log-power spectrogram."""
    win = np.hanning(nfft).astype(np.float32)
    n_frames = max(1, (len(iq) - nfft) // hop)
    if n_frames < 2:
        return np.zeros((nfft, 2), dtype=np.float32)

    starts = np.arange(n_frames) * hop
    idx = starts[:, None] + np.arange(nfft)[None, :]
    idx = np.clip(idx, 0, len(iq) - 1)
    frames = iq[idx] * win[None, :]
    F = np.fft.fftshift(np.fft.fft(frames, axis=1), axes=1)
    spec = (np.abs(F) ** 2).T.astype(np.float32)

    spec = 10.0 * np.log10(spec + 1e-12)
    vmin = np.percentile(spec, 2)
    vmax = np.percentile(spec, 99.5)
    if vmax - vmin < 1.0:
        vmax = vmin + 1.0
    return np.clip((spec - vmin) / (vmax - vmin), 0, 1)
