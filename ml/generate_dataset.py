"""
generate_dataset.py — Generate synthetic training data for all Meshtastic configs.

Covers all 30 SF×BW combinations (SF7-12 × 31.25k/62.5k/125k/250k/500k).
Window duration auto-scales so low-BW signals produce enough spectrogram frames.

Usage:
  python generate_dataset.py --train 15000 --val 3000 --output ../data
"""

import numpy as np
import os
import json
import time
import argparse
from config import SF_LIST, BW_LIST, SF_TO_CLASS, nb_sample_rate, symbol_duration
from lora_synth import (LoRaParams, generate_lora_preamble, apply_cfo,
                         apply_multipath, apply_phase_noise, add_noise, compute_spectrogram)


NFFT = 256
HOP = 128
PATCH_SIZE = 128
MIN_SPEC_FRAMES = 48  # minimum spectrogram frames for a usable patch


def window_duration(sf, bw):
    """
    Compute window duration in seconds.
    Must be long enough for:
      a) 4+ chirp symbols (to see the pattern)
      b) enough spectrogram frames to fill 128×128 patch
    """
    sym_dur = symbol_duration(sf, bw)
    fs = nb_sample_rate(bw)
    min_samples = NFFT + MIN_SPEC_FRAMES * HOP  # ~6400 at nfft=256/hop=128
    min_dur = min_samples / fs
    return max(sym_dur * 4, min_dur, 0.05)  # at least 50ms


def generate_lora_patch(sf, bw, snr_db, rng):
    """Generate one spectrogram patch of a LoRa signal."""
    fs = nb_sample_rate(bw)
    dur = window_duration(sf, bw)
    n_samples = int(fs * dur)

    # Generate enough preamble chirps that it's always LONGER than the window.
    # This forces the model to learn from partial views — it never sees the
    # complete preamble, just like in live detection where we might catch
    # the middle or tail end of a transmission.
    sym_samples = int(round(fs / bw)) * (2 ** sf)
    min_chirps = max(16, (n_samples // sym_samples) + 8)

    params = LoRaParams(sf=sf, bw=bw, fs=fs, n_preamble=min_chirps)
    preamble = generate_lora_preamble(params)

    # Always slice a random window from the preamble — partial view every time
    max_start = max(0, len(preamble) - n_samples)
    s = rng.integers(0, max_start + 1)
    sig = preamble[s:s + n_samples]
    if len(sig) < n_samples:
        sig = np.pad(sig, (0, n_samples - len(sig)))

    # Domain randomization
    cfo_hz = rng.uniform(-0.15 * bw, 0.15 * bw)
    sig = apply_cfo(sig, cfo_hz, fs)
    if rng.random() > 0.3:
        sig = apply_multipath(sig, n_taps=rng.integers(2, 5), rng=rng)
    if rng.random() > 0.5:
        sig = apply_phase_noise(sig, std_deg=rng.uniform(0.5, 5.0), rng=rng)
    sig = add_noise(sig, snr_db, rng=rng)

    # Compute spectrogram and resize to patch
    spec = compute_spectrogram(sig, nfft=NFFT, hop=HOP)
    return resize_to_patch(spec)


def generate_noise_patch(bw, rng):
    """Generate one spectrogram patch of noise only."""
    fs = nb_sample_rate(bw)
    dur = window_duration(7, bw)  # use SF7 duration (shortest) — doesn't matter much
    n_samples = int(fs * dur)
    sig = 0.01 * (rng.standard_normal(n_samples) + 1j * rng.standard_normal(n_samples)).astype(np.complex64)
    spec = compute_spectrogram(sig, nfft=NFFT, hop=HOP)
    return resize_to_patch(spec)


def resize_to_patch(spec):
    """Resize spectrogram to PATCH_SIZE × PATCH_SIZE."""
    h, w = spec.shape
    ri = np.clip((np.arange(PATCH_SIZE) * h / PATCH_SIZE).astype(int), 0, h - 1)
    ci = np.clip((np.arange(PATCH_SIZE) * w / PATCH_SIZE).astype(int), 0, w - 1)
    return spec[np.ix_(ri, ci)].astype(np.float32)


def generate_dataset(n_samples, output_dir, seed=42):
    os.makedirs(output_dir, exist_ok=True)
    rng = np.random.default_rng(seed)

    # 20% noise, 80% LoRa — balanced across 6 SF classes
    n_noise = n_samples // 5
    n_lora = n_samples - n_noise

    n_sf_classes = len(SF_LIST)  # 6
    per_sf = n_lora // n_sf_classes
    remainder = n_lora - per_sf * n_sf_classes

    patches = np.zeros((n_samples, PATCH_SIZE, PATCH_SIZE), dtype=np.float32)
    labels = []

    print(f"Generating {n_lora} LoRa ({per_sf}/SF × {n_sf_classes} SFs) + {n_noise} noise = {n_samples}")
    print(f"  SF: {SF_LIST}")
    print(f"  BW: random per sample (all BWs produce identical spectrograms at same SF)")
    t0 = time.time()

    idx = 0

    # Noise examples
    for i in range(n_noise):
        if i % 500 == 0:
            print(f"  Noise: {i}/{n_noise}")
        bw = int(rng.choice(BW_LIST))
        patches[idx] = generate_noise_patch(bw=bw, rng=rng)
        labels.append({'has_lora': 0, 'sf': 0, 'bw': 0, 'snr_db': -99.0})
        idx += 1

    # LoRa examples — balanced across SFs, random BW each
    for sf_idx, sf in enumerate(SF_LIST):
        n_this = per_sf + (1 if sf_idx < remainder else 0)

        for i in range(n_this):
            if idx % 1000 == 0:
                elapsed = time.time() - t0
                rate = idx / elapsed if elapsed > 0 else 0
                eta = (n_samples - idx) / rate if rate > 0 else 0
                print(f"  [{idx}/{n_samples}] SF{sf} "
                      f"({rate:.0f}/s, ETA {eta:.0f}s)")

            bw = int(rng.choice(BW_LIST))
            snr_db = float(rng.uniform(-5, 30))
            patches[idx] = generate_lora_patch(sf=sf, bw=bw, snr_db=snr_db, rng=rng)
            labels.append({
                'has_lora': 1, 'sf': int(sf), 'bw': int(bw), 'snr_db': snr_db
            })
            idx += 1

    dt = time.time() - t0

    # Shuffle
    order = rng.permutation(n_samples)
    patches = patches[order]
    labels = [labels[i] for i in order]

    # Save
    np.savez_compressed(os.path.join(output_dir, 'patches.npz'), patches=patches)
    with open(os.path.join(output_dir, 'labels.json'), 'w') as f:
        json.dump(labels, f)

    print(f"\nSaved {n_samples} in {dt:.1f}s ({n_samples / dt:.0f}/s) → {output_dir}/")

    from collections import Counter
    ll = [l for l in labels if l['has_lora']]
    sf_d = Counter(l['sf'] for l in ll)
    print(f"  Per SF: {dict(sorted(sf_d.items()))}")
    snrs = [l['snr_db'] for l in ll]
    print(f"  SNR range: {min(snrs):.1f} to {max(snrs):.1f} dB")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--train', type=int, default=12000)
    p.add_argument('--val', type=int, default=2400)
    p.add_argument('--output', default='data')
    p.add_argument('--seed', type=int, default=42)
    a = p.parse_args()

    print("=== Training Data ===\n")
    generate_dataset(a.train, os.path.join(a.output, 'train'), seed=a.seed)

    print("\n\n=== Validation Data ===\n")
    generate_dataset(a.val, os.path.join(a.output, 'val'), seed=a.seed + 1000)
