"""
extract_patches.py — Extract labeled spectrogram patches from real SDR recordings.

Finds packets via energy envelope, slices into spectrogram patches for training.

Usage:
  python extract_patches.py \
      --file recordings/sf11_bw250k_close_001.raw \
      --sf 11 --bw 250000 --rate 1000000 --output data/real

  python extract_patches.py \
      --file recordings/noise_1msps_001.raw \
      --noise --rate 1000000 --output data/real
"""

import numpy as np
import os, json, argparse, sys
from config import symbol_duration


NFFT = 256
HOP = 128
PSZ = 128


def load_iq(filepath):
    raw = np.fromfile(filepath, dtype=np.int16)
    iq = (raw[0::2] + 1j * raw[1::2]).astype(np.complex64) / 2048.0
    print(f"  Loaded {len(iq)/1e6:.1f}M samples from {filepath}")
    return iq


def compute_patch(iq, nfft=NFFT, hop=HOP, psz=PSZ):
    win = np.hanning(nfft).astype(np.float32)
    nf = max(1, (len(iq) - nfft) // hop)
    if nf < 4: return None
    starts = np.arange(nf) * hop
    idx = starts[:, None] + np.arange(nfft)[None, :]
    idx = np.clip(idx, 0, len(iq) - 1)
    F = np.fft.fftshift(np.fft.fft(iq[idx] * win[None, :], axis=1), axes=1)
    spec = (np.abs(F) ** 2).T.astype(np.float32)
    spec = 10.0 * np.log10(spec + 1e-12)
    vmin, vmax = np.percentile(spec, [2, 99.5])
    if vmax - vmin < 1: vmax = vmin + 1
    spec = np.clip((spec - vmin) / (vmax - vmin), 0, 1)
    h, w = spec.shape
    ri = np.clip((np.arange(psz) * h / psz).astype(int), 0, h - 1)
    ci = np.clip((np.arange(psz) * w / psz).astype(int), 0, w - 1)
    return spec[np.ix_(ri, ci)].astype(np.float32)


def find_active_regions(iq, fs, sf=11, bw=250000):
    sym_samp = max(256, int(symbol_duration(sf, bw) * fs) // 4)
    power = np.abs(iq) ** 2
    cs = np.cumsum(power); cs = np.insert(cs, 0, 0)
    env = (cs[sym_samp:] - cs[:-sym_samp]) / sym_samp

    nf = np.median(env)
    thresh = nf * 3.0
    above = env > thresh
    min_gap = int(symbol_duration(sf, bw) * fs * 2)

    regions = []
    in_r, start = False, 0
    for i in range(len(above)):
        if above[i] and not in_r: start = i; in_r = True
        elif not above[i] and in_r:
            ge = min(len(above), i + min_gap)
            if not np.any(above[i:ge]):
                regions.append((start, i)); in_r = False
    if in_r: regions.append((start, len(above)))
    return regions, nf, thresh


def extract(iq, sf, bw, fs, is_noise=False):
    results = []

    if is_noise:
        window = int(fs * 0.15)
        hop = window // 2
        n = (len(iq) - window) // hop
        print(f"  Noise: extracting {n} windows")
        for i in range(n):
            p = compute_patch(iq[i * hop:i * hop + window])
            if p is not None:
                results.append((p, {'has_lora': 0, 'sf': 0, 'bw': 0, 'snr_db': -99.0}))
        return results

    regions, nf, thresh = find_active_regions(iq, fs, sf, bw)
    print(f"  Found {len(regions)} active regions")

    if not regions:
        print(f"  WARNING: No signals found!")
        return results

    sym_samp = int(symbol_duration(sf, bw) * fs)
    window = max(2048, min(sym_samp * 4, int(fs * 0.5)))
    hop = window // 2

    for ri, (rs, re) in enumerate(regions):
        rlen = re - rs
        sig_env = np.mean(np.abs(iq[rs:re]) ** 2)
        snr = 10 * np.log10(sig_env / (nf + 1e-15)) if nf > 0 else 30.0

        n_patches = max(1, (rlen - window) // hop + 1)
        for p in range(n_patches):
            s = rs + p * hop
            if s + window > len(iq): break
            patch = compute_patch(iq[s:s + window])
            if patch is not None:
                results.append((patch, {
                    'has_lora': 1, 'sf': sf, 'bw': bw,
                    'snr_db': round(float(snr), 1), 'source': 'real',
                }))

        if ri < 3:
            print(f"    Region {ri}: {rlen/fs*1000:.0f}ms SNR~{snr:.1f}dB → {n_patches}p")

    # Quiet regions for noise patches
    quiet_pos = []
    if regions:
        if regions[0][0] > window: quiet_pos.append((0, regions[0][0]))
        for i in range(len(regions) - 1):
            if regions[i+1][0] - regions[i][1] > window:
                quiet_pos.append((regions[i][1], regions[i+1][0]))
        if len(iq) - regions[-1][1] > window:
            quiet_pos.append((regions[-1][1], len(iq)))

    n_noise = 0
    max_noise = len([r for r in results if r[1]['has_lora']]) // 3
    for qs, qe in quiet_pos:
        pos = qs
        while pos + window < qe and n_noise < max_noise:
            p = compute_patch(iq[pos:pos + window])
            if p is not None:
                results.append((p, {'has_lora': 0, 'sf': 0, 'bw': 0, 'snr_db': -99.0}))
                n_noise += 1
            pos += window

    lora_n = sum(1 for _, l in results if l['has_lora'])
    noise_n = sum(1 for _, l in results if not l['has_lora'])
    print(f"  Total: {lora_n} LoRa + {noise_n} noise")
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--file', required=True)
    p.add_argument('--sf', type=int, default=0)
    p.add_argument('--bw', type=int, default=0)
    p.add_argument('--rate', type=int, required=True)
    p.add_argument('--noise', action='store_true')
    p.add_argument('--output', default='data/real')
    a = p.parse_args()

    is_noise = a.noise or (a.sf == 0 and a.bw == 0)
    if not is_noise and (a.sf == 0 or a.bw == 0):
        print("ERROR: specify --sf and --bw, or use --noise"); sys.exit(1)

    iq = load_iq(a.file)
    results = extract(iq, a.sf, a.bw, a.rate, is_noise)
    if not results: print("No patches!"); return

    os.makedirs(a.output, exist_ok=True)
    pf = os.path.join(a.output, 'patches.npz')
    lf = os.path.join(a.output, 'labels.json')

    new_p = np.array([r[0] for r in results], dtype=np.float32)
    new_l = [r[1] for r in results]

    if os.path.exists(pf) and os.path.exists(lf):
        old_p = np.load(pf)['patches']
        with open(lf) as f: old_l = json.load(f)
        all_p = np.concatenate([old_p, new_p])
        all_l = old_l + new_l
        print(f"  Appended: {len(old_p)} → {len(all_p)}")
    else:
        all_p, all_l = new_p, new_l

    np.savez_compressed(pf, patches=all_p)
    with open(lf, 'w') as f: json.dump(all_l, f)
    print(f"  Saved → {a.output}/")


if __name__ == '__main__': main()
