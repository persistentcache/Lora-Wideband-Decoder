"""
add_real_noise.py — Replace synthetic noise with real RF environment noise.

Usage:
  python add_real_noise.py \
      --noise-files recordings/noise_001.raw recordings/noise_002.raw \
      --rate 1000000 \
      --dataset data/train --val-dataset data/val
"""

import numpy as np
import os, json, argparse, sys


def load_iq(filepath):
    raw = np.fromfile(filepath, dtype=np.int16)
    return (raw[0::2] + 1j * raw[1::2]).astype(np.complex64) / 2048.0


def compute_patch(iq, nfft=256, hop=128, psz=128):
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


def extract_noise_patches(filepath, rate, window_ms=150, psz=128):
    iq = load_iq(filepath)
    print(f"  {filepath}: {len(iq)/1e6:.1f}M samples ({len(iq)/rate:.1f}s)")
    window = int(rate * window_ms / 1000)
    hop = window // 2
    patches = []
    pos = 0
    while pos + window <= len(iq):
        p = compute_patch(iq[pos:pos + window], psz=psz)
        if p is not None: patches.append(p)
        pos += hop
    print(f"    Extracted {len(patches)} noise patches")
    return patches


def merge_noise(dataset_dir, real_noise, rng):
    pf = os.path.join(dataset_dir, 'patches.npz')
    lf = os.path.join(dataset_dir, 'labels.json')
    if not os.path.exists(pf):
        print(f"  WARNING: {dataset_dir} not found, skipping"); return

    patches = np.load(pf)['patches']
    with open(lf) as f: labels = json.load(f)

    lora_idx = [i for i, l in enumerate(labels) if l['has_lora']]
    old_noise = sum(1 for l in labels if not l['has_lora'])
    n_lora = len(lora_idx)

    n_new = min(len(real_noise), max(len(real_noise), n_lora // 4))
    if n_new < len(real_noise):
        sel = rng.choice(len(real_noise), n_new, replace=False)
        noise = [real_noise[i] for i in sel]
    else:
        noise = real_noise

    new_patches = np.zeros((n_lora + n_new, 128, 128), dtype=np.float32)
    new_labels = []
    for i, idx in enumerate(lora_idx):
        new_patches[i] = patches[idx]; new_labels.append(labels[idx])
    for i, p in enumerate(noise):
        new_patches[n_lora + i] = p
        new_labels.append({'has_lora': 0, 'sf': 0, 'bw': 0, 'snr_db': -99.0, 'source': 'real_noise'})

    order = rng.permutation(len(new_patches))
    new_patches = new_patches[order]
    new_labels = [new_labels[i] for i in order]

    np.savez_compressed(pf, patches=new_patches)
    with open(lf, 'w') as f: json.dump(new_labels, f)
    print(f"  {dataset_dir}: {old_noise} synthetic → {n_new} real noise "
          f"(kept {n_lora} LoRa, total {len(new_patches)})")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--noise-files', nargs='+', required=True)
    p.add_argument('--rate', type=int, required=True)
    p.add_argument('--dataset', default='data/train')
    p.add_argument('--val-dataset', default='data/val')
    p.add_argument('--seed', type=int, default=42)
    a = p.parse_args()

    rng = np.random.default_rng(a.seed)
    print("Extracting noise patches...")
    all_noise = []
    for f in a.noise_files:
        if not os.path.exists(f):
            print(f"  WARNING: {f} not found"); continue
        all_noise.extend(extract_noise_patches(f, a.rate))
    if not all_noise:
        print("ERROR: No noise patches"); sys.exit(1)

    rng.shuffle(all_noise)
    n_val = max(1, int(len(all_noise) * 0.15))
    print(f"\nTotal: {len(all_noise)} patches ({len(all_noise)-n_val} train, {n_val} val)")

    print("\nMerging into training dataset...")
    merge_noise(a.dataset, all_noise[n_val:], rng)
    print("Merging into validation dataset...")
    merge_noise(a.val_dataset, all_noise[:n_val], rng)

    print(f"\nDone! Retrain:")
    print(f"  python3 scripts/train.py --train {a.dataset} --val {a.val_dataset} \\")
    print(f"      --epochs 30 --checkpoint models/best.pth --export models/lora_detector.onnx")

if __name__ == '__main__': main()
