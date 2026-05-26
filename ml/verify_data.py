"""
verify_data.py — Inspect recordings and extracted patch datasets.

Usage:
  python verify_data.py --recording recordings/sf11_bw250k_close.raw --rate 1000000 --sf 11 --bw 250000
  python verify_data.py --dataset data/train --save-images verify_output/
"""

import numpy as np, os, json, argparse
from collections import Counter, defaultdict
from config import symbol_duration


def inspect_recording(filepath, rate, sf=0, bw=0):
    raw = np.fromfile(filepath, dtype=np.int16)
    iq = (raw[0::2] + 1j * raw[1::2]).astype(np.complex64) / 2048.0
    dur = len(iq) / rate
    print(f"\n{'='*60}\nRecording: {filepath}\n{'='*60}")
    print(f"  Samples: {len(iq):,}  Duration: {dur:.1f}s  Rate: {rate:,}")
    if sf: print(f"  Expected: SF{sf} BW{bw/1000:.2f}k")

    power = np.abs(iq) ** 2
    mean_p, peak_p = np.mean(power), np.max(power)
    dyn = 10 * np.log10(peak_p / (mean_p + 1e-15))
    print(f"  Power: mean={10*np.log10(mean_p+1e-15):.1f}dB peak={10*np.log10(peak_p+1e-15):.1f}dB dynamic={dyn:.1f}dB")
    if dyn < 3: print(f"  WARNING: Low dynamic range — signal may be absent")

    ew = max(256, rate // 4000)
    cs = np.cumsum(power); cs = np.insert(cs, 0, 0)
    env = (cs[ew:] - cs[:-ew]) / ew
    nf = np.median(env); above = env > nf * 3
    regions, in_r, start = [], False, 0
    mg = rate // 10
    for i in range(len(above)):
        if above[i] and not in_r: start = i; in_r = True
        elif not above[i] and in_r:
            if not np.any(above[i:min(len(above), i + mg)]):
                regions.append((start, i)); in_r = False
    if in_r: regions.append((start, len(above)))

    print(f"  Packets: {len(regions)}")
    if sf and bw: print(f"  Symbol: {symbol_duration(sf,bw)*1000:.1f}ms")
    for i, (s, e) in enumerate(regions[:8]):
        snr = 10 * np.log10(np.mean(env[s:e]) / (nf + 1e-15))
        print(f"    [{i}] {s/rate:.3f}-{e/rate:.3f}s ({(e-s)/rate*1000:.0f}ms) SNR~{snr:.1f}dB")
    if not regions: print(f"  → No packets! Check frequency/gain/transmitter.")
    elif len(regions) >= 3:
        snrs = [10*np.log10(np.mean(env[s:e])/(nf+1e-15)) for s, e in regions]
        print(f"  → GOOD: {len(regions)} packets, SNR {min(snrs):.0f}-{max(snrs):.0f}dB")


def inspect_dataset(dataset_dir, save_dir=None):
    pf = os.path.join(dataset_dir, 'patches.npz')
    if not os.path.exists(pf): print(f"ERROR: {pf} not found"); return
    patches = np.load(pf)['patches']
    with open(os.path.join(dataset_dir, 'labels.json')) as f: labels = json.load(f)

    print(f"\n{'='*60}\nDataset: {dataset_dir}\n{'='*60}")
    print(f"  Patches: {len(patches)}  Shape: {patches[0].shape}")
    lora = [l for l in labels if l['has_lora']]
    noise = [l for l in labels if not l['has_lora']]
    print(f"  LoRa: {len(lora)} ({len(lora)/len(labels)*100:.0f}%)  Noise: {len(noise)}")

    if lora:
        combo = Counter((l['sf'], l['bw']) for l in lora)
        print(f"\n  Per SF x BW:")
        for (sf, bw) in sorted(combo.keys()):
            n = combo[(sf, bw)]
            st = "OK" if n >= 50 else ("LOW" if n >= 20 else "NEED MORE")
            print(f"    SF{sf:2d} x {bw/1000:6.2f}k: {n:4d}  [{st}]")

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        by_cls = defaultdict(list)
        for i, l in enumerate(labels):
            k = f"sf{l['sf']}_bw{l['bw']}" if l['has_lora'] else "noise"
            by_cls[k].append(i)
        for cls, idxs in sorted(by_cls.items()):
            for j, idx in enumerate(idxs[:3]):
                img = (patches[idx] * 255).astype(np.uint8)
                fp = os.path.join(save_dir, f"{cls}_{j+1}.pgm")
                with open(fp, 'wb') as f:
                    f.write(f"P5\n{img.shape[1]} {img.shape[0]}\n255\n".encode())
                    f.write(img.tobytes())
        print(f"\n  Saved samples to {save_dir}/ — view with: feh {save_dir}/")
        print(f"  LoRa patches: diagonal lines. Noise: uniform texture.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--recording'); p.add_argument('--rate', type=int)
    p.add_argument('--sf', type=int, default=0); p.add_argument('--bw', type=int, default=0)
    p.add_argument('--dataset'); p.add_argument('--save-images')
    a = p.parse_args()
    if not a.recording and not a.dataset: p.print_help(); return
    if a.recording:
        if not a.rate: print("ERROR: --rate required"); return
        inspect_recording(a.recording, a.rate, a.sf, a.bw)
    if a.dataset: inspect_dataset(a.dataset, a.save_images)

if __name__ == '__main__': main()
