"""
split_dataset.py — Stratified train/val split.
Usage: python split_dataset.py --input data/real --val-frac 0.15
"""

import numpy as np, os, json, argparse
from collections import defaultdict

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True)
    p.add_argument('--val-frac', type=float, default=0.15)
    p.add_argument('--seed', type=int, default=42)
    a = p.parse_args()

    patches = np.load(os.path.join(a.input, 'patches.npz'))['patches']
    with open(os.path.join(a.input, 'labels.json')) as f: labels = json.load(f)
    print(f"Dataset: {len(patches)} patches")
    rng = np.random.default_rng(a.seed)

    groups = defaultdict(list)
    for i, l in enumerate(labels):
        groups[(l['has_lora'], l.get('sf', 0), l.get('bw', 0))].append(i)

    train_idx, val_idx = [], []
    for key in sorted(groups.keys()):
        idx = np.array(groups[key]); rng.shuffle(idx)
        nv = max(1, int(len(idx) * a.val_frac))
        val_idx.extend(idx[:nv].tolist()); train_idx.extend(idx[nv:].tolist())
        h, sf, bw = key
        name = f"SF{sf} BW{bw/1000:.2f}k" if h else "noise"
        print(f"  {name:22s}: {len(idx):4d} → {len(idx)-nv:4d} train + {nv:3d} val")

    rng.shuffle(train_idx); rng.shuffle(val_idx)
    for name, idxs in [('train', train_idx), ('val', val_idx)]:
        d = os.path.join(a.input, name); os.makedirs(d, exist_ok=True)
        np.savez_compressed(os.path.join(d, 'patches.npz'), patches=patches[idxs])
        with open(os.path.join(d, 'labels.json'), 'w') as f:
            json.dump([labels[i] for i in idxs], f)
        print(f"\n{name}: {len(idxs)} → {d}/")

if __name__ == '__main__': main()
