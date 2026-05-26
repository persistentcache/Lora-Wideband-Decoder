"""
train.py — Train LoRa detector CNN on synthetic preamble data.

Trains detection (LoRa vs noise) and SF classification.
BW is not trained — it's determined by the detection pipeline.

Usage:
  python train.py --train ../data/train --val ../data/val --epochs 30 \
      --checkpoint ../models/best.pth --export ../models/lora_detector.onnx
"""

import os, json, time, argparse, numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from config import SF_TO_CLASS, CLASS_TO_SF
from lora_model import LoRaDetector, count_parameters, export_onnx


class LoRaDataset(Dataset):
    def __init__(self, d):
        self.patches = np.load(os.path.join(d, 'patches.npz'))['patches']
        with open(os.path.join(d, 'labels.json')) as f:
            self.labels = json.load(f)

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, i):
        l = self.labels[i]
        return (torch.from_numpy(self.patches[i]).unsqueeze(0),
                1 if l['has_lora'] else 0,
                SF_TO_CLASS.get(l['sf'], 0))


def run_epoch(model, loader, opt, device, train=True):
    model.train() if train else model.eval()
    tot_loss = cd = cs = n = 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for x, det, sf in loader:
            x = x.to(device)
            det = det.to(device, dtype=torch.long)
            sf = sf.to(device, dtype=torch.long)
            out = model(x)
            loss = (F.cross_entropy(out['detect'], det) +
                    F.cross_entropy(out['sf'], sf))
            if train:
                opt.zero_grad(); loss.backward(); opt.step()
            tot_loss += loss.item() * len(x)
            cd += (out['detect'].argmax(1) == det).sum().item()
            cs += (out['sf'].argmax(1) == sf).sum().item()
            n += len(x)
    return {'loss': tot_loss / n, 'det': cd / n, 'sf': cs / n}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--train', default='data/train')
    p.add_argument('--val', default='data/val')
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--batch', type=int, default=64)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--device', default='cpu')
    p.add_argument('--checkpoint', default='models/best.pth')
    p.add_argument('--export', default=None)
    a = p.parse_args()

    os.makedirs(os.path.dirname(a.checkpoint) or '.', exist_ok=True)
    device = torch.device(a.device)

    tr_ds, va_ds = LoRaDataset(a.train), LoRaDataset(a.val)
    print(f"Train: {len(tr_ds)}  Val: {len(va_ds)}")
    tr_dl = DataLoader(tr_ds, batch_size=a.batch, shuffle=True)
    va_dl = DataLoader(va_ds, batch_size=a.batch, shuffle=False)

    model = LoRaDetector().to(device)
    print(f"Model: {count_parameters(model):,} params")
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)

    best = 0.0
    print(f"\n{'Ep':>3} {'TrL':>6} {'TrDet':>6} {'TrSF':>6} | "
          f"{'VaL':>6} {'VaDet':>6} {'VaSF':>6} {'t':>4}")
    print("-" * 55)

    for ep in range(1, a.epochs + 1):
        t0 = time.time()
        tr = run_epoch(model, tr_dl, opt, device, train=True)
        va = run_epoch(model, va_dl, None, device, train=False)
        sched.step()
        dt = time.time() - t0
        print(f"{ep:3d} {tr['loss']:6.3f} {tr['det']:6.1%} {tr['sf']:6.1%} | "
              f"{va['loss']:6.3f} {va['det']:6.1%} {va['sf']:6.1%} {dt:4.1f}s")
        avg = (va['det'] + va['sf']) / 2
        if avg > best:
            best = avg
            torch.save({'epoch': ep, 'model_state_dict': model.state_dict(), 'val': va},
                       a.checkpoint)
            print(f"    → saved ({avg:.1%})")

    print(f"\nBest: {best:.1%}")
    if a.export:
        ckpt = torch.load(a.checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(ckpt['model_state_dict'])
        export_onnx(model, a.export)


if __name__ == '__main__':
    main()
