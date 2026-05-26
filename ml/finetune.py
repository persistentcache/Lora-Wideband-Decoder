"""
finetune.py — Fine-tune synthetic-pretrained model on real SDR captures.

Usage:
  python finetune.py --real-data data/finetune/train \
      --model models/best.pth --output models/finetuned.pth \
      --export models/lora_detector.onnx
"""

import os, json, argparse, numpy as np
import torch, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from config import SF_TO_CLASS
from lora_model import LoRaDetector, export_onnx
from lora_synth import compute_spectrogram


class RealCaptureDataset(Dataset):
    def __init__(self, data_dir, n_patches_per_file=10, psz=128):
        with open(os.path.join(data_dir, 'labels.json')) as f:
            self.file_labels = json.load(f)
        self.patches, self.labels = [], []
        for entry in self.file_labels:
            fp = os.path.join(data_dir, entry['file'])
            if not os.path.exists(fp): continue
            raw = np.fromfile(fp, dtype=np.int16)
            iq = (raw[0::2] + 1j * raw[1::2]).astype(np.complex64) / 2048.0
            wn = int(entry['sample_rate'] * 0.15)
            for _ in range(n_patches_per_file):
                s = np.random.randint(0, max(1, len(iq) - wn))
                chunk = iq[s:s + wn]
                if len(chunk) < 256: continue
                spec = compute_spectrogram(chunk, nfft=256, hop=128)
                h, w = spec.shape
                ri = np.clip((np.arange(psz) * h / psz).astype(int), 0, h - 1)
                ci = np.clip((np.arange(psz) * w / psz).astype(int), 0, w - 1)
                self.patches.append(spec[np.ix_(ri, ci)].astype(np.float32))
                self.labels.append({'has_lora': int(entry['has_lora']), 'sf': entry['sf']})
        print(f"Loaded {len(self.patches)} patches from {len(self.file_labels)} files")

    def __len__(self): return len(self.patches)
    def __getitem__(self, i):
        l = self.labels[i]
        return (torch.from_numpy(self.patches[i]).unsqueeze(0),
                1 if l['has_lora'] else 0,
                SF_TO_CLASS.get(l['sf'], 0))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--real-data', required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--output', default='models/finetuned.pth')
    p.add_argument('--export', default=None)
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--batch', type=int, default=32)
    a = p.parse_args()

    model = LoRaDetector()
    ckpt = torch.load(a.model, map_location='cpu', weights_only=True)
    model.load_state_dict(ckpt['model_state_dict'])
    ds = RealCaptureDataset(a.real_data)
    dl = DataLoader(ds, batch_size=a.batch, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)

    for ep in range(1, a.epochs + 1):
        model.train()
        tot, cor, n = 0, 0, 0
        for x, det, sf in dl:
            out = model(x)
            loss = (F.cross_entropy(out['detect'], det.long()) +
                    F.cross_entropy(out['sf'], sf.long()))
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(x)
            cor += (out['sf'].argmax(1) == sf).sum().item()
            n += len(x)
        print(f"  Epoch {ep}/{a.epochs}: loss={tot / n:.4f} sf_acc={cor / n:.1%}")

    os.makedirs(os.path.dirname(a.output) or '.', exist_ok=True)
    torch.save({'model_state_dict': model.state_dict()}, a.output)
    print(f"Saved: {a.output}")
    if a.export: export_onnx(model, a.export)


if __name__ == '__main__': main()
