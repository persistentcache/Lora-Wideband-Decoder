"""
lora_model.py — Lightweight CNN for LoRa detection + SF classification.

~200K params, <1MB INT8, ~2ms inference per patch on CPU.
Heads:
  detect: 2 classes (no_signal, lora_present)
  SF:     7 classes (noise, SF7, SF8, SF9, SF10, SF11, SF12)

BW is determined by the detection pipeline, not the CNN. All BWs produce
identical spectrograms at the same SF (since fs=2×BW), so the CNN has no
discriminative features for bandwidth.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import N_DET_CLASSES, N_SF_CLASSES, CLASS_TO_SF


class DSConv(nn.Module):
    """Depthwise separable convolution."""
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, 3, stride, 1, groups=in_ch, bias=False)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        return F.relu(self.bn(self.pw(self.dw(x))), inplace=True)


class LoRaDetector(nn.Module):
    """Input: (B, 1, 128, 128) → detect + SF logits."""
    def __init__(self):
        super().__init__()
        self.conv0 = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        self.b1 = DSConv(32, 64, stride=2)
        self.b2 = DSConv(64, 128, stride=2)
        self.b3 = DSConv(128, 128, stride=2)
        self.b4 = DSConv(128, 256, stride=2)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(0.3)
        self.head_det = nn.Linear(256, N_DET_CLASSES)
        self.head_sf = nn.Linear(256, N_SF_CLASSES)

    def forward(self, x):
        x = self.b4(self.b3(self.b2(self.b1(self.conv0(x)))))
        x = self.drop(self.gap(x).flatten(1))
        return {'detect': self.head_det(x), 'sf': self.head_sf(x)}

    def predict(self, x):
        with torch.no_grad():
            out = self.forward(x)
            results = []
            for i in range(x.shape[0]):
                dc = torch.argmax(out['detect'][i]).item()
                sc = torch.argmax(out['sf'][i]).item()
                results.append({
                    'has_lora': dc == 1,
                    'sf': CLASS_TO_SF[sc],
                    'detect_conf': F.softmax(out['detect'][i], dim=0)[1].item(),
                    'sf_conf': F.softmax(out['sf'][i], dim=0)[sc].item(),
                })
            return results


def count_parameters(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def export_onnx(model, path='models/lora_detector.onnx'):
    model.eval()
    class W(nn.Module):
        def __init__(self, m): super().__init__(); self.m = m
        def forward(self, x):
            o = self.m(x); return o['detect'], o['sf']
    torch.onnx.export(W(model), torch.randn(1, 1, 128, 128), path,
        input_names=['spectrogram'], output_names=['detect', 'sf'],
        dynamic_axes={'spectrogram': {0: 'batch'}, 'detect': {0: 'batch'},
                      'sf': {0: 'batch'}}, opset_version=18)
    print(f"Exported ONNX: {path} ({count_parameters(model):,} params)")
