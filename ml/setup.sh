#!/bin/bash
set -e
cd "$(dirname "$0")"
mkdir -p data models recordings

echo "============================================"
echo "  LoRa ML Detector — Setup"
echo "  Detects LoRa + classifies SF 7-12"
echo "  BW determined by detection pipeline"
echo "  Preamble-only training"
echo "============================================"
echo ""

echo "[1/3] Installing Python packages..."
if python3 -c "import torch" 2>/dev/null; then
    echo "  PyTorch already installed"
else
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu 2>/dev/null || \
    pip install torch torchvision 2>/dev/null || \
    pip install torch torchvision --break-system-packages --index-url https://download.pytorch.org/whl/cpu || \
    { echo "ERROR: Could not install PyTorch."; exit 1; }
fi
pip install onnxruntime numpy 2>/dev/null || \
pip install onnxruntime numpy --break-system-packages 2>/dev/null || true
python3 -c "import torch; print(f'  PyTorch:      {torch.__version__}')"
python3 -c "import onnxruntime; print(f'  ONNX Runtime: {onnxruntime.__version__}')"
echo ""

echo "[2/3] Generating synthetic preamble data..."
echo "  12,000 train + 2,400 val (6 SF classes + noise)"
echo ""
cd scripts
python3 generate_dataset.py --train 12000 --val 2400 --output ../data
echo ""

echo "[3/3] Training CNN (30 epochs)..."
echo ""
python3 train.py \
    --train ../data/train \
    --val ../data/val \
    --epochs 30 \
    --batch 64 \
    --lr 0.001 \
    --checkpoint ../models/best.pth \
    --export ../models/lora_detector.onnx
echo ""

cd ..
echo "============================================"
echo "  Setup complete!"
echo "============================================"
echo ""
echo "  Model: models/lora_detector.onnx"
echo ""
echo "  TEST:"
echo "    bladeRF-cli -e \""
echo "        set frequency rx 906875000; set samplerate rx 1000000;"
echo "        set bandwidth rx 1000000; set gain rx 40;"
echo "        rx config file=/dev/stdout format=bin n=0;"
echo "        rx start; rx wait\" \\"
echo "    | python3 scripts/lora_detect.py \\"
echo "        -r 1000000 -b 1000000 -c 906.875 \\"
echo "        --model models/lora_detector.onnx -t sc16 -d 2"
echo ""
