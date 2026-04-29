#!/bin/bash
# ──────────────────────────────────────────────
# REI-Bench: Auto-detect device and run baselines
# ──────────────────────────────────────────────
# Usage: bash run_all.sh
#        bash run_all.sh --max-examples 5   # Quick test
# ──────────────────────────────────────────────

set -e

EXTRA_ARGS="${@}"

echo "╔══════════════════════════════════════════════════╗"
echo "║     REI-Bench Baseline Evaluation Pipeline       ║"
echo "╚══════════════════════════════════════════════════╝"
echo ""

# Auto-detect device
DEVICE="cpu"
if command -v hl-smi &> /dev/null; then
    echo "✓ Detected Intel Gaudi HPU"
    DEVICE="hpu"
    hl-smi 2>/dev/null | head -5 || true
elif command -v nvidia-smi &> /dev/null; then
    echo "✓ Detected NVIDIA GPU"
    DEVICE="cuda"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
else
    echo "⚠ No GPU/HPU detected, using CPU (will be slow)"
fi

echo ""
echo "Device: $DEVICE"
echo ""

# Set HF token (must be set before running)
if [ -z "$HF_TOKEN" ]; then
    echo "ERROR: HF_TOKEN environment variable not set"
    echo "Run: export HF_TOKEN=your_huggingface_token"
    exit 1
fi

# Install requirements
echo "Installing requirements..."
pip install -r phase1/requirements.txt -q
if [ "$DEVICE" = "hpu" ]; then
    pip install optimum-habana -q 2>/dev/null || echo "Note: optimum-habana install skipped"
fi

# Step 1: Generate dataset
echo ""
echo "━━━ Step 1: Generating REI-Bench Dataset ━━━"
python -m phase1.scripts.generate_dataset

# Step 2: Run Llama-3.2-1B-Instruct
echo ""
echo "━━━ Step 2: Evaluating Llama-3.2-1B-Instruct ━━━"
python -m phase1.scripts.run_baseline \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --planning-mode full \
    $EXTRA_ARGS

# Step 3: Run Llama-3.2-3B-Instruct
echo ""
echo "━━━ Step 3: Evaluating Llama-3.2-3B-Instruct ━━━"
python -m phase1.scripts.run_baseline \
    --model meta-llama/Llama-3.2-3B-Instruct \
    --planning-mode full \
    $EXTRA_ARGS

# Step 4: Analyze errors
echo ""
echo "━━━ Step 4: Error Analysis ━━━"
python -m phase1.scripts.analyze_errors

echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║            All Evaluations Complete!              ║"
echo "║   Results saved in: phase1/results/              ║"
echo "╚══════════════════════════════════════════════════╝"
