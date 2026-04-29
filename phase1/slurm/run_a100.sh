#!/bin/bash
#SBATCH --job-name=rei-bench-a100
#SBATCH --output=logs/rei_a100_%j.out
#SBATCH --error=logs/rei_a100_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00

# ──────────────────────────────────────────────
# REI-Bench Baseline Evaluation on NVIDIA A100
# ──────────────────────────────────────────────

echo "============================================"
echo "REI-Bench Evaluation - NVIDIA A100"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Date: $(date)"
echo "============================================"

# Navigate to project root
cd $SLURM_SUBMIT_DIR/..
mkdir -p logs

# Load modules (adjust for your cluster)
module load cuda/12.1 2>/dev/null || true
module load python/3.10 2>/dev/null || true

# Activate virtual environment if exists
if [ -d "venv" ]; then
    source venv/bin/activate
elif [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Install requirements
pip install -r phase1/requirements.txt 2>/dev/null

# Check GPU
echo ""
nvidia-smi
echo ""

# Set HuggingFace token (must be set before running)
if [ -z "$HF_TOKEN" ]; then
    echo "ERROR: HF_TOKEN environment variable not set"
    echo "Run: export HF_TOKEN=your_huggingface_token"
    exit 1
fi

# Step 1: Generate dataset
echo "Generating dataset..."
python -m phase1.scripts.generate_dataset

# Step 2: Run Llama-3.2-1B-Instruct
echo ""
echo "============================================"
echo "Running Llama-3.2-1B-Instruct..."
echo "============================================"
python -m phase1.scripts.run_baseline \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --planning-mode full

# Step 3: Run Llama-3.2-3B-Instruct
echo ""
echo "============================================"
echo "Running Llama-3.2-3B-Instruct..."
echo "============================================"
python -m phase1.scripts.run_baseline \
    --model meta-llama/Llama-3.2-3B-Instruct \
    --planning-mode full

# Step 4: Analyze errors
echo ""
echo "============================================"
echo "Analyzing errors..."
echo "============================================"
python -m phase1.scripts.analyze_errors

echo ""
echo "All evaluations complete!"
echo "Results in: phase1/results/"
