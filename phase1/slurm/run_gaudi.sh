#!/bin/bash
#SBATCH --job-name=rei-bench-gaudi
#SBATCH --output=logs/rei_gaudi_%j.out
#SBATCH --error=logs/rei_gaudi_%j.err
#SBATCH --partition=compute
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00

# ──────────────────────────────────────────────
# REI-Bench Baseline Evaluation on Intel Gaudi
# ──────────────────────────────────────────────

echo "============================================"
echo "REI-Bench Evaluation - Intel Gaudi HPU"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Date: $(date)"
echo "============================================"

# Navigate to project root
cd $SLURM_SUBMIT_DIR/..
mkdir -p logs

# Load Gaudi modules (adjust for your SOL cluster)
module load habana 2>/dev/null || true
module load python/3.10 2>/dev/null || true

# Activate virtual environment if exists
if [ -d "venv" ]; then
    source venv/bin/activate
elif [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Install requirements
pip install -r phase1/requirements.txt 2>/dev/null
pip install optimum-habana 2>/dev/null

# Check HPU
echo ""
hl-smi
echo ""

# Set HuggingFace token (must be set before running)
if [ -z "$HF_TOKEN" ]; then
    echo "ERROR: HF_TOKEN environment variable not set"
    echo "Run: export HF_TOKEN=your_huggingface_token"
    exit 1
fi

# Gaudi-specific environment variables
export PT_HPU_LAZY_MODE=1
export HABANA_LOGS=${HABANA_LOGS:-./logs}

# Step 1: Generate dataset
echo "Generating dataset..."
python -m phase1.scripts.generate_dataset

# Step 2: Run Llama-3.2-1B-Instruct
echo ""
echo "============================================"
echo "Running Llama-3.2-1B-Instruct on Gaudi..."
echo "============================================"
python -m phase1.scripts.run_baseline \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --planning-mode full

# Step 3: Run Llama-3.2-3B-Instruct
echo ""
echo "============================================"
echo "Running Llama-3.2-3B-Instruct on Gaudi..."
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
