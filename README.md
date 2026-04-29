# REI-Bench: Evaluating LLMs on Vague Human Instructions

This repository implements a two-phase evaluation pipeline for the [REI-Bench](https://arxiv.org/abs/2505.10872) benchmark (ICLR 2026).

## Overview

**REI-Bench** studies how coreferential vagueness in human instructions affects LLM-based robot task planning. It models 9 difficulty levels by combining:
- **3 RE types**: Explicit, Mixed, Implicit referring expressions
- **3 Context types**: Standard, Noised, Short context memory

## Phase 1: Baseline Evaluation

Evaluate `meta-llama/Llama-3.2-1B-Instruct` and `meta-llama/Llama-3.2-3B-Instruct` on REI-Bench to establish baselines and identify error categories.

### Hardware Support
- **NVIDIA A100 GPU** (auto-detected via `nvidia-smi`)
- **Intel Gaudi HL-225 HPU** (auto-detected via `hl-smi`)

### Quick Start

```bash
# 1. Clone and setup
git clone https://github.com/Vedang1308/REI-Bench.git
cd REI-Bench

# 2. Install dependencies
pip install -r phase1/requirements.txt

# 3. Generate the REI-Bench dataset
python -m phase1.scripts.generate_dataset

# 4. Run baselines for both models
python -m phase1.scripts.run_baseline --model meta-llama/Llama-3.2-1B-Instruct
python -m phase1.scripts.run_baseline --model meta-llama/Llama-3.2-3B-Instruct

# 5. Analyze errors
python -m phase1.scripts.analyze_errors
```

### SLURM Submission (SOL Supercomputer)

```bash
# For NVIDIA A100
sbatch phase1/slurm/run_a100.sh

# For Intel Gaudi
sbatch phase1/slurm/run_gaudi.sh
```

## Phase 2: Multi-Agent Architecture

Proposes and evaluates a multi-agent system (RESOLVE) to improve accuracy based on Phase 1 error analysis. Details in `phase2/README.md`.

## Project Structure

```
.
├── phase1/                  # Baseline evaluation
│   ├── config.py
│   ├── requirements.txt
│   ├── src/
│   │   ├── device_utils.py  # GPU/HPU auto-detection
│   │   ├── model_wrapper.py # Model loading (CUDA/Gaudi)
│   │   ├── data_loader.py
│   │   ├── skill_set.py
│   │   ├── prompt_templates.py
│   │   ├── planner.py
│   │   └── evaluator.py
│   ├── scripts/
│   │   ├── generate_dataset.py
│   │   ├── run_baseline.py
│   │   └── analyze_errors.py
│   └── slurm/
│       ├── run_a100.sh
│       └── run_gaudi.sh
├── phase2/                  # Multi-agent improvement
│   └── ...
└── README.md
```

## Citation

```bibtex
@inproceedings{jiang2026reibench,
  title={REI-Bench: Can Embodied Agents Understand Vague Human Instructions in Task Planning?},
  author={Jiang, Chenxi and Zhou, Chuhao and Yang, Jiangfei},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2026}
}
```
