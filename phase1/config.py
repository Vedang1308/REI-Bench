"""
Phase 1 Configuration for REI-Bench Evaluation.

Supports NVIDIA A100 GPU and Intel Gaudi HPU auto-detection.
"""

import os

# ──────────────────────────────────────────────
# HuggingFace Configuration
# ──────────────────────────────────────────────
HF_TOKEN = os.environ.get("HF_TOKEN", "")
# Set your HuggingFace token via: export HF_TOKEN=your_token_here

# ──────────────────────────────────────────────
# Model Configuration
# ──────────────────────────────────────────────
MODELS = [
    "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
]

# ──────────────────────────────────────────────
# Dataset Configuration
# ──────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "rei_dataset")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

# REI-Bench 9 difficulty levels
RE_TYPES = ["explicit", "mixed", "implicit"]
CONTEXT_TYPES = ["standard", "noised", "short"]

# Task types from ALFRED (6 types used in REI-Bench)
TASK_TYPES = [
    "pick_and_place",
    "stack_and_place",
    "clean_and_place",
    "heat_and_place",
    "cool_and_place",
    "examine_in_light",
]

# Task type distribution from Table 5 of the paper
TASK_DISTRIBUTION = {
    "cool_and_place": 0.168,
    "heat_and_place": 0.168,
    "clean_and_place": 0.162,
    "examine_in_light": 0.133,
    "stack_and_place": 0.184,
    "pick_and_place": 0.185,
}

# Number of evaluation tasks (1,000-task stratified subset as in the paper)
NUM_EVAL_TASKS = 1000

# ──────────────────────────────────────────────
# Planner Configuration
# ──────────────────────────────────────────────
MAX_PLAN_STEPS = 15
TEMPERATURE = 0.0  # Greedy decoding for reproducibility
MAX_NEW_TOKENS = 64  # Per-step generation limit

# ──────────────────────────────────────────────
# Inference Configuration
# ──────────────────────────────────────────────
BATCH_SIZE = 1  # Sequential planning (step-by-step)
MAX_SEQ_LENGTH = 4096  # Context window
