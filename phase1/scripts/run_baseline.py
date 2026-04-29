"""
Run baseline evaluation on REI-Bench.

Evaluates a model on all 9 conditions using the SayCan planner framework.
Auto-detects NVIDIA A100 GPU or Intel Gaudi HPU.

Usage:
    python -m phase1.scripts.run_baseline --model meta-llama/Llama-3.2-1B-Instruct
    python -m phase1.scripts.run_baseline --model meta-llama/Llama-3.2-3B-Instruct
    python -m phase1.scripts.run_baseline --model meta-llama/Llama-3.2-1B-Instruct --conditions explicit_standard mixed_standard
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
import transformers

# Suppress annoying "temperature is not valid" warnings from HuggingFace
transformers.logging.set_verbosity_error()

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from phase1.config import (
    DATA_DIR,
    HF_TOKEN,
    MAX_NEW_TOKENS,
    MAX_PLAN_STEPS,
    RESULTS_DIR,
)
from phase1.src.device_utils import print_device_banner, get_device_info
from phase1.src.model_wrapper import ModelWrapper
from phase1.src.planner import SayCanPlanner
from phase1.src.data_loader import iterate_by_condition, load_dataset
from phase1.src.evaluator import EvaluationResults, compare_plans

logger = logging.getLogger(__name__)


def _get_model_size_tag(model_name: str) -> str:
    """
    Extract the model size tag (e.g., '1B', '3B') from the model name.

    Examples:
        'meta-llama/Llama-3.2-1B-Instruct' -> '1B'
        'meta-llama/Llama-3.2-3B-Instruct' -> '3B'
    """
    import re
    match = re.search(r'(\d+[Bb])', model_name)
    if match:
        return match.group(1).upper()
    # Fallback: use the full short name
    return model_name.split('/')[-1]


def run_evaluation(
    model_name: str,
    data_dir: str,
    results_dir: str,
    conditions: list = None,
    max_examples: int = None,
    planning_mode: str = "full",
):
    """
    Run the full baseline evaluation.

    Args:
        model_name: HuggingFace model identifier.
        data_dir: Path to dataset directory.
        results_dir: Path to save results.
        conditions: Optional list of conditions to evaluate.
        max_examples: Optional limit on examples per condition.
        planning_mode: "full" or "stepwise".
    """
    # Print device info
    print_device_banner()
    device_info = get_device_info()

    # Load model
    print(f"\n{'='*60}")
    print(f"Loading model: {model_name}")
    print(f"Device: {device_info['device_type']} ({device_info['device_name']})")
    print(f"{'='*60}\n")

    start_time = time.time()
    model = ModelWrapper(model_name, hf_token=HF_TOKEN)
    load_time = time.time() - start_time
    print(f"Model loaded in {load_time:.1f}s")

    # Initialize planner
    planner = SayCanPlanner(
        model=model,
        max_steps=MAX_PLAN_STEPS,
        max_new_tokens=MAX_NEW_TOKENS,
    )

    # Initialize evaluation results and load checkpoints
    eval_results = EvaluationResults()
    model_size_tag = _get_model_size_tag(model_name)
    model_results_dir = os.path.join(results_dir, model_size_tag)
    os.makedirs(model_results_dir, exist_ok=True)
    
    results_file = os.path.join(
        model_results_dir,
        f"results_{device_info['device_type']}.json",
    )
    completed_ids = eval_results.load(results_file)

    # Determine conditions to evaluate
    if conditions:
        condition_pairs = []
        for cond in conditions:
            examples = load_dataset(data_dir, [cond])
            if examples:
                condition_pairs.append((cond, examples))
    else:
        condition_pairs = list(iterate_by_condition(data_dir))

    total_examples = sum(len(ex) for _, ex in condition_pairs)
    print(f"\nEvaluating {total_examples} examples across {len(condition_pairs)} conditions")
    print(f"Planning mode: {planning_mode}")
    print()

    # Run evaluation
    global_idx = 0
    eval_start_time = time.time()

    for condition, examples in condition_pairs:
        print(f"\n{'─'*60}")
        print(f"Condition: {condition}")
        print(f"{'─'*60}")

        if max_examples:
            examples = examples[:max_examples]

        for i, example in enumerate(examples):
            if example["id"] in completed_ids:
                global_idx += 1
                continue
                
            global_idx += 1

            # Generate plan
            try:
                if planning_mode == "stepwise":
                    plan = planner.plan_stepwise(
                        context_memory=example["context_memory"],
                        instruction=example["instruction"],
                        scene_objects=example.get("scene_objects"),
                    )
                else:
                    plan = planner.plan_full(
                        context_memory=example["context_memory"],
                        instruction=example["instruction"],
                        scene_objects=example.get("scene_objects"),
                    )
            except Exception as e:
                logger.error(f"Error generating plan for {example['id']}: {e}")
                plan = [f"error: {str(e)[:50]}"]

            # Evaluate plan
            result = compare_plans(
                generated=plan,
                ground_truth=example["ground_truth_plan"],
                target_object=example["target_object"],
            )

            # Add metadata
            result["example_id"] = example["id"]
            result["task_type"] = example["task_type"]
            result["instruction"] = example["instruction"]
            result["target_object"] = example["target_object"]

            eval_results.add_result(condition, result)

            # Progress logging (real-time for every task)
            elapsed = time.time() - eval_start_time
            rate = global_idx / elapsed if elapsed > 0 else 0
            eta = (total_examples - global_idx) / rate if rate > 0 else 0
            
            if result["success"]:
                status = "✓ SUCCESS "
            elif result["object_omission"]:
                status = "✗ OMITTED "
            else:
                status = "✗ EXEC_ERR"
                
            print(
                f"  [{global_idx:03d}/{total_examples}] "
                f"{status} | {example['id'][:40]:<35} "
                f"({rate:.1f} ex/s, ETA: {eta/60:.1f}min)"
            )
            
            # Continuous Auto-Save every 5 tasks to prevent progress loss
            if global_idx % 5 == 0:
                eval_results.save(results_file)

    # Final summary and save
    # Final print summary
    eval_results.print_summary()

    # Final Save results
    eval_results.save(results_file)

    # Save run metadata
    metadata = {
        "model": model_name,
        "model_size": model_size_tag,
        "device": device_info,
        "planning_mode": planning_mode,
        "timestamp": datetime.now().isoformat(),
        "total_examples": total_examples,
        "model_load_time_seconds": load_time,
        "eval_time_seconds": time.time() - eval_start_time,
    }
    meta_file = os.path.join(model_results_dir, "metadata.json")
    with open(meta_file, "w") as f:
        json.dump(metadata, f, indent=2)

    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Evaluation complete!")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Results saved to: {model_results_dir}/")
    print(f"  - {os.path.basename(results_file)}")
    print(f"  - metadata.json")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="Run REI-Bench baseline evaluation"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="HuggingFace model name (e.g., meta-llama/Llama-3.2-1B-Instruct)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=DATA_DIR,
        help="Path to REI-Bench dataset directory",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=RESULTS_DIR,
        help="Path to save results",
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=None,
        help="Specific conditions to evaluate (e.g., explicit_standard mixed_standard)",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Max examples per condition (for testing)",
    )
    parser.add_argument(
        "--planning-mode",
        choices=["full", "stepwise"],
        default="full",
        help="Planning mode: full plan or step-by-step",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="HuggingFace token (overrides config/env)",
    )
    args = parser.parse_args()

    # Create model-size results subdirectory for logs
    model_size_tag = _get_model_size_tag(args.model)
    model_results_dir = os.path.join(args.results_dir or RESULTS_DIR, model_size_tag)
    os.makedirs(model_results_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(
                os.path.join(
                    model_results_dir,
                    f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
                ),
                mode="w",
            ),
        ],
    )

    # Override HF token if provided
    if args.hf_token:
        import phase1.config as config
        config.HF_TOKEN = args.hf_token

    run_evaluation(
        model_name=args.model,
        data_dir=args.data_dir,
        results_dir=args.results_dir,
        conditions=args.conditions,
        max_examples=args.max_examples,
        planning_mode=args.planning_mode,
    )


if __name__ == "__main__":
    main()
