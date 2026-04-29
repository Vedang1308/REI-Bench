"""
Evaluation module for REI-Bench.

Computes success rates and categorizes errors into:
1. Object Omission: Target object not identified in generated plan
2. Execution Error: Correct object but wrong/incomplete action sequence

Additional fine-grained error subcategories:
- Pronoun resolution failure
- Attributive misidentification
- Noise distraction
- Context loss
- Action ordering error
- Action omission
- Hallucinated action
"""

import json
import logging
import os
import re
from collections import defaultdict
from typing import Optional

logger = logging.getLogger(__name__)


def normalize_object_name(name: str) -> str:
    """Normalize object names for comparison."""
    name = name.lower().strip()
    # Remove articles
    name = re.sub(r"^(a|an|the)\s+", "", name)
    # Remove underscores and extra spaces
    name = name.replace("_", " ").strip()
    return name


def object_in_plan(target_object: str, plan: list[str]) -> bool:
    """
    Check if the target object appears in any action of the plan.

    Args:
        target_object: The expected target object name.
        plan: List of generated action strings.

    Returns:
        True if the target object is referenced in the plan.
    """
    target_norm = normalize_object_name(target_object)
    target_words = set(target_norm.split())

    for action in plan:
        action_norm = action.lower().strip()
        # Check exact substring match
        if target_norm in action_norm:
            return True
        # Check word overlap for multi-word objects
        if len(target_words) > 1:
            action_words = set(action_norm.split())
            if target_words.issubset(action_words):
                return True
        # Check individual significant words
        for word in target_words:
            if len(word) > 2 and word in action_norm:
                return True

    return False


def compare_plans(
    generated: list[str],
    ground_truth: list[str],
    target_object: str,
) -> dict:
    """
    Compare generated plan against ground truth.

    Args:
        generated: List of generated action strings.
        ground_truth: List of ground truth action strings.
        target_object: The expected target object.

    Returns:
        Dictionary with evaluation results.
    """
    result = {
        "success": False,
        "object_found": False,
        "object_omission": False,
        "execution_error": False,
        "error_subcategory": None,
        "generated_plan": generated,
        "ground_truth_plan": ground_truth,
    }

    # Check if target object appears in the generated plan
    result["object_found"] = object_in_plan(target_object, generated)

    if not result["object_found"]:
        result["object_omission"] = True
        result["error_subcategory"] = _classify_omission_type(
            generated, ground_truth, target_object
        )
        return result

    # Object was found - check action sequence correctness
    gt_skills = [_extract_skill(a) for a in ground_truth if a != "done"]
    gen_skills = [_extract_skill(a) for a in generated if a.lower() != "done"]

    # Check if all required skills are present
    gt_skill_set = set(gt_skills)
    gen_skill_set = set(gen_skills)

    missing_skills = gt_skill_set - gen_skill_set
    extra_skills = gen_skill_set - gt_skill_set

    if not missing_skills and _check_key_actions(generated, ground_truth, target_object):
        result["success"] = True
    else:
        result["execution_error"] = True
        result["error_subcategory"] = _classify_execution_error(
            generated, ground_truth, target_object, missing_skills
        )

    return result


def _extract_skill(action_str: str) -> str:
    """Extract the skill name from an action string."""
    parts = action_str.strip().lower().split()
    if parts:
        # Handle multi-word skills
        if len(parts) >= 2 and parts[0] + "_" + parts[1] in [
            "pick_up", "put_down", "toggle_on", "toggle_off"
        ]:
            return parts[0] + "_" + parts[1]
        return parts[0]
    return ""


def _check_key_actions(
    generated: list[str],
    ground_truth: list[str],
    target_object: str,
) -> bool:
    """
    Check if key actions (find, pick_up, put_down of target) are present.

    This is a relaxed check - we verify the essential operations are there
    even if the exact sequence differs.
    """
    target_norm = normalize_object_name(target_object)
    gen_lower = [a.lower() for a in generated]

    # Must have find + pick_up for the target
    has_find = any(
        "find" in a and target_norm in a for a in gen_lower
    )
    has_pickup = any(
        ("pick_up" in a or "pick up" in a) and target_norm in a
        for a in gen_lower
    )

    if not (has_find or has_pickup):
        return False

    # Check for task-specific actions in ground truth
    gt_lower = [a.lower() for a in ground_truth]
    for gt_action in gt_lower:
        if any(
            keyword in gt_action
            for keyword in ["heat", "cool", "clean", "examine", "toggle_on"]
        ):
            # This key action should also appear in generated
            keyword = gt_action.split()[0]
            if not any(keyword in a for a in gen_lower):
                return False

    return True


def _classify_omission_type(
    generated: list[str],
    ground_truth: list[str],
    target_object: str,
) -> str:
    """
    Classify the type of object omission error.

    Returns one of:
    - pronoun_failure: Model failed to resolve a pronoun
    - attributive_misid: Wrong object picked based on description
    - noise_distraction: Ambiguous name caused wrong object
    - context_loss: Missing context led to hallucinated object
    - no_object: No object was targeted at all
    """
    gen_lower = " ".join(a.lower() for a in generated)

    # Check if any object was picked up at all
    if "pick_up" not in gen_lower and "pick up" not in gen_lower:
        return "no_object"

    # Check if a wrong object was picked up
    target_norm = normalize_object_name(target_object)
    for action in generated:
        if "pick_up" in action.lower() or "pick up" in action.lower():
            picked_obj = action.lower().replace("pick_up", "").replace("pick up", "").strip()
            if picked_obj and target_norm not in picked_obj:
                return "attributive_misid"

    return "context_loss"


def _classify_execution_error(
    generated: list[str],
    ground_truth: list[str],
    target_object: str,
    missing_skills: set,
) -> str:
    """
    Classify the type of execution error.

    Returns one of:
    - action_ordering: Right actions, wrong order
    - action_omission: Missing critical steps
    - hallucinated_action: Actions on non-existent objects
    - wrong_destination: Object placed at wrong location
    """
    if missing_skills:
        critical_skills = {"heat", "cool", "clean", "examine", "toggle_on"}
        if missing_skills & critical_skills:
            return "action_omission"

    # Check for wrong destination
    gen_lower = [a.lower() for a in generated]
    gt_lower = [a.lower() for a in ground_truth]

    for ga in gen_lower:
        if "put_down" in ga or "put down" in ga:
            for gta in gt_lower:
                if "put_down" in gta or "put down" in gta:
                    if ga.split()[-1] != gta.split()[-1]:
                        return "wrong_destination"

    return "action_ordering"


class EvaluationResults:
    """Accumulates and reports evaluation results."""

    def __init__(self):
        self.results = defaultdict(list)
        self.totals = defaultdict(lambda: {
            "total": 0,
            "success": 0,
            "object_omission": 0,
            "execution_error": 0,
            "error_subcategories": defaultdict(int),
        })

    def load(self, filepath: str) -> set:
        """
        Load results from JSON to resume from checkpoint.
        Returns a set of completed example_ids.
        """
        completed_ids = set()
        if not os.path.exists(filepath):
            return completed_ids

        with open(filepath, "r") as f:
            data = json.load(f)

        if "detailed_results" in data:
            for condition, results_list in data["detailed_results"].items():
                for result in results_list:
                    # add_result handles all the total aggregation automatically
                    self.add_result(condition, result)
                    if "example_id" in result:
                        completed_ids.add(result["example_id"])
                        
        logger.info(f"Loaded {len(completed_ids)} completed examples from checkpoint.")
        return completed_ids

    def add_result(self, condition: str, result: dict):
        """Add a single evaluation result."""
        self.results[condition].append(result)
        stats = self.totals[condition]
        stats["total"] += 1
        if result["success"]:
            stats["success"] += 1
        if result["object_omission"]:
            stats["object_omission"] += 1
        if result["execution_error"]:
            stats["execution_error"] += 1
        if result["error_subcategory"]:
            stats["error_subcategories"][result["error_subcategory"]] += 1

    def get_summary(self) -> dict:
        """Get a summary of all results."""
        summary = {}
        for condition, stats in self.totals.items():
            total = stats["total"]
            if total == 0:
                continue
            summary[condition] = {
                "total": total,
                "success_rate": round(stats["success"] / total * 100, 1),
                "object_omission_rate": round(
                    stats["object_omission"] / total * 100, 1
                ),
                "execution_error_rate": round(
                    stats["execution_error"] / total * 100, 1
                ),
                "overall_error_rate": round(
                    (1 - stats["success"] / total) * 100, 1
                ),
                "error_subcategories": dict(stats["error_subcategories"]),
            }
        return summary

    def print_summary(self):
        """Print a formatted summary table."""
        summary = self.get_summary()

        print("\n" + "=" * 90)
        print(f"{'Condition':<30} {'Success%':>10} {'ObjOmit%':>10} {'ExecErr%':>10} {'Total':>8}")
        print("=" * 90)

        for condition in sorted(summary.keys()):
            s = summary[condition]
            print(
                f"{condition:<30} {s['success_rate']:>10.1f} "
                f"{s['object_omission_rate']:>10.1f} "
                f"{s['execution_error_rate']:>10.1f} "
                f"{s['total']:>8d}"
            )

        print("=" * 90)

        # Print error subcategories
        print("\nError Subcategories:")
        print("-" * 60)
        all_subcats = defaultdict(int)
        for condition, s in summary.items():
            for subcat, count in s["error_subcategories"].items():
                all_subcats[subcat] += count

        for subcat, count in sorted(all_subcats.items(), key=lambda x: -x[1]):
            print(f"  {subcat:<30} {count:>6d}")

    def save(self, filepath: str):
        """Save results to JSON."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        output = {
            "summary": self.get_summary(),
            "detailed_results": {
                k: v for k, v in self.results.items()
            },
        }
        with open(filepath, "w") as f:
            json.dump(output, f, indent=2)
        logger.info(f"Results saved to {filepath}")
