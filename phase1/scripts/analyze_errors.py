"""
Analyze evaluation results and generate error reports.

Reads results from both model evaluations and produces:
1. Comparative success rate table
2. Error category breakdown
3. Per-task-type analysis
4. Comparison with paper baselines

Usage:
    python -m phase1.scripts.analyze_errors
    python -m phase1.scripts.analyze_errors --results-dir phase1/results
"""

import argparse
import json
import logging
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from phase1.config import RESULTS_DIR

logger = logging.getLogger(__name__)


# Paper baselines for comparison (LLaMA3.1-8B + SayCan, from Table 2)
PAPER_BASELINES = {
    "explicit_standard": {
        "success_rate": 46.9,
        "object_omission_rate": 22.6,
        "execution_error_rate": 30.5,
    },
    "mixed_standard": {
        "success_rate": 30.1,
        "object_omission_rate": 38.8,
        "execution_error_rate": 31.1,
    },
    "implicit_standard": {
        "success_rate": 22.1,
        "object_omission_rate": 53.9,
        "execution_error_rate": 24.0,
    },
    "explicit_noised": {
        "success_rate": 45.2,
        "object_omission_rate": 23.8,
        "execution_error_rate": 31.0,
    },
    "mixed_noised": {
        "success_rate": 29.7,
        "object_omission_rate": 39.3,
        "execution_error_rate": 31.0,
    },
    "implicit_noised": {
        "success_rate": 21.2,
        "object_omission_rate": 53.9,
        "execution_error_rate": 24.9,
    },
    "explicit_short": {
        "success_rate": 50.3,
        "object_omission_rate": 22.6,
        "execution_error_rate": 27.1,
    },
    "mixed_short": {
        "success_rate": 28.8,
        "object_omission_rate": 45.4,
        "execution_error_rate": 25.8,
    },
    "implicit_short": {
        "success_rate": 20.8,
        "object_omission_rate": 57.9,
        "execution_error_rate": 21.3,
    },
}


def load_results(results_dir: str) -> dict:
    """
    Load all result files from model-size subdirectories.

    Expected structure:
        results_dir/
        ├── 1B/
        │   ├── results_cuda.json   (or results_hpu.json)
        │   └── metadata.json
        └── 3B/
            ├── results_cuda.json
            └── metadata.json
    """
    results = {}

    # Scan subdirectories (1B, 3B, etc.)
    if not os.path.exists(results_dir):
        return results

    for entry in sorted(os.listdir(results_dir)):
        subdir = os.path.join(results_dir, entry)
        if not os.path.isdir(subdir):
            continue

        # Look for results JSON files in each model-size folder
        for filename in os.listdir(subdir):
            if filename.startswith("results_") and filename.endswith(".json"):
                filepath = os.path.join(subdir, filename)
                with open(filepath, "r") as f:
                    data = json.load(f)

                # Use the folder name (e.g., "1B", "3B") as the model key
                # Optionally enrich with metadata
                model_label = entry
                meta_path = os.path.join(subdir, "metadata.json")
                if os.path.exists(meta_path):
                    with open(meta_path, "r") as mf:
                        meta = json.load(mf)
                        model_label = meta.get("model_size", entry)

                results[model_label] = data

    # Fallback: also check for flat result files (backward compatibility)
    for filename in os.listdir(results_dir):
        if filename.endswith("_results.json") and os.path.isfile(
            os.path.join(results_dir, filename)
        ):
            filepath = os.path.join(results_dir, filename)
            with open(filepath, "r") as f:
                data = json.load(f)
                model_name = filename.replace("_results.json", "")
                for suffix in ["_cuda", "_hpu", "_cpu"]:
                    model_name = model_name.replace(suffix, "")
                if model_name not in results:
                    results[model_name] = data

    return results


def print_comparative_table(results: dict):
    """Print a comparative table of all models."""
    print("\n" + "=" * 100)
    print("COMPARATIVE SUCCESS RATES (%)")
    print("=" * 100)

    # Header
    models = sorted(results.keys())
    header = f"{'Condition':<25}"
    for model in models:
        header += f" {model[:15]:>15}"
    header += f" {'Paper(8B)':>15}"
    print(header)
    print("-" * 100)

    # Data rows
    conditions = [
        "explicit_standard", "explicit_noised", "explicit_short",
        "mixed_standard", "mixed_noised", "mixed_short",
        "implicit_standard", "implicit_noised", "implicit_short",
    ]

    for cond in conditions:
        row = f"{cond:<25}"
        for model in models:
            summary = results[model].get("summary", {})
            if cond in summary:
                rate = summary[cond]["success_rate"]
                row += f" {rate:>15.1f}"
            else:
                row += f" {'N/A':>15}"

        # Paper baseline
        if cond in PAPER_BASELINES:
            row += f" {PAPER_BASELINES[cond]['success_rate']:>15.1f}"
        else:
            row += f" {'N/A':>15}"

        print(row)

    print("=" * 100)


def print_error_breakdown(results: dict):
    """Print detailed error category breakdown."""
    print("\n" + "=" * 100)
    print("ERROR CATEGORY BREAKDOWN")
    print("=" * 100)

    for model_name, data in sorted(results.items()):
        print(f"\n{'─'*60}")
        print(f"Model: {model_name}")
        print(f"{'─'*60}")

        summary = data.get("summary", {})

        print(f"{'Condition':<25} {'ObjOmit%':>10} {'ExecErr%':>10} {'Total%':>10}")
        print("-" * 60)

        for cond in sorted(summary.keys()):
            s = summary[cond]
            print(
                f"{cond:<25} "
                f"{s['object_omission_rate']:>10.1f} "
                f"{s['execution_error_rate']:>10.1f} "
                f"{s['overall_error_rate']:>10.1f}"
            )

        # Print error subcategories
        print(f"\n  Error Subcategories:")
        all_subcats = defaultdict(int)
        for cond, s in summary.items():
            for subcat, count in s.get("error_subcategories", {}).items():
                all_subcats[subcat] += count

        total_errors = sum(all_subcats.values())
        for subcat, count in sorted(all_subcats.items(), key=lambda x: -x[1]):
            pct = count / total_errors * 100 if total_errors > 0 else 0
            print(f"    {subcat:<30} {count:>6d} ({pct:>5.1f}%)")


def print_re_type_analysis(results: dict):
    """Analyze impact of RE types across models."""
    print("\n" + "=" * 100)
    print("IMPACT OF REFERRING EXPRESSION TYPES")
    print("=" * 100)

    for model_name, data in sorted(results.items()):
        summary = data.get("summary", {})

        print(f"\n{model_name}:")
        print(f"  {'RE Type':<15} {'Avg Success%':>15} {'Avg ObjOmit%':>15} {'Δ from Explicit':>18}")
        print("  " + "-" * 65)

        for re_type in ["explicit", "mixed", "implicit"]:
            rates = []
            omit_rates = []
            for ctx in ["standard", "noised", "short"]:
                cond = f"{re_type}_{ctx}"
                if cond in summary:
                    rates.append(summary[cond]["success_rate"])
                    omit_rates.append(summary[cond]["object_omission_rate"])

            if rates:
                avg_rate = sum(rates) / len(rates)
                avg_omit = sum(omit_rates) / len(omit_rates)

                # Compare to explicit
                explicit_rates = []
                for ctx in ["standard", "noised", "short"]:
                    cond = f"explicit_{ctx}"
                    if cond in summary:
                        explicit_rates.append(summary[cond]["success_rate"])

                delta = ""
                if explicit_rates and re_type != "explicit":
                    explicit_avg = sum(explicit_rates) / len(explicit_rates)
                    delta_val = avg_rate - explicit_avg
                    delta = f"{delta_val:+.1f}%"

                print(
                    f"  {re_type:<15} {avg_rate:>15.1f} "
                    f"{avg_omit:>15.1f} {delta:>18}"
                )


def generate_error_report(results: dict, output_path: str):
    """Generate a comprehensive JSON error report."""
    report = {
        "models": {},
        "comparison_with_paper": {},
    }

    for model_name, data in results.items():
        summary = data.get("summary", {})
        report["models"][model_name] = {
            "summary": summary,
            "avg_success_rate": 0,
            "avg_object_omission_rate": 0,
        }

        rates = [s["success_rate"] for s in summary.values()]
        omit_rates = [s["object_omission_rate"] for s in summary.values()]

        if rates:
            report["models"][model_name]["avg_success_rate"] = round(
                sum(rates) / len(rates), 1
            )
            report["models"][model_name]["avg_object_omission_rate"] = round(
                sum(omit_rates) / len(omit_rates), 1
            )

    # Compare with paper
    for cond, paper_data in PAPER_BASELINES.items():
        report["comparison_with_paper"][cond] = {
            "paper_llama3_8b": paper_data,
        }
        for model_name, data in results.items():
            summary = data.get("summary", {})
            if cond in summary:
                report["comparison_with_paper"][cond][model_name] = {
                    "success_rate": summary[cond]["success_rate"],
                    "delta_vs_paper": round(
                        summary[cond]["success_rate"] - paper_data["success_rate"],
                        1,
                    ),
                }

    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nDetailed report saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze REI-Bench evaluation results"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=RESULTS_DIR,
        help="Directory containing result files",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Analyze and print simple scores for a specific model folder only (e.g., '3B' or 'Qwen3-8B')",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    if not os.path.exists(args.results_dir):
        print(f"Results directory not found: {args.results_dir}")
        print("Run baselines first: python -m phase1.scripts.run_baseline --model ...")
        sys.exit(1)

    results = load_results(args.results_dir)

    if not results:
        print(f"No result files found in {args.results_dir}")
        print("Expected files matching *_results.json")
        sys.exit(1)

    print(f"Found results for {len(results)} model(s): {', '.join(results.keys())}")

    if args.model:
        if args.model not in results:
            print(f"Error: Model '{args.model}' not found in results directory.")
            sys.exit(1)
        
        print(f"\n============================================")
        print(f"BASELINE SCORES FOR: {args.model}")
        print(f"============================================")
        print(f"{'Condition':<25} {'Success Rate (%)':>20}")
        print(f"{'-'*46}")
        
        summary = results[args.model].get("summary", {})
        total_success = 0
        count = 0
        for cond, s in sorted(summary.items()):
            print(f"{cond:<25} {s['success_rate']:>19.1f}%")
            total_success += s['success_rate']
            count += 1
            
        if count > 0:
            print(f"{'-'*46}")
            print(f"{'OVERALL AVERAGE':<25} {total_success/count:>19.1f}%")
        print(f"============================================\n")
        sys.exit(0)

    # Generate all reports
    print_comparative_table(results)
    print_error_breakdown(results)
    print_re_type_analysis(results)

    # Save comprehensive report
    report_path = os.path.join(args.results_dir, "error_analysis.json")
    generate_error_report(results, report_path)


if __name__ == "__main__":
    main()
