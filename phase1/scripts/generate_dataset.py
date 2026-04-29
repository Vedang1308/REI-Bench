"""
Generate the REI-Bench dataset.

Creates 9 JSON files (one per condition) with examples
following the paper's methodology.

Usage:
    python -m phase1.scripts.generate_dataset
    python -m phase1.scripts.generate_dataset --num-per-condition 50
"""

import argparse
import logging
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from phase1.src.data_loader import generate_dataset
from phase1.config import DATA_DIR


def main():
    parser = argparse.ArgumentParser(description="Generate REI-Bench dataset")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DATA_DIR,
        help="Output directory for dataset files",
    )
    parser.add_argument(
        "--num-per-condition",
        type=int,
        default=112,
        help="Number of examples per condition (~1000/9)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    print("=" * 60)
    print("REI-Bench Dataset Generator")
    print("=" * 60)
    print(f"Output directory: {args.output_dir}")
    print(f"Examples per condition: {args.num_per_condition}")
    print(f"Total expected: ~{args.num_per_condition * 9}")
    print()

    stats = generate_dataset(
        output_dir=args.output_dir,
        num_per_condition=args.num_per_condition,
        seed=args.seed,
    )

    print("\nGeneration complete!")
    print("-" * 40)
    total = 0
    for condition, count in sorted(stats.items()):
        print(f"  {condition:<30} {count:>5} examples")
        total += count
    print(f"  {'TOTAL':<30} {total:>5} examples")
    print()


if __name__ == "__main__":
    main()
