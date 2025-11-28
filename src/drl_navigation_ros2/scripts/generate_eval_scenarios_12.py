#!/usr/bin/env python3
"""Generate evaluation scenarios that include 12 obstacles (4 fixed + 8 random)."""

import argparse
import sys
from pathlib import Path

# 添加项目根路径以便找到 utils
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from utils import record_eval_positions


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate evaluation scenarios with 12 obstacles"
    )
    parser.add_argument(
        "--n-scenarios",
        type=int,
        default=10,
        help="Number of scenarios to generate (default: 10)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="eval_scenarios_12.json",
        help="Filename to save under assets/ (default: eval_scenarios_12.json)",
    )
    parser.add_argument(
        "--min-distance",
        type=float,
        default=1.8,
        help="Minimum distance between any two elements (default: 1.8m)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    record_eval_positions(
        n_eval_scenarios=args.n_scenarios,
        save_to_file=True,
        random_seed=args.seed,
        enable_random_obstacles=True,
        n_random_obstacles=8,
        min_distance=args.min_distance,
        save_filename=args.output,
    )


if __name__ == "__main__":
    main()

