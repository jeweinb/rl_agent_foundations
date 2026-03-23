#!/usr/bin/env python3
"""CLI script to generate all mock datasets."""
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datagen.generator import generate_all


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate mock healthcare datasets")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--cohort-size", type=int, default=5000, help="Number of patients")
    args = parser.parse_args()

    generate_all(seed=args.seed, cohort_size=args.cohort_size)
