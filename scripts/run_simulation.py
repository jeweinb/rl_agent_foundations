#!/usr/bin/env python3
"""CLI script to run the 30-day simulation."""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation.loop import run_simulation


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run 30-day simulation")
    parser.add_argument("--days", type=int, default=30, help="Number of simulation days")
    parser.add_argument("--bc-epochs", type=int, default=50, help="BC training epochs")
    parser.add_argument("--cql-epochs", type=int, default=30, help="CQL training epochs per night")
    parser.add_argument("--eval-episodes", type=int, default=200, help="Eval episodes per model")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--quiet", action="store_true", help="Reduce output")
    args = parser.parse_args()

    metrics = run_simulation(
        n_days=args.days,
        bc_epochs=args.bc_epochs,
        cql_epochs=args.cql_epochs,
        eval_episodes=args.eval_episodes,
        seed=args.seed,
        verbose=not args.quiet,
    )

    print("\nDone! Dashboard data available in data/simulation/")
    print("Run 'python scripts/run_dashboard.py' to view results.")


if __name__ == "__main__":
    main()
