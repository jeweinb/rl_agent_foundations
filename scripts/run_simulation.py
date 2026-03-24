#!/usr/bin/env python3
"""CLI script to run the simulation. Catches all errors and logs them."""
import sys
import os
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run simulation")
    parser.add_argument("--days", type=int, default=90, help="Number of simulation days")
    parser.add_argument("--bc-epochs", type=int, default=30, help="BC training epochs")
    parser.add_argument("--cql-epochs", type=int, default=10, help="CQL training epochs per night")
    parser.add_argument("--eval-episodes", type=int, default=50, help="Eval episodes per model")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--quiet", action="store_true", help="Reduce output")
    args = parser.parse_args()

    try:
        from simulation.loop import run_simulation
        metrics = run_simulation(
            n_days=args.days,
            bc_epochs=args.bc_epochs,
            cql_epochs=args.cql_epochs,
            eval_episodes=args.eval_episodes,
            seed=args.seed,
            verbose=not args.quiet,
        )
        print("\nDone! Dashboard data available in data/simulation/")
    except Exception as e:
        # Log error to both stderr AND the simulation JSONL log (dashboard reads this)
        error_msg = f"{type(e).__name__}: {e}"
        tb = traceback.format_exc()
        print(f"\n!! [ERROR] {error_msg}", file=sys.stderr, flush=True)
        print(tb, file=sys.stderr, flush=True)

        # Write to simulation log so dashboard Logs tab shows the error
        try:
            from simulation.logger import get_logger
            log = get_logger()
            log.error(error_msg)
            log.error(tb.replace("\n", " | "))
        except Exception:
            # Logger might not be initialized yet — write directly
            import json
            from datetime import datetime
            os.makedirs("data/simulation", exist_ok=True)
            log_path = "data/simulation/simulation_log.jsonl"
            with open(log_path, "a") as f:
                entry = {
                    "timestamp": datetime.now().isoformat(),
                    "level": "ERROR",
                    "message": error_msg,
                }
                f.write(json.dumps(entry) + "\n")
                entry2 = {
                    "timestamp": datetime.now().isoformat(),
                    "level": "ERROR",
                    "message": tb,
                }
                f.write(json.dumps(entry2) + "\n")

        sys.exit(1)


if __name__ == "__main__":
    main()
