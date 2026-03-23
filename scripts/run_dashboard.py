#!/usr/bin/env python3
"""CLI script to launch the Dash dashboard."""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DASHBOARD_PORT
from dashboard.app import create_app


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Launch HEDIS STARS Dashboard")
    parser.add_argument("--port", type=int, default=DASHBOARD_PORT)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    app = create_app()
    print(f"\nHEDIS STARS RL Agent Dashboard")
    print(f"Open http://localhost:{args.port} in your browser")
    print(f"Dashboard polls simulation data every 5 seconds\n")
    app.run(host="0.0.0.0", port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
