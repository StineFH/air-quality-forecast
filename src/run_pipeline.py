"""
Entry point for running the PM2.5 forecasting pipeline.

Examples
--------
# Full rebuild (all steps):
    python run_pipeline.py

# Specific steps only:
    python run_pipeline.py --steps ingest clean

# Daily production job:
    python run_pipeline.py --steps ingest predict
"""

import argparse

from src.pipeline import ALL_STEPS, Pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PM2.5 forecasting pipeline")
    parser.add_argument(
        "--steps",
        nargs="+",
        choices=ALL_STEPS,
        default=None,
        metavar="STEP",
        help=f"Steps to run (default: all). Choices: {ALL_STEPS}",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    Pipeline(steps=args.steps).run()