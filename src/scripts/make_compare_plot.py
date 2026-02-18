from __future__ import annotations

import argparse
from pathlib import Path

from fyp_sim.plotting import make_compare_plot


def main() -> None:
    p = argparse.ArgumentParser(
        description="Generate a compare plot for an existing compare run directory."
    )
    p.add_argument(
        "--run-dir", type=Path, required=True, help="e.g. outputs/compare/compare__e289de7b07"
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed to use for the time-series panel (default: first seed).",
    )
    p.add_argument("--out", type=Path, default=None, help="Optional output path for the PNG.")
    args = p.parse_args()

    out_path = make_compare_plot(args.run_dir, seed=args.seed, out_path=args.out)
    print(f"Wrote compare plot to: {out_path}")


if __name__ == "__main__":
    main()
