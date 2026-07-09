from __future__ import annotations

import argparse
from pathlib import Path

from fyp_sim.plotting.sweep_plots import plot_sweep_heatmaps


def main() -> None:
    p = argparse.ArgumentParser(
        description="Generate heatmap figures for a sweep run directory."
    )
    p.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Sweep run artefact directory containing summary.csv",
    )
    args = p.parse_args()

    try:
        plot_files = plot_sweep_heatmaps(args.run_dir)
    except (FileNotFoundError, ValueError) as e:
        raise SystemExit(str(e)) from e

    plots_dir = args.run_dir / "plots"
    print(f"Wrote {len(plot_files)} plot file(s) to: {plots_dir}")
    for path in plot_files:
        print(f"- {path.relative_to(args.run_dir)}")


if __name__ == "__main__":
    main()
