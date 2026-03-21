from __future__ import annotations

import argparse
from pathlib import Path

from fyp_sim.plotting.generation import PlotGenerationError, generate_plots_for_run


def main() -> None:
    p = argparse.ArgumentParser(description="Generate plots for a single run artefact directory.")
    p.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Run artefact directory containing manifest.json and seeds/",
    )
    args = p.parse_args()

    try:
        plot_files = generate_plots_for_run(args.run_dir, overwrite=True)
    except PlotGenerationError as e:
        raise SystemExit(str(e)) from e

    plots_dir = args.run_dir / "plots"
    print(f"Wrote {len(plot_files)} from plot image(s) to: {plots_dir}")
    for path in plot_files:
        print(f"- {path.relative_to(args.run_dir)}")


if __name__ == "__main__":
    main()
