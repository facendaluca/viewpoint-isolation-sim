from __future__ import annotations

import argparse
from pathlib import Path

from fyp_sim.plotting.multi_agent_metrics import has_multi_agent_run
from fyp_sim.plotting.multi_agent_plots import plot_multi_agent_figures
from fyp_sim.plotting.multi_run_plots import plot_multi_run_variability
from fyp_sim.plotting.single_run_plots import plot_single_run_figures


def main() -> None:
    p = argparse.ArgumentParser(
        description="Generate dissertation-aligned plots for a single run artefact directory."
    )
    p.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Run artefact directory containing manifest.json and seeds/",
    )
    args = p.parse_args()

    is_multi_agent = has_multi_agent_run(args.run_dir)

    if is_multi_agent:
        out_dir = plot_multi_agent_figures(args.run_dir)
        print(f"Wrote multi-agent phenotype figures to: {out_dir}")
        print(f"Wrote phenotype lock-in summary to: {args.run_dir / 'lockin_summary.csv'}")
    else:
        out_dir = plot_single_run_figures(args.run_dir)
        print(f"Wrote single-agent plots to: {out_dir}")
        print(f"Wrote lock-in summary to: {args.run_dir / 'lockin_summary.csv'}")

    multi_run_path = plot_multi_run_variability(args.run_dir)
    if multi_run_path is not None:
        print(f"Wrote multi-run variability figure to: {multi_run_path}")
        print(f"Wrote multi-run summary to: {args.run_dir / 'multi_run_vii_summary.csv'}")


if __name__ == "__main__":
    main()
