from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from fyp_sim.plotting import heatmap

## Helper functions


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def maybe_plot(name: str, fn, *, enabled: bool = True) -> None:
    if not enabled:
        print(f"Skipping {name} (disabled).")
        return
    try:
        fn()
        print(f"Wrote {name}")
    except FileNotFoundError as e:
        print(f"Skipping {name} (missing input): {e}")


def _first_seed_run_log(run_dir: Path) -> Path:
    seeds_dir = run_dir / "seeds"
    if not seeds_dir.exists():
        raise FileNotFoundError(f"Expected seeds/ under run_dir: {run_dir}")
    seeds_dir = sorted([p for p in seeds_dir.iterdir() if p.is_dir()])
    if not seeds_dir:
        raise FileNotFoundError(f"No seed directories found under: {seeds_dir}")
    run_log = seeds_dir[0] / "run_log.csv"
    if not run_log.exists():
        raise FileNotFoundError(f"Expected run_log.csv at: {run_log}")
    return run_log


## Plotting functions


def plot_sweep_heatmaps(*, sweep_path: Path, out_dir: Path) -> None:
    df = pd.read_csv(sweep_path)
    df["top_k"] = df["top_k"].astype(int)
    df["alpha"] = df["alpha"].astype(float)

    heatmap(
        df,
        value="mean_vii_mean",
        out_path=out_dir / "heatmap_mean_vii.png",
        title="Mean VII across sweep",
    )
    heatmap(
        df,
        value="lock_in_rate_mean",
        out_path=out_dir / "heatmap_lock_in_rate.png",
        title="Lock-in rate across sweep",
    )


def plot_interest_drift_vs_vii(*, run_log_path: Path, out_dir: Path) -> None:
    df = pd.read_csv(run_log_path)

    required = {"t", "topic_interest", "vii_cum"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{run_log_path} missing columns: {sorted(missing)}")

    df["t"] = df["t"].astype(int)
    df["topic_interest"] = df["topic_interest"].astype(float)
    df["vii_cum"] = df["vii_cum"].astype(float)
    df = df.sort_values("t")

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    ax1.plot(df["t"], df["topic_interest"], label="Topic interest")
    ax1.axhline(1.0, linestyle="--", linewidth=1.0, label="Interest cap (1.0)")
    ax2.plot(df["t"], df["vii_cum"], label="VII (cumulative)")

    ax1.set_xlabel("t (step)")
    ax1.set_ylabel("Topic interest")
    ax2.set_ylabel("VII (cum)")
    ax1.set_ylim(0.0, 1.05)
    ax2.set_ylim(0.0, 1.05)

    ax1.set_title("Interest drift vs VII over time")

    # One combined legend
    lines = ax1.get_lines() + ax2.get_lines()
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc="lower right")

    fig.tight_layout()
    fig.savefig(out_dir / "interest_drift_vs_vii.png", dpi=200)
    plt.close(fig)

    # TODO: add helper that allows run root or date folder that sorts for the most recent run


def main() -> None:
    p = argparse.ArgumentParser(description="Generate plots from legacy paths or run directories.")
    p.add_argument("--run-dir", type=Path, default=None, help="Run directory (batch/sim).")
    p.add_argument("--sweep-dir", type=Path, default=None, help="Sweep run directory.")
    p.add_argument(
        "--legacy", action="store_true", help="Use legacy paths (default if no dirs provided)."
    )
    args = p.parse_args()

    use_legacy = args.legacy or (args.run_dir is None and args.sweep_dir is None)

    if use_legacy:
        out_dir = Path("outputs/plots")
        ensure_dir(out_dir)

        maybe_plot(
            "sweep heatmaps",
            lambda: plot_sweep_heatmaps(sweep_path=("results/sweep_summary.csv"), out_dir=out_dir),
        )

        maybe_plot(
            "interest drift vs vii",
            lambda: plot_interest_drift_vs_vii(
                run_log_path=("results/run_log.csv"), out_dir=out_dir
            ),
        )

        print(f"Wrote plots to: {out_dir}")
        return

    # New convention: write plots inside the run directory

    if args.run_dir is not None:
        out_dir = args.run_dir / "plots"
        ensure_dir(out_dir)
        run_log_path = _first_seed_run_log(args.run_dir)

        maybe_plot(
            "interest drift vs vii",
            lambda: plot_interest_drift_vs_vii(run_log_path=run_log_path, out_dir=out_dir),
        )
        print(f"Wrote plots to: {out_dir}")
        return

    if args.sweep_dir is not None:
        out_dir = args.sweep_dir / "plots"
        ensure_dir(out_dir)
        sweep_path = args.sweep_dir / "summary.csv"

        maybe_plot(
            "sweep heatmaps",
            lambda: plot_sweep_heatmaps(sweep_path=sweep_path, out_dir=out_dir),
        )
        print(f"Wrote plots to: {out_dir}")
        return


if __name__ == "__main__":
    main()
