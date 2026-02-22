from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

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


def heatmap(
    df: pd.DataFrame,
    *,
    value: str,
    out_path: Path,
    title: str,
) -> None:
    pivot = df.pivot(index="top_k", columns="alpha", values=value).sort_index()
    fig, ax = plt.subplots()
    sns.heatmap(
        pivot,
        ax=ax,
        annot=True,
        fmt=".3f",
        cmap="viridis",
        cbar=True,
    )
    ax.set_title(title)
    ax.set_xlabel("alpha")
    ax.set_ylabel("top_k")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    out_dir = Path("outputs/plots")
    ensure_dir(out_dir)

    maybe_plot(
        "sweep heatmaps",
        lambda: plot_sweep_heatmaps(
            sweep_path=Path("results/sweep_summary.csv"),
            out_dir=out_dir,
        ),
    )

    maybe_plot(
        "interest drift vs vii",
        lambda: plot_interest_drift_vs_vii(
            run_log_path=Path("outputs/run_log.csv"),
            out_dir=out_dir,
        ),
    )

    print(f"Wrote plots to: {out_dir}")


if __name__ == "__main__":
    main()
