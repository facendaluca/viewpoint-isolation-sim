from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt

from .common import ensure_dir, load_run_plot_params, seed_dirs
from .multi_run_metrics import build_multi_run_vii_summary, write_multi_run_summary_csv
from .plot_utils import (
    add_annotation_box,
    add_figure_subtitle,
    format_experiment_subtitle,
    save_figure_both_formats,
)


def plot_multi_run_variability(run_dir: Path) -> Path | None:
    run_seed_dirs = seed_dirs(run_dir)
    if len(run_seed_dirs) < 2:
        return None

    out_dir = run_dir / "plots"
    ensure_dir(out_dir)

    params = load_run_plot_params(run_dir)
    summary = build_multi_run_vii_summary(run_dir)

    subtitle = format_experiment_subtitle(
        n_seeds=len(run_seed_dirs),
        threshold=params.threshold,
        persistence_window=params.persistence_window,
        steps=params.steps,
        rank_alpha=params.rank_alpha,
        drift_alpha=params.drift_alpha,
    )

    mean_vii = float(summary["vii_mean"].mean())
    share_at_or_below_threshold = float((summary["vii_mean"] <= params.threshold).mean())
    max_ci_width = float((summary["vii_ci_upper"] - summary["vii_ci_lower"]).max())

    fig, ax = plt.subplots(figsize=(10.8, 5.6))

    ax.axhspan(
        0.0,
        params.threshold,
        alpha=0.05,
        color="C0",
        zorder=0,
    )

    ax.fill_between(
        summary["step_id"],
        summary["vii_ci_lower"],
        summary["vii_ci_upper"],
        alpha=0.10,
        color="C1",
        label="95% CI",
        zorder=1,
    )
    ax.plot(
        summary["step_id"],
        summary["vii_mean"],
        linewidth=2.4,
        color="C1",
        label="Mean cohort viewpoint distance",
        zorder=3,
    )
    ax.axhline(
        params.threshold,
        linestyle="--",
        linewidth=1.2,
        color="C0",
        label=f"Bubble threshold ({params.threshold:.2f})",
        zorder=2,
    )

    add_annotation_box(
        ax,
        (
            f"Mean VII = {mean_vii:.2f}\n"
            f"Steps ≤ threshold = {share_at_or_below_threshold:.0%}\n"
            f"Max CI width = {max_ci_width:.2f}"
        ),
        x=0.02,
        y=0.92,
        va="top",
        fontsize=9,
    )

    ax.set_xlabel("Step ID")
    ax.set_ylabel("Mean viewpoint distance (lower = stronger isolation)")
    ax.set_ylim(0.0, 1.0)
    ax.grid(axis="y", alpha=0.18)

    fig.suptitle("Figure D — Multi-run variability in stance distance", y=0.985)
    add_figure_subtitle(fig, subtitle, y=0.942)
    ax.legend(loc="upper right")

    fig.tight_layout(rect=[0, 0, 1, 0.89])

    out_path = out_dir / "figure_d_multi_run_variability.png"
    save_figure_both_formats(fig, out_path)
    plt.close(fig)

    write_multi_run_summary_csv(run_dir)
    return out_path
