from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import pandas as pd

from .common import LockInEpisode
from .plot_utils import (
    add_annotation_box,
    add_figure_subtitle,
    format_lockin_summary,
    save_figure_both_formats,
)
from .single_run_metrics import (
    ACTION_ORDER,
    ActionDistributionData,
    build_single_run_context,
    compute_action_distribution,
    write_lockin_summary_csv,
)


def plot_vii_trajectory(
    *,
    df: pd.DataFrame,
    out_path: Path,
    threshold: float,
    subtitle: str,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(
        df["step_id"],
        df["viewpoint_distance"],
        label="Viewpoint distance (VII_t)",
    )
    ax.plot(
        df["step_id"],
        df["isolation_index"],
        label="Isolation index (running mean VII)",
    )
    ax.axhline(
        threshold,
        linestyle="--",
        linewidth=1.2,
        label=f"Bubble threshold ({threshold:.2f})",
    )

    ax.set_xlabel("Step ID")
    ax.set_ylabel("Viewpoint distance / isolation index")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Figure A - Stance-distance convergence over time", pad=18)
    fig.text(0.5, 0.94, subtitle, ha="center", va="center", fontsize=10)
    ax.legend(loc="upper right")

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save_figure_both_formats(fig, out_path)
    plt.close(fig)


def plot_action_distribution(
    *,
    step_ids: pd.Series,
    action_data: ActionDistributionData,
    out_path: Path,
    subtitle: str,
) -> None:
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(13, 4.8))

    ax_left.bar(
        list(ACTION_ORDER),
        [action_data.proportions.get(action, 0.0) for action in ACTION_ORDER],
    )
    ax_left.set_ylim(0.0, 1.0)
    ax_left.set_ylabel("Proportion")
    ax_left.set_title("Overall action mix")

    for action in ACTION_ORDER:
        ax_right.plot(step_ids, action_data.rolling_rates[action], label=action)

    ax_right.set_ylim(0.0, 1.0)
    ax_right.set_xlabel("Step ID")
    ax_right.set_ylabel(f"Rolling rate (window={action_data.window})")
    ax_right.set_title("Rolling action rates")
    ax_right.legend(loc="upper right")

    fig.suptitle("Figure B — Action distribution over time", y=0.98)
    add_figure_subtitle(fig, subtitle, y=0.92)

    fig.tight_layout(rect=[0, 0, 1, 0.88])
    save_figure_both_formats(fig, out_path)
    plt.close(fig)


def plot_lockin_episodes(
    *,
    df: pd.DataFrame,
    out_path: Path,
    threshold: float,
    episodes: list[LockInEpisode],
    summary: dict[str, int],
    subtitle: str,
) -> None:
    fig, ax = plt.subplots(figsize=(11, 5.8))

    ax.plot(
        df["step_id"],
        df["viewpoint_distance"],
        linewidth=1.8,
        label="Viewpoint distance (VII_t)",
        zorder=3,
    )
    ax.axhline(
        threshold,
        linestyle="--",
        linewidth=1.4,
        label=f"Bubble threshold ({threshold:.2f})",
        zorder=2,
    )

    for idx, episode in enumerate(episodes):
        label = "Detected lock-in episode" if idx == 0 else None
        ax.axvspan(
            episode.start_step,
            episode.end_step,
            alpha=0.12,
            label=label,
            zorder=1,
        )

    if episodes:
        first_start = episodes[0].start_step
        ax.axvline(
            first_start,
            linestyle=":",
            linewidth=1.4,
            label="First lock-in start",
            zorder=2,
        )
        ax.annotate(
            f"Start = {first_start}",
            xy=(first_start, threshold),
            xytext=(first_start + 6, min(0.92, threshold + 0.14)),
            arrowprops={"arrowstyle": "->", "lw": 1.0},
            fontsize=9,
        )
    else:
        add_annotation_box(
            ax,
            "No lock-in episodes detected",
            x=0.02,
            y=0.90,
            va="top",
            fontsize=10,
        )

    add_annotation_box(
        ax,
        format_lockin_summary(summary),
        x=0.02,
        y=0.98,
        va="top",
        fontsize=10,
    )

    ax.set_xlabel("Step ID")
    ax.set_ylabel("Viewpoint distance (VII_t; lower = stronger isolation)")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Figure C — Operational lock-in over time", pad=18)
    add_figure_subtitle(fig, subtitle, y=0.94)
    ax.legend(loc="upper right")

    fig.tight_layout(rect=[0, 0, 1, 0.92])
    save_figure_both_formats(fig, out_path)
    plt.close(fig)


def plot_single_run_figures(run_dir: Path) -> Path:
    context = build_single_run_context(run_dir)
    action_data = compute_action_distribution(context.df)

    plot_vii_trajectory(
        df=context.df,
        out_path=context.out_dir / "figure_a_vii_trajectory.png",
        threshold=context.params.threshold,
        subtitle=context.subtitle,
    )
    plot_action_distribution(
        step_ids=context.df["step_id"],
        action_data=action_data,
        out_path=context.out_dir / "figure_b_action_distribution.png",
        subtitle=context.subtitle,
    )
    plot_lockin_episodes(
        df=context.df,
        out_path=context.out_dir / "figure_c_lockin_episodes.png",
        threshold=context.params.threshold,
        episodes=context.episodes,
        summary=context.lockin_summary,
        subtitle=context.subtitle,
    )
    write_lockin_summary_csv(run_dir)
    return context.out_dir
