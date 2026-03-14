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
    fig, ax = plt.subplots(figsize=(10.8, 5.6))

    mean_vii = float(df["viewpoint_distance"].mean())
    final_isolation_index = float(df["isolation_index"].iloc[-1])
    share_at_or_below_threshold = float((df["viewpoint_distance"] <= threshold).mean())

    ax.axhspan(
        0.0,
        threshold,
        alpha=0.05,
        color="C0",
        zorder=0,
    )

    ax.plot(
        df["step_id"],
        df["viewpoint_distance"],
        linewidth=0.9,
        alpha=0.35,
        color="0.55",
        label="Per-step viewpoint distance (VII_t)",
        zorder=1,
    )
    ax.plot(
        df["step_id"],
        df["isolation_index"],
        linewidth=2.4,
        color="C1",
        label="Isolation index (running mean VII)",
        zorder=3,
    )
    ax.axhline(
        threshold,
        linestyle="--",
        linewidth=1.2,
        color="C0",
        label=f"Bubble threshold ({threshold:.2f})",
        zorder=2,
    )

    add_annotation_box(
        ax,
        (
            f"Mean VII = {mean_vii:.2f}\n"
            f"Final isolation index = {final_isolation_index:.2f}\n"
            f"Steps ≤ threshold = {share_at_or_below_threshold:.0%}"
        ),
        x=0.02,
        y=0.92,
        va="top",
        fontsize=9,
    )

    ax.set_xlabel("Step ID")
    ax.set_ylabel("Viewpoint distance / isolation index (lower = stronger isolation)")
    ax.set_ylim(0.0, 1.0)
    ax.grid(axis="y", alpha=0.18)

    fig.suptitle("Figure A — Stance-distance convergence over time", y=0.985)
    add_figure_subtitle(fig, subtitle, y=0.942)
    ax.legend(loc="upper right")

    fig.tight_layout(rect=[0, 0, 1, 0.89])
    save_figure_both_formats(fig, out_path)
    plt.close(fig)


def plot_action_distribution(
    *,
    step_ids: pd.Series,
    action_data: ActionDistributionData,
    out_path: Path,
    subtitle: str,
) -> None:
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(13.2, 5.2))

    action_colors = {
        "Watch": "C0",
        "Sample": "C1",
        "Avoid": "C2",
    }

    actions = list(ACTION_ORDER)
    proportions = [action_data.proportions.get(action, 0.0) for action in actions]

    bars = ax_left.bar(
        actions,
        proportions,
        color=[action_colors[action] for action in actions],
    )
    ax_left.set_ylim(0.0, 1.0)
    ax_left.set_ylabel("Proportion")
    ax_left.set_title("Overall action mix", fontsize=12, pad=10)
    ax_left.grid(axis="y", alpha=0.18)

    for bar, value in zip(bars, proportions, strict=True):
        ax_left.text(
            bar.get_x() + (bar.get_width() / 2.0),
            min(0.97, bar.get_height() + 0.03),
            f"{value:.0%}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    for action in ACTION_ORDER:
        ax_right.plot(
            step_ids,
            action_data.rolling_rates[action],
            label=action,
            linewidth=2.0,
            color=action_colors[action],
        )

    add_annotation_box(
        ax_right,
        f"Rolling window = {action_data.window} steps",
        x=0.02,
        y=0.92,
        va="top",
        fontsize=9,
    )

    ax_right.set_ylim(0.0, 1.0)
    ax_right.set_xlabel("Step ID")
    ax_right.set_ylabel("Rolling action rate")
    ax_right.set_title("Rolling action rates", fontsize=12, pad=10)
    ax_right.grid(axis="y", alpha=0.18)
    ax_right.legend(loc="upper right")

    fig.suptitle("Figure B — Action distribution over time", y=0.985)
    add_figure_subtitle(fig, subtitle, y=0.942)

    fig.tight_layout(rect=[0, 0, 1, 0.89])
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
    fig, ax = plt.subplots(figsize=(10.8, 5.8))

    ax.axhspan(
        0.0,
        threshold,
        alpha=0.05,
        color="C0",
        zorder=0,
    )

    ax.plot(
        df["step_id"],
        df["viewpoint_distance"],
        linewidth=1.1,
        alpha=0.55,
        color="0.40",
        label="Per-step viewpoint distance (VII_t)",
        zorder=3,
    )
    ax.axhline(
        threshold,
        linestyle="--",
        linewidth=1.2,
        color="C0",
        label=f"Bubble threshold ({threshold:.2f})",
        zorder=2,
    )

    for idx, episode in enumerate(episodes):
        ax.axvspan(
            episode.start_step,
            episode.end_step,
            alpha=0.10,
            color="C0",
            label="Detected lock-in episode" if idx == 0 else None,
            zorder=1,
        )

    if episodes:
        first_start = episodes[0].start_step
        ax.axvline(
            first_start,
            linestyle=":",
            linewidth=1.1,
            color="C0",
            label="First lock-in start",
            zorder=2,
        )
    else:
        add_annotation_box(
            ax,
            "No lock-in episodes detected",
            x=0.02,
            y=0.82,
            va="top",
            fontsize=9,
        )

    add_annotation_box(
        ax,
        format_lockin_summary(summary),
        x=0.02,
        y=0.94,
        va="top",
        fontsize=10,
    )

    ax.set_xlabel("Step ID")
    ax.set_ylabel("Viewpoint distance (lower = stronger isolation)")
    ax.set_ylim(0.0, 1.0)
    ax.grid(axis="y", alpha=0.18)

    fig.suptitle("Figure C — Operational lock-in over time", y=0.985)
    add_figure_subtitle(fig, subtitle, y=0.942)
    ax.legend(loc="upper right")

    fig.tight_layout(rect=[0, 0, 1, 0.89])
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
