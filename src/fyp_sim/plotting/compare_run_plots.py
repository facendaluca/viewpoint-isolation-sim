from __future__ import annotations

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from .compare_run_data import CompareRunData
from .single_run_metrics import ACTION_ORDER


def plot_vii_overlay(a: CompareRunData, b: CompareRunData) -> Figure:
    """Overlay viewpoint-distance trajectories for runs A and B."""
    fig, ax = plt.subplots(figsize=(10.8, 5.6))

    ax.plot(
        a.df["step_id"],
        a.df["viewpoint_distance"],
        label=f"Run A — {a.primary_seed}",
        color="C0",
        linewidth=1.6,
        alpha=0.85,
    )
    ax.plot(
        b.df["step_id"],
        b.df["viewpoint_distance"],
        label=f"Run B — {b.primary_seed}",
        color="C1",
        linewidth=1.6,
        alpha=0.85,
    )

    if a.params.threshold == b.params.threshold:
        ax.axhline(
            a.params.threshold,
            ls="--",
            lw=1.2,
            color="grey",
            label=f"Threshold ({a.params.threshold:.2f})",
        )
    else:
        ax.axhline(
            a.params.threshold,
            ls="--",
            lw=1.2,
            color="C0",
            label=f"Threshold A ({a.params.threshold:.2f})",
        )
        ax.axhline(
            b.params.threshold,
            ls="--",
            lw=1.2,
            color="C1",
            label=f"Threshold B ({b.params.threshold:.2f})",
        )

    ax.set_xlabel("Step")
    ax.set_ylabel("Viewpoint distance (lower = stronger isolation)")
    ax.set_ylim(0.0, 1.0)
    ax.grid(axis="y", alpha=0.18)
    ax.legend(loc="upper right")
    fig.suptitle("Viewpoint Distance Trajectory — Run A vs Run B")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def plot_lockin_timeline(a: CompareRunData, b: CompareRunData) -> Figure:
    """Stacked lock-in timeline showing episodes for both runs."""
    fig, (ax_a, ax_b) = plt.subplots(2, 1, figsize=(10.8, 7), sharex=True)

    for ax, data, color, run_label in [
        (ax_a, a, "C0", "Run A"),
        (ax_b, b, "C1", "Run B"),
    ]:
        ax.plot(
            data.df["step_id"],
            data.df["viewpoint_distance"],
            color=color,
            linewidth=1.2,
        )
        ax.axhline(data.params.threshold, ls="--", lw=1.0, color="grey")
        for episode in data.episodes:
            ax.axvspan(episode.start_step, episode.end_step, alpha=0.15, color="red")
        ax.set_ylabel(f"{run_label}\n({data.primary_seed})", fontsize=9)
        ax.set_ylim(0.0, 1.0)
        ax.grid(axis="y", alpha=0.18)

    ax_b.set_xlabel("Step")
    fig.suptitle("Lock-in Timeline Comparison")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def plot_action_mix(a: CompareRunData, b: CompareRunData) -> Figure:
    """Grouped bar chart of action proportions and rolling watch-rate overlay."""
    fig, (ax_bar, ax_roll) = plt.subplots(1, 2, figsize=(12, 5))

    x = range(len(ACTION_ORDER))
    width = 0.35

    values_a = [a.action_dist.proportions[action] for action in ACTION_ORDER]
    values_b = [b.action_dist.proportions[action] for action in ACTION_ORDER]

    ax_bar.bar([i - width / 2 for i in x], values_a, width, label="Run A", color="C0")
    ax_bar.bar([i + width / 2 for i in x], values_b, width, label="Run B", color="C1")
    ax_bar.set_xticks(list(x))
    ax_bar.set_xticklabels(ACTION_ORDER)
    ax_bar.set_ylabel("Proportion")
    ax_bar.set_ylim(0.0, 1.0)
    ax_bar.legend()
    ax_bar.set_title("Action Distribution")

    ax_roll.plot(
        a.action_dist.rolling_rates["step_id"],
        a.action_dist.rolling_rates["Watch"],
        label="Run A",
        color="C0",
        linewidth=1.4,
    )
    ax_roll.plot(
        b.action_dist.rolling_rates["step_id"],
        b.action_dist.rolling_rates["Watch"],
        label="Run B",
        color="C1",
        linewidth=1.4,
    )
    ax_roll.set_xlabel("Step")
    ax_roll.set_ylabel("Rolling Watch rate")
    ax_roll.set_ylim(0.0, 1.0)
    ax_roll.legend()
    ax_roll.set_title(f"Rolling Watch Rate (window={a.action_dist.window})")

    fig.tight_layout()
    return fig
