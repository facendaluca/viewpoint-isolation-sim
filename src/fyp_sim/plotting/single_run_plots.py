from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import pandas as pd

from .common import (
    LockInEpisode,
    detect_lock_in_episodes,
    ensure_dir,
    first_seed_run_log,
    load_run_log_df,
    load_run_plot_params,
    seed_dirs,
)
from .plot_utils import format_lockin_summary, format_run_subtitle, save_figure_both_formats

ACTION_ORDER = ("Watch", "Sample", "Avoid")
ROLLING_WINDOW = 25


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
    df: pd.DataFrame,
    out_path: Path,
    subtitle: str,
) -> None:
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(13, 4.8))

    action_series = df["user_action"].astype(str).str.title()
    proportions = action_series.value_counts(normalize=True)

    ax_left.bar(
        list(ACTION_ORDER),
        [float(proportions.get(action, 0.0)) for action in ACTION_ORDER],
    )
    ax_left.set_ylim(0.0, 1.0)
    ax_left.set_ylabel("Proportion")
    ax_left.set_title("Overall action mix")

    window = min(ROLLING_WINDOW, max(1, len(df)))
    for action in ACTION_ORDER:
        rate = action_series.eq(action).astype(float).rolling(window, min_periods=1).mean()
        ax_right.plot(df["step_id"], rate, label=action)

    ax_right.set_ylim(0.0, 1.0)
    ax_right.set_xlabel("Step ID")
    ax_right.set_ylabel(f"Rolling rate (window={window})")
    ax_right.set_title("Rolling action rates")
    ax_right.legend(loc="upper right")

    fig.suptitle("Figure B - Action distribution over time", y=0.98)
    fig.text(0.5, 0.92, subtitle, ha="center", va="center", fontsize=10)

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
        ax.text(
            0.02,
            0.90,
            "No lock-in episodes detected",
            transform=ax.transAxes,
            fontsize=10,
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.9},
        )

    ax.text(
        0.02,
        0.98,
        format_lockin_summary(summary),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.9},
    )

    ax.set_xlabel("Step ID")
    ax.set_ylabel("Viewpoint distance (VII_t; lower = stronger isolation)")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Figure C - Operational lock-in over time", pad=18)
    fig.text(0.5, 0.94, subtitle, ha="center", va="center", fontsize=10)
    ax.legend(loc="upper right")

    fig.tight_layout(rect=[0, 0, 1, 0.92])
    save_figure_both_formats(fig, out_path)
    plt.close(fig)


def write_lockin_summary_csv(run_dir: Path) -> Path:
    params = load_run_plot_params(run_dir)
    rows: list[dict[str, int | float | str]] = []

    for seed_dir in seed_dirs(run_dir):
        run_log_path = seed_dir / "run_log.csv"
        if not run_log_path.exists():
            raise FileNotFoundError(f"Expected run_log.csv at: {run_log_path}")

        df = load_run_log_df(run_log_path, run_dir=run_dir)
        _, summary = detect_lock_in_episodes(
            df,
            threshold=params.threshold,
            persistence_window=params.persistence_window,
        )

        rows.append(
            {
                "seed": seed_dir.name,
                "time_to_lock_in": summary["time_to_lock_in"],
                "n_lockin_episodes": summary["n_lockin_episodes"],
                "total_lockin_steps": summary["total_lockin_steps"],
                "filter_bubble_threshold": params.threshold,
                "persistence_window": params.persistence_window,
            }
        )

    out_path = run_dir / "lockin_summary.csv"
    pd.DataFrame(rows).sort_values("seed").to_csv(out_path, index=False)
    return out_path


def plot_single_run_figures(run_dir: Path) -> Path:
    out_dir = run_dir / "plots"
    ensure_dir(out_dir)

    params = load_run_plot_params(run_dir)
    primary_run_log = first_seed_run_log(run_dir)
    primary_seed = primary_run_log.parent.name

    df = load_run_log_df(primary_run_log, run_dir=run_dir)

    subtitle = format_run_subtitle(
        seed_label=primary_seed,
        threshold=params.threshold,
        persistence_window=params.persistence_window,
        steps=params.steps,
        rank_alpha=params.rank_alpha,
        drift_alpha=params.drift_alpha,
    )

    episodes, summary = detect_lock_in_episodes(
        df,
        threshold=params.threshold,
        persistence_window=params.persistence_window,
    )

    plot_vii_trajectory(
        df=df,
        out_path=out_dir / "figure_a_vii_trajectory.png",
        threshold=params.threshold,
        subtitle=subtitle,
    )
    plot_action_distribution(
        df=df,
        out_path=out_dir / "figure_b_action_distribution.png",
        subtitle=subtitle,
    )
    plot_lockin_episodes(
        df=df,
        out_path=out_dir / "figure_c_lockin_episodes.png",
        threshold=params.threshold,
        episodes=episodes,
        summary=summary,
        subtitle=subtitle,
    )
    write_lockin_summary_csv(run_dir)
    return out_dir
