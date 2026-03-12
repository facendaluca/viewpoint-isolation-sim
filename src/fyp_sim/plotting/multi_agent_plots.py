from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt

from .common import ensure_dir, load_run_plot_params, seed_dirs
from .multi_agent_metrics import (
    ACTION_ORDER,
    build_phenotype_action_summary,
    build_phenotype_lockin_data,
    build_phenotype_seed_trajectories,
    build_phenotype_trajectory_summary,
    has_multi_agent_run,
    ordered_phenotypes,
    write_phenotype_lockin_summary_csv,
)
from .plot_utils import _format_multi_run_subtitle, add_figure_subtitle, save_figure_both_formats


def plot_phenotype_vii_trajectories(
    *,
    seed_traces,
    summary,
    out_path: Path,
    threshold: float,
    subtitle: str,
) -> None:
    phenotypes = ordered_phenotypes(summary["phenotype"].tolist())

    fig, axes = plt.subplots(
        len(phenotypes),
        1,
        figsize=(10.8, max(6.4, 2.8 * len(phenotypes) + 1.6)),
        sharex=True,
        sharey=True,
    )
    if len(phenotypes) == 1:
        axes = [axes]

    used_seed_label = False
    used_mean_label = False
    used_ci_label = False
    used_threshold_label = False

    for ax, phenotype in zip(axes, phenotypes, strict=False):
        phenotype_traces = seed_traces[seed_traces["phenotype"] == phenotype]
        phenotype_summary = summary[summary["phenotype"] == phenotype]

        for _, seed_df in phenotype_traces.groupby("seed", sort=True):
            ax.plot(
                seed_df["step_id"],
                seed_df["viewpoint_distance"],
                linewidth=1.0,
                alpha=0.22,
                label="Seed trace" if not used_seed_label else None,
            )
            used_seed_label = True

        ax.fill_between(
            phenotype_summary["step_id"],
            phenotype_summary["vii_ci_lower"],
            phenotype_summary["vii_ci_upper"],
            alpha=0.10,
            label="95% CI" if not used_ci_label else None,
        )
        used_ci_label = True

        ax.plot(
            phenotype_summary["step_id"],
            phenotype_summary["vii_mean"],
            linewidth=2.0,
            label="Phenotype mean" if not used_mean_label else None,
        )
        used_mean_label = True

        ax.axhline(
            threshold,
            linestyle="--",
            linewidth=1.2,
            label=f"Bubble threshold ({threshold:.2f})" if not used_threshold_label else None,
        )
        used_threshold_label = True

        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel("Viewpoint distance")
        ax.set_title(phenotype.capitalize(), loc="left", fontsize=11)

    axes[-1].set_xlabel("Step ID")
    axes[0].legend(loc="upper right")

    fig.suptitle("Figure E - Phenotype stance-distance trajectories", y=0.99)
    add_figure_subtitle(fig, subtitle, y=0.955)

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save_figure_both_formats(fig, out_path)
    plt.close(fig)


def plot_phenotype_action_mix(
    *,
    action_summary,
    out_path: Path,
    subtitle: str,
) -> None:
    phenotypes = ordered_phenotypes(action_summary["phenotype"].tolist())
    fig, ax = plt.subplots(figsize=(10.5, max(4.6, 1.2 * len(phenotypes) + 2.0)))

    left_offsets = [0.0] * len(phenotypes)

    for action in ACTION_ORDER:
        action_rows = action_summary[action_summary["action"] == action]
        values = [
            float(action_rows.loc[action_rows["phenotype"] == phenotype, "proportion_mean"].iloc[0])
            for phenotype in phenotypes
        ]

        bars = ax.barh(
            [phenotype.capitalize() for phenotype in phenotypes],
            values,
            left=left_offsets,
            label=action,
        )

        for bar, value in zip(bars, values, strict=False):
            if value >= 0.88:
                ax.text(
                    bar.get_x() + (bar.get_width() / 2.0),
                    bar.get_y() + (bar.get_height() / 2.0),
                    f"{value:.0%}",
                    ha="center",
                    va="center",
                    fontsize=9,
                )

        left_offsets = [left + value for left, value in zip(left_offsets, values, strict=False)]

    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("Proportion of actions")
    ax.set_ylabel("Phenotype")
    ax.set_title("Figure F - Phenotype action mix", pad=18)
    add_figure_subtitle(fig, subtitle, y=0.94)
    ax.legend(loc="lower right")
    ax.invert_yaxis()

    fig.tight_layout(rect=[0, 0, 1, 0.92])
    save_figure_both_formats(fig, out_path)
    plt.close(fig)


def plot_phenotype_lockin_timeline(
    *,
    lockin_summary,
    episode_rows,
    out_path: Path,
    threshold: float,
    persistence_window: int,
    subtitle: str,
) -> None:
    phenotypes = ordered_phenotypes(lockin_summary["phenotype"].tolist())

    fig, axes = plt.subplots(
        len(phenotypes),
        1,
        figsize=(11.2, max(7.2, 2.6 * len(phenotypes) + 2.2)),
        sharex=True,
    )
    if len(phenotypes) == 1:
        axes = [axes]

    used_episode_label = False
    used_onset_label = False

    for ax, phenotype in zip(axes, phenotypes, strict=False):
        phenotype_summary = (
            lockin_summary[lockin_summary["phenotype"] == phenotype]
            .sort_values("seed")
            .reset_index(drop=True)
        )
        phenotype_episodes = episode_rows[episode_rows["phenotype"] == phenotype]

        seeds = phenotype_summary["seed"].tolist()
        y_positions = list(range(len(seeds)))

        for y_pos, seed in zip(y_positions, seeds, strict=False):
            seed_episodes = phenotype_episodes[phenotype_episodes["seed"] == seed]

            for _, episode in seed_episodes.iterrows():
                ax.broken_barh(
                    [(int(episode["start_step"]), int(episode["length"]))],
                    (y_pos - 0.35, 0.7),
                    alpha=0.25,
                    label="Lock-in episode" if not used_episode_label else None,
                )
                used_episode_label = True

            onset = int(
                phenotype_summary.loc[phenotype_summary["seed"] == seed, "time_to_lock_in"].iloc[0]
            )
            if onset >= 0:
                ax.scatter(
                    onset,
                    y_pos,
                    marker="|",
                    s=240,
                    label="First lock-in onset" if not used_onset_label else None,
                    zorder=3,
                )
                used_onset_label = True

        if phenotype_episodes.empty:
            ax.text(
                0.02,
                0.88,
                "No lock-in episodes detected",
                transform=ax.transAxes,
                fontsize=10,
                bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.9},
            )

        ax.set_yticks(y_positions, seeds)
        ax.set_ylabel("Seed")
        ax.set_title(
            f"{phenotype.capitalize()} (threshold={threshold:.2f}, persistence={persistence_window})",
            loc="left",
            fontsize=11,
        )

    axes[-1].set_xlabel("Step ID")

    if used_episode_label or used_onset_label:
        axes[0].legend(loc="upper right")

    fig.suptitle("Figure G - Phenotype lock-in timelines", y=0.99)
    add_figure_subtitle(fig, subtitle, y=0.955)

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save_figure_both_formats(fig, out_path)
    plt.close(fig)


def plot_multi_agent_figures(run_dir: Path) -> Path | None:
    if not has_multi_agent_run(run_dir):
        return None

    out_dir = run_dir / "plots"
    ensure_dir(out_dir)

    params = load_run_plot_params(run_dir)
    subtitle = _format_multi_run_subtitle(
        n_seeds=len(seed_dirs(run_dir)),
        threshold=params.threshold,
        persistence_window=params.persistence_window,
        steps=params.steps,
        rank_alpha=params.rank_alpha,
        drift_alpha=params.drift_alpha,
    )

    seed_traces = build_phenotype_seed_trajectories(run_dir)
    trajectory_summary = build_phenotype_trajectory_summary(run_dir)
    action_summary = build_phenotype_action_summary(run_dir)
    lockin_summary, episode_rows = build_phenotype_lockin_data(run_dir)

    plot_phenotype_vii_trajectories(
        seed_traces=seed_traces,
        summary=trajectory_summary,
        out_path=out_dir / "figure_e_phenotype_vii_trajectories.png",
        threshold=params.threshold,
        subtitle=subtitle,
    )
    plot_phenotype_action_mix(
        action_summary=action_summary,
        out_path=out_dir / "figure_f_phenotype_action_mix.png",
        subtitle=subtitle,
    )
    plot_phenotype_lockin_timeline(
        lockin_summary=lockin_summary,
        episode_rows=episode_rows,
        out_path=out_dir / "figure_g_phenotype_lockin_timeline.png",
        threshold=params.threshold,
        persistence_window=params.persistence_window,
        subtitle=subtitle,
    )
    write_phenotype_lockin_summary_csv(run_dir)
    return out_dir
