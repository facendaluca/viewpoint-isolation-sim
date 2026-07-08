from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt

from .common import ensure_dir, load_run_plot_params, seed_dirs
from .multi_agent_metrics import (
    ACTION_ORDER,
    build_phenotype_action_dynamics_summary,
    build_phenotype_lockin_data,
    build_phenotype_lockin_outcome_summary,
    build_phenotype_seed_trajectories,
    build_phenotype_trajectory_summary,
    has_multi_agent_run,
    ordered_phenotypes,
    write_phenotype_lockin_summary_csv,
)
from .plot_utils import (
    add_annotation_box,
    add_figure_subtitle,
    format_experiment_subtitle,
    save_figure_both_formats,
)


def _resolve_trajectory_smoothing_window(persistence_window: int) -> int:
    window = max(3, persistence_window // 2)
    if window % 2 == 0:
        window += 1
    return min(window, 9)


def _seed_point_offsets(n_points: int, *, spread: float = 0.12) -> list[float]:
    if n_points <= 1:
        return [0.0]

    step = (2.0 * spread) / float(n_points - 1)
    return [(-spread + (idx * step)) for idx in range(n_points)]


def _plot_box_and_points(
    ax,
    *,
    grouped_values: list[list[float]],
    positions: list[int],
) -> None:
    non_empty_values: list[list[float]] = []
    non_empty_positions: list[int] = []

    for position, values in zip(positions, grouped_values, strict=False):
        if values:
            non_empty_values.append(values)
            non_empty_positions.append(position)

    if non_empty_positions:
        ax.boxplot(
            non_empty_values,
            positions=non_empty_positions,
            widths=0.55,
            showfliers=False,
        )

    for position, values in zip(positions, grouped_values, strict=False):
        if not values:
            continue

        offsets = _seed_point_offsets(len(values))
        x_values = [position + offset for offset in offsets]
        ax.scatter(
            x_values,
            values,
            s=28,
            alpha=0.85,
            zorder=3,
        )


def plot_phenotype_vii_trajectories(
    *,
    seed_traces,
    summary,
    out_path: Path,
    threshold: float,
    subtitle: str,
    smoothing_window: int,
) -> None:
    phenotypes = ordered_phenotypes(summary["phenotype"].tolist())

    fig, axes = plt.subplots(
        len(phenotypes),
        1,
        figsize=(10.8, max(6.6, 2.9 * len(phenotypes) + 1.8)),
        sharex=True,
        sharey=True,
    )
    if len(phenotypes) == 1:
        axes = [axes]

    used_seed_label = False
    used_ci_label = False
    used_raw_mean_label = False
    used_smooth_mean_label = False
    used_threshold_label = False

    for ax, phenotype in zip(axes, phenotypes, strict=False):
        phenotype_traces = (
            seed_traces[seed_traces["phenotype"] == phenotype]
            .sort_values(["seed", "step_id"], kind="stable")
            .reset_index(drop=True)
        )
        phenotype_summary = (
            summary[summary["phenotype"] == phenotype]
            .sort_values("step_id", kind="stable")
            .reset_index(drop=True)
            .copy()
        )

        phenotype_summary["vii_mean_smooth"] = (
            phenotype_summary["vii_mean"]
            .astype(float)
            .rolling(window=smoothing_window, center=True, min_periods=1)
            .mean()
        )

        for _, seed_df in phenotype_traces.groupby("seed", sort=True):
            ax.plot(
                seed_df["step_id"],
                seed_df["viewpoint_distance"],
                linewidth=0.7,
                alpha=0.08,
                color="0.70",
                label="Seed trace" if not used_seed_label else None,
                zorder=1,
            )
            used_seed_label = True

        ax.fill_between(
            phenotype_summary["step_id"],
            phenotype_summary["vii_ci_lower"],
            phenotype_summary["vii_ci_upper"],
            alpha=0.08,
            color="C1",
            label="95% CI" if not used_ci_label else None,
            zorder=2,
        )
        used_ci_label = True

        ax.plot(
            phenotype_summary["step_id"],
            phenotype_summary["vii_mean"],
            linewidth=1.1,
            alpha=0.45,
            color="C1",
            label="Per-step mean" if not used_raw_mean_label else None,
            zorder=3,
        )
        used_raw_mean_label = True

        ax.plot(
            phenotype_summary["step_id"],
            phenotype_summary["vii_mean_smooth"],
            linewidth=2.5,
            color="C1",
            label=f"Smoothed mean ({smoothing_window}-step)"
            if not used_smooth_mean_label
            else None,
            zorder=4,
        )
        used_smooth_mean_label = True

        ax.axhline(
            threshold,
            linestyle="--",
            linewidth=1.1,
            color="C0",
            label=f"Bubble threshold ({threshold:.2f})" if not used_threshold_label else None,
            zorder=2,
        )
        used_threshold_label = True

        mean_vii = float(phenotype_summary["vii_mean"].mean())
        share_at_or_above_threshold = float((phenotype_summary["vii_mean"] >= threshold).mean())

        add_annotation_box(
            ax,
            (f"Mean VII = {mean_vii:.2f}\nSteps ≥ threshold = {share_at_or_above_threshold:.0%}"),
            x=0.02,
            y=0.90,
            va="top",
            fontsize=9,
        )

        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel("Viewpoint distance")
        ax.set_title(phenotype.capitalize(), loc="left", fontsize=11)
        ax.grid(axis="y", alpha=0.18)

    axes[-1].set_xlabel("Step ID")
    axes[0].legend(loc="upper right")

    fig.suptitle("Figure E — Phenotype stance-distance trajectories", y=0.99)
    add_figure_subtitle(fig, subtitle, y=0.955)

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save_figure_both_formats(fig, out_path)
    plt.close(fig)


def plot_phenotype_action_dynamics(
    *,
    action_dynamics_summary,
    out_path: Path,
    subtitle: str,
) -> None:
    phenotypes = ordered_phenotypes(action_dynamics_summary["phenotype"].tolist())
    rolling_window = int(action_dynamics_summary["rolling_window"].iloc[0])

    fig, axes = plt.subplots(
        len(phenotypes),
        1,
        figsize=(10.8, max(6.6, 2.9 * len(phenotypes) + 1.8)),
        sharex=True,
        sharey=True,
    )
    if len(phenotypes) == 1:
        axes = [axes]

    for ax, phenotype in zip(axes, phenotypes, strict=False):
        phenotype_df = (
            action_dynamics_summary[action_dynamics_summary["phenotype"] == phenotype]
            .sort_values(["action", "step_id"], kind="stable")
            .reset_index(drop=True)
        )

        for action in ACTION_ORDER:
            action_df = phenotype_df[phenotype_df["action"] == action].sort_values(
                "step_id",
                kind="stable",
            )

            ax.fill_between(
                action_df["step_id"],
                action_df["action_rate_ci_lower"],
                action_df["action_rate_ci_upper"],
                alpha=0.06,
                zorder=1,
            )
            ax.plot(
                action_df["step_id"],
                action_df["action_rate_mean"],
                linewidth=2.0,
                label=action if phenotype == phenotypes[0] else None,
                zorder=2,
            )

        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel("Rolling action rate")
        ax.set_title(phenotype.capitalize(), loc="left", fontsize=11)
        ax.grid(axis="y", alpha=0.20)

    add_annotation_box(
        axes[0],
        f"Rolling window = {rolling_window} steps",
        x=0.02,
        y=0.92,
        va="top",
        fontsize=9,
    )

    axes[-1].set_xlabel("Step ID")
    axes[0].legend(loc="upper right")

    fig.suptitle("Figure F — Phenotype action dynamics", y=0.99)
    add_figure_subtitle(fig, subtitle, y=0.955)

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save_figure_both_formats(fig, out_path)
    plt.close(fig)


def plot_phenotype_lockin_outcomes(
    *,
    lockin_outcomes,
    out_path: Path,
    subtitle: str,
) -> None:
    phenotypes = ordered_phenotypes(lockin_outcomes["phenotype"].tolist())
    phenotype_labels = [phenotype.capitalize() for phenotype in phenotypes]
    positions = list(range(1, len(phenotypes) + 1))

    fig, axes = plt.subplots(1, 3, figsize=(13.4, 5.6))

    locked_counts = [
        int(lockin_outcomes.loc[lockin_outcomes["phenotype"] == phenotype, "locked_in"].sum())
        for phenotype in phenotypes
    ]
    total_counts = [
        int((lockin_outcomes["phenotype"] == phenotype).sum()) for phenotype in phenotypes
    ]

    locked_note_lines = [
        f"{label} {locked}/{total} locked"
        for label, locked, total in zip(phenotype_labels, locked_counts, total_counts, strict=True)
    ]
    metric_specs = [
        (
            "time_to_lock_in_plot",
            "Time to first lock-in",
            "Step ID",
            "\n".join(["Non-locking seeds omitted", *locked_note_lines]),
        ),
        (
            "n_lockin_episodes",
            "Lock-in episodes",
            "Episodes",
            None,
        ),
        (
            "total_lockin_steps",
            "Total lock-in steps",
            "Steps",
            None,
        ),
    ]

    for ax, (metric_col, panel_title, y_label, note_text) in zip(axes, metric_specs, strict=False):
        grouped_values: list[list[float]] = []

        for phenotype in phenotypes:
            values = (
                lockin_outcomes.loc[lockin_outcomes["phenotype"] == phenotype, metric_col]
                .dropna()
                .astype(float)
                .tolist()
            )
            grouped_values.append(values)

        _plot_box_and_points(
            ax,
            grouped_values=grouped_values,
            positions=positions,
        )

        ax.set_xticks(positions, phenotype_labels)
        ax.set_xlim(0.5, len(phenotypes) + 0.5)
        ax.set_title(panel_title, fontsize=11, pad=10)
        ax.set_ylabel(y_label)
        ax.grid(axis="y", alpha=0.20)

        if note_text is not None:
            add_annotation_box(
                ax,
                note_text,
                x=0.03,
                y=0.92,
                va="top",
                fontsize=8,
            )

    fig.suptitle("Figure G — Phenotype lock-in outcome comparison", y=0.985)
    add_figure_subtitle(fig, subtitle, y=0.94)

    fig.tight_layout(rect=[0, 0, 1, 0.86])
    save_figure_both_formats(fig, out_path)
    plt.close(fig)


def plot_phenotype_lockin_timeline(
    *,
    lockin_summary,
    episode_rows,
    out_path: Path,
    subtitle: str,
) -> None:
    phenotypes = ordered_phenotypes(lockin_summary["phenotype"].tolist())

    fig, axes = plt.subplots(
        len(phenotypes),
        1,
        figsize=(11.2, max(7.2, 2.5 * len(phenotypes) + 2.2)),
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
                    alpha=0.30,
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
                    s=180,
                    label="First lock-in onset" if not used_onset_label else None,
                    zorder=3,
                )
                used_onset_label = True

        if phenotype_episodes.empty:
            add_annotation_box(
                ax,
                "No lock-in episodes detected",
                x=0.02,
                y=0.88,
                va="top",
                fontsize=10,
            )

        ax.set_yticks(y_positions, seeds)
        ax.set_ylabel("Seed")
        ax.set_title(phenotype.capitalize(), loc="left", fontsize=11)

    axes[-1].set_xlabel("Step ID")

    if used_episode_label or used_onset_label:
        axes[0].legend(loc="upper right")

    fig.suptitle("Supplementary — Phenotype lock-in timeline", y=0.99)
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
    subtitle = format_experiment_subtitle(
        n_seeds=len(seed_dirs(run_dir)),
        threshold=params.threshold,
        persistence_window=params.persistence_window,
        steps=params.steps,
        rank_alpha=params.rank_alpha,
        drift_alpha=params.drift_alpha,
    )

    trajectory_smoothing_window = _resolve_trajectory_smoothing_window(params.persistence_window)

    seed_traces = build_phenotype_seed_trajectories(run_dir)
    trajectory_summary = build_phenotype_trajectory_summary(run_dir)
    action_dynamics_summary = build_phenotype_action_dynamics_summary(run_dir)
    lockin_outcomes = build_phenotype_lockin_outcome_summary(run_dir)
    lockin_summary, episode_rows = build_phenotype_lockin_data(run_dir)

    plot_phenotype_vii_trajectories(
        seed_traces=seed_traces,
        summary=trajectory_summary,
        out_path=out_dir / "figure_e_phenotype_vii_trajectories.png",
        threshold=params.threshold,
        subtitle=subtitle,
        smoothing_window=trajectory_smoothing_window,
    )
    plot_phenotype_action_dynamics(
        action_dynamics_summary=action_dynamics_summary,
        out_path=out_dir / "figure_f_phenotype_action_dynamics.png",
        subtitle=subtitle,
    )
    plot_phenotype_lockin_outcomes(
        lockin_outcomes=lockin_outcomes,
        out_path=out_dir / "figure_g_phenotype_lockin_outcomes.png",
        subtitle=subtitle,
    )
    plot_phenotype_lockin_timeline(
        lockin_summary=lockin_summary,
        episode_rows=episode_rows,
        out_path=out_dir / "figure_g_sup_phenotype_lockin_timeline.png",
        subtitle=subtitle,
    )

    write_phenotype_lockin_summary_csv(run_dir)
    return out_dir
