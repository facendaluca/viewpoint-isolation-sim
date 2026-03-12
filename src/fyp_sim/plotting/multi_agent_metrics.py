from __future__ import annotations

from pathlib import Path

import pandas as pd

from .common import detect_lock_in_episodes, load_run_log_df, load_run_plot_params, seed_dirs
from .single_run_metrics import ACTION_ORDER

PREFERRED_PHENOTYPE_ORDER = ("watcher", "sampler", "avoider")


def _normalise_phenotype(value: object) -> str:
    return str(value).strip().lower()


def ordered_phenotypes(values: pd.Series | list[str]) -> list[str]:
    seen = {_normalise_phenotype(value) for value in values}
    preferred = [name for name in PREFERRED_PHENOTYPE_ORDER if name in seen]
    remaining = sorted(seen - set(preferred))
    return preferred + remaining


def _load_multi_agent_seed_df(run_dir: Path, seed_dir: Path) -> pd.DataFrame:
    run_log_path = seed_dir / "run_log.csv"
    if not run_log_path.exists():
        raise FileNotFoundError(f"Expected run_log.csv at: {run_log_path}")

    df = load_run_log_df(run_log_path, run_dir=run_dir)
    if "agent_id" not in df.columns:
        raise ValueError(
            f"Multi-agent plotting requires 'agent_id' in the run logs: {run_log_path}"
        )

    out = df.copy()
    out["phenotype"] = out["agent_id"].map(_normalise_phenotype)
    return out


def has_multi_agent_run(run_dir: Path) -> bool:
    first_seed_dir = seed_dirs(run_dir)[0]
    df = _load_multi_agent_seed_df(run_dir, first_seed_dir)
    return df["phenotype"].nunique() > 1


def build_phenotype_seed_trajectories(run_dir: Path) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []

    for seed_dir in seed_dirs(run_dir):
        df = _load_multi_agent_seed_df(run_dir, seed_dir)

        grouped = (
            df.groupby(["phenotype", "step_id"], as_index=False)
            .agg(viewpoint_distance=("viewpoint_distance", "mean"))
            .sort_values(["phenotype", "step_id"], kind="stable")
            .reset_index(drop=True)
        )
        grouped["seed"] = seed_dir.name
        rows.append(grouped[["seed", "phenotype", "step_id", "viewpoint_distance"]])

    if not rows:
        raise ValueError(f"No multi-agent seed trajectories found in: {run_dir}")

    return pd.concat(rows, ignore_index=True)


def build_phenotype_trajectory_summary(run_dir: Path) -> pd.DataFrame:
    seed_traces = build_phenotype_seed_trajectories(run_dir)

    summary = (
        seed_traces.groupby(["phenotype", "step_id"], as_index=False)
        .agg(
            n_seeds=("seed", "nunique"),
            vii_mean=("viewpoint_distance", "mean"),
            vii_std=("viewpoint_distance", lambda s: s.std(ddof=1) if len(s) > 1 else 0.0),
        )
        .sort_values(["phenotype", "step_id"], kind="stable")
        .reset_index(drop=True)
    )

    sqrt_n = summary["n_seeds"].where(summary["n_seeds"] > 0, 1).astype(float).pow(0.5)
    stderr = summary["vii_std"] / sqrt_n
    ci_half_width = (1.96 * stderr).where(summary["n_seeds"] > 1, 0.0)

    summary["vii_ci_lower"] = (summary["vii_mean"] - ci_half_width).clip(0.0, 1.0)
    summary["vii_ci_upper"] = (summary["vii_mean"] + ci_half_width).clip(0.0, 1.0)

    return summary


def build_phenotype_action_summary(run_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, str | float]] = []

    for seed_dir in seed_dirs(run_dir):
        df = _load_multi_agent_seed_df(run_dir, seed_dir)

        for phenotype, group in df.groupby("phenotype", sort=False):
            action_series = group["user_action"].astype(str).str.title()

            for action in ACTION_ORDER:
                rows.append(
                    {
                        "seed": seed_dir.name,
                        "phenotype": phenotype,
                        "action": action,
                        "proportion": float(action_series.eq(action).mean()),
                    }
                )

    if not rows:
        raise ValueError(f"No multi-agent action rows found in: {run_dir}")

    seed_level = pd.DataFrame(rows)

    summary = (
        seed_level.groupby(["phenotype", "action"], as_index=False)
        .agg(
            proportion_mean=("proportion", "mean"),
            proportion_std=("proportion", lambda s: s.std(ddof=1) if len(s) > 1 else 0.0),
            n_seeds=("seed", "nunique"),
        )
        .sort_values(["phenotype", "action"], kind="stable")
        .reset_index(drop=True)
    )

    return summary


def build_phenotype_lockin_data(run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    params = load_run_plot_params(run_dir)

    summary_rows: list[dict[str, str | int | float]] = []
    episode_rows: list[dict[str, str | int]] = []

    for seed_dir in seed_dirs(run_dir):
        df = _load_multi_agent_seed_df(run_dir, seed_dir)

        for phenotype, group in df.groupby("phenotype", sort=False):
            group = group.sort_values("step_id", kind="stable").reset_index(drop=True)

            episodes, summary = detect_lock_in_episodes(
                group,
                threshold=params.threshold,
                persistence_window=params.persistence_window,
            )

            summary_rows.append(
                {
                    "seed": seed_dir.name,
                    "phenotype": phenotype,
                    "time_to_lock_in": summary["time_to_lock_in"],
                    "n_lockin_episodes": summary["n_lockin_episodes"],
                    "total_lockin_steps": summary["total_lockin_steps"],
                    "filter_bubble_threshold": params.threshold,
                    "persistence_window": params.persistence_window,
                }
            )

            for episode in episodes:
                episode_rows.append(
                    {
                        "seed": seed_dir.name,
                        "phenotype": phenotype,
                        "start_step": episode.start_step,
                        "end_step": episode.end_step,
                        "length": episode.length,
                    }
                )

    summary_df = (
        pd.DataFrame(summary_rows).sort_values(["phenotype", "seed"]).reset_index(drop=True)
    )

    if episode_rows:
        episodes_df = (
            pd.DataFrame(episode_rows)
            .sort_values(["phenotype", "seed", "start_step"])
            .reset_index(drop=True)
        )
    else:
        episodes_df = pd.DataFrame(
            columns=["seed", "phenotype", "start_step", "end_step", "length"]
        )

    return summary_df, episodes_df


def write_phenotype_lockin_summary_csv(run_dir: Path) -> Path:
    summary_df, _ = build_phenotype_lockin_data(run_dir)
    out_path = run_dir / "phenotype_lockin_summary.csv"
    summary_df.to_csv(out_path, index=False)
    return out_path
