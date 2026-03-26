from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from fyp_sim.analysis import LockInMetrics, compute_lock_in_metrics

from .common import (
    LockInEpisode,
    RunPlotParams,
    detect_lock_in_episodes,
    first_seed_run_log,
    load_run_log_df,
    load_run_plot_params,
)
from .single_run_metrics import (
    ActionDistributionData,
    compute_action_distribution,
)


@dataclass(frozen=True, slots=True)
class CompareRunData:
    """Pre-computed data for one side of a run comparison."""

    label: str
    run_dir: Path
    display_path: str
    params: RunPlotParams
    primary_seed: str
    df: pd.DataFrame
    episodes: list[LockInEpisode]
    action_dist: ActionDistributionData
    lock_in: LockInMetrics


def load_compare_run(label: str, display_path: str, run_dir: Path) -> CompareRunData:
    """Load and validate a single run for comparison.

    Raises FileNotFoundError, ValueError, or TypeError on invalid run data.
    """
    params = load_run_plot_params(run_dir)
    run_log_path = first_seed_run_log(run_dir)
    primary_seed = run_log_path.parent.name
    df = load_run_log_df(run_log_path, run_dir=run_dir)

    episodes, _ = detect_lock_in_episodes(
        df, threshold=params.threshold, persistence_window=params.persistence_window
    )
    action_dist = compute_action_distribution(df)
    lock_in = compute_lock_in_metrics(
        df["viewpoint_distance"].tolist(),
        lock_in_threshold=params.threshold,
        persistence_window=params.persistence_window,
    )

    return CompareRunData(
        label=label,
        run_dir=run_dir,
        display_path=display_path,
        params=params,
        primary_seed=primary_seed,
        df=df,
        episodes=episodes,
        action_dist=action_dist,
        lock_in=lock_in,
    )
