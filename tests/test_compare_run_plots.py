from pathlib import Path

import pandas as pd
from matplotlib.figure import Figure

from fyp_sim.analysis import LockInMetrics
from fyp_sim.plotting.common import LockInEpisode, RunPlotParams
from fyp_sim.plotting.compare_run_data import CompareRunData
from fyp_sim.plotting.compare_run_plots import (
    plot_action_mix,
    plot_lockin_timeline,
    plot_vii_overlay,
)
from fyp_sim.plotting.single_run_metrics import ActionDistributionData


def _make_compare_run_data(
    *,
    label: str,
    watch_rate: float,
    threshold: float = 0.2,
    seed: str = "s00000",
) -> CompareRunData:
    df = pd.DataFrame(
        {
            "step_id": [0, 1, 2, 3, 4],
            "viewpoint_distance": [0.6, 0.1, 0.1, 0.5, 0.1],
            "isolation_index": [0.6, 0.35, 0.2667, 0.325, 0.28],
        }
    )

    params = RunPlotParams(
        steps=5,
        threshold=threshold,
        persistence_window=2,
        rank_alpha=0.5,
        drift_alpha=0.1,
        seeds=[seed],
    )

    episodes = [LockInEpisode(start_step=1, end_step=2, length=2)]

    action_dist = ActionDistributionData(
        proportions={
            "Watch": watch_rate,
            "Sample": 1.0 - watch_rate if watch_rate < 1.0 else 0.0,
            "Avoid": 0.0,
        },
        rolling_rates=pd.DataFrame(
            {
                "step_id": [0, 1, 2, 3, 4],
                "Watch": [watch_rate] * 5,
            }
        ),
        window=25,
    )

    lock_in = LockInMetrics(
        lock_in_events=1,
        time_to_first_lock_in=1,
        max_consecutive_lock_in_steps=2,
        total_lock_in_steps=2,
        lock_in_rate=0.4,
    )

    return CompareRunData(
        label=label,
        run_dir=Path("."),
        display_path="test",
        params=params,
        primary_seed=seed,
        df=df,
        episodes=episodes,
        action_dist=action_dist,
        lock_in=lock_in,
    )


def test_plot_vii_overlay_returns_figure():
    a = _make_compare_run_data(label="A", watch_rate=0.5)
    b = _make_compare_run_data(label="B", watch_rate=1.0)

    fig = plot_vii_overlay(a, b)

    assert isinstance(fig, Figure)
    assert len(fig.axes) == 1


def test_plot_lockin_timeline_returns_two_axes():
    a = _make_compare_run_data(label="A", watch_rate=0.5)
    b = _make_compare_run_data(label="B", watch_rate=1.0)

    fig = plot_lockin_timeline(a, b)

    assert isinstance(fig, Figure)
    assert len(fig.axes) == 2


def test_plot_action_mix_returns_two_axes():
    run_a = _make_compare_run_data(label="A", watch_rate=0.5)
    run_b = _make_compare_run_data(label="B", watch_rate=1.0)

    fig = plot_action_mix(run_a, run_b)

    assert isinstance(fig, Figure)
    assert len(fig.axes) == 2
