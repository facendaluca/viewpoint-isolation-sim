from pathlib import Path
from unittest.mock import sentinel

import pandas as pd
import pytest

from fyp_sim.plotting.compare_run_data import CompareRunData, load_compare_run


def test_load_compare_run_returns_prepared_data(monkeypatch: pytest.MonkeyPatch):
    run_dir = Path("outputs/runs/test_run")
    run_log_path = run_dir / "seeds" / "s00042" / "run_log.csv"
    df = pd.DataFrame(
        {
            "step_id": [0, 1, 2],
            "viewpoint_distance": [0.6, 0.2, 0.1],
            "isolation_index": [0.6, 0.4, 0.3],
        }
    )

    class DummyParams:
        threshold = 0.25
        persistence_window = 3

    params = DummyParams()
    episodes = sentinel.episodes
    action_dist = sentinel.action_dist
    lock_in = sentinel.lock_in

    monkeypatch.setattr(
        "fyp_sim.plotting.compare_run_data.load_run_plot_params",
        lambda actual_run_dir: params,
    )
    monkeypatch.setattr(
        "fyp_sim.plotting.compare_run_data.first_seed_run_log",
        lambda actual_run_dir: run_log_path,
    )
    monkeypatch.setattr(
        "fyp_sim.plotting.compare_run_data.load_run_log_df",
        lambda actual_run_log_path, *, run_dir: df,
    )
    monkeypatch.setattr(
        "fyp_sim.plotting.compare_run_data.detect_lock_in_episodes",
        lambda actual_df, *, threshold, persistence_window: (episodes, sentinel.summary),
    )
    monkeypatch.setattr(
        "fyp_sim.plotting.compare_run_data.compute_action_distribution",
        lambda actual_df: action_dist,
    )
    monkeypatch.setattr(
        "fyp_sim.plotting.compare_run_data.compute_lock_in_metrics",
        lambda distances, *, lock_in_threshold, persistence_window: lock_in,
    )

    data = load_compare_run("A", "outputs/runs/test_run", run_dir)

    assert isinstance(data, CompareRunData)
    assert data.label == "A"
    assert data.run_dir == run_dir
    assert data.display_path == "outputs/runs/test_run"
    assert data.params is params
    assert data.primary_seed == "s00042"
    assert data.df is df
    assert data.episodes is episodes
    assert data.action_dist is action_dist
    assert data.lock_in is lock_in


def test_load_compare_run_passes_expected_inputs_to_helpers(
    monkeypatch: pytest.MonkeyPatch,
):
    run_dir = Path("outputs/runs/test_run")
    run_log_path = run_dir / "seeds" / "s00007" / "run_log.csv"
    df = pd.DataFrame(
        {
            "step_id": [0, 1, 2],
            "viewpoint_distance": [0.7, 0.15, 0.1],
            "isolation_index": [0.7, 0.425, 0.3167],
        }
    )

    calls: dict[str, object] = {}

    class DummyParams:
        threshold = 0.25
        persistence_window = 3

    monkeypatch.setattr(
        "fyp_sim.plotting.compare_run_data.load_run_plot_params",
        lambda actual_run_dir: DummyParams(),
    )

    monkeypatch.setattr(
        "fyp_sim.plotting.compare_run_data.first_seed_run_log",
        lambda actual_run_dir: run_log_path,
    )

    def fake_load_run_log_df(actual_run_log_path: Path, *, run_dir: Path) -> pd.DataFrame:
        calls["run_log_path"] = actual_run_log_path
        calls["run_dir"] = run_dir
        return df

    def fake_detect_lock_in_episodes(
        actual_df: pd.DataFrame,
        *,
        threshold: float,
        persistence_window: int,
    ):
        calls["episodes_df"] = actual_df
        calls["episodes_threshold"] = threshold
        calls["episodes_window"] = persistence_window
        return ({}, sentinel.summary)

    def fake_compute_action_distribution(actual_df: pd.DataFrame):
        calls["action_df"] = actual_df
        return sentinel.action_dist

    def fake_compute_lock_in_metrics(
        distances: list[float],
        *,
        lock_in_threshold: float,
        persistence_window: int,
    ):
        calls["distances"] = distances
        calls["lock_in_threshold"] = lock_in_threshold
        calls["lock_in_window"] = persistence_window
        return sentinel.lock_in

    monkeypatch.setattr(
        "fyp_sim.plotting.compare_run_data.load_run_log_df",
        fake_load_run_log_df,
    )
    monkeypatch.setattr(
        "fyp_sim.plotting.compare_run_data.detect_lock_in_episodes",
        fake_detect_lock_in_episodes,
    )
    monkeypatch.setattr(
        "fyp_sim.plotting.compare_run_data.compute_action_distribution",
        fake_compute_action_distribution,
    )
    monkeypatch.setattr(
        "fyp_sim.plotting.compare_run_data.compute_lock_in_metrics",
        fake_compute_lock_in_metrics,
    )

    data = load_compare_run("B", "outputs/runs/test_run", run_dir)

    assert data.primary_seed == "s00007"
    assert calls["run_log_path"] == run_log_path
    assert calls["run_dir"] == run_dir
    assert calls["episodes_df"] is df
    assert calls["episodes_threshold"] == 0.25
    assert calls["episodes_window"] == 3
    assert calls["action_df"] is df
    assert calls["distances"] == [0.7, 0.15, 0.1]
    assert calls["lock_in_threshold"] == 0.25
    assert calls["lock_in_window"] == 3


def test_load_compare_run_propagates_missing_run_artefacts(monkeypatch: pytest.MonkeyPatch):
    run_dir = Path("outputs/runs/missing_run")

    monkeypatch.setattr(
        "fyp_sim.plotting.compare_run_data.load_run_plot_params",
        lambda actual_run_dir: (_ for _ in ()).throw(FileNotFoundError("missing config")),
    )

    with pytest.raises(FileNotFoundError, match="missing config"):
        load_compare_run("A", "outputs/runs/missing_run", run_dir)
