from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from fyp_sim.plotting.multi_agent_metrics import (
    build_phenotype_action_dynamics_summary,
    build_phenotype_lockin_outcome_summary,
    build_phenotype_trajectory_summary,
    has_multi_agent_run,
    write_phenotype_lockin_summary_csv,
)


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def _write_run_log(seed_dir: Path, rows: list[dict]) -> None:
    seed_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(seed_dir / "run_log.csv", index=False)


def _build_run_dir(
    base_dir: Path,
    *,
    seed_rows_by_name: dict[str, list[dict]],
    threshold: float = 0.20,
    persistence_window: int = 2,
    steps: int = 4,
) -> Path:
    run_dir = base_dir
    (run_dir / "seeds").mkdir(parents=True, exist_ok=True)

    _write_json(
        run_dir / "manifest.json",
        {
            "seeds": list(range(len(seed_rows_by_name))),
            "key_params": {
                "filter_bubble_threshold": threshold,
                "persistence_window": persistence_window,
                "steps": steps,
                "rank_alpha": 0.30,
                "drift_alpha": 0.02,
            },
        },
    )
    _write_json(
        run_dir / "config_resolved.json",
        {
            "filter_bubble_threshold": threshold,
            "persistence_window": persistence_window,
            "steps": steps,
            "rank_alpha": 0.30,
            "drift_alpha": 0.02,
        },
    )

    for seed_name, rows in seed_rows_by_name.items():
        _write_run_log(run_dir / "seeds" / seed_name, rows)

    return run_dir


def _multi_agent_seed_rows(seed_index: int) -> list[dict]:
    if seed_index == 0:
        phenotype_specs = {
            "watcher": {
                "distances": [0.10, 0.10, 0.30, 0.30],
                "actions": ["Watch", "Watch", "Sample", "Watch"],
            },
            "sampler": {
                "distances": [0.30, 0.10, 0.10, 0.30],
                "actions": ["Sample", "Watch", "Watch", "Sample"],
            },
            "avoider": {
                "distances": [0.30, 0.30, 0.30, 0.30],
                "actions": ["Avoid", "Avoid", "Sample", "Avoid"],
            },
        }
    elif seed_index == 1:
        phenotype_specs = {
            "watcher": {
                "distances": [0.10, 0.10, 0.10, 0.30],
                "actions": ["Watch", "Watch", "Watch", "Sample"],
            },
            "sampler": {
                "distances": [0.30, 0.30, 0.10, 0.10],
                "actions": ["Sample", "Sample", "Watch", "Watch"],
            },
            "avoider": {
                "distances": [0.30, 0.10, 0.30, 0.10],
                "actions": ["Avoid", "Sample", "Avoid", "Sample"],
            },
        }
    else:
        raise ValueError(f"Unsupported seed index for test data: {seed_index}")

    rows: list[dict] = []
    for phenotype, spec in phenotype_specs.items():
        for step_id, (distance, action) in enumerate(
            zip(spec["distances"], spec["actions"], strict=True)
        ):
            rows.append(
                {
                    "step_id": step_id,
                    "user_action": action,
                    "viewpoint_distance": distance,
                    "isolation_index": distance,
                    "agent_id": phenotype,
                }
            )
    return rows


def _single_phenotype_seed_rows() -> list[dict]:
    rows: list[dict] = []
    distances = [0.10, 0.10, 0.30, 0.30]
    actions = ["Watch", "Watch", "Sample", "Watch"]

    for step_id, (distance, action) in enumerate(zip(distances, actions, strict=True)):
        rows.append(
            {
                "step_id": step_id,
                "user_action": action,
                "viewpoint_distance": distance,
                "isolation_index": distance,
                "agent_id": "watcher",
            }
        )
    return rows


def test_has_multi_agent_run_returns_true_for_multiple_phenotypes(tmp_path: Path) -> None:
    run_dir = _build_run_dir(
        tmp_path / "multi_agent_run",
        seed_rows_by_name={"s00000": _multi_agent_seed_rows(0)},
    )

    assert has_multi_agent_run(run_dir) is True


def test_has_multi_agent_run_returns_false_for_single_phenotype(tmp_path: Path) -> None:
    run_dir = _build_run_dir(
        tmp_path / "single_phenotype_run",
        seed_rows_by_name={"s00000": _single_phenotype_seed_rows()},
    )

    assert has_multi_agent_run(run_dir) is False


def test_build_phenotype_trajectory_summary_returns_expected_grain_and_columns(
    tmp_path: Path,
) -> None:
    run_dir = _build_run_dir(
        tmp_path / "trajectory_run",
        seed_rows_by_name={
            "s00000": _multi_agent_seed_rows(0),
            "s00001": _multi_agent_seed_rows(1),
        },
    )

    summary = build_phenotype_trajectory_summary(run_dir)

    expected_columns = {
        "phenotype",
        "step_id",
        "n_seeds",
        "vii_mean",
        "vii_std",
        "vii_ci_lower",
        "vii_ci_upper",
    }
    assert expected_columns.issubset(summary.columns)

    # 3 phenotypes x 4 steps
    assert len(summary) == 12
    assert summary.groupby(["phenotype", "step_id"]).size().eq(1).all()
    assert summary["n_seeds"].eq(2).all()
    assert summary["vii_ci_lower"].between(0.0, 1.0).all()
    assert summary["vii_ci_upper"].between(0.0, 1.0).all()

    watcher_step0 = summary[(summary["phenotype"] == "watcher") & (summary["step_id"] == 0)].iloc[0]
    sampler_step0 = summary[(summary["phenotype"] == "sampler") & (summary["step_id"] == 0)].iloc[0]
    avoider_step0 = summary[(summary["phenotype"] == "avoider") & (summary["step_id"] == 0)].iloc[0]

    assert watcher_step0["vii_mean"] == pytest.approx(0.10)
    assert sampler_step0["vii_mean"] == pytest.approx(0.30)
    assert avoider_step0["vii_mean"] == pytest.approx(0.30)


def test_build_phenotype_action_dynamics_summary_returns_expected_rows_and_ranges(
    tmp_path: Path,
) -> None:
    run_dir = _build_run_dir(
        tmp_path / "action_dynamics_run",
        seed_rows_by_name={
            "s00000": _multi_agent_seed_rows(0),
            "s00001": _multi_agent_seed_rows(1),
        },
        persistence_window=2,
    )

    summary = build_phenotype_action_dynamics_summary(run_dir)

    expected_columns = {
        "phenotype",
        "action",
        "step_id",
        "n_seeds",
        "action_rate_mean",
        "action_rate_std",
        "action_rate_ci_lower",
        "action_rate_ci_upper",
        "rolling_window",
    }
    assert expected_columns.issubset(summary.columns)

    # 3 phenotypes x 3 actions x 4 steps
    assert len(summary) == 36
    assert summary.groupby(["phenotype", "action", "step_id"]).size().eq(1).all()
    assert summary["n_seeds"].eq(2).all()
    assert summary["rolling_window"].eq(2).all()

    for column in ("action_rate_mean", "action_rate_ci_lower", "action_rate_ci_upper"):
        assert summary[column].between(0.0, 1.0).all()

    watcher_watch_step0 = summary[
        (summary["phenotype"] == "watcher")
        & (summary["action"] == "Watch")
        & (summary["step_id"] == 0)
    ].iloc[0]
    avoider_watch_step0 = summary[
        (summary["phenotype"] == "avoider")
        & (summary["action"] == "Watch")
        & (summary["step_id"] == 0)
    ].iloc[0]

    assert watcher_watch_step0["action_rate_mean"] == pytest.approx(1.0)
    assert avoider_watch_step0["action_rate_mean"] == pytest.approx(0.0)


def test_build_phenotype_lockin_outcome_summary_marks_non_locking_cases_with_nan(
    tmp_path: Path,
) -> None:
    run_dir = _build_run_dir(
        tmp_path / "lockin_outcomes_run",
        seed_rows_by_name={
            "s00000": _multi_agent_seed_rows(0),
            "s00001": _multi_agent_seed_rows(1),
        },
        threshold=0.20,
        persistence_window=2,
    )

    outcomes = build_phenotype_lockin_outcome_summary(run_dir)

    expected_columns = {
        "seed",
        "phenotype",
        "time_to_lock_in",
        "n_lockin_episodes",
        "total_lockin_steps",
        "filter_bubble_threshold",
        "persistence_window",
        "locked_in",
        "time_to_lock_in_plot",
    }
    assert expected_columns.issubset(outcomes.columns)

    # 2 seeds x 3 phenotypes
    assert len(outcomes) == 6
    assert outcomes.groupby(["seed", "phenotype"]).size().eq(1).all()

    watcher_s00000 = outcomes[
        (outcomes["seed"] == "s00000") & (outcomes["phenotype"] == "watcher")
    ].iloc[0]
    sampler_s00001 = outcomes[
        (outcomes["seed"] == "s00001") & (outcomes["phenotype"] == "sampler")
    ].iloc[0]
    avoider_rows = outcomes[outcomes["phenotype"] == "avoider"].sort_values("seed")

    assert bool(watcher_s00000["locked_in"]) is True
    assert watcher_s00000["time_to_lock_in"] == 0
    assert watcher_s00000["time_to_lock_in_plot"] == pytest.approx(0.0)

    assert bool(sampler_s00001["locked_in"]) is True
    assert sampler_s00001["time_to_lock_in"] == 2
    assert sampler_s00001["time_to_lock_in_plot"] == pytest.approx(2.0)

    assert avoider_rows["locked_in"].eq(False).all()
    assert avoider_rows["time_to_lock_in"].eq(-1).all()
    assert avoider_rows["time_to_lock_in_plot"].isna().all()


def test_write_phenotype_lockin_summary_csv_creates_expected_file(tmp_path: Path) -> None:
    run_dir = _build_run_dir(
        tmp_path / "lockin_csv_run",
        seed_rows_by_name={
            "s00000": _multi_agent_seed_rows(0),
            "s00001": _multi_agent_seed_rows(1),
        },
    )

    out_path = write_phenotype_lockin_summary_csv(run_dir)

    assert out_path == run_dir / "phenotype_lockin_summary.csv"
    assert out_path.exists()

    written = pd.read_csv(out_path)
    expected_columns = {
        "seed",
        "phenotype",
        "time_to_lock_in",
        "n_lockin_episodes",
        "total_lockin_steps",
        "filter_bubble_threshold",
        "persistence_window",
    }
    assert expected_columns.issubset(written.columns)
    assert len(written) == 6
