from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from fyp_sim.plotting.multi_run_metrics import (
    build_multi_run_vii_summary,
    write_multi_run_summary_csv,
)


def _write_run_log(seed_dir: Path, rows: list[dict]) -> None:
    seed_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(seed_dir / "run_log.csv", index=False)


def _build_run_dir(base_dir: Path, *, seed_rows_by_name: dict[str, list[dict]]) -> Path:
    run_dir = base_dir
    (run_dir / "seeds").mkdir(parents=True, exist_ok=True)

    for seed_name, rows in seed_rows_by_name.items():
        _write_run_log(run_dir / "seeds" / seed_name, rows)

    return run_dir


def _rows_for_seed_zero_with_repeated_steps() -> list[dict]:
    return [
        {
            "step_id": 0,
            "user_action": "Watch",
            "viewpoint_distance": 0.10,
            "isolation_index": 0.10,
            "agent_id": "watcher",
        },
        {
            "step_id": 0,
            "user_action": "Sample",
            "viewpoint_distance": 0.30,
            "isolation_index": 0.20,
            "agent_id": "sampler",
        },
        {
            "step_id": 1,
            "user_action": "Watch",
            "viewpoint_distance": 0.20,
            "isolation_index": 0.15,
            "agent_id": "watcher",
        },
        {
            "step_id": 1,
            "user_action": "Avoid",
            "viewpoint_distance": 0.40,
            "isolation_index": 0.25,
            "agent_id": "avoider",
        },
        {
            "step_id": 2,
            "user_action": "Watch",
            "viewpoint_distance": 0.50,
            "isolation_index": 0.30,
            "agent_id": "watcher",
        },
    ]


def _rows_for_seed_one() -> list[dict]:
    return [
        {
            "step_id": 0,
            "user_action": "Watch",
            "viewpoint_distance": 0.40,
            "isolation_index": 0.40,
            "agent_id": "watcher",
        },
        {
            "step_id": 1,
            "user_action": "Sample",
            "viewpoint_distance": 0.60,
            "isolation_index": 0.50,
            "agent_id": "sampler",
        },
        {
            "step_id": 2,
            "user_action": "Avoid",
            "viewpoint_distance": 0.80,
            "isolation_index": 0.60,
            "agent_id": "avoider",
        },
        {
            "step_id": 3,
            "user_action": "Watch",
            "viewpoint_distance": 0.90,
            "isolation_index": 0.70,
            "agent_id": "watcher",
        },
    ]


def test_build_multi_run_vii_summary_collapses_repeated_step_ids_within_each_seed(
    tmp_path: Path,
) -> None:
    run_dir = _build_run_dir(
        tmp_path / "multi_run_repeated_steps",
        seed_rows_by_name={
            "s00000": _rows_for_seed_zero_with_repeated_steps(),
            "s00001": _rows_for_seed_one(),
        },
    )

    summary = build_multi_run_vii_summary(run_dir)

    assert summary["step_id"].tolist() == [0, 1, 2, 3]
    assert bool(summary.groupby("step_id").size().eq(1).all())

    step0 = summary.loc[summary["step_id"] == 0].iloc[0]
    step1 = summary.loc[summary["step_id"] == 1].iloc[0]
    step2 = summary.loc[summary["step_id"] == 2].iloc[0]

    # Seed s00000 should first collapse repeated rows:
    # step 0: mean(0.10, 0.30) = 0.20
    # step 1: mean(0.20, 0.40) = 0.30
    # Then merge across seeds:
    # step 0 mean across seeds = mean(0.20, 0.40) = 0.30
    # step 1 mean across seeds = mean(0.30, 0.60) = 0.45
    # step 2 mean across seeds = mean(0.50, 0.80) = 0.65
    assert step0["vii_mean"] == pytest.approx(0.30)
    assert step1["vii_mean"] == pytest.approx(0.45)
    assert step2["vii_mean"] == pytest.approx(0.65)

    assert step0["n_runs"] == 2
    assert step1["n_runs"] == 2
    assert step2["n_runs"] == 2


def test_build_multi_run_vii_summary_uses_outer_merge_and_counts_available_runs(
    tmp_path: Path,
) -> None:
    run_dir = _build_run_dir(
        tmp_path / "multi_run_sparse_steps",
        seed_rows_by_name={
            "s00000": _rows_for_seed_zero_with_repeated_steps(),
            "s00001": _rows_for_seed_one(),
        },
    )

    summary = build_multi_run_vii_summary(run_dir)

    step3 = summary.loc[summary["step_id"] == 3].iloc[0]

    assert step3["n_runs"] == 1
    assert step3["vii_mean"] == pytest.approx(0.90)
    assert step3["vii_std"] == pytest.approx(0.0)
    assert step3["vii_ci_lower"] == pytest.approx(0.90)
    assert step3["vii_ci_upper"] == pytest.approx(0.90)


def test_build_multi_run_vii_summary_returns_expected_columns_and_bounded_ci(
    tmp_path: Path,
) -> None:
    run_dir = _build_run_dir(
        tmp_path / "multi_run_summary_schema",
        seed_rows_by_name={
            "s00000": _rows_for_seed_zero_with_repeated_steps(),
            "s00001": _rows_for_seed_one(),
        },
    )

    summary = build_multi_run_vii_summary(run_dir)

    expected_columns = {
        "step_id",
        "n_runs",
        "vii_mean",
        "vii_std",
        "vii_ci_lower",
        "vii_ci_upper",
    }
    assert expected_columns.issubset(summary.columns)

    assert summary["step_id"].tolist() == sorted(summary["step_id"].tolist())
    assert summary["n_runs"].between(1, 2).all()
    assert summary["vii_mean"].between(0.0, 1.0).all()
    assert summary["vii_ci_lower"].between(0.0, 1.0).all()
    assert summary["vii_ci_upper"].between(0.0, 1.0).all()
    assert (summary["vii_ci_lower"] <= summary["vii_mean"]).all()
    assert (summary["vii_mean"] <= summary["vii_ci_upper"]).all()


def test_build_multi_run_vii_summary_requires_at_least_two_seed_runs(tmp_path: Path) -> None:
    run_dir = _build_run_dir(
        tmp_path / "single_seed_run",
        seed_rows_by_name={"s00000": _rows_for_seed_zero_with_repeated_steps()},
    )

    with pytest.raises(ValueError, match="at least two seed runs"):
        build_multi_run_vii_summary(run_dir)


def test_write_multi_run_summary_csv_creates_expected_file(tmp_path: Path) -> None:
    run_dir = _build_run_dir(
        tmp_path / "multi_run_csv",
        seed_rows_by_name={
            "s00000": _rows_for_seed_zero_with_repeated_steps(),
            "s00001": _rows_for_seed_one(),
        },
    )

    out_path = write_multi_run_summary_csv(run_dir)

    assert out_path == run_dir / "multi_run_vii_summary.csv"
    assert out_path is not None
    assert out_path.exists()

    written = pd.read_csv(out_path)
    expected_columns = {
        "step_id",
        "n_runs",
        "vii_mean",
        "vii_std",
        "vii_ci_lower",
        "vii_ci_upper",
    }
    assert expected_columns.issubset(written.columns)
    assert len(written) == 4


def test_write_multi_run_summary_csv_returns_none_for_single_seed_run(tmp_path: Path) -> None:
    run_dir = _build_run_dir(
        tmp_path / "single_seed_csv",
        seed_rows_by_name={"s00000": _rows_for_seed_zero_with_repeated_steps()},
    )

    out_path = write_multi_run_summary_csv(run_dir)

    assert out_path is None
    assert not (run_dir / "multi_run_vii_summary.csv").exists()
