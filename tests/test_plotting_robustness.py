from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from fyp_sim.plotting.common import load_run_plot_params
from fyp_sim.plotting.multi_agent_metrics import (
    build_phenotype_lockin_outcome_summary,
    has_multi_agent_run,
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
    seed_rows_by_name: dict[str, list[dict]] | None = None,
    manifest: dict | None = None,
    config_resolved: dict | None = None,
) -> Path:
    run_dir = base_dir
    run_dir.mkdir(parents=True, exist_ok=True)

    if manifest is not None:
        _write_json(run_dir / "manifest.json", manifest)

    if config_resolved is not None:
        _write_json(run_dir / "config_resolved.json", config_resolved)

    if seed_rows_by_name is not None:
        (run_dir / "seeds").mkdir(parents=True, exist_ok=True)
        for seed_name, rows in seed_rows_by_name.items():
            _write_run_log(run_dir / "seeds" / seed_name, rows)

    return run_dir


def _single_agent_rows() -> list[dict]:
    return [
        {
            "step_id": 0,
            "user_action": "Watch",
            "viewpoint_distance": 0.10,
            "isolation_index": 0.10,
        },
        {
            "step_id": 1,
            "user_action": "Sample",
            "viewpoint_distance": 0.25,
            "isolation_index": 0.18,
        },
        {
            "step_id": 2,
            "user_action": "Watch",
            "viewpoint_distance": 0.15,
            "isolation_index": 0.16,
        },
    ]


def test_has_multi_agent_run_returns_false_when_agent_id_column_is_missing(
    tmp_path: Path,
) -> None:
    run_dir = _build_run_dir(
        tmp_path / "single_agent_run",
        seed_rows_by_name={"s00000": _single_agent_rows()},
        manifest={"seeds": [0]},
        config_resolved={"filter_bubble_threshold": 0.20, "persistence_window": 2},
    )

    assert has_multi_agent_run(run_dir) is False


def test_build_phenotype_lockin_outcome_summary_requires_agent_id_for_multi_agent_inputs(
    tmp_path: Path,
) -> None:
    run_dir = _build_run_dir(
        tmp_path / "strict_multi_agent_builder",
        seed_rows_by_name={"s00000": _single_agent_rows()},
        manifest={"seeds": [0]},
        config_resolved={"filter_bubble_threshold": 0.20, "persistence_window": 2},
    )

    with pytest.raises(ValueError, match="requires 'agent_id'"):
        build_phenotype_lockin_outcome_summary(run_dir)


def test_has_multi_agent_run_raises_clear_error_when_run_log_is_missing(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "missing_run_log"
    seed_dir = run_dir / "seeds" / "s00000"
    seed_dir.mkdir(parents=True, exist_ok=True)

    with pytest.raises(FileNotFoundError, match="Expected run_log.csv"):
        has_multi_agent_run(run_dir)


def test_load_run_plot_params_resolves_threshold_from_manifest_lock_in_threshold(
    tmp_path: Path,
    capsys,
) -> None:
    run_dir = _build_run_dir(
        tmp_path / "manifest_threshold_fallback",
        manifest={
            "seeds": [0, 1],
            "key_params": {
                "lock_in_threshold": 0.25,
                "steps": 120,
                "rank_alpha": 0.30,
                "drift_alpha": 0.02,
            },
        },
    )

    params = load_run_plot_params(run_dir)
    captured = capsys.readouterr()

    assert params.threshold == pytest.approx(0.25)
    assert params.persistence_window == 10
    assert params.steps == 120
    assert params.rank_alpha == pytest.approx(0.30)
    assert params.drift_alpha == pytest.approx(0.02)
    assert params.seeds == [0, 1]
    assert "Warning: persistence_window not found" in captured.out


def test_load_run_plot_params_prefers_config_resolved_over_manifest_key_params(
    tmp_path: Path,
) -> None:
    run_dir = _build_run_dir(
        tmp_path / "config_precedence",
        manifest={
            "seeds": [0],
            "key_params": {
                "filter_bubble_threshold": 0.40,
                "persistence_window": 9,
                "steps": 50,
            },
        },
        config_resolved={
            "filter_bubble_threshold": 0.20,
            "persistence_window": 3,
            "steps": 80,
            "rank_alpha": 0.10,
            "drift_alpha": 0.01,
        },
    )

    params = load_run_plot_params(run_dir)

    assert params.threshold == pytest.approx(0.20)
    assert params.persistence_window == 3
    assert params.steps == 80
    assert params.rank_alpha == pytest.approx(0.10)
    assert params.drift_alpha == pytest.approx(0.01)


def test_load_run_plot_params_raises_when_threshold_cannot_be_resolved(
    tmp_path: Path,
) -> None:
    run_dir = _build_run_dir(
        tmp_path / "missing_threshold",
        manifest={"seeds": [0], "key_params": {"steps": 100}},
        config_resolved={"persistence_window": 4},
    )

    with pytest.raises(ValueError, match="Could not resolve a bubble threshold"):
        load_run_plot_params(run_dir)
