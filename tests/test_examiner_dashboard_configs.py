from __future__ import annotations

from pathlib import Path

import pytest

from fyp_sim.examiner_dashboard.configs import (
    build_resolved_config,
    resolve_config,
    scenario_to_config_path,
)


def test_scenario_to_config_path(tmp_path: Path) -> None:
    p = scenario_to_config_path("experiment_baseline", configs_dir=tmp_path)
    assert p == tmp_path / "experiment_baseline.json"


def test_scenario_to_config_path_json_name(tmp_path: Path) -> None:
    p = scenario_to_config_path("experiment_baseline.json", configs_dir=tmp_path)
    assert p == tmp_path / "experiment_baseline.json"


def test_resolve_config_shallow_merge_and_seed_override() -> None:
    base = {"steps": 10, "seeds": [1, 2, 3], "user": {"phenotype": "watcher"}}
    overrides = {"steps": 99, "seed": 42}
    resolved = resolve_config(base, overrides)

    assert resolved["steps"] == 99
    # seed override becomes seeds=[seed]
    assert resolved["seeds"] == [42]
    # shallow merge: user dict stays as is unless explicitly overriden
    assert resolved["user"]["phenotype"] == "watcher"


def test_build_resolved_config_loads_and_sets_scenario(tmp_path: Path) -> None:
    cfg_dir = tmp_path / "configs"
    cfg_dir.mkdir()
    with pytest.raises(FileNotFoundError):
        build_resolved_config(
            "experiment_missing",
            overrides={},
            configs_dir=cfg_dir,
            repo_root=tmp_path,
        )
