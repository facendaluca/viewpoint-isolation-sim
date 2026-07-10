from __future__ import annotations

import json
from pathlib import Path

import pytest

from fyp_sim.config_validation import validate_experiment_config
from fyp_sim.models import ConfigValidationError


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_all_authoritative_experiment_configs_validate() -> None:
    root = Path(__file__).resolve().parents[1]
    runners = {
        "experiment_baseline.json": "batch",
        "experiment_baseline_drift.json": "batch",
        "experiment_compare.json": "compare",
        "experiment_generated.json": "batch",
        "experiment_sweep.json": "sweep",
    }
    for name, runner in runners.items():
        path = root / "configs" / name
        validate_experiment_config(_load(path), runner=runner, cfg_path=path)


def test_unknown_top_level_field_fails_fast() -> None:
    root = Path(__file__).resolve().parents[1]
    path = root / "configs" / "experiment_baseline.json"
    cfg = _load(path)
    cfg["topkk"] = 5

    with pytest.raises(ConfigValidationError, match="topkk"):
        validate_experiment_config(cfg, runner="batch", cfg_path=path)


def test_conflicting_drift_aliases_fail_fast() -> None:
    root = Path(__file__).resolve().parents[1]
    path = root / "configs" / "experiment_baseline_drift.json"
    cfg = _load(path)
    cfg["viewpoint_drift_rate"] = 0.5

    with pytest.raises(ConfigValidationError, match="conflict"):
        validate_experiment_config(cfg, runner="batch", cfg_path=path)


def test_compare_requires_separate_rng_streams() -> None:
    root = Path(__file__).resolve().parents[1]
    path = root / "configs" / "experiment_compare.json"
    cfg = _load(path)
    cfg["separate_rng_streams"] = False

    with pytest.raises(ConfigValidationError, match="separate_rng_streams=true"):
        validate_experiment_config(cfg, runner="compare", cfg_path=path)


def test_dormant_llm_block_is_reported() -> None:
    root = Path(__file__).resolve().parents[1]
    path = root / "configs" / "experiment_baseline_drift.json"
    audit = validate_experiment_config(_load(path), runner="batch", cfg_path=path)

    assert any("policy.llm is dormant" in warning for warning in audit.warnings)
    assert any("redundant legacy alias" in warning for warning in audit.warnings)
