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
    cfg = _load(path)
    cfg["policy"]["mode"] = "heuristic"
    cfg["viewpoint_drift_rate"] = cfg["drift_alpha"]
    audit = validate_experiment_config(cfg, runner="batch", cfg_path=path)

    assert any("policy.llm is dormant" in warning for warning in audit.warnings)
    assert any("redundant legacy alias" in warning for warning in audit.warnings)


def test_warns_when_exploit_only_run_starts_inside_watch_region() -> None:
    # The frozen compare condition: sampler with meme=0.72 >= the 0.70 sampler
    # watch threshold, and run_compare never applies exploration. The heuristic
    # arm must be flagged as an exploit-only baseline, not silently accepted as
    # a mixed-behaviour user model.
    root = Path(__file__).resolve().parents[1]
    path = root / "configs" / "experiment_compare.json"
    audit = validate_experiment_config(_load(path), runner="compare", cfg_path=path)

    assert any("exploit-only" in warning for warning in audit.warnings)


def test_saturation_warning_respects_threshold_and_exploration() -> None:
    root = Path(__file__).resolve().parents[1]
    path = root / "configs" / "experiment_compare.json"

    # Below the sampler watch threshold at t=0: no saturation warning.
    cfg = _load(path)
    cfg["user"]["interest_vector"] = {"meme": 0.65, "comedy_memes": 0.55}
    audit = validate_experiment_config(cfg, runner="compare", cfg_path=path)
    assert not any("exploit-only" in warning for warning in audit.warnings)

    # Batch runner with active exploration: curiosity serves random videos, so
    # the exploit-only saturation regime does not apply.
    cfg = _load(path)
    cfg["policy"] = {"mode": "heuristic", "curiosity": 0.3}
    audit = validate_experiment_config(cfg, runner="batch", cfg_path=path)
    assert not any("exploit-only" in warning for warning in audit.warnings)

    # The same batch config without exploration warns.
    cfg["policy"] = {"mode": "heuristic", "curiosity": 0.0}
    audit = validate_experiment_config(cfg, runner="batch", cfg_path=path)
    assert any("exploit-only" in warning for warning in audit.warnings)


def test_saturation_warning_names_the_affected_agent_in_cohorts() -> None:
    root = Path(__file__).resolve().parents[1]
    path = root / "configs" / "experiment_compare.json"
    cfg = _load(path)
    user = cfg.pop("user")
    safe_agent = dict(user, agent_id="calm", interest_vector={"meme": 0.4})
    saturated_agent = dict(user, agent_id="hooked", interest_vector={"meme": 0.9})
    cfg["agents"] = [safe_agent, saturated_agent]
    cfg["policy"] = {"mode": "heuristic"}

    audit = validate_experiment_config(cfg, runner="batch", cfg_path=path)

    saturation = [warning for warning in audit.warnings if "exploit-only" in warning]
    assert len(saturation) == 1
    assert "agents[1]" in saturation[0]
