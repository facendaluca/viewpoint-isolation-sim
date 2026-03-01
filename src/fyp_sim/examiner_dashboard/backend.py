from __future__ import annotations

from pathlib import Path
from typing import Any

from fyp_sim.artefacts import create_run_artefacts
from fyp_sim.runners.seed_sweep import run_seed_sweep


def _scenario_to_mode(scenario: str) -> str:
    scenario = scenario.strip() or "experiment_baseline"
    if scenario.startswith("experiment_"):
        return scenario.removeprefix("experiment_")
    return scenario


def _extract_seeds(cfg: dict[str, Any]) -> list[int]:
    # Prefer explicit seeds list, else single seed, else default [0]
    seeds_raw = cfg.get("seeds")
    if isinstance(seeds_raw, list) and all(isinstance(s, int) for s in seeds_raw):
        seeds = seeds_raw
    else:
        seed_raw = cfg.get("seed")
        seeds = [seed_raw] if isinstance(seed_raw, int) else [0]

    # Stable + unique
    return sorted(set(seeds))


def run_heuristic(
    config: dict[str, Any],
    *,
    output_root: str | Path = "outputs/runs",
    cfg_path: str | Path | None = None,
) -> Path:
    """
    Backend placeholder for the dashboard

    Contract (v1 Scaffold):
    - Accepts a resolved config dict (dict[str, Any])
    - Creates a run directory matching the project's artefact convention
    - Writes config_resolved.json + manifest.json via create_run_artefacts
    - Returns the run directory path
    """
    if not isinstance(config, dict):
        raise TypeError("config must be a dict[str, Any]")

    cfg = dict(config)  # copy to avoid mutating caller

    scenario = cfg.get("scenario")
    if not isinstance(scenario, str) or not scenario.strip():
        scenario = "experiment_baseline"
    cfg["scenario"] = scenario

    mode = _scenario_to_mode(scenario)
    seeds = _extract_seeds(cfg)

    artefacts = create_run_artefacts(
        cfg=cfg,
        cfg_path=Path(cfg_path) if cfg_path is not None else None,
        mode=mode,
        seeds=seeds,
        outputs_root=Path(output_root),
        write_config_snapshot=True,
    )

    # Ensure placeholder files exist to match expected tree
    artefacts.summary_path.touch(exist_ok=True)

    for seed in seeds:
        seed_dir = artefacts.seeds_dir / f"s{seed:05d}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        (seed_dir / "run_log.csv").touch(exist_ok=True)

    return artefacts.root_dir


def run_heuristic_real(
    config: dict[str, Any],
    *,
    output_root: str | Path | None = None,
    cfg_path: str | Path | None = None,
) -> Path:
    return run_seed_sweep(
        config,
        cfg_path=Path(cfg_path),
        outputs_root=Path(output_root),
    )
