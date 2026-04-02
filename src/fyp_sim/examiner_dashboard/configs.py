from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fyp_sim.artefacts import _fail_fast_old_alpha


def _default_repo_root() -> Path:
    """
    Best-effort repo root discovery.

    In-repo it finds the parent of 'src/'.
    Otherwise it falls back to current working directory.
    """
    here = Path(__file__).resolve()
    for parent in here.parents:
        if parent.name == "src":
            return parent.parent
    return Path.cwd()


def scenario_to_config_path(
    scenario: str,
    *,
    configs_dir: str | Path | None = None,
    repo_root: str | Path | None = None,
) -> Path:
    """
    Map a scenario identifier to a JSON config path.

    Accepts:
    - "experiment_baseline"
    - "experiment_baseline.json"
    - "configs/experiment_baseline.json"
    - nested scenario names such as "E1-baseline_single_watcher"
      when they uniquely exist under configs/
    - absolute paths
    """
    scenario = scenario.strip()
    if not scenario:
        raise ValueError("scenario must be a non-empty string")

    root = Path(repo_root) if repo_root is not None else _default_repo_root()
    cfg_dir = Path(configs_dir) if configs_dir is not None else (root / "configs")

    if not cfg_dir.is_absolute():
        cfg_dir = root / cfg_dir

    # Treat as explicit path if it looks like one
    has_sep = ("/" in scenario) or ("\\" in scenario)

    if has_sep:
        p = Path(scenario)
        if not p.is_absolute():
            p = root / p
        if p.suffix != ".json":
            p = p.with_suffix(".json")
        return p

    if scenario.endswith(".json"):
        return cfg_dir / scenario

    direct = cfg_dir / f"{scenario}.json"
    if direct.exists():
        return direct

    matches = sorted(cfg_dir.rglob(f"{scenario}.json"))
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        joined = ", ".join(str(p.relative_to(cfg_dir)) for p in matches)
        raise FileNotFoundError(
            f"Scenario `{scenario}` matched the multiple config files under `{cfg_dir}`: {joined}"
        )

    return direct


def load_config(path: Path) -> dict[str, Any]:
    """Load a JSON config dict and fail fast on deprecated keys (e.g. old 'alpha')."""
    with path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config must be a JSON object, got {type(cfg).__name__} from {path}")
    _fail_fast_old_alpha(cfg, path)
    return cfg


def resolve_config(base_cfg: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """
    Produce a resolved config dict.

    Special-case:
        - if overrides has "seed": int -> set "seeds" = [seed]
        - if overrides has "seeds": list[int] -> use it (sorted unique)
    """
    if not isinstance(base_cfg, dict):
        raise TypeError("base_cfg must be a dict[str, Any]")
    if not isinstance(overrides, dict):
        raise TypeError("overrides must be a dict[str, Any]")

    cfg = dict(base_cfg)
    cfg.update(overrides)

    # Seed semantics for dashboard runs:
    # UI typically edits "seed" (single); translate into "seeds" for consistency
    if "seeds" in overrides:
        seeds_raw = overrides.get("seeds")
        if isinstance(seeds_raw, list) and all(isinstance(x, int) for x in seeds_raw):
            cfg["seeds"] = sorted(set(seeds_raw))
        # if invalid, leave as-is (caller/UI should validate)
    elif "seed" in overrides:
        seed_raw = overrides.get("seed")
        if isinstance(seed_raw, int):
            cfg["seeds"] = [seed_raw]

    return cfg


def build_resolved_config(
    scenario: str,
    overrides: dict[str, Any],
    *,
    configs_dir: str | Path | None = None,
    repo_root: str | Path | None = None,
) -> tuple[dict[str, Any], Path]:
    """
    Convenience for the dashboard:
    - locate + load scenario config
    - apply overrides
    - ensure cfg["scenario"] is set (required by dashboard backend)
    Returns (resolved_cfg, cfg_path).
    """
    cfg_path = scenario_to_config_path(scenario, configs_dir=configs_dir, repo_root=repo_root)
    base = load_config(cfg_path)
    resolved = resolve_config(base, overrides)
    resolved["scenario"] = scenario.strip()
    return resolved, cfg_path
