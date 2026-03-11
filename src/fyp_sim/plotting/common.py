from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

REQUIRED_LOG_COLUMNS = {
    "step_id",
    "user_action",
    "viewpoint_distance",
    "isolation_index",
}

LOG_COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "step_id": ("step_id", "t"),
    "user_action": ("user_action", "action"),
    "viewpoint_distance": ("viewpoint_distance", "vii_t"),
    "isolation_index": ("isolation_index", "vii_cum"),
}

DEFAULT_PERSISTENCE_WINDOW = 10


@dataclass(frozen=True, slots=True)
class RunPlotParams:
    threshold: float
    persistence_window: int
    steps: int | None
    rank_alpha: float | None
    drift_alpha: float | None
    seeds: list[int]


@dataclass(frozen=True, slots=True)
class LockInEpisode:
    start_step: int
    end_step: int
    length: int


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _read_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return {}

    with path.open("r") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise TypeError(f"Expected JSON object at: {path}")

    return data


def _first_present(sources: list[dict], keys: tuple[str, ...]) -> object | None:
    for source in sources:
        if not isinstance(source, dict):
            continue
        for key in keys:
            if key in source and source[key] is not None:
                return source[key]
    return None


def seed_dirs(run_dir: Path) -> list[Path]:
    seeds_root = run_dir / "seeds"
    if not seeds_root.exists():
        raise FileNotFoundError(f"Expected seeds/ under run_dir: {run_dir}")

    out = sorted([p for p in seeds_root.iterdir() if p.is_dir()])
    if not out:
        raise FileNotFoundError(f"No seed directories found under: {seeds_root}")

    return out


def first_seed_run_log(run_dir: Path) -> Path:
    first_seed_dir = seed_dirs(run_dir)[0]
    run_log = first_seed_dir / "run_log.csv"
    if not run_log.exists():
        raise FileNotFoundError(f"Expected run_log.csv at: {run_log}")

    return run_log


def load_run_plot_params(run_dir: Path) -> RunPlotParams:
    manifest = _read_json_if_exists(run_dir / "manifest.json")
    cfg = _read_json_if_exists(run_dir / "config_resolved.json")
    key_params = (
        manifest.get("key_params", {}) if isinstance(manifest.get("key_params"), dict) else {}
    )

    sources = [cfg, key_params, manifest]

    threshold_raw = _first_present(
        sources,
        ("filter_bubble_threshold", "lock_in_threshold"),
    )
    if threshold_raw is None:
        raise ValueError(
            "Could not resolve a bubble threshold for plotting."
            f"Run directory: {run_dir}. "
            "Expected one of: filter_bubble_threshold, lock_in_threshold"
        )

    persistence_raw = _first_present(sources, ("persistence_window",))
    if persistence_raw is None:
        persistence_window = DEFAULT_PERSISTENCE_WINDOW
        print(
            "Warning: persistence_window not found in manifest/config for "
            f"{run_dir}. Using default={DEFAULT_PERSISTENCE_WINDOW}."
        )
    else:
        persistence_window = int(persistence_raw)

    seeds_raw = manifest.get("seeds", [])
    seeds = [int(seed) for seed in seeds_raw] if isinstance(seeds_raw, list) else []

    steps_raw = _first_present(sources, ("steps",))
    rank_alpha_raw = _first_present(sources, ("rank_alpha",))
    drift_alpha_raw = _first_present(sources, ("drift_alpha",))

    return RunPlotParams(
        threshold=float(threshold_raw),
        persistence_window=int(persistence_window),
        steps=int(steps_raw) if steps_raw is not None else None,
        rank_alpha=float(rank_alpha_raw) if rank_alpha_raw is not None else None,
        drift_alpha=float(drift_alpha_raw) if drift_alpha_raw is not None else None,
        seeds=seeds,
    )


def validate_and_normalise_log(
    df: pd.DataFrame,
    *,
    run_dir: Path,
    run_log_path: Path,
) -> pd.DataFrame:
    out = df.copy()
    missing: list[str] = []

    for canonical, aliases in LOG_COLUMN_ALIASES.items():
        present = next((name for name in aliases if name in df.columns), None)
        if present is None:
            missing.append(canonical)
        else:
            out[canonical] = df[present]

    if missing:
        alias_text = ", ".join(
            f"{canonical} <- {list(aliases)}" for canonical, aliases in LOG_COLUMN_ALIASES.items()
        )
        raise ValueError(
            "Run log schema validation failed. "
            f"run_dir={run_dir}, log={run_log_path}. "
            f"Missing required columns: {sorted(missing)}. "
            f"Found columns: {sorted(df.columns.tolist())}. "
            f"Accepted aliases: {alias_text}"
        )

    out["step_id"] = pd.to_numeric(out["step_id"], errors="raise").astype(int)
    out["viewpoint_distance"] = pd.to_numeric(out["viewpoint_distance"], errors="raise").astype(
        float
    )
    out["isolation_index"] = pd.to_numeric(out["isolation_index"], errors="raise").astype(float)
    out["user_action"] = out["user_action"].astype(str).str.strip().str.title()

    out = out.sort_values("step_id", kind="stable").reset_index(drop=True)
    return out


def load_run_log_df(run_log_path: Path, *, run_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(run_log_path)
    return validate_and_normalise_log(df, run_dir=run_dir, run_log_path=run_log_path)


def detect_lock_in_episodes(
    df: pd.DataFrame,
    *,
    threshold: float,
    persistence_window: int,
) -> tuple[list[LockInEpisode], dict[str, int]]:
    if persistence_window <= 0:
        raise ValueError("persistence_window must be > 0")
    if not (0.0 <= threshold <= 1.0):
        raise ValueError("threshold must be within [0.0, 1.0]")

    locked = df["viewpoint_distance"].le(threshold).tolist()
    step_ids = df["step_id"].astype(int).tolist()

    episodes: list[LockInEpisode] = []
    i = 0
    n = len(df)

    while i < n:
        if not locked[i]:
            i += 1
            continue

        start_idx = i
        while i < n and locked[i]:
            i += 1
        end_idx = i - 1

        run_len = end_idx - start_idx + 1
        if run_len >= persistence_window:
            episodes.append(
                LockInEpisode(
                    start_step=int(step_ids[start_idx]),
                    end_step=int(step_ids[end_idx]),
                    length=int(run_len),
                )
            )

    summary = {
        "time_to_lock_in": int(episodes[0].start_step) if episodes else -1,
        "n_lockin_episodes": int(len(episodes)),
        "total_lockin_steps": int(sum(ep.length for ep in episodes)),
    }
    return episodes, summary
