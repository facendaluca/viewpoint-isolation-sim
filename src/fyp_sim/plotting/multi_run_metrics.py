from __future__ import annotations

from pathlib import Path

import pandas as pd

from .common import load_run_log_df, seed_dirs


def _build_seed_level_vii_frame(run_dir: Path, seed_dir: Path) -> pd.DataFrame:
    run_log_path = seed_dir / "run_log.csv"
    if not run_log_path.exists():
        raise FileNotFoundError(f"Expected run_log.csv at: {run_log_path}")

    df = load_run_log_df(run_log_path, run_dir=run_dir)

    # Single-agent logs already have one row per step_id.
    # Multi-agent logs can contain multiple rows per step_id.
    # into one cohort-level value per step before merging across seeds.
    seed_level = (
        df.groupby("step_id", as_index=False)["viewpoint_distance"]
        .mean()
        .sort_values("step_id", kind="stable")
        .reset_index(drop=True)
    )

    vii_column = f"vii_{seed_dir.name}"
    return seed_level.rename(columns={"viewpoint_distance": vii_column})


def build_multi_run_vii_summary(run_dir: Path) -> pd.DataFrame:
    run_seed_dirs = seed_dirs(run_dir)
    if len(run_seed_dirs) < 2:
        raise ValueError("Multi-run variability requires at least two seed runs.")

    merged: pd.DataFrame | None = None
    vii_columns: list[str] = []

    for seed_dir in run_seed_dirs:
        frame = _build_seed_level_vii_frame(run_dir, seed_dir)
        vii_column = f"vii_{seed_dir.name}"

        if merged is None:
            merged = frame
        else:
            merged = merged.merge(frame, on="step_id", how="outer", validate="1:1")

        vii_columns.append(vii_column)

    if merged is None:
        raise ValueError(f"No run logs found for multi-run summary: {run_dir}")

    merged = merged.sort_values("step_id").reset_index(drop=True)

    summary = pd.DataFrame({"step_id": merged["step_id"]})
    summary["n_runs"] = merged[vii_columns].count(axis=1).astype(int)
    summary["vii_mean"] = merged[vii_columns].mean(axis=1)
    summary["vii_std"] = merged[vii_columns].std(axis=1, ddof=1).fillna(0.0)

    sqrt_n = summary["n_runs"].where(summary["n_runs"] > 0, 1).astype(float).pow(0.5)
    stderr = summary["vii_std"] / sqrt_n
    ci_half_width = (1.96 * stderr).where(summary["n_runs"] > 1, 0.0)

    summary["vii_ci_lower"] = (summary["vii_mean"] - ci_half_width).clip(0.0, 1.0)
    summary["vii_ci_upper"] = (summary["vii_mean"] + ci_half_width).clip(0.0, 1.0)

    return summary


def write_multi_run_summary_csv(run_dir: Path) -> Path | None:
    run_seed_dirs = seed_dirs(run_dir)
    if len(run_seed_dirs) < 2:
        return None

    summary = build_multi_run_vii_summary(run_dir)
    out_path = run_dir / "multi_run_vii_summary.csv"
    summary.to_csv(out_path, index=False)
    return out_path
