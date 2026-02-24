from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class Series:
    t: list[int]
    mean: list[float]
    lo: list[float]
    hi: list[float]


def _load_manifest(run_dir: Path) -> dict:
    path = run_dir / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"manifest.json not found in run_dir: {run_dir}")
    return json.load(path.open("r"))


def _seed_log_path(run_dir: Path, seed: int) -> Path:
    return run_dir / "seeds" / f"s{seed:05d}" / "run_log.csv"


def _read_seed_log(run_dir: Path, seed: int) -> pd.DataFrame:
    p = _seed_log_path(run_dir, seed)
    if not p.is_file():
        raise FileNotFoundError(f"run_log.csv not found for seed {seed}: {p}")
    df = pd.read_csv(p)
    if "t" not in df.columns:
        raise ValueError(f"{p} missing required column: 't'")
    df["t"] = df["t"].astype(int)
    return df.sort_values("t")


def _mean_ci95(xs: list[float]) -> tuple[float, float, float]:
    n = len(xs)
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))
    m = sum(xs) / n
    if n == 1:
        return (m, m, m)
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    se = math.sqrt(var) / math.sqrt(n)
    z = 1.96
    return (m, m - z * se, m + z * se)


def _aggregate_by_t(frames: list[pd.DataFrame], col: str) -> Series:
    if not frames:
        raise ValueError("No seed logs provided for aggregation.")
    if col not in frames[0].columns:
        raise KeyError(f"Missing column {col} in seed log.")

    T = len(frames[0])
    tvals = frames[0]["t"].tolist()

    for df in frames:
        if len(df) != T:
            raise ValueError("Seed logs have different lengths; expected equal steps.")
        if df["t"].tolist() != tvals:
            raise ValueError(
                "Seed logs have different timesteps; expected identical 't' sequences."
            )
        if col not in df.columns:
            raise KeyError(f"Missing column '{col}' in seed logs.")

    ts: list[int] = []
    mean: list[float] = []
    lo: list[float] = []
    hi: list[float] = []

    for i in range(T):
        ts.append(int(tvals[i]))
        vals = [float(df.iloc[i][col]) for df in frames]
        m, ci_lo, ci_hi = _mean_ci95(vals)
        mean.append(m)
        lo.append(ci_lo)
        hi.append(ci_hi)

    return Series(t=ts, mean=mean, lo=lo, hi=hi)


def _plot_series_with_ci(
    series_list: list[tuple[Series, str]],
    *,
    xlabel: str,
    ylabel: str,
    out_path: Path,
) -> None:
    plt.figure()
    for s, label in series_list:
        plt.plot(s.t, s.mean, label=label)
        plt.fill_between(s.t, s.lo, s.hi, alpha=0.15)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def make_drift_compare_plots(*, baseline_run_dir: Path, drift_run_dir: Path, out_dir: Path) -> Path:
    """
    Compare baseline (drift off) vs drift-on runs using the new run directory convention.

    Expects:
        - <run_dir>/manifest.json containing seeds
        - <run_dir>/seeds/s00042/run_log.csv

    Drift run logs must include:
        - user_viewpoint_post
        - video_viewpoint_score
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    base_manifest = _load_manifest(baseline_run_dir)
    drift_manifest = _load_manifest(drift_run_dir)

    base_seeds = [int(s) for s in base_manifest.get("seeds", [])]
    drift_seeds = [int(s) for s in drift_manifest.get("seeds", [])]

    if not base_seeds:
        raise ValueError(f"No seeds found in baseline manifest: {baseline_run_dir}")
    if not drift_seeds:
        raise ValueError(f"No seeds found in drift manifest: {drift_run_dir}")

    # Use intersection so comparison is fair and robust.
    seeds = sorted(set(base_seeds).intersection(drift_seeds))
    if not seeds:
        raise ValueError(
            "No overlapping seeds between baseline and drift runs."
            f"(baseline={base_seeds}, drift={drift_seeds})"
        )

    base_frames = [_read_seed_log(baseline_run_dir, s) for s in seeds]
    drift_frames = [_read_seed_log(drift_run_dir, s) for s in seeds]

    # Pre-distance is already logged as VII_t = |user_pre - video|
    base_vii = _aggregate_by_t(base_frames, "vii_t")
    drift_vii = _aggregate_by_t(drift_frames, "vii_t")

    # Drift post-distance: |video - user_post|
    required = {"video_viewpoint_score", "user_viewpoint_post"}
    missing = required - set(drift_frames[0].columns)
    if missing:
        raise ValueError(
            f"Drift run logs missing columns {sorted(missing)} in {drift_run_dir}. "
            "Re-run drift with viewpoint columns enabled."
        )

    for df in drift_frames:
        df["post_dist"] = (
            df["video_viewpoint_score"].astype(float) - df["user_viewpoint_post"].astype(float)
        ).abs()

    drift_post = _aggregate_by_t(drift_frames, "post_dist")

    # Write plots
    _plot_series_with_ci(
        [(base_vii, "baseline: VII_t (pre)"), (drift_vii, "drift on: VII_t (pre)")],
        xlabel="timestep",
        ylabel="distance",
        out_path=out_dir / "vii_t_over_time.png",
    )

    _plot_series_with_ci(
        [(drift_post, "drift on: |video - user_post|")],
        xlabel="timestep",
        ylabel="post distance",
        out_path=out_dir / "post_distance_over_time.png",
    )

    _plot_series_with_ci(
        [
            (base_vii, "baseline: VII_t (pre)"),
            (drift_vii, "drift on: VII_t (pre)"),
            (drift_post, "drift on: |video - user_post|"),
        ],
        xlabel="timestep",
        ylabel="distance",
        out_path=out_dir / "overlay_pre_vs_post.png",
    )

    return out_dir
