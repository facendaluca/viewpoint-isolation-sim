"""
Batch experiment runner (seeds-only).

Reads a single experiment config from 'configs/experiment_baseline.json, then:
    - runs one simulation per seed (deterministic per seed),
    - writes per-run logs to 'outputs/runs/run_seed_<seed>.csv' (generated artifacts not commited),
    - writes a compact summary to 'results/summary.csv' (tracked).

This script is intentionally simple and no parameter sweeps yet (added in later scripts).
"""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Any

from fyp_sim.analysis import compute_lock_in_metrics
from fyp_sim.models import User, UserPhenotype, Video
from fyp_sim.simulation.engine import run_simulation


def phenotype_from_str(s: str) -> UserPhenotype:
    """Map config string -> UserPhenotype enum (validated)."""
    s = s.strip().lower()
    if s == "watcher":
        return UserPhenotype.WATCHER
    if s == "sampler":
        return UserPhenotype.SAMPLER
    if s == "avoider":
        return UserPhenotype.AVOIDER
    raise ValueError(f"Unknown phenotype: {s!r} (expected watcher/sampler/avoider)")


def load_config(path: Path) -> dict[str, Any]:
    """Load experiment configuration JSON.

    Expected keys: steps, top_k, alpha, seeds, user, video_pool.
    """
    with path.open("r") as f:
        return json.load(f)


def build_user(cfg: dict[str, Any]) -> User:
    """
    Build User from config dict.

    Notes:
        - interest_vector supports both topic categories and free-form tags.
        - sentiment_threshold gates negative content (policy layer).
    """
    u = cfg["user"]
    return User(
        phenotype=phenotype_from_str(u["phenotype"]),
        viewpoint_score=float(u["viewpoint_score"]),
        interest_vector={str(k): float(v) for k, v in u["interest_vector"].items()},
        sentiment_threshold=float(u["sentiment_threshold"]),
    )


def build_video_pool(cfg: dict[str, Any]) -> list[Video]:
    """Construct the list of Video objects from the config.

    Tags are free-form strings; they are not normalised here to keep the simulation flexible.
    """
    pool: list[Video] = []
    for v in cfg["video_pool"]:
        tags = tuple(v.get("tags", []))
        pool.append(
            Video(
                int(v["video_id"]),
                str(v["topic_category"]),
                float(v["viewpoint_score"]),
                float(v["sentiment_score"]),
                int(v["duration_s"]),
                tags=tags,
            )
        )
    return pool


def write_run_log(path: Path, logs) -> None:
    """Write per-step logs for one seed to CSV. (generated artifacts, gitignored)"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "video_id", "action", "watch_time_s", "interest", "vii_t", "vii_cum"])
        for row in logs:
            w.writerow(
                [
                    row.t,
                    row.video_id,
                    row.action,
                    row.watch_time_s,
                    f"{row.interest:.4f}",
                    f"{row.vii_t:.4f}",
                    f"{row.vii_cum:.4f}",
                ]
            )


def summarise(
    logs,
    *,
    lock_in_threshold: float,
    persistence_window: int,
) -> dict[str, float | int]:
    """Compute per-step logs into metrics suitable for tables/plots.

    Designed for reporting:
        - mean_vii: average viewpoint distance over steps (exposure diversity proxy)
        - final_vii_cum: running mean at final timestep (should ~ mean_vii for long runs)
        - action rates + unique videos + mean watch time
    """
    n = len(logs)
    if n == 0:
        raise ValueError("No logs to summarise")

    vii_mean = sum(r.vii_t for r in logs) / n
    final_vii_cum = logs[-1].vii_cum

    actions = [str(r.action).lower() for r in logs]
    watch_rate = actions.count("watch") / n
    sample_rate = actions.count("sample") / n
    avoid_rate = actions.count("avoid") / n

    unique_videos_seen = len({r.video_id for r in logs})
    mean_watch_time = sum(r.watch_time_s for r in logs) / n

    vii_series = [r.vii_t for r in logs]
    li = compute_lock_in_metrics(
        vii_series,
        lock_in_threshold=lock_in_threshold,
        persistence_window=persistence_window,
    )

    return {
        "mean_vii": float(vii_mean),
        "final_vii_cum": float(final_vii_cum),
        "watch_rate": float(watch_rate),
        "sample_rate": float(sample_rate),
        "avoid_rate": float(avoid_rate),
        "unique_videos_seen": int(unique_videos_seen),
        "mean_watch_time_s": float(mean_watch_time),
        "lock_in_events": li.lock_in_events,
        "time_to_first_lock_in": li.time_to_first_lock_in,
        "max_consecutive_lock_in_steps": li.max_consecutive_lock_in_steps,
        "total_lock_in_steps": li.total_lock_in_steps,
        "lock_in_rate": float(li.lock_in_rate),
    }


def main() -> None:
    cfg_path = Path("configs/experiment_baseline.json")
    cfg = load_config(cfg_path)

    steps = int(cfg["steps"])
    top_k = int(cfg["top_k"])
    alpha = float(cfg["alpha"])
    lock_in_threshold = float(cfg["lock_in_threshold"])
    persistence_window = int(cfg["persistence_window"])
    seeds = [int(x) for x in cfg["seeds"]]

    user = build_user(cfg)
    pool = build_video_pool(cfg)

    outputs_dir = Path("outputs") / "runs"
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    summary_path = results_dir / "summary.csv"

    rows: list[dict[str, Any]] = []

    # Reproducibility: each seed produces a deterministic run given fixed config + code version.
    for seed in seeds:
        rng = random.Random(seed)
        logs = run_simulation(
            user=user,
            video_pool=pool,
            steps=steps,
            rng=rng,
            top_k=top_k,
            alpha=alpha,
        )

        # Per-run logs are generated artifacts (gitignored) for inspection/debugging.
        write_run_log(outputs_dir / f"run_seed_{seed}.csv", logs)

        # Summary row (tracked)
        s = summarise(
            logs,
            lock_in_threshold=lock_in_threshold,
            persistence_window=persistence_window,
        )
        rows.append(
            {
                "seed": seed,
                "steps": steps,
                "top_k": top_k,
                "alpha": alpha,
                "lock_in_threshold": lock_in_threshold,
                "persistence_window": persistence_window,
                **s,
            }
        )

    fieldnames = list(rows[0].keys())
    with summary_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote per-run logs to: {outputs_dir}")
    print(f"Wrote summary to: {summary_path}")


if __name__ == "__main__":
    main()
