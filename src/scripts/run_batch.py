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

from fyp_sim.analysis import summarise_logs
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
        s = summarise_logs(
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
