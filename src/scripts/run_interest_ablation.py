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
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_user(cfg: dict[str, Any]) -> User:
    u = cfg["user"]
    return User(
        phenotype=phenotype_from_str(u["phenotype"]),
        viewpoint_score=float(u["viewpoint_score"]),
        interest_vector={str(k): float(v) for k, v in u["interest_vector"].items()},
        sentiment_threshold=float(u["sentiment_threshold"]),
    )


def build_video_pool(cfg: dict[str, Any]) -> list[Video]:
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
        w.writerow(
            [
                "t",
                "video_id",
                "action",
                "watch_time_s",
                "interest",
                "vii_t",
                "vii_cum",
                "topic_interest",
                "interest_keys",
            ]
        )
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
                    f"{row.topic_interest:.4f}",
                    row.interest_keys,
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

    pool = build_video_pool(cfg)

    outputs_dir = Path("outputs") / "runs"
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    out_path = results_dir / "interest_ablation.csv"

    # Shared interest params (kept identical in OFF vs ON)
    interest_params = {
        "interest_topic_alpha": float(cfg.get("interest_topic_alpha", 0.10)),
        "interest_tag_alpha": float(cfg.get("interest_tag_alpha", 0.05)),
        "interest_decay": float(cfg.get("interest_decay", 0.02)),
        "interest_normalise": bool(cfg.get("interest_normalise", False)),
        "interest_prune_below": float(cfg.get("interest_prune_below", 0.001)),
    }

    rows: list[dict[str, Any]] = []

    for enabled in (False, True):
        for seed in seeds:
            rng = random.Random(seed)
            user = build_user(cfg)

            logs = run_simulation(
                user=user,
                video_pool=pool,
                steps=steps,
                rng=rng,
                top_k=top_k,
                alpha=alpha,
                enable_interest_updates=enabled,
                **interest_params,
            )

            write_run_log(
                outputs_dir / f"ablation{'on' if enabled else 'off'}_seed_{seed}.csv", logs
            )

            s = summarise_logs(
                logs,
                lock_in_threshold=lock_in_threshold,
                persistence_window=persistence_window,
            )
            rows.append(
                {
                    "condition": "interest_updates_on" if enabled else "interest_updates_off",
                    "seed": seed,
                    "steps": steps,
                    "top_k": top_k,
                    "alpha": alpha,
                    "lock_in_threshold": lock_in_threshold,
                    "persistence_window": persistence_window,
                    **interest_params,
                    **s,
                }
            )

    fieldnames = list(rows[0].keys())
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote: {out_path}")
    print(f"Wrote per-run logs to: {outputs_dir}")


if __name__ == "__main__":
    main()
