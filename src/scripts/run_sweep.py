from __future__ import annotations

import csv
import json
import random
import statistics
from pathlib import Path
from typing import Any

from fyp_sim.analysis import summarise_logs
from fyp_sim.corpus import build_corpus
from fyp_sim.models import User, UserPhenotype
from fyp_sim.simulation.engine import run_simulation


def phenotype_from_str(s: str) -> UserPhenotype:
    s = s.strip().lower()
    if s == "watcher":
        return UserPhenotype.WATCHER
    if s == "sampler":
        return UserPhenotype.SAMPLER
    if s == "avoider":
        return UserPhenotype.AVOIDER
    raise ValueError(f"Unknown phenotype: {s!r} (expected watcher/sampler/avoider)")


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r") as f:
        return json.load(f)


def build_user(cfg: dict[str, Any]) -> User:
    u = cfg["user"]
    return User(
        phenotype=phenotype_from_str(u["phenotype"]),
        viewpoint_score=float(u["viewpoint_score"]),
        interest_vector={str(k): float(v) for k, v in u["interest_vector"].items()},
        sentiment_threshold=float(u["sentiment_threshold"]),
    )


def mean_std(xs: list[float]) -> tuple[float, float]:
    if not xs:
        return 0.0, 0.0
    if len(xs) == 1:
        return float(xs[0]), 0.0
    return float(statistics.mean(xs)), float(statistics.pstdev(xs))


def main() -> None:
    cfg = load_config(Path("configs/experiment_sweep.json"))

    steps = int(cfg["steps"])
    seeds = [int(x) for x in cfg["seeds"]]
    alpha_grid = [float(x) for x in cfg["alpha_grid"]]
    top_k_grid = [int(x) for x in cfg["top_k_grid"]]
    lock_in_threshold = float(cfg["lock_in_threshold"])
    persistence_window = int(cfg["persistence_window"])

    user = build_user(cfg)

    # Use shared corpus builder
    pool = build_corpus(cfg)

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    out_path = results_dir / "sweep_summary.csv"

    # Aggregate per (top_k, alpha) across seeds
    rows: list[dict[str, Any]] = []

    for top_k in top_k_grid:
        for alpha in alpha_grid:
            per_seed: list[dict[str, float | int]] = []

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
                per_seed.append(
                    summarise_logs(
                        logs,
                        lock_in_threshold=lock_in_threshold,
                        persistence_window=persistence_window,
                    )
                )

            # Aggregate a few key metrics (mean + std)
            mean_vii_mu, mean_vii_sd = mean_std([float(r["mean_vii"]) for r in per_seed])
            final_vii_mu, final_vii_sd = mean_std([float(r["final_vii_cum"]) for r in per_seed])
            lock_rate_mu, lock_rate_sd = mean_std([float(r["lock_in_rate"]) for r in per_seed])

            rows.append(
                {
                    "steps": steps,
                    "top_k": top_k,
                    "alpha": alpha,
                    "lock_in_threshold": lock_in_threshold,
                    "persistence_window": persistence_window,
                    "n_seeds": len(seeds),
                    "mean_vii_mean": mean_vii_mu,
                    "mean_vii_std": mean_vii_sd,
                    "final_vii_mean": final_vii_mu,
                    "final_vii_std": final_vii_sd,
                    "lock_in_rate_mean": lock_rate_mu,
                    "lock_in_rate_std": lock_rate_sd,
                }
            )

    fieldnames = list(rows[0].keys())
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote sweep summary to: {out_path}")


if __name__ == "__main__":
    main()
