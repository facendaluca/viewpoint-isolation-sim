from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
from pathlib import Path
from typing import Any

from fyp_sim.analysis import summarise_logs
from fyp_sim.artefacts import create_run_artefacts
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


def main() -> None:
    p = argparse.ArgumentParser(description="Run a parameter sweep over (top_k, alpha).")
    p.add_argument("config", nargs="?", type=Path, default=Path("configs/experiment_sweep.json"))
    p.add_argument("--legacy", action="store_true", help="Write outputs to legacy locations.")
    args = p.parse_args()

    cfg_path = args.config
    cfg = load_config(cfg_path)

    steps = int(cfg["steps"])
    seeds = [int(x) for x in cfg["seeds"]]
    alpha_grid = [float(x) for x in cfg["alpha_grid"]]
    top_k_grid = [int(x) for x in cfg["top_k_grid"]]
    lock_in_threshold = float(cfg["lock_in_threshold"])
    persistence_window = int(cfg["persistence_window"])

    user = build_user(cfg)
    pool = build_corpus(cfg)

    legacy_results_dir = Path("results")
    legacy_results_dir.mkdir(exist_ok=True)
    legacy_out_path = legacy_results_dir / "sweep_summary.csv"

    artefacts = None
    if not args.legacy:
        artefacts = create_run_artefacts(
            cfg=cfg,
            cfg_path=cfg_path,
            mode="sweep",
            seeds=seeds,
            outputs_root=Path("outputs/runs"),
        )

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
                    enable_interest_updates=bool(cfg.get("enable_interest_updates", False)),
                    interest_topic_alpha=float(cfg.get("interest_topic_alpha", 0.10)),
                    interest_tag_alpha=float(cfg.get("interest_tag_alpha", 0.05)),
                    interest_decay=float(cfg.get("interest_decay", 0.02)),
                    interest_normalise=bool(cfg.get("interest_normalise", False)),
                    interest_prune_below=float(cfg.get("interest_prune_below", 0.001)),
                )
                s = summarise_logs(
                    logs,
                    lock_in_threshold=lock_in_threshold,
                    persistence_window=persistence_window,
                )
                s["seed"] = seed
                per_seed.append(s)

            # mean/std across seeds for each metric
            agg: dict[str, Any] = {"top_k": top_k, "alpha": alpha}
            keys = [k for k in per_seed[0].keys() if k != "seed"]
            for k in keys:
                vals = [float(d[k]) for d in per_seed]
                agg[f"{k}_mean"] = statistics.fmean(vals)
                agg[f"{k}_std"] = statistics.pstdev(vals) if len(vals) > 1 else 0.0

            rows.append(agg)

    out_path = legacy_out_path if args.legacy else artefacts.summary_path  # type: ignore[union-attr]
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(rows[0].keys())
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    if args.legacy:
        print(f"Wrote sweep summary to: {out_path}")
    else:
        assert artefacts is not None
        print(f"Wrote run directory to: {artefacts.root_dir}")
        print(f"Wrote sweep summary to: {out_path}")


if __name__ == "__main__":
    main()
