"""
Sweep experiment runner with structured outputs, manifests, and robust logging.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import statistics
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

from fyp_sim.analysis import summarise_logs
from fyp_sim.corpus import build_corpus
from fyp_sim.models import User, UserPhenotype
from fyp_sim.simulation.engine import run_simulation
from fyp_sim.utils.run_artifacts import (
    make_run_id,
    slugify,
    stable_config_hash,
    write_json,
    write_manifest,
)

# --- Configuration & Setup ---


def setup_logging(batch_log_path: Path, verbose: bool = False) -> None:
    """Configure root logger to write to batch log file and stderr."""
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)  # Capture everything, handlers filter

    # File handler (batch log)
    fh = logging.FileHandler(batch_log_path, mode="w")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    root.addHandler(fh)

    # Console handler
    ch = logging.StreamHandler(sys.stderr)
    ch.setLevel(logging.DEBUG if verbose else logging.INFO)
    ch.setFormatter(logging.Formatter("%(message)s"))
    root.addHandler(ch)


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r") as f:
        return json.load(f)


def phenotype_from_str(s: str) -> UserPhenotype:
    s = s.strip().lower()
    if s == "watcher":
        return UserPhenotype.WATCHER
    if s == "sampler":
        return UserPhenotype.SAMPLER
    if s == "avoider":
        return UserPhenotype.AVOIDER
    raise ValueError(f"Unknown phenotype: {s!r}")


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


def write_run_log(path: Path, logs: list[Any]) -> None:
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


# --- Main Execution ---


def main() -> int:
    parser = argparse.ArgumentParser(description="Run sweep experiment with structured outputs.")
    parser.add_argument("config_path", type=Path, help="Path to experiment config JSON.")
    parser.add_argument("--name", type=str, help="Experiment name (default: config filename stem).")
    parser.add_argument("--seed", type=int, help="Base random seed (overrides config).")

    # Corpus overrides
    parser.add_argument(
        "--corpus-mode", choices=["file", "generated"], help="Override corpus source."
    )
    parser.add_argument("--corpus-size", type=int, help="Override N videos (generated only).")
    parser.add_argument("--corpus-seed", type=int, help="Override corpus seed (generated only).")

    # Execution flags
    parser.add_argument(
        "--dry-run", action="store_true", help="Plan/print manifest but do not execute."
    )
    parser.add_argument("--fail-fast", action="store_true", help="Stop on first failure.")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose console output.")

    args = parser.parse_args()

    # 1. Load & Resolve Config
    if not args.config_path.exists():
        print(f"Error: Config not found: {args.config_path}", file=sys.stderr)
        return 1

    raw_config = load_config(args.config_path)

    # Apply Overrides (Mutation!)
    # Base seed
    if args.seed is not None:
        if "seeds" in raw_config:
            # We'll treat this as: run list is just [base_seed]
            print(f"Override: replacing seeds list with single seed {args.seed}", file=sys.stderr)
            raw_config["seeds"] = [args.seed]
        else:
            raw_config["seed"] = args.seed

    # Corpus overrides
    if "corpus" not in raw_config:
        raw_config["corpus"] = {}

    if args.corpus_mode:
        raw_config["corpus"]["source"] = args.corpus_mode

    if args.corpus_size is not None:
        raw_config["corpus"]["n_videos"] = args.corpus_size

    if args.corpus_seed is not None:
        raw_config["corpus"]["seed"] = args.corpus_seed

    # 2. Prepare Output Structure
    exp_name = args.name or args.config_path.stem
    config_slug = slugify(args.config_path.stem)
    config_hash = stable_config_hash(raw_config)

    # Determine base info for ID
    seeds = raw_config.get("seeds", [])
    if not seeds and "seed" in raw_config:
        seeds = [int(raw_config["seed"])]
    if not seeds:
        print("Error: No seeds found in config.", file=sys.stderr)
        return 1

    base_seed = seeds[0]
    # For sweep, n_runs is combinatorial
    steps = int(raw_config["steps"])
    alpha_grid = [float(x) for x in raw_config["alpha_grid"]]
    top_k_grid = [int(x) for x in raw_config["top_k_grid"]]
    lock_in_threshold = float(raw_config["lock_in_threshold"])
    persistence_window = int(raw_config["persistence_window"])

    n_combinations = len(alpha_grid) * len(top_k_grid)
    n_runs_total = n_combinations * len(seeds)

    corpus_mode = raw_config.get("corpus", {}).get("source", "file")

    run_id = make_run_id(
        exp_name, config_slug, corpus_mode, base_seed, n_runs_total, config_hash[:8]
    )

    output_root = Path("outputs/runs") / run_id
    runs_dir = output_root / "runs"
    agg_dir = output_root / "aggregate"

    if not args.dry_run:
        output_root.mkdir(parents=True, exist_ok=True)
        runs_dir.mkdir(exist_ok=True)
        agg_dir.mkdir(exist_ok=True)

        # Write resolved config
        write_json(output_root / "resolved_config.json", raw_config)

        # Setup logging
        setup_logging(output_root / "batch.log", args.verbose)

    # Note: We must get logger AFTER setup_logging
    logger = logging.getLogger("sweep")
    if args.dry_run:
        # Configure basic stderr logging for dry-run
        logging.basicConfig(level=logging.INFO, format="%(message)s")

    logger.info(f"Starting sweep experiment: {run_id}")
    logger.info(f"Config: {args.config_path} (hash: {config_hash[:8]})")
    logger.info(f"Corpus: {corpus_mode}")
    logger.info(f"Grid: top_k={top_k_grid}, alpha={alpha_grid} ({n_combinations} combos)")
    logger.info(f"Seeds per combo: {len(seeds)} (Total runs: {n_runs_total})")

    # 3. Create Manifest
    manifest = {
        "run_id": run_id,
        "created_at": datetime.now().isoformat(),
        "exp_name": exp_name,
        "config_path": str(args.config_path),
        "resolved_config_hash": config_hash,
        "corpus": raw_config.get("corpus", {}),
        "base_seed": base_seed,
        "seeds": seeds,
        "grid": {"top_k": top_k_grid, "alpha": alpha_grid},
        "output_root": str(output_root),
        "runs": [],
    }

    # Initialize run entries
    run_index = 0
    # Order: top_k -> alpha -> seed (matches original logic somewhat)
    for top_k in top_k_grid:
        for alpha in alpha_grid:
            for seed in seeds:
                run_entry = {
                    "index": run_index,
                    "top_k": top_k,
                    "alpha": alpha,
                    "seed": seed,
                    "status": "pending",
                }
                manifest["runs"].append(run_entry)
                run_index += 1

    if not args.dry_run:
        write_manifest(output_root, manifest)

    if args.dry_run:
        print(f"Dry run complete. Would write to: {output_root}")
        print(f"Manifest preview (first 5 runs):\n{json.dumps(manifest['runs'][:5], indent=2)}...")
        return 0

    # 4. Build Shared Resources
    try:
        user = build_user(raw_config)
        logger.info("Building corpus...")
        pool = build_corpus(raw_config)
        logger.info(f"Corpus loaded: {len(pool)} videos")
    except Exception as e:
        logger.critical(f"Failed to build shared resources: {e}", exc_info=True)
        return 1

    # 5. Execute Runs
    agg_rows: list[dict[str, Any]] = []

    # We iterate grid again to run
    run_idx = 0
    for top_k in top_k_grid:
        for alpha in alpha_grid:
            combo_metrics = []

            for seed in seeds:
                current_run_idx = run_idx
                run_idx += 1

                run_dir = runs_dir / f"run_{current_run_idx:04d}"
                run_dir.mkdir(exist_ok=True)

                # Update manifest
                manifest["runs"][current_run_idx]["status"] = "running"
                manifest["runs"][current_run_idx]["started_at"] = datetime.now().isoformat()
                manifest["runs"][current_run_idx]["output_dir"] = str(run_dir)
                write_manifest(output_root, manifest)

                logger.info(
                    f"Running run {current_run_idx + 1}/{n_runs_total}: top_k={top_k}, alpha={alpha}, seed={seed}"
                )

                t0 = time.time()
                try:
                    rng = random.Random(seed)
                    logs = run_simulation(
                        user=user,
                        video_pool=pool,
                        steps=steps,
                        rng=rng,
                        top_k=top_k,
                        alpha=alpha,
                    )

                    # Write per-run logs
                    write_run_log(run_dir / "run.log.csv", logs)

                    # Summarise
                    s = summarise_logs(
                        logs,
                        lock_in_threshold=lock_in_threshold,
                        persistence_window=persistence_window,
                    )
                    combo_metrics.append(s)

                    duration = time.time() - t0
                    manifest["runs"][current_run_idx]["status"] = "success"
                    manifest["runs"][current_run_idx]["ended_at"] = datetime.now().isoformat()
                    manifest["runs"][current_run_idx]["duration_s"] = duration

                except Exception as e:
                    duration = time.time() - t0
                    logger.error(f"Run {current_run_idx} failed: {e}", exc_info=True)
                    manifest["runs"][current_run_idx]["status"] = "failed"
                    manifest["runs"][current_run_idx]["ended_at"] = datetime.now().isoformat()
                    manifest["runs"][current_run_idx]["duration_s"] = duration
                    manifest["runs"][current_run_idx]["error"] = {
                        "msg": str(e),
                        "traceback": traceback.format_exc(),
                    }

                    if args.fail_fast:
                        logger.warning("Fail-fast enabled, stopping sweep.")
                        write_manifest(output_root, manifest)
                        return 1

            # Aggregate metrics for this (top_k, alpha) combo
            if combo_metrics:
                mean_vii_mu, mean_vii_sd = mean_std([float(r["mean_vii"]) for r in combo_metrics])
                final_vii_mu, final_vii_sd = mean_std(
                    [float(r["final_vii_cum"]) for r in combo_metrics]
                )
                lock_rate_mu, lock_rate_sd = mean_std(
                    [float(r["lock_in_rate"]) for r in combo_metrics]
                )

                agg_rows.append(
                    {
                        "steps": steps,
                        "top_k": top_k,
                        "alpha": alpha,
                        "lock_in_threshold": lock_in_threshold,
                        "persistence_window": persistence_window,
                        "n_seeds": len(combo_metrics),
                        "mean_vii_mean": mean_vii_mu,
                        "mean_vii_std": mean_vii_sd,
                        "final_vii_mean": final_vii_mu,
                        "final_vii_std": final_vii_sd,
                        "lock_in_rate_mean": lock_rate_mu,
                        "lock_in_rate_std": lock_rate_sd,
                    }
                )

            write_manifest(output_root, manifest)

    # 6. Aggregate Results
    if agg_rows:
        summary_path = agg_dir / "sweep_summary.csv"
        keys = list(agg_rows[0].keys())
        with summary_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(agg_rows)
        logger.info(f"Sweep summary written to: {summary_path}")

    logger.info("Sweep execution complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
