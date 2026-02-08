"""
Batch experiment runner with structured outputs, manifests, and robust logging.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
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
    import json

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


def write_run_log(path: Path, logs: list[Any]) -> None:
    import csv

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
    parser = argparse.ArgumentParser(description="Run batch experiment with structured outputs.")
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
    n_runs = len(seeds)
    corpus_mode = raw_config.get("corpus", {}).get("source", "file")

    run_id = make_run_id(exp_name, config_slug, corpus_mode, base_seed, n_runs, config_hash[:8])

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
    logger = logging.getLogger("batch")
    if args.dry_run:
        # Configure basic stderr logging for dry-run
        logging.basicConfig(level=logging.INFO, format="%(message)s")

    logger.info(f"Starting batch experiment: {run_id}")
    logger.info(f"Config: {args.config_path} (hash: {config_hash[:8]})")
    logger.info(f"Corpus: {corpus_mode}")
    logger.info(f"Runs: {n_runs} (seeds: {seeds})")

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
        "output_root": str(output_root),
        "runs": [],
    }

    # initialize run entries
    for i, seed in enumerate(seeds):
        run_entry = {
            "index": i,
            "seed": seed,
            "status": "pending",
        }
        manifest["runs"].append(run_entry)

    if not args.dry_run:
        write_manifest(output_root, manifest)

    if args.dry_run:
        print(f"Dry run complete. Would write to: {output_root}")
        print(f"Manifest preview:\n{json.dumps(manifest, indent=2)}")
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
    results = []

    for i, seed in enumerate(seeds):
        run_dir = runs_dir / f"run_{i:03d}"
        run_dir.mkdir(exist_ok=True)

        # Update manifest to running
        manifest["runs"][i]["status"] = "running"
        manifest["runs"][i]["started_at"] = datetime.now().isoformat()
        manifest["runs"][i]["output_dir"] = str(run_dir)
        write_manifest(output_root, manifest)

        logger.info(f"Running seed {seed} ({i + 1}/{n_runs})...")

        t0 = time.time()
        try:
            # Per-run logger setup could go here, for now relying on batch log capturing exceptions

            rng = random.Random(seed)
            logs = run_simulation(
                user=user,
                video_pool=pool,
                steps=int(raw_config["steps"]),
                rng=rng,
                top_k=int(raw_config["top_k"]),
                alpha=float(raw_config["alpha"]),
            )

            # Write logs
            write_run_log(run_dir / "run.log.csv", logs)

            # Summarise
            s = summarise_logs(
                logs,
                lock_in_threshold=float(raw_config["lock_in_threshold"]),
                persistence_window=int(raw_config["persistence_window"]),
            )

            duration = time.time() - t0
            manifest["runs"][i]["status"] = "success"
            manifest["runs"][i]["ended_at"] = datetime.now().isoformat()
            manifest["runs"][i]["duration_s"] = duration

            result_row = {"seed": seed, "config_hash": config_hash[:8], **s}
            results.append(result_row)

        except Exception as e:
            duration = time.time() - t0
            logger.error(f"Run {i} failed: {e}", exc_info=True)
            manifest["runs"][i]["status"] = "failed"
            manifest["runs"][i]["ended_at"] = datetime.now().isoformat()
            manifest["runs"][i]["duration_s"] = duration
            manifest["runs"][i]["error"] = {"msg": str(e), "traceback": traceback.format_exc()}

            if args.fail_fast:
                logger.warning("Fail-fast enabled, stopping batch.")
                write_manifest(output_root, manifest)
                return 1

        write_manifest(output_root, manifest)

    # 6. Aggregate Results
    if results:
        import csv

        summary_path = agg_dir / "summary.csv"
        keys = list(results[0].keys())
        with summary_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(results)
        logger.info(f"Summary written to: {summary_path}")

    logger.info("Batch execution complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
