from __future__ import annotations

from pathlib import Path
from typing import Any

from fyp_sim.analysis import summarise_logs
from fyp_sim.corpus import build_corpus
from fyp_sim.runners.csv_io import write_run_log_csv, write_summary_csv
from fyp_sim.runners.seed_sweep import build_decider, build_user, extract_seeds
from fyp_sim.simulation.engine import run_simulation


def run_seed_sweep_legacy(
    cfg: dict[str, Any],
    *,
    cfg_path: Path | None,
    legacy_outputs_dir: Path = Path("outputs") / "runs",
    legacy_results_dir: Path = Path("results"),
) -> tuple[Path, Path]:
    """
    Legacy runner:
      outputs/runs/run_seed_<seed>.csv
      results/summary.csv
    Returns (legacy_outputs_dir, summary_path).
    """
    steps = int(cfg["steps"])
    top_k = int(cfg["top_k"])
    rank_alpha = float(cfg["rank_alpha"])
    lock_in_threshold = float(cfg["lock_in_threshold"])
    persistence_window = int(cfg["persistence_window"])
    seeds = [int(x) for x in extract_seeds(cfg)]

    enable_interest_updates = bool(cfg.get("enable_interest_updates", False))

    enable_viewpoint_drift = bool(cfg.get("enable_viewpoint_drift", False))
    drift_alpha = float(cfg.get("drift_alpha", cfg.get("viewpoint_drift_rate", 0.0)))
    viewpoint_drift_rate = drift_alpha
    drift_active = enable_viewpoint_drift and drift_alpha > 0.0

    mutates_user = drift_active or enable_interest_updates

    legacy_outputs_dir.mkdir(parents=True, exist_ok=True)
    legacy_results_dir.mkdir(parents=True, exist_ok=True)
    summary_path = legacy_results_dir / "summary.csv"

    base_user = build_user(cfg)
    pool = build_corpus(cfg)
    decider = build_decider(cfg)

    rows: list[dict[str, Any]] = []

    for seed in seeds:
        # Keep determinism: your engine takes an rng built from seed
        import random

        rng = random.Random(seed)
        user = build_user(cfg) if mutates_user else base_user

        logs = run_simulation(
            user=user,
            video_pool=pool,
            steps=steps,
            rng=rng,
            top_k=top_k,
            rank_alpha=rank_alpha,
            drift_alpha=drift_alpha,
            decider=decider,
            enable_interest_updates=enable_interest_updates,
            interest_topic_alpha=float(cfg.get("interest_topic_alpha", 0.10)),
            interest_tag_alpha=float(cfg.get("interest_tag_alpha", 0.05)),
            interest_decay=float(cfg.get("interest_decay", 0.02)),
            interest_normalise=bool(cfg.get("interest_normalise", False)),
            interest_prune_below=float(cfg.get("interest_prune_below", 0.001)),
            enable_viewpoint_drift=enable_viewpoint_drift,
            viewpoint_drift_rate=viewpoint_drift_rate,
        )

        write_run_log_csv(
            legacy_outputs_dir / f"run_seed_{seed}.csv",
            logs,
            include_viewpoint=drift_active,
        )

        s = summarise_logs(
            logs, lock_in_threshold=lock_in_threshold, persistence_window=persistence_window
        )
        rows.append(
            {
                "seed": seed,
                "steps": steps,
                "top_k": top_k,
                "rank_alpha": rank_alpha,
                "drift_alpha": drift_alpha,
                "lock_in_threshold": lock_in_threshold,
                "persistence_window": persistence_window,
                **s,
            }
        )

    write_summary_csv(summary_path, rows)
    return legacy_outputs_dir, summary_path
