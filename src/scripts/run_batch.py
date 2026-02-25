"""
Batch experiment runner (seeds-only).

Default (new convention):
    outputs/runs/YYYYMMDD/HHMMSSZ_<mode>_<hash8>/
        manifest.json
        summary.csv
        seeds/s00042/run_log.csv

Legacy (--legacy):
    outputs/runs/run_seed_<seed>.csv
    results/summary.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any

from fyp_sim.agents.clients import OpenAICompatClient
from fyp_sim.agents.deciders import HeuristicDecider, LLMDecider
from fyp_sim.analysis import summarise_logs
from fyp_sim.artefacts import _fail_fast_old_alpha, create_run_artefacts
from fyp_sim.cli import run_cli
from fyp_sim.corpus import build_corpus
from fyp_sim.models import User, UserPhenotype
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
    """Load experiment configuration JSON."""
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


def _policy_mode(cfg: dict[str, Any]) -> str:
    policy = cfg.get("policy", {}) or {}
    mode = policy.get("mode", "heuristic")
    return str(mode).strip().lower()


def write_run_log(path: Path, logs, *, include_viewpoint: bool = False) -> None:
    """Write per-step logs for one seed to CSV. (generated artefacts, gitignored)"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        header = [
            "t",
            "video_id",
            "action",
            "watch_time_s",
            "interest",
            "topic_interest",
            "vii_t",
            "vii_cum",
        ]
        if include_viewpoint:
            header += ["user_viewpoint_pre", "user_viewpoint_post", "video_viewpoint_score"]
        w.writerow(header)
        for row in logs:
            base = [
                row.t,
                row.video_id,
                row.action,
                row.watch_time_s,
                f"{row.interest:.4f}",
                f"{getattr(row, 'topic_interest', 0.0):.4f}",
                f"{row.vii_t:.4f}",
                f"{row.vii_cum:.4f}",
            ]
            if include_viewpoint:
                base += [
                    f"{row.user_viewpoint_pre:.4f}",
                    f"{row.user_viewpoint_post:.4f}",
                    f"{row.video_viewpoint_score:.4f}",
                ]
            w.writerow(base)


def main() -> None:
    p = argparse.ArgumentParser(description="Run a seed sweep for a single experiment config.")
    p.add_argument("config", nargs="?", type=Path, default=Path("configs/experiment_baseline.json"))
    p.add_argument("--legacy", action="store_true", help="Write outputs to legacy locations.")
    args = p.parse_args()

    cfg_path = args.config
    cfg = load_config(cfg_path)

    _fail_fast_old_alpha(cfg, cfg_path)

    steps = int(cfg["steps"])
    top_k = int(cfg["top_k"])
    rank_alpha = float(cfg["rank_alpha"])
    lock_in_threshold = float(cfg["lock_in_threshold"])
    persistence_window = int(cfg["persistence_window"])
    seeds = [int(x) for x in cfg["seeds"]]

    enable_interest_updates = bool(cfg.get("enable_interest_updates", False))

    # Drift config (backwards compatible defaults)
    enable_viewpoint_drift = bool(cfg.get("enable_viewpoint_drift", False))
    drift_alpha = float(cfg.get("drift_alpha", cfg.get("viewpoint_drift_rate", 0.0)))
    viewpoint_drift_rate = drift_alpha

    drift_active = enable_viewpoint_drift and drift_alpha > 0.0
    # If we mutate user state, rebuild per seed to avoid cross-seed leakage
    mutates_user = drift_active or enable_interest_updates

    base_user = build_user(cfg)
    pool = build_corpus(cfg)

    mode = _policy_mode(cfg)
    policy = cfg.get("policy", {}) or {}
    llm_cfg = policy.get("llm", {}) or {}

    if mode == "llm":
        if "model" not in llm_cfg:
            raise ValueError("policy.llm.model is required when policy.mode='llm'")

        client = OpenAICompatClient(
            base_url=(llm_cfg.get("base_url") or "http://localhost:1234/v1"),
            model=str(llm_cfg["model"]),
            api_key=llm_cfg.get("api_key"),
            temperature=float(llm_cfg.get("temperature", 0.0)),
            max_tokens=llm_cfg.get("max_tokens"),
        )

        decider = LLMDecider(
            prompt_id=str(llm_cfg.get("prompt_id", "decision_v1")),
            client=client,
            timeout_s=float(llm_cfg.get("timeout_s", 10.0)),
            fallback=HeuristicDecider(),
        )
    elif mode == "heuristic":
        decider = HeuristicDecider()
    else:
        raise ValueError("policy.mode must be 'heuristic' or 'llm'")

    # Legacy paths (kept for compatibility)
    legacy_outputs_dir = Path("outputs") / "runs"
    legacy_results_dir = Path("results")
    legacy_results_dir.mkdir(exist_ok=True)
    legacy_summary_path = legacy_results_dir / "summary.csv"

    # New run folder (default)
    artefacts = None
    if not args.legacy:
        artefacts = create_run_artefacts(
            cfg=cfg,
            cfg_path=cfg_path,
            mode=mode,
            seeds=seeds,
            outputs_root=Path("outputs/runs"),
        )

    rows: list[dict[str, Any]] = []

    for seed in seeds:
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
            enable_interest_updates=bool(cfg.get("enable_interest_updates", False)),
            interest_topic_alpha=float(cfg.get("interest_topic_alpha", 0.10)),
            interest_tag_alpha=float(cfg.get("interest_tag_alpha", 0.05)),
            interest_decay=float(cfg.get("interest_decay", 0.02)),
            interest_normalise=bool(cfg.get("interest_normalise", False)),
            interest_prune_below=float(cfg.get("interest_prune_below", 0.001)),
            enable_viewpoint_drift=enable_viewpoint_drift,
            viewpoint_drift_rate=viewpoint_drift_rate,
        )

        if args.legacy:
            write_run_log(
                legacy_outputs_dir / f"run_seed_{seed}.csv", logs, include_viewpoint=drift_active
            )
        else:
            assert artefacts is not None
            seed_dir = artefacts.seeds_dir / f"s{seed:05d}"
            write_run_log(seed_dir / "run_log.csv", logs, include_viewpoint=drift_active)

        s = summarise_logs(
            logs,
            lock_in_threshold=lock_in_threshold,
            persistence_window=persistence_window,
        )
        row = {
            "seed": seed,
            "steps": steps,
            "top_k": top_k,
            "rank_alpha": rank_alpha,
            "drift_alpha": drift_alpha,
            "lock_in_threshold": lock_in_threshold,
            "persistence_window": persistence_window,
            **s,
        }

        rows.append(row)

    fieldnames = list(rows[0].keys())
    summary_path = legacy_summary_path if args.legacy else artefacts.summary_path  # type: ignore[union-attr]
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    with summary_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    if args.legacy:
        print(f"Wrote per-run logs to: {legacy_outputs_dir}")
        print(f"Wrote summary to: {summary_path}")
    else:
        assert artefacts is not None
        print(f"Wrote run directory to: {artefacts.root_dir}")
        print(f"Wrote summary to: {summary_path}")


if __name__ == "__main__":
    run_cli(main)
