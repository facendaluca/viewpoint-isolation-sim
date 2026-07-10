from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
import time
from pathlib import Path
from typing import Any

from fyp_sim.agents import llm_diagnostics_delta, llm_diagnostics_snapshot
from fyp_sim.analysis import summarise_logs
from fyp_sim.artefacts import _fail_fast_old_alpha, create_run_artefacts
from fyp_sim.cli import run_cli
from fyp_sim.config_validation import validate_experiment_config
from fyp_sim.corpus import build_corpus
from fyp_sim.models import User, UserPhenotype
from fyp_sim.runners.csv_io import write_run_log_csv, write_summary_csv
from fyp_sim.runners.seed_sweep import build_decider, make_chooser
from fyp_sim.runners.seed_sweep_parsing import policy_curiosity, policy_mode
from fyp_sim.runtime_overrides import apply_runtime_overrides
from fyp_sim.simulation.engine import choose_video_weighted_top_k, run_simulation
from fyp_sim.simulation.engine_opt import WeightedTopKChooserOpt


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
    p = argparse.ArgumentParser(description="Run a parameter sweep over (top_k, rank_alpha).")
    p.add_argument("config", nargs="?", type=Path, default=Path("configs/experiment_sweep.json"))
    p.add_argument("--legacy", action="store_true", help="Write outputs to legacy locations.")
    p.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/runs"),
        help="Root directory for isolated run artefacts.",
    )
    p.add_argument("--steps", type=int, default=None, help="Temporary runtime step override.")
    p.add_argument("--seeds", type=int, nargs="+", default=None, help="Temporary seed list.")
    p.add_argument("--top-k-grid", type=int, nargs="+", default=None)
    p.add_argument("--rank-alpha-grid", type=float, nargs="+", default=None)
    p.add_argument("--policy-mode", choices=["heuristic", "llm"], default=None)
    p.add_argument("--llm-base-url", default=None)
    p.add_argument("--llm-model", default=None)
    p.add_argument("--prompt-id", default=None)
    p.add_argument(
        "--llm-rerank-slate", action=argparse.BooleanOptionalAction, default=None
    )
    p.add_argument(
        "--separate-rng-streams", action=argparse.BooleanOptionalAction, default=None
    )
    args = p.parse_args()

    cfg_path = args.config
    cfg = load_config(cfg_path)
    cfg, runtime_overrides = apply_runtime_overrides(
        cfg,
        steps=args.steps,
        seeds=args.seeds,
        policy_mode=args.policy_mode,
        llm_base_url=args.llm_base_url,
        llm_model=args.llm_model,
        prompt_id=args.prompt_id,
        llm_rerank_slate=args.llm_rerank_slate,
        separate_rng_streams=args.separate_rng_streams,
        top_k_grid=args.top_k_grid,
        rank_alpha_grid=args.rank_alpha_grid,
    )
    if runtime_overrides:
        print(f"[runtime overrides] {json.dumps(runtime_overrides, sort_keys=True)}")
    _fail_fast_old_alpha(cfg, cfg_path)
    config_audit = validate_experiment_config(cfg, runner="sweep", cfg_path=cfg_path)
    for warning in config_audit.warnings:
        print(f"[config warning] {warning}")

    steps = int(cfg["steps"])
    seeds = [int(x) for x in cfg["seeds"]]
    rank_alpha_grid = [float(x) for x in cfg["rank_alpha_grid"]]
    top_k_grid = [int(x) for x in cfg["top_k_grid"]]
    lock_in_threshold = float(cfg["lock_in_threshold"])
    persistence_window = int(cfg["persistence_window"])

    # Drift config (backwards compatible defaults)
    enable_viewpoint_drift = bool(cfg.get("enable_viewpoint_drift", False))
    drift_alpha = float(cfg.get("drift_alpha", cfg.get("viewpoint_drift_rate", 0.0)))
    viewpoint_drift_rate = drift_alpha
    drift_active = enable_viewpoint_drift and drift_alpha > 0.0

    enable_interest_updates = bool(cfg.get("enable_interest_updates", False))

    # If we mutate user state, rebuild per seed to avoid cross-seed leakage
    mutates_user = drift_active or enable_interest_updates

    base_user = build_user(cfg)
    pool = build_corpus(cfg)

    engine_name = str(cfg.get("engine", "baseline")).lower()
    if engine_name == "opt":
        base_chooser = WeightedTopKChooserOpt.from_pool(pool)
    else:
        base_chooser = choose_video_weighted_top_k

    curiosity = policy_curiosity(cfg)
    chooser = make_chooser(curiosity, base_chooser)
    decider = build_decider(cfg)
    llm_policy = policy_mode(cfg) == "llm"
    llm_cfg = (cfg.get("policy") or {}).get("llm") or {}
    llm_rerank = llm_policy and bool(llm_cfg.get("rerank_slate", False))
    separate_rng_streams = bool(cfg.get("separate_rng_streams", False))

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
            outputs_root=args.out,
            corpus=pool,
            runner="src.scripts.run_sweep",
        )

    rows: list[dict[str, Any]] = []
    per_seed_rows: list[dict[str, Any]] = []
    run_started = time.perf_counter()

    for top_k in top_k_grid:
        for rank_alpha in rank_alpha_grid:
            per_seed: list[dict[str, float | int]] = []

            for seed in seeds:
                rng = random.Random(seed)
                engagement_rng = (
                    random.Random(f"{seed}:engagement") if separate_rng_streams else None
                )

                # Rebuild the user when state mutates so seeds/grid cells stay independent
                user = build_user(cfg) if mutates_user else base_user

                diagnostics_before = llm_diagnostics_snapshot(decider)
                seed_started = time.perf_counter()
                logs = run_simulation(
                    user=user,
                    video_pool=pool,
                    steps=steps,
                    rng=rng,
                    engagement_rng=engagement_rng,
                    top_k=top_k,
                    rank_alpha=rank_alpha,
                    drift_alpha=drift_alpha,
                    chooser=chooser,
                    decider=decider,
                    llm_rerank=llm_rerank,
                    enable_interest_updates=bool(cfg.get("enable_interest_updates", False)),
                    interest_topic_alpha=float(cfg.get("interest_topic_alpha", 0.10)),
                    interest_tag_alpha=float(cfg.get("interest_tag_alpha", 0.05)),
                    interest_decay=float(cfg.get("interest_decay", 0.02)),
                    interest_normalise=bool(cfg.get("interest_normalise", False)),
                    interest_prune_below=float(cfg.get("interest_prune_below", 0.001)),
                    enable_viewpoint_drift=drift_active,
                    viewpoint_drift_rate=viewpoint_drift_rate,
                )
                seed_runtime_s = time.perf_counter() - seed_started
                diagnostics = llm_diagnostics_delta(
                    diagnostics_before, llm_diagnostics_snapshot(decider)
                )
                s = summarise_logs(
                    logs,
                    lock_in_threshold=lock_in_threshold,
                    persistence_window=persistence_window,
                )
                expected_calls = (
                    steps * min(top_k, len(pool)) if llm_policy and llm_rerank else steps
                ) if llm_policy else 0
                calls = diagnostics["llm_call_count"]
                seed_row: dict[str, float | int] = {
                    "top_k": top_k,
                    "rank_alpha": rank_alpha,
                    "seed": seed,
                    "runtime_s": seed_runtime_s,
                    **s,
                    "llm_expected_call_count": expected_calls,
                    **diagnostics,
                    "llm_valid_rate": diagnostics["llm_valid_count"] / calls if calls else 0.0,
                    "llm_fallback_rate": (
                        diagnostics["llm_fallback_count"] / calls if calls else 0.0
                    ),
                }
                per_seed.append(seed_row)
                per_seed_rows.append(seed_row)

                if artefacts is not None:
                    cell_dir = (
                        artefacts.root_dir
                        / "cells"
                        / f"top_k={top_k}"
                        / f"rank_alpha={rank_alpha:.4f}"
                        / "seeds"
                        / f"s{seed:05d}"
                    )
                    write_run_log_csv(
                        cell_dir / "run_log.csv",
                        logs,
                        include_viewpoint=drift_active,
                        include_llm_meta=llm_policy,
                    )

            # mean/std across seeds for each metric
            if args.legacy:
                agg: dict[str, Any] = {"top_k": top_k, "alpha": rank_alpha}
            else:
                agg: dict[str, Any] = {
                    "top_k": top_k,
                    "rank_alpha": rank_alpha,
                    "drift_alpha": drift_alpha,
                }
            keys = [k for k in per_seed[0] if k not in {"top_k", "rank_alpha", "seed"}]
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
        write_summary_csv(artefacts.root_dir / "per_seed_summary.csv", per_seed_rows)

        total_runtime_s = time.perf_counter() - run_started
        diagnostics_path: str | None = None
        if llm_policy:
            count_keys = [
                "llm_expected_call_count",
                "llm_call_count",
                "llm_valid_count",
                "llm_fallback_count",
                "llm_retry_count",
                "llm_prompt_tokens",
                "llm_completion_tokens",
                "llm_total_tokens",
                "llm_token_estimated_calls",
                "llm_fallback_no_client",
                "llm_fallback_timeout",
                "llm_fallback_client_error",
                "llm_fallback_invalid_output",
            ]
            totals = {key: sum(int(row[key]) for row in per_seed_rows) for key in count_keys}
            calls = totals["llm_call_count"]
            diagnostics_payload: dict[str, Any] = {
                **totals,
                "llm_valid_rate": totals["llm_valid_count"] / calls if calls else 0.0,
                "llm_fallback_rate": totals["llm_fallback_count"] / calls if calls else 0.0,
                "llm_prompt_id": str(llm_cfg.get("prompt_id", "decision_v1")),
                "llm_model": str(llm_cfg.get("model", "")),
                "llm_rerank_slate": llm_rerank,
                "token_usage_source": (
                    "provider"
                    if totals["llm_token_estimated_calls"] == 0
                    else "mixed_or_character_estimate"
                ),
            }
            diagnostics_file = artefacts.root_dir / "llm_diagnostics.json"
            diagnostics_file.write_text(
                json.dumps(diagnostics_payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            diagnostics_path = diagnostics_file.name

        manifest = json.loads(artefacts.manifest_path.read_text(encoding="utf-8"))
        manifest["runtime_s"] = total_runtime_s
        manifest["config_warnings"] = list(config_audit.warnings)
        manifest["per_seed_summary_path"] = "per_seed_summary.csv"
        manifest["llm_diagnostics_path"] = diagnostics_path
        manifest["separate_rng_streams"] = separate_rng_streams
        manifest["runtime_overrides"] = runtime_overrides
        artefacts.manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"Wrote run directory to: {artefacts.root_dir}")
        print(f"Wrote sweep summary to: {out_path}")


if __name__ == "__main__":
    run_cli(main)
