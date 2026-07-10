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
import json
from pathlib import Path
from typing import Any

from fyp_sim.artefacts import _fail_fast_old_alpha
from fyp_sim.runners.legacy_run_sweep import run_seed_sweep_legacy
from fyp_sim.runners.seed_sweep import run_seed_sweep
from fyp_sim.runtime_overrides import apply_runtime_overrides


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    if not isinstance(cfg, dict):
        raise TypeError(f"Config must be a JSON object: {path}")
    return cfg


def main() -> None:
    p = argparse.ArgumentParser(description="Run a seed sweep for a single experiment config.")
    p.add_argument("config", nargs="?", type=Path, default=Path("configs/experiment_baseline.json"))
    p.add_argument("--legacy", action="store_true", help="Write outputs to legacy locations.")
    p.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/runs"),
        help="Root directory for isolated run artefacts.",
    )
    p.add_argument(
        "--engine",
        choices=["baseline", "opt"],
        default=None,
        help="Override cfg.engine for this run (baseline or opt).",
    )
    p.add_argument("--steps", type=int, default=None, help="Temporary runtime step override.")
    p.add_argument("--seeds", type=int, nargs="+", default=None, help="Temporary seed list.")
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
    )
    _fail_fast_old_alpha(cfg, cfg_path)

    if args.engine is not None:
        cfg["engine"] = str(args.engine)
        runtime_overrides["engine"] = str(args.engine)

    if runtime_overrides:
        print(f"[runtime overrides] {json.dumps(runtime_overrides, sort_keys=True)}")

    if args.legacy:
        if args.out != Path("outputs/runs"):
            raise ValueError("--out is not supported with --legacy")
        if args.engine == "opt":
            raise ValueError("--engine is not supported with --legacy")
        out_dir, summary_path = run_seed_sweep_legacy(cfg, cfg_path=cfg_path)
        print(f"Wrote per-run logs to: {out_dir}")
        print(f"Wrote summary to: {summary_path}")
        return

    run_dir = run_seed_sweep(
        cfg,
        cfg_path=cfg_path,
        outputs_root=args.out,
        runtime_overrides=runtime_overrides,
    )
    print(f"Wrote run artefacts to: {run_dir}")
    print(f"Wrote summary to: {run_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
