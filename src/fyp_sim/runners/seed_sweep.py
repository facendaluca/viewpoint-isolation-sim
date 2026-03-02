from __future__ import annotations

import random
from collections.abc import Callable
from pathlib import Path
from typing import Any

from fyp_sim.agents.clients import OpenAICompatClient
from fyp_sim.agents.deciders import HeuristicDecider, LLMDecider
from fyp_sim.analysis import summarise_logs
from fyp_sim.artefacts import _fail_fast_old_alpha, create_run_artefacts
from fyp_sim.corpus import build_corpus
from fyp_sim.models import User, UserPhenotype
from fyp_sim.runners.csv_io import write_run_log_csv, write_summary_csv
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


def build_user(cfg: dict[str, Any]) -> User:
    u = cfg["user"]
    return User(
        phenotype=phenotype_from_str(u["phenotype"]),
        viewpoint_score=float(u["viewpoint_score"]),
        interest_vector={str(k): float(v) for k, v in u["interest_vector"].items()},
        sentiment_threshold=float(u["sentiment_threshold"]),
    )


def policy_mode(cfg: dict[str, Any]) -> str:
    policy = cfg.get("policy", {}) or {}
    return str(policy.get("mode", "heuristic")).strip().lower()


def build_decider(cfg: dict[str, Any]):
    mode = policy_mode(cfg)
    policy = cfg.get("policy", {}) or {}
    llm_cfg = policy.get("llm", {}) or {}

    if mode == "heuristic":
        return HeuristicDecider()

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
        return LLMDecider(
            prompt_id=str(llm_cfg.get("prompt_id", "decision_v1")),
            client=client,
            timeout_s=float(llm_cfg.get("timeout_s", 10.0)),
            fallback=HeuristicDecider(),
        )

    raise ValueError("policy.mode must be 'heuristic' or 'llm'")


def extract_seeds(cfg: dict[str, Any]) -> list[int]:
    seeds_raw = cfg.get("seeds")
    if isinstance(seeds_raw, list) and all(isinstance(x, int) for x in seeds_raw):
        return [int(x) for x in seeds_raw]

    seed_raw = cfg.get("seed")
    if isinstance(seed_raw, int):
        return [seed_raw]

    return [0]


def _scenario_from(cfg: dict[str, Any], cfg_path: Path | None) -> str:
    s = cfg.get("scenario")
    if isinstance(s, str) and s.strip():
        return s.strip()
    if cfg_path is not None:
        return cfg_path.stem
    return "experiment_baseline"


def _scenario_to_mode(scenario: str) -> str:
    scenario = scenario.strip() or "experiment_baseline"
    return scenario.removeprefix("experiment_") if scenario.startswith("experiment_") else scenario


def run_seed_sweep(
    cfg: dict[str, Any],
    *,
    cfg_path: Path | None,
    outputs_root: Path,
    progress_cb: Callable[[int, int, int], None] | None = None,
    force_heuristic: bool = False,
) -> Path:
    """
    Real experiment execution: run all seeds and write:
        - config_resolved.json / manifest.json via create_run_artefacts
        - seeds/sXXXXX/run_log.csv
        - summary.csv
    Returns the created run directory.
    """
    if not isinstance(cfg, dict):
        raise TypeError("cfg must be a dict[str, Any]")

    # Avoid mutating caller
    cfg = dict(cfg)

    if force_heuristic:
        cfg["policy"] = {"mode": "heuristic"}

    _fail_fast_old_alpha(cfg, cfg_path)

    scenario = _scenario_from(cfg, cfg_path)
    cfg["scenario"] = scenario
    mode = _scenario_to_mode(scenario)

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

    base_user = build_user(cfg)
    pool = build_corpus(cfg)
    decider = build_decider(cfg)

    artefacts = create_run_artefacts(
        cfg=cfg,
        cfg_path=cfg_path,
        mode=mode,
        seeds=seeds,
        outputs_root=outputs_root,
        write_config_snapshot=True,
        corpus=pool,
    )

    rows: list[dict[str, Any]] = []

    for i, seed in enumerate(seeds, start=1):
        if progress_cb is not None:
            progress_cb(i, len(seeds), seed)

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

        seed_dir = artefacts.seeds_dir / f"s{seed:05d}"
        write_run_log_csv(seed_dir / "run_log.csv", logs, include_viewpoint=drift_active)

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

    write_summary_csv(artefacts.summary_path, rows)
    return artefacts.root_dir
