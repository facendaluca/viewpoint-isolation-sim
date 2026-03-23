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
from fyp_sim.runners.csv_io import write_run_log_csv, write_summary_csv
from fyp_sim.runners.seed_sweep_parsing import (
    build_user_from_dict,
    derive_agent_seed,
    extract_seeds,
    parse_agents,
    policy_curiosity,
    policy_mode,
    scenario_from,
    scenario_to_mode,
)
from fyp_sim.simulation.engine import choose_video_weighted_top_k, run_simulation
from fyp_sim.simulation.engine_opt import WeightedTopKChooserOpt


def build_decider(cfg: dict[str, Any]):
    """Build the action decider from cfg.policy (heuristic or LLM)."""
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
            model=str(llm_cfg.get("model")),
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


def _make_chooser(curiosity: float, base_chooser):
    """Return a chooser function with run_simulation(chooser=...)."""
    if curiosity <= 0.0:
        return base_chooser

    def chooser(user, pool, rng, *, top_k: int, rank_alpha: float):
        if rng.random() < curiosity:
            return rng.choice(pool)
        return base_chooser(user, pool, rng, top_k=top_k, rank_alpha=rank_alpha)

    return chooser


def run_seed_sweep(
    cfg: dict[str, Any],
    *,
    cfg_path: Path | None,
    outputs_root: Path,
    runner: str | None = None,
    progress_cb: Callable[[int, int, int], None] | None = None,
    force_heuristic: bool = False,
) -> Path:
    """Run all seeds (and optionally multiple agents) and write artefacts + logs + summary.

    Outputs:
        - <run>/manifest.json + config_resolved.json
        - <run>/seeds/sXXXXX/run_log.csv
        - <run>/summary.csv
    """
    if not isinstance(cfg, dict):
        raise TypeError("cfg must be a dict[str, Any]")

    # Avoid mutating caller
    cfg = dict(cfg)

    if force_heuristic:
        cfg["policy"] = {"mode": "heuristic"}

    _fail_fast_old_alpha(cfg, cfg_path)

    scenario = scenario_from(cfg, cfg_path)
    cfg["scenario"] = scenario
    mode = scenario_to_mode(scenario)

    # Required core params
    steps = int(cfg["steps"])
    top_k = int(cfg["top_k"])
    rank_alpha = float(cfg["rank_alpha"])
    lock_in_threshold = float(cfg["lock_in_threshold"])
    persistence_window = int(cfg["persistence_window"])
    seeds = [int(x) for x in extract_seeds(cfg)]

    # Drift + interest update toggles
    enable_interest_updates = bool(cfg.get("enable_interest_updates", False))
    enable_viewpoint_drift = bool(cfg.get("enable_viewpoint_drift", False))
    drift_alpha = float(cfg.get("drift_alpha", cfg.get("viewpoint_drift_rate", 0.0)))
    drift_active = enable_viewpoint_drift and drift_alpha > 0.0
    mutates_user = drift_active or enable_interest_updates

    # Fixed corpus for entire run (required for evaluation comparability)
    pool = build_corpus(cfg)

    # Engine selection (default baseline)
    engine_name = str(cfg.get("engine", "baseline")).lower()
    if engine_name == "opt":
        base_chooser = WeightedTopKChooserOpt.from_pool(pool)
    elif engine_name == "baseline":
        base_chooser = choose_video_weighted_top_k
    else:
        raise ValueError("cfg['engine] must be 'baseline' or 'opt'")

    # Policy
    curiosity = policy_curiosity(cfg)
    chooser = _make_chooser(curiosity, base_chooser)
    decider = build_decider(cfg)

    # Cohort support
    agents = parse_agents(cfg)
    include_agent_id = len(agents) > 1

    # Base user optimisation (only if user never mutates and only one agent)
    base_user = None
    if (not mutates_user) and (not include_agent_id):
        base_user = build_user_from_dict(cfg["user"])

    artefacts = create_run_artefacts(
        cfg=cfg,
        cfg_path=cfg_path,
        mode=mode,
        seeds=seeds,
        outputs_root=outputs_root,
        runner=runner,
        write_config_snapshot=True,
        corpus=pool,
    )

    interest_kwargs = {
        "enable_interest_updates": enable_interest_updates,
        "interest_topic_alpha": float(cfg.get("interest_topic_alpha", 0.10)),
        "interest_tag_alpha": float(cfg.get("interest_tag_alpha", 0.05)),
        "interest_decay": float(cfg.get("interest_decay", 0.02)),
        "interest_normalise": bool(cfg.get("interest_normalise", False)),
        "interest_prune_below": float(cfg.get("interest_prune_below", 0.001)),
    }

    rows: list[dict[str, Any]] = []

    for i, seed in enumerate(seeds, start=1):
        if progress_cb is not None:
            progress_cb(i, len(seeds), seed)

        seed_dir = artefacts.seeds_dir / f"s{seed:05d}"

        combined_logs: list[Any] = []

        for agent_id, agent_user_dict in agents:
            agent_rng = random.Random(derive_agent_seed(seed, agent_id))

            if base_user is not None:
                user = base_user
            else:
                user = build_user_from_dict(agent_user_dict)

            logs = run_simulation(
                user=user,
                video_pool=pool,
                steps=steps,
                rng=agent_rng,
                top_k=top_k,
                rank_alpha=rank_alpha,
                drift_alpha=drift_alpha,
                chooser=chooser,
                decider=decider,
                enable_viewpoint_drift=enable_viewpoint_drift,
                viewpoint_drift_rate=drift_alpha,
                **interest_kwargs,
            )

            if include_agent_id:
                for r in logs:
                    r.agent_id = agent_id
                combined_logs.extend(logs)
            else:
                combined_logs = logs

            # Summary row is per-(seed, agent) when cohort mode is enabled
            s = summarise_logs(
                logs, lock_in_threshold=lock_in_threshold, persistence_window=persistence_window
            )
            row: dict[str, Any] = {
                "seed": seed,
                "steps": steps,
                "top_k": top_k,
                "rank_alpha": rank_alpha,
                "drift_alpha": drift_alpha,
                "lock_in_threshold": lock_in_threshold,
                "persistence_window": persistence_window,
                **s,
            }
            if include_agent_id:
                row = {"agent_id": agent_id, **row}
            rows.append(row)

        write_run_log_csv(
            seed_dir / "run_log.csv",
            combined_logs,
            include_viewpoint=drift_active,
            include_agent_id=include_agent_id,
        )

    write_summary_csv(artefacts.root_dir / "summary.csv", rows)
    return artefacts.root_dir
