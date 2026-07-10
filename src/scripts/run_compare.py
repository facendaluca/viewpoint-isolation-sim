from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import random
import time
from pathlib import Path
from typing import Any

from fyp_sim.agents import (
    HeuristicDecider,
    LLMDecider,
    llm_diagnostics_delta,
    llm_diagnostics_snapshot,
)
from fyp_sim.agents.clients import OpenAICompatClient
from fyp_sim.analysis import summarise_logs
from fyp_sim.artefacts import _fail_fast_old_alpha, create_run_artefacts
from fyp_sim.cli import run_cli
from fyp_sim.config_validation import validate_experiment_config
from fyp_sim.corpus import build_corpus
from fyp_sim.models import User, UserPhenotype, Video
from fyp_sim.plotting import make_compare_plot
from fyp_sim.simulation.engine import run_simulation

# -------------------------------
# Config -> domain objects
# -------------------------------


def phenotype_from_str(s: str) -> UserPhenotype:
    s = s.strip().lower()
    if s == "watcher":
        return UserPhenotype.WATCHER
    if s == "sampler":
        return UserPhenotype.SAMPLER
    if s == "avoider":
        return UserPhenotype.AVOIDER
    raise ValueError(f"Unknown phenotype: {s} (expected watcher/sampler/avoider)")


def load_config(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_user(cfg: dict[str, Any]) -> User:
    u = cfg["user"]
    return User(
        phenotype=phenotype_from_str(u["phenotype"]),
        viewpoint_score=float(u["viewpoint_score"]),
        interest_vector={k: float(v) for k, v in u["interest_vector"].items()},
        sentiment_threshold=float(u["sentiment_threshold"]),
    )


def build_video_pool(cfg: dict[str, Any]) -> list[Video]:
    pool: list[Video] = []
    for v in cfg["video_pool"]:
        tags = tuple(v.get("tags", []))
        pool.append(
            Video(
                int(v["video_id"]),
                str(v["topic_category"]),
                float(v["viewpoint_score"]),
                float(v["sentiment_score"]),
                int(v["duration_s"]),
                tags=tags,
            )
        )
    return pool


# -------------------------------
# Deterministic run id
# -------------------------------


def _deepcopy_jsonable(obj: Any) -> Any:
    # safe for dict/list primitives coming from json.load
    return json.loads(json.dumps(obj))


def _normalise_cfg_for_hash(cfg: dict[str, Any]) -> dict[str, Any]:
    """
    Make the hash stable and meaningful for "same conditions:
    - policy.mode doesn't matter because the script always runs both
    - api_key should not affect hashes or be written to artifacts
    """
    c = _deepcopy_jsonable(cfg)
    policy = c.get("policy", {}) or {}
    policy["mode"] = "compare"
    llm = policy.get("llm", {}) or {}
    if "api_key" in llm:
        llm["api_key"] = None
    policy["llm"] = llm
    c["policy"] = policy
    return c


def config_hash(cfg: dict[str, Any], *, n: int = 10) -> str:
    c = _normalise_cfg_for_hash(cfg)
    payload = json.dumps(c, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
        "utf-8"
    )
    return hashlib.sha256(payload).hexdigest()[:n]


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)


# -------------------------------
# Logging / artifacts
# -------------------------------

_LOG_HEADERS = [
    "t",
    "video_id",
    "action",
    "watch_time_s",
    "interest",
    "vii_t",
    "vii_cum",
    "topic_interest",
    "interest_pre",
    "interest_post",
    "topic_interest_pre",
    "topic_interest_post",
    "interest_state_hash_pre",
    "interest_state_hash_post",
    "interest_keys",
    "user_viewpoint_pre",
    "user_viewpoint_post",
    "video_viewpoint_score",
    # optional LLM/policy metadata
    "policy_mode",
    "llm_prompt_id",
    "llm_valid",
    "llm_fallback_reason",
    "llm_action",
    "llm_confidence",
    "llm_prompt_tokens",
    "llm_completion_tokens",
    "llm_total_tokens",
    "llm_token_count_estimated",
]


def write_step_logs_csv(path: Path, logs: list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=_LOG_HEADERS)
        w.writeheader()
        for r in logs:
            policy_mode = str(getattr(r, "policy_mode", "") or "").strip().lower()
            is_llm = policy_mode == "llm"
            row = {
                "t": getattr(r, "t", ""),
                "video_id": getattr(r, "video_id", ""),
                "action": getattr(r, "action", ""),
                "watch_time_s": getattr(r, "watch_time_s", ""),
                "interest": _fmt_float(getattr(r, "interest", None)),
                "vii_t": _fmt_float(getattr(r, "vii_t", None)),
                "vii_cum": _fmt_float(getattr(r, "vii_cum", None)),
                "topic_interest": _fmt_float(getattr(r, "topic_interest", None)),
                "interest_pre": _fmt_float(getattr(r, "interest_pre", None)),
                "interest_post": _fmt_float(getattr(r, "interest_post", None)),
                "topic_interest_pre": _fmt_float(getattr(r, "topic_interest_pre", None)),
                "topic_interest_post": _fmt_float(getattr(r, "topic_interest_post", None)),
                "interest_state_hash_pre": getattr(r, "interest_state_hash_pre", ""),
                "interest_state_hash_post": getattr(r, "interest_state_hash_post", ""),
                "interest_keys": getattr(r, "interest_keys", ""),
                "user_viewpoint_pre": _fmt_float(getattr(r, "user_viewpoint_pre", None)),
                "user_viewpoint_post": _fmt_float(getattr(r, "user_viewpoint_post", None)),
                "video_viewpoint_score": _fmt_float(
                    getattr(r, "video_viewpoint_score", None)
                ),
                "policy_mode": getattr(r, "policy_mode", ""),
                # LLM-only columns: blank unless policy_mode == "llm"
                "llm_prompt_id": getattr(r, "llm_prompt_id", "") if is_llm else "",
                "llm_valid": getattr(r, "llm_valid", "") if is_llm else "",
                "llm_fallback_reason": getattr(r, "llm_fallback_reason", "") if is_llm else "",
                "llm_action": getattr(r, "llm_action", "") if is_llm else "",
                "llm_confidence": getattr(r, "llm_confidence", None) if is_llm else "",
                "llm_prompt_tokens": getattr(r, "llm_prompt_tokens", 0) if is_llm else "",
                "llm_completion_tokens": (
                    getattr(r, "llm_completion_tokens", 0) if is_llm else ""
                ),
                "llm_total_tokens": getattr(r, "llm_total_tokens", 0) if is_llm else "",
                "llm_token_count_estimated": (
                    getattr(r, "llm_token_count_estimated", False) if is_llm else ""
                ),
            }
            w.writerow(row)


def _fmt_float(x: float | None) -> str:
    if x is None or x == "":
        return ""
    try:
        return f"{float(x):.4f}"
    except (ValueError, TypeError):
        return str(x)


def compare_seed_logs(heuristic_logs: list[Any], llm_logs: list[Any]) -> dict[str, int | float]:
    paired = list(zip(heuristic_logs, llm_logs, strict=False))
    aligned_steps = len(paired)
    action_difference_steps = sum(h.action != llm.action for h, llm in paired)
    same_video_steps = sum(h.video_id == llm.video_id for h, llm in paired)
    same_video_action_difference_steps = sum(
        h.video_id == llm.video_id and h.action != llm.action for h, llm in paired
    )
    same_context_steps = sum(
        h.video_id == llm.video_id
        and h.interest_state_hash_pre == llm.interest_state_hash_pre
        and h.user_viewpoint_pre == llm.user_viewpoint_pre
        for h, llm in paired
    )
    same_context_action_difference_steps = sum(
        h.video_id == llm.video_id
        and h.interest_state_hash_pre == llm.interest_state_hash_pre
        and h.user_viewpoint_pre == llm.user_viewpoint_pre
        and h.action != llm.action
        for h, llm in paired
    )
    return {
        "aligned_steps": aligned_steps,
        "action_difference_steps": action_difference_steps,
        "action_difference_rate": action_difference_steps / aligned_steps if aligned_steps else 0.0,
        "same_video_steps": same_video_steps,
        "same_video_rate": same_video_steps / aligned_steps if aligned_steps else 0.0,
        "same_video_action_difference_steps": same_video_action_difference_steps,
        "same_context_steps": same_context_steps,
        "same_context_action_difference_steps": same_context_action_difference_steps,
    }


def _run_simulation_compat(
    *,
    user: User,
    video_pool: list[Video],
    steps: int,
    rng: random.Random,
    top_k: int,
    rank_alpha: float,
    drift_alpha: float,
    enable_viewpoint_drift: bool,
    decider: Any,
    engagement_rng: random.Random | None = None,
    llm_rerank: bool = False,
    interest_kwargs: dict[str, Any] | None = None,
) -> list[Any]:
    """
    Calls run_simulation and returns only the logs.
    """
    sig = inspect.signature(run_simulation)
    kwargs: dict[str, Any] = dict(
        user=user,
        video_pool=video_pool,
        steps=steps,
        rng=rng,
        top_k=top_k,
        rank_alpha=rank_alpha,
        drift_alpha=drift_alpha,
        enable_viewpoint_drift=enable_viewpoint_drift,
    )
    if "decider" in sig.parameters:
        kwargs["decider"] = decider
    if "engagement_rng" in sig.parameters:
        kwargs["engagement_rng"] = engagement_rng
    if "llm_rerank" in sig.parameters:
        kwargs["llm_rerank"] = llm_rerank
    if interest_kwargs and "enable_interest_updates" in sig.parameters:
        kwargs.update(interest_kwargs)
    return run_simulation(**kwargs)


# -------------------------------
# Decider builders
# -------------------------------


def build_llm_decider(cfg: dict[str, Any]) -> Any:
    policy = cfg.get("policy", {}) or {}
    llm_cfg = policy.get("llm", {}) or {}

    model = llm_cfg.get("model")
    if not model:
        raise ValueError("policy.llm.model is required to run the LLM baseline")

    client = OpenAICompatClient(
        base_url=str(llm_cfg.get("base_url", "http://100.127.102.30:1234/v1")),
        model=str(model),
        api_key=llm_cfg.get("api_key"),
        temperature=float(llm_cfg.get("temperature", 0.0)),
        max_tokens=llm_cfg.get("max_tokens"),
    )

    # only pass kwargs that exist in the LLMDecider constructor
    llm_kwargs: dict[str, Any] = {
        "prompt_id": str(llm_cfg.get("prompt_id", "decision_v1")),
        "client": client,
    }

    sig = inspect.signature(LLMDecider)
    if "timeout_s" in sig.parameters:
        llm_kwargs["timeout_s"] = float(llm_cfg.get("timeout_s", 10.0))
    if "fallback" in sig.parameters:
        llm_kwargs["fallback"] = HeuristicDecider()

    return LLMDecider(**llm_kwargs)

    # -------------------------------
    # Main
    # -------------------------------


def main() -> None:
    p = argparse.ArgumentParser(
        description="Run a fair Heuristic vs LLM baseline (same seeds, same corpus)."
    )
    p.add_argument("--config", type=Path, default=Path("configs/experiment_baseline.json"))
    p.add_argument("--out", type=Path, default=Path("outputs/compare"))

    args = p.parse_args()

    cfg = load_config(args.config)
    _fail_fast_old_alpha(cfg, args.config)
    config_audit = validate_experiment_config(cfg, runner="compare", cfg_path=args.config)
    for warning in config_audit.warnings:
        print(f"[config warning] {warning}")

    steps = int(cfg["steps"])
    top_k = int(cfg["top_k"])
    rank_alpha = float(cfg["rank_alpha"])
    drift_alpha = float(cfg.get("drift_alpha", cfg.get("viewpoint_drift_rate", 0.0)))
    enable_viewpoint_drift = bool(cfg.get("enable_viewpoint_drift", False))
    lock_in_threshold = float(cfg["lock_in_threshold"])
    persistence_window = int(cfg["persistence_window"])
    seeds = [int(x) for x in cfg["seeds"]]
    separate_rng_streams = bool(cfg["separate_rng_streams"])

    # Interest/state updates: honoured from config (same keys as run_batch),
    # so LLM actions can shape future recommendations when enabled.
    enable_interest_updates = bool(cfg.get("enable_interest_updates", False))
    interest_kwargs = {
        "enable_interest_updates": enable_interest_updates,
        "interest_topic_alpha": float(cfg.get("interest_topic_alpha", 0.10)),
        "interest_tag_alpha": float(cfg.get("interest_tag_alpha", 0.05)),
        "interest_decay": float(cfg.get("interest_decay", 0.02)),
        "interest_normalise": bool(cfg.get("interest_normalise", False)),
        "interest_prune_below": float(cfg.get("interest_prune_below", 0.001)),
    }

    # LLM-in-loop: rerank the top_k slate with the LLM (LLM arm only).
    llm_cfg = (cfg.get("policy") or {}).get("llm") or {}
    rerank_slate = bool(llm_cfg.get("rerank_slate", False))

    print(
        f"[run_compare] config={args.config} "
        f"steps={steps} top_k={top_k} rank_alpha={rank_alpha} drift_alpha={drift_alpha} "
        f"enable_viewpoint_drift={enable_viewpoint_drift} "
        f"enable_interest_updates={enable_interest_updates} rerank_slate={rerank_slate} "
        f"separate_rng_streams={separate_rng_streams} "
        f"lock_in_threshold={lock_in_threshold} persistence_window={persistence_window} "
        f"seeds={seeds}"
    )

    if persistence_window <= 0:
        raise ValueError(
            f"Config error: persistence_window must be > 0, got {persistence_window}"
            f"in {args.config}"
        )

    pool = build_corpus(cfg)

    cfg_h = config_hash(cfg, n=10)
    artefacts = create_run_artefacts(
        cfg=_normalise_cfg_for_hash(cfg),
        cfg_path=args.config,
        mode="compare",
        seeds=seeds,
        outputs_root=args.out,
        corpus=pool,
        runner="src.scripts.run_compare",
    )
    run_id = artefacts.run_id
    run_dir = artefacts.root_dir

    deciders: dict[str, Any] = {
        "heuristic": HeuristicDecider(),
        "llm": build_llm_decider(cfg),
    }

    rows: list[dict[str, Any]] = []
    logs_by_agent: dict[str, dict[int, list[Any]]] = {"heuristic": {}, "llm": {}}
    run_started = time.perf_counter()

    for agent_name, decider in deciders.items():
        for seed in seeds:
            rng = random.Random(seed)
            # Separate stream for watch-time draws: otherwise arms that act
            # differently consume different amounts of shared randomness and
            # exposure diverges for accidental reasons.
            engagement_rng = (
                random.Random(f"{seed}:engagement") if separate_rng_streams else None
            )
            # Fresh user per (agent, seed): drift/interest updates mutate state,
            # so a shared instance would leak state across runs and break fairness.
            user = build_user(cfg)
            diagnostics_before = llm_diagnostics_snapshot(decider)
            seed_started = time.perf_counter()
            logs = _run_simulation_compat(
                user=user,
                video_pool=pool,
                steps=steps,
                rng=rng,
                top_k=top_k,
                rank_alpha=rank_alpha,
                drift_alpha=drift_alpha,
                enable_viewpoint_drift=enable_viewpoint_drift,
                decider=decider,
                engagement_rng=engagement_rng,
                llm_rerank=(agent_name == "llm" and rerank_slate),
                interest_kwargs=interest_kwargs,
            )
            seed_runtime_s = time.perf_counter() - seed_started
            diagnostics = llm_diagnostics_delta(
                diagnostics_before, llm_diagnostics_snapshot(decider)
            )
            logs_by_agent[agent_name][seed] = logs

            # Per-run logs (generated artifacts)
            log_path = run_dir / "logs" / agent_name / f"run_seed_{seed}.csv"
            write_step_logs_csv(log_path, logs)

            # Per-run summary (tracked within compare output dir)
            s = summarise_logs(
                logs,
                lock_in_threshold=lock_in_threshold,
                persistence_window=persistence_window,
            )
            rows.append(
                {
                    "run_id": run_id,
                    "config_hash": cfg_h,
                    "agent": agent_name,
                    "seed": seed,
                    "steps": steps,
                    "top_k": top_k,
                    "rank_alpha": rank_alpha,
                    "drift_alpha": drift_alpha,
                    "lock_in_threshold": lock_in_threshold,
                    "persistence_window": persistence_window,
                    "runtime_s": seed_runtime_s,
                    "llm_expected_call_count": (
                        steps * min(top_k, len(pool)) if agent_name == "llm" and rerank_slate else steps
                    )
                    if agent_name == "llm"
                    else 0,
                    **diagnostics,
                    "llm_valid_rate": (
                        diagnostics["llm_valid_count"] / diagnostics["llm_call_count"]
                        if diagnostics["llm_call_count"]
                        else 0.0
                    ),
                    "llm_fallback_rate": (
                        diagnostics["llm_fallback_count"] / diagnostics["llm_call_count"]
                        if diagnostics["llm_call_count"]
                        else 0.0
                    ),
                    **s,
                }
            )

    # Combined summary for easy downstream plotting
    summary_path = run_dir / "summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    llm_rows = [row for row in rows if row["agent"] == "llm"]
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
    llm_totals = {key: sum(int(row[key]) for row in llm_rows) for key in count_keys}
    llm_calls = llm_totals["llm_call_count"]
    llm_diagnostics = {
        **llm_totals,
        "llm_valid_rate": llm_totals["llm_valid_count"] / llm_calls if llm_calls else 0.0,
        "llm_fallback_rate": llm_totals["llm_fallback_count"] / llm_calls if llm_calls else 0.0,
        "llm_prompt_id": str(llm_cfg.get("prompt_id", "decision_v1")),
        "llm_model": str(llm_cfg.get("model", "")),
        "llm_rerank_slate": rerank_slate,
        "token_usage_source": (
            "provider"
            if llm_totals["llm_token_estimated_calls"] == 0
            else "mixed_or_character_estimate"
        ),
    }
    write_json(run_dir / "llm_diagnostics.json", llm_diagnostics)

    per_seed_comparison = []
    for seed in seeds:
        per_seed_comparison.append(
            {
                "seed": seed,
                **compare_seed_logs(logs_by_agent["heuristic"][seed], logs_by_agent["llm"][seed]),
            }
        )
    aggregate_count_keys = [
        "aligned_steps",
        "action_difference_steps",
        "same_video_steps",
        "same_video_action_difference_steps",
        "same_context_steps",
        "same_context_action_difference_steps",
    ]
    aggregate = {
        key: sum(int(row[key]) for row in per_seed_comparison) for key in aggregate_count_keys
    }
    aligned_steps = aggregate["aligned_steps"]
    aggregate["action_difference_rate"] = (
        aggregate["action_difference_steps"] / aligned_steps if aligned_steps else 0.0
    )
    aggregate["same_video_rate"] = (
        aggregate["same_video_steps"] / aligned_steps if aligned_steps else 0.0
    )
    write_json(
        run_dir / "comparison_diagnostics.json",
        {"per_seed": per_seed_comparison, "aggregate": aggregate},
    )

    total_runtime_s = time.perf_counter() - run_started
    manifest = json.loads(artefacts.manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        {
            "config_hash_short": cfg_h,
            "config_warnings": list(config_audit.warnings),
            "runtime_s": total_runtime_s,
            "enable_viewpoint_drift": enable_viewpoint_drift,
            "enable_interest_updates": enable_interest_updates,
            "rerank_slate": rerank_slate,
            "separate_rng_streams": separate_rng_streams,
            "llm_diagnostics_path": "llm_diagnostics.json",
            "comparison_diagnostics_path": "comparison_diagnostics.json",
        }
    )
    write_json(artefacts.manifest_path, manifest)

    print(f"Wrote compare run to: {run_dir}")
    print(f"- logs: {run_dir / 'logs'}")
    print(f"- summary: {summary_path}")

    # auto-generate a compare plot
    plot_path = make_compare_plot(run_dir, seed=seeds[0] if seeds else 0)
    print(f"- compare plot: {plot_path}")


if __name__ == "__main__":
    run_cli(main)
