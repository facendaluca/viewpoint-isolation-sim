from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import random
from pathlib import Path
from typing import Any

from fyp_sim.agents import HeuristicDecider, LLMDecider
from fyp_sim.agents.clients import OpenAICompatClient
from fyp_sim.analysis import summarise_logs
from fyp_sim.artefacts import _fail_fast_old_alpha
from fyp_sim.cli import run_cli
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
    # optional LLM/policy metadata
    "policy_mode",
    "llm_prompt_id",
    "llm_valid",
    "llm_fallback_reason",
    "llm_action",
    "llm_confidence",
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
                "policy_mode": getattr(r, "policy_mode", ""),
                # LLM-only columns: blank unless policy_mode == "llm"
                "llm_prompt_id": getattr(r, "llm_prompt_id", "") if is_llm else "",
                "llm_valid": getattr(r, "llm_valid", "") if is_llm else "",
                "llm_fallback_reason": getattr(r, "llm_fallback_reason", "") if is_llm else "",
                "llm_action": getattr(r, "llm_action", "") if is_llm else "",
                "llm_confidence": getattr(r, "llm_confidence", None) if is_llm else "",
            }
            w.writerow(row)


def _fmt_float(x: float | None) -> str:
    if x is None or x == "":
        return ""
    try:
        return f"{float(x):.4f}"
    except (ValueError, TypeError):
        return str(x)


def _run_simulation_compat(
    *,
    user: User,
    video_pool: list[Video],
    steps: int,
    rng: random.Random,
    top_k: int,
    rank_alpha: float,
    drift_alpha: float,
    decider: Any,
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
    )
    if "decider" in sig.parameters:
        kwargs["decider"] = decider
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
        base_url=str(llm_cfg.get("base_url", "http://localhost:1234/v1")),
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

    steps = int(cfg["steps"])
    top_k = int(cfg["top_k"])
    rank_alpha = float(cfg["rank_alpha"])
    drift_alpha = float(cfg.get("drift_alpha", cfg.get("viewpoint_drift_rate", 0.0)))
    lock_in_threshold = float(cfg["lock_in_threshold"])
    persistence_window = int(cfg["persistence_window"])
    seeds = [int(x) for x in cfg["seeds"]]

    print(
        f"[run_compare] config={args.config} "
        f"steps={steps} top_k={top_k} rank_alpha={rank_alpha} drift_alpha={drift_alpha} "
        f"lock_in_threshold={lock_in_threshold} persistence_window={persistence_window} "
        f"seeds={seeds}"
    )

    if persistence_window <= 0:
        raise ValueError(
            f"Config error: persistence_window must be > 0, got {persistence_window}"
            f"in {args.config}"
        )

    user = build_user(cfg)
    pool = build_video_pool(cfg)

    cfg_h = config_hash(cfg, n=10)
    run_id = f"compare__{cfg_h}"
    run_dir = args.out / run_id

    # Save a snapshot (api_key removed, policy.mode normalised)
    write_json(run_dir / "config.redacted.json", _normalise_cfg_for_hash(cfg))
    write_json(
        run_dir / "manifest.json",
        {
            "run_id": run_id,
            "config_hash": cfg_h,
            "config_path": str(args.config),
            "seeds": seeds,
            "steps": steps,
            "top_k": top_k,
            "rank_alpha": rank_alpha,
            "drift_alpha": drift_alpha,
        },
    )

    deciders: dict[str, Any] = {
        "heuristic": HeuristicDecider(),
        "llm": build_llm_decider(cfg),
    }

    rows: list[dict[str, Any]] = []

    for agent_name, decider in deciders.items():
        for seed in seeds:
            rng = random.Random(seed)
            logs = _run_simulation_compat(
                user=user,
                video_pool=pool,
                steps=steps,
                rng=rng,
                top_k=top_k,
                rank_alpha=rank_alpha,
                drift_alpha=drift_alpha,
                decider=decider,
            )

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

    print(f"Wrote compare run to: {run_dir}")
    print(f"- logs: {run_dir / 'logs'}")
    print(f"- summary: {summary_path}")

    # auto-generate a compare plot
    plot_path = make_compare_plot(run_dir, seed=seeds[0] if seeds else 0)
    print(f"- compare plot: {plot_path}")


if __name__ == "__main__":
    run_cli(main)
