from __future__ import annotations

from copy import deepcopy
from typing import Any


def apply_runtime_overrides(
    cfg: dict[str, Any],
    *,
    steps: int | None = None,
    seeds: list[int] | None = None,
    policy_mode: str | None = None,
    llm_base_url: str | None = None,
    llm_model: str | None = None,
    prompt_id: str | None = None,
    llm_rerank_slate: bool | None = None,
    separate_rng_streams: bool | None = None,
    top_k_grid: list[int] | None = None,
    rank_alpha_grid: list[float] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Apply explicit CLI overrides and return both config and an audit record."""
    resolved = deepcopy(cfg)
    overrides: dict[str, Any] = {}

    if steps is not None:
        resolved["steps"] = int(steps)
        overrides["steps"] = int(steps)
    if seeds is not None:
        resolved["seeds"] = [int(seed) for seed in seeds]
        overrides["seeds"] = resolved["seeds"]
    if separate_rng_streams is not None:
        resolved["separate_rng_streams"] = bool(separate_rng_streams)
        overrides["separate_rng_streams"] = bool(separate_rng_streams)
    if top_k_grid is not None:
        resolved["top_k_grid"] = [int(value) for value in top_k_grid]
        overrides["top_k_grid"] = resolved["top_k_grid"]
    if rank_alpha_grid is not None:
        resolved["rank_alpha_grid"] = [float(value) for value in rank_alpha_grid]
        overrides["rank_alpha_grid"] = resolved["rank_alpha_grid"]

    llm_override_requested = any(
        value is not None
        for value in (llm_base_url, llm_model, prompt_id, llm_rerank_slate)
    )
    if policy_mode is not None or llm_override_requested:
        policy = dict(resolved.get("policy", {}) or {})
        if policy_mode is not None:
            policy["mode"] = policy_mode
            overrides["policy.mode"] = policy_mode
        if llm_override_requested:
            llm = dict(policy.get("llm", {}) or {})
            for key, value in (
                ("base_url", llm_base_url),
                ("model", llm_model),
                ("prompt_id", prompt_id),
                ("rerank_slate", llm_rerank_slate),
            ):
                if value is not None:
                    llm[key] = value
                    overrides[f"policy.llm.{key}"] = value
            policy["llm"] = llm
        resolved["policy"] = policy

    return resolved, overrides
