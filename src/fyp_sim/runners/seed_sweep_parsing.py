from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from fyp_sim.models import User, UserPhenotype


def build_user(cfg: dict[str, Any]) -> User:
    """Legacy user builder"""
    user_cfg = cfg.get("user")
    if not isinstance(user_cfg, dict):
        raise ValueError("user must be an object/dict")
    return build_user_from_dict(user_cfg)


def phenotype_from_str(s: str) -> UserPhenotype:
    s = s.strip().lower()
    if s == "watcher":
        return UserPhenotype.WATCHER
    if s == "sampler":
        return UserPhenotype.SAMPLER
    if s == "avoider":
        return UserPhenotype.AVOIDER
    raise ValueError(f"Unknown phenotype: {s!r} (expected watcher/sampler/avoider)")


def build_user_from_dict(u: dict[str, Any]) -> User:
    """Build a User from a config dict"""
    if not isinstance(u, dict):
        raise ValueError(f"User config must be an object/dict, got {type(u)}.__name__")

    try:
        phenotype = phenotype_from_str(u["phenotype"])
        viewpoint = float(u["viewpoint_score"])
        sentiment_threshold = float(u["sentiment_threshold"])
    except KeyError as e:
        raise ValueError(f"User config muissing required key: {e.args[0]!r}") from e

    iv_raw = u.get("interest_vector", {}) or {}
    if not isinstance(iv_raw, dict):
        raise ValueError("user.interest_vector must be an object/dict")

    interest_vector = {str(k): float(v) for k, v in iv_raw.items()}

    return User(
        phenotype=phenotype,
        viewpoint_score=viewpoint,
        sentiment_threshold=sentiment_threshold,
        interest_vector=interest_vector,
    )


def extract_seeds(cfg: dict[str, Any]) -> list[int]:
    seeds_raw = cfg.get("seeds")
    if isinstance(seeds_raw, list) and all(isinstance(x, int) for x in seeds_raw):
        return [int(x) for x in seeds_raw]

    seed_raw = cfg.get("seed")
    if isinstance(seed_raw, int):
        return [int(seed_raw)]

    return [0]


def scenario_from(cfg: dict[str, Any], cfg_path: Path | None) -> str:
    s = cfg.get("scenario")
    if isinstance(s, str) and s.strip():
        return s.strip()
    if cfg_path is not None:
        return cfg_path.stem
    return "experiment_baseline"


def scenario_to_mode(scenario: str) -> str:
    scenario = scenario.strip() or "experiment_baseline"
    return scenario.removeprefix("experiment_") if scenario.startswith("experiment_") else scenario


def policy_mode(cfg: dict[str, Any]) -> str:
    policy = cfg.get("policy", {}) or {}
    return str(policy.get("mode", "heuristic")).strip().lower()


def policy_curiosity(cfg: dict[str, Any]) -> float:
    """Probability of exploring a random video instead of ranking-based exposure."""
    policy = cfg.get("policy", {}) or {}
    c = float(policy.get("curiosity", 0.0))
    if c < 0.0 or c > 1.0:
        raise ValueError("policy.curiosity must be within [0.0, 1.0]")
    return c


def derive_agent_seed(seed: int, agent_id: str) -> int:
    """Stable derived seed so agent order can't affect randomness."""
    h = hashlib.sha256(f"{seed}:{agent_id}".encode()).digest()
    return int.from_bytes(h[:8], "big", signed=False)


def parse_agents(cfg: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """Return list of (agent_id, agent_user_dict)

    NOTE: Backwards compatabile with single agent mode
        - If 'agents' is missing, treat as single-agent config using 'user'.
        - If n_agents > 1, 'agents' must be provided
    """
    n_agents = int(cfg.get("n_agents", 1))
    agents_raw = cfg.get("agents", None)

    # Single-agent mode
    if agents_raw is None:
        if n_agents != 1:
            raise ValueError("If n_agents > 1 you must provide an 'agents' list.")
        user = cfg.get("user")
        if not isinstance(user, dict):
            raise ValueError("Single-agent configs must include a 'user' object.")
        return [("agent0", user)]

    # Cohort mode
    if not isinstance(agents_raw, list) or not agents_raw:
        raise ValueError("'agents' must be a non-empty list when provided.")

    agents: list[tuple[str, dict[str, Any]]] = []
    seen: set[str] = set()

    for i, a in enumerate(agents_raw):
        if not isinstance(a, dict):
            raise ValueError(f"agents[{i}] must be an object/dict.")
        agent_id = str(a.get("agent_id") or f"agent{i}").strip()
        if not agent_id:
            raise ValueError(f"agents[{i}].agent_id must be a non-empty string.")
        if agent_id in seen:
            raise ValueError(f"Duplicate agent_id: {agent_id!r}")
        seen.add(agent_id)
        agents.append((agent_id, a))

    # Only enforce if user explicitly provided n_agents
    if "n_agents" in cfg and n_agents != len(agents_raw):
        raise ValueError(f"n_agents={n_agents} but agents has len={len(agents_raw)}")

    return agents
