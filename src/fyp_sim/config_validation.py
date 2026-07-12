from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from fyp_sim.llm.prompting import load_prompt_template
from fyp_sim.models import ConfigValidationError
from fyp_sim.policy import INTEREST_BRACKET_THRESHOLD, SAMPLER_WATCH_THRESHOLD

RunnerKind = Literal["batch", "compare", "sweep"]

# An initial affinity at or above these values puts a user inside the
# platform's predicted-Watch region from step 0 (see fyp_sim.policy.predicted_action).
_WATCH_THRESHOLDS = {
    "watcher": INTEREST_BRACKET_THRESHOLD,
    "sampler": SAMPLER_WATCH_THRESHOLD,
    "avoider": INTEREST_BRACKET_THRESHOLD,
}

_COMMON_KEYS = {
    "scenario",
    "policy",
    "steps",
    "engine",
    "lock_in_threshold",
    "persistence_window",
    "seeds",
    "seed",
    "user",
    "n_agents",
    "agents",
    "corpus",
    "video_pool",
    "drift_alpha",
    "viewpoint_drift_rate",
    "enable_interest_updates",
    "interest_topic_alpha",
    "interest_tag_alpha",
    "interest_decay",
    "interest_normalise",
    "interest_prune_below",
    "enable_viewpoint_drift",
    "separate_rng_streams",
}
_POLICY_KEYS = {"mode", "curiosity", "llm"}
_LLM_KEYS = {
    "base_url",
    "model",
    "prompt_id",
    "timeout_s",
    "temperature",
    "max_tokens",
    "rerank_slate",
    "api_key",
}
_USER_KEYS = {"phenotype", "viewpoint_score", "sentiment_threshold", "interest_vector"}
_AGENT_KEYS = {"agent_id", *_USER_KEYS}
_CORPUS_KEYS = {"source", "n_videos", "seed", "generator"}
_GENERATOR_KEYS = {"topic", "sentiment", "viewpoint", "duration", "tags"}


@dataclass(frozen=True, slots=True)
class ConfigAudit:
    warnings: tuple[str, ...] = ()


def _fail(message: str) -> None:
    raise ConfigValidationError(message)


def _unknown_keys(obj: dict[str, Any], allowed: set[str], path: str) -> None:
    unknown = sorted(set(obj) - allowed)
    if unknown:
        _fail(f"{path} contains unsupported field(s): {', '.join(unknown)}")


def _object(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        _fail(f"{path} must be a JSON object")
    return value


def _finite_number(value: Any, path: str) -> float:
    if isinstance(value, bool):
        _fail(f"{path} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError):
        _fail(f"{path} must be a finite number")
    if not math.isfinite(number):
        _fail(f"{path} must be a finite number")
    return number


def _bounded(value: Any, path: str, lower: float, upper: float) -> float:
    number = _finite_number(value, path)
    if not lower <= number <= upper:
        _fail(f"{path} must be within [{lower}, {upper}], got {number}")
    return number


def _positive_int(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        _fail(f"{path} must be a positive integer")
    return value


def _boolean_if_present(cfg: dict[str, Any], key: str) -> None:
    if key in cfg and not isinstance(cfg[key], bool):
        _fail(f"{key} must be a boolean")


def _validate_user(user: Any, path: str, *, agent: bool = False) -> None:
    user_obj = _object(user, path)
    _unknown_keys(user_obj, _AGENT_KEYS if agent else _USER_KEYS, path)
    if user_obj.get("phenotype") not in {"watcher", "sampler", "avoider"}:
        _fail(f"{path}.phenotype must be watcher, sampler, or avoider")
    _bounded(user_obj.get("viewpoint_score"), f"{path}.viewpoint_score", 0.0, 1.0)
    _bounded(user_obj.get("sentiment_threshold"), f"{path}.sentiment_threshold", -1.0, 1.0)
    interests = _object(user_obj.get("interest_vector"), f"{path}.interest_vector")
    if not interests:
        _fail(f"{path}.interest_vector must not be empty")
    for key, value in interests.items():
        if not isinstance(key, str) or not key:
            _fail(f"{path}.interest_vector keys must be non-empty strings")
        _bounded(value, f"{path}.interest_vector[{key!r}]", 0.0, 1.0)


def _watch_saturation_warning(user_obj: dict[str, Any], path: str) -> str | None:
    """Flag users who already sit inside the predicted-Watch region before any feedback.

    Under exploit-only exposure (no exploration), a top-k ranker can then fill
    every slate with videos the platform predicts will be watched, so serving
    saturates by construction (risk-01 audit finding). For samplers and
    avoiders the realised actions saturate too, because their behaviour matches
    the prediction. Watchers can still refuse tag-hooked videos whose topic is
    outside their established taste, since that rule is hidden from the ranker,
    so for them this flags saturated serving rather than guaranteed
    Watch-only behaviour.
    """
    phenotype = str(user_obj["phenotype"])
    threshold = _WATCH_THRESHOLDS[phenotype]
    top_affinity = max(float(v) for v in user_obj["interest_vector"].values())
    if top_affinity < threshold:
        return None
    if phenotype == "watcher":
        behaviour_note = (
            "realised watcher actions can still mix because the taste rule in "
            "policy.decide_action is hidden from the ranker's predicted_action"
        )
    else:
        behaviour_note = (
            "report the heuristic arm of this condition as an exploit-only baseline, "
            "not a mixed-behaviour user model"
        )
    return (
        f"{path} starts inside the platform's predicted-Watch region (max initial affinity "
        f"{top_affinity:.2f} >= {phenotype} watch threshold {threshold:.2f}) and no exploration "
        "is active, so exploit-only top-k exposure can fill every slate with predicted-Watch "
        f"videos; {behaviour_note}"
    )


def _validate_weights(value: Any, path: str, *, score_range: tuple[float, float] | None = None) -> None:
    weights = _object(value, path)
    if not weights:
        _fail(f"{path} must not be empty")
    total = 0.0
    for key, value in weights.items():
        weight = _finite_number(value, f"{path}[{key!r}]")
        if weight < 0.0:
            _fail(f"{path}[{key!r}] must be non-negative")
        total += weight
        if score_range is not None:
            _bounded(key, f"{path} key {key!r}", *score_range)
    if total <= 0.0:
        _fail(f"{path} must contain at least one positive weight")


def _validate_generator(generator: Any) -> None:
    gen = _object(generator, "corpus.generator")
    _unknown_keys(gen, _GENERATOR_KEYS, "corpus.generator")

    topic = _object(gen.get("topic"), "corpus.generator.topic")
    _unknown_keys(topic, {"weights"}, "corpus.generator.topic")
    _validate_weights(topic.get("weights"), "corpus.generator.topic.weights")

    sentiment = _object(gen.get("sentiment"), "corpus.generator.sentiment")
    _unknown_keys(sentiment, {"weights"}, "corpus.generator.sentiment")
    _validate_weights(
        sentiment.get("weights"),
        "corpus.generator.sentiment.weights",
        score_range=(-1.0, 1.0),
    )

    viewpoint = _object(gen.get("viewpoint"), "corpus.generator.viewpoint")
    _unknown_keys(viewpoint, {"weights"}, "corpus.generator.viewpoint")
    _validate_weights(
        viewpoint.get("weights"),
        "corpus.generator.viewpoint.weights",
        score_range=(0.0, 1.0),
    )

    duration = _object(gen.get("duration"), "corpus.generator.duration")
    _unknown_keys(duration, {"dist", "params", "min", "max"}, "corpus.generator.duration")
    if duration.get("dist") not in {"uniform", "normal", "lognormal", "choice"}:
        _fail("corpus.generator.duration.dist is unsupported")
    _object(duration.get("params", {}), "corpus.generator.duration.params")
    duration_min = _finite_number(duration.get("min"), "corpus.generator.duration.min")
    duration_max = _finite_number(duration.get("max"), "corpus.generator.duration.max")
    if duration_min < 0 or duration_max < duration_min:
        _fail("corpus.generator.duration requires 0 <= min <= max")

    tags = _object(gen.get("tags"), "corpus.generator.tags")
    _unknown_keys(tags, {"vocab", "per_video"}, "corpus.generator.tags")
    vocab = tags.get("vocab")
    if not isinstance(vocab, list) or not all(isinstance(tag, str) and tag for tag in vocab):
        _fail("corpus.generator.tags.vocab must be a list of non-empty strings")
    if len(set(vocab)) != len(vocab):
        _fail("corpus.generator.tags.vocab must not contain duplicates")
    per_video = _object(tags.get("per_video"), "corpus.generator.tags.per_video")
    _unknown_keys(per_video, {"min", "max"}, "corpus.generator.tags.per_video")
    minimum = per_video.get("min")
    maximum = per_video.get("max")
    if not isinstance(minimum, int) or not isinstance(maximum, int):
        _fail("corpus.generator.tags.per_video min/max must be integers")
    if minimum < 0 or maximum < minimum or maximum > len(vocab):
        _fail("corpus.generator.tags.per_video requires 0 <= min <= max <= len(vocab)")


def _validate_corpus(cfg: dict[str, Any]) -> int:
    corpus = _object(cfg.get("corpus", {}), "corpus")
    _unknown_keys(corpus, _CORPUS_KEYS, "corpus")
    source = corpus.get("source", "file")
    if source == "generated":
        n_videos = _positive_int(corpus.get("n_videos"), "corpus.n_videos")
        if not isinstance(corpus.get("seed", 42), int):
            _fail("corpus.seed must be an integer")
        _validate_generator(corpus.get("generator"))
        return n_videos
    if source == "file":
        videos = cfg.get("video_pool")
        if not isinstance(videos, list) or not videos:
            _fail("video_pool must be a non-empty list when corpus.source='file'")
        return len(videos)
    _fail(f"corpus.source is unsupported: {source!r}")


def validate_experiment_config(
    cfg: dict[str, Any],
    *,
    runner: RunnerKind,
    cfg_path: Path | None = None,
) -> ConfigAudit:
    if not isinstance(cfg, dict):
        _fail("Experiment config must be a JSON object")

    runner_keys = {"top_k_grid", "rank_alpha_grid"} if runner == "sweep" else {"top_k", "rank_alpha"}
    _unknown_keys(cfg, _COMMON_KEYS | runner_keys, str(cfg_path or "config"))
    warnings: list[str] = []

    steps = _positive_int(cfg.get("steps"), "steps")
    persistence_window = _positive_int(cfg.get("persistence_window"), "persistence_window")
    if persistence_window > steps:
        warnings.append("persistence_window exceeds steps; this run cannot observe lock-in")
    _bounded(cfg.get("lock_in_threshold"), "lock_in_threshold", 0.0, 1.0)

    seeds = cfg.get("seeds", [cfg.get("seed", 0)])
    if not isinstance(seeds, list) or not seeds or not all(type(seed) is int for seed in seeds):
        _fail("seeds must be a non-empty list of integers")
    if len(set(seeds)) != len(seeds):
        _fail("seeds must not contain duplicates")

    engine = cfg.get("engine", "baseline")
    if engine not in {"baseline", "opt"}:
        _fail("engine must be 'baseline' or 'opt'")
    if runner == "compare" and engine != "baseline":
        _fail("run_compare currently requires engine='baseline'")

    for key in ("enable_interest_updates", "interest_normalise", "enable_viewpoint_drift", "separate_rng_streams"):
        _boolean_if_present(cfg, key)
    for key in ("interest_topic_alpha", "interest_tag_alpha", "interest_decay", "interest_prune_below"):
        if key in cfg and _finite_number(cfg[key], key) < 0.0:
            _fail(f"{key} must be non-negative")

    drift_alpha = _finite_number(
        cfg.get("drift_alpha", cfg.get("viewpoint_drift_rate", 0.0)), "drift_alpha"
    )
    if drift_alpha < 0.0:
        _fail("drift_alpha must be non-negative")
    if "drift_alpha" in cfg and "viewpoint_drift_rate" in cfg:
        alias = _finite_number(cfg["viewpoint_drift_rate"], "viewpoint_drift_rate")
        if alias != drift_alpha:
            _fail("drift_alpha and viewpoint_drift_rate conflict")

    policy = _object(cfg.get("policy", {}), "policy")
    _unknown_keys(policy, _POLICY_KEYS, "policy")
    mode = str(policy.get("mode", "heuristic")).strip().lower()
    if mode not in {"heuristic", "llm"}:
        _fail("policy.mode must be 'heuristic' or 'llm'")
    curiosity = _bounded(policy.get("curiosity", 0.0), "policy.curiosity", 0.0, 1.0)

    llm = policy.get("llm")
    if llm is not None:
        llm_obj = _object(llm, "policy.llm")
        _unknown_keys(llm_obj, _LLM_KEYS, "policy.llm")
        if not isinstance(llm_obj.get("model"), str) or not llm_obj["model"].strip():
            _fail("policy.llm.model must be a non-empty string")
        prompt_id = llm_obj.get("prompt_id", "decision_v1")
        if not isinstance(prompt_id, str) or not prompt_id:
            _fail("policy.llm.prompt_id must be a non-empty string")
        load_prompt_template(prompt_id)
        if _finite_number(llm_obj.get("timeout_s", 10.0), "policy.llm.timeout_s") <= 0:
            _fail("policy.llm.timeout_s must be positive")
        if _finite_number(llm_obj.get("temperature", 0.0), "policy.llm.temperature") < 0:
            _fail("policy.llm.temperature must be non-negative")
        if "max_tokens" in llm_obj:
            _positive_int(llm_obj["max_tokens"], "policy.llm.max_tokens")
        if "rerank_slate" in llm_obj and not isinstance(llm_obj["rerank_slate"], bool):
            _fail("policy.llm.rerank_slate must be a boolean")
        if llm_obj.get("rerank_slate", False) and curiosity > 0.0:
            _fail("policy.curiosity is incompatible with policy.llm.rerank_slate")
        if mode == "heuristic":
            warnings.append("policy.llm is dormant because policy.mode='heuristic'")
    elif mode == "llm" or runner == "compare":
        _fail("policy.llm is required for LLM or compare execution")

    if runner == "compare" and cfg.get("separate_rng_streams") is not True:
        _fail("run_compare requires separate_rng_streams=true for fair paired execution")

    if "agents" in cfg:
        agents = cfg["agents"]
        if not isinstance(agents, list) or not agents:
            _fail("agents must be a non-empty list")
        for index, agent in enumerate(agents):
            _validate_user(agent, f"agents[{index}]", agent=True)
        if "n_agents" in cfg and cfg["n_agents"] != len(agents):
            _fail("n_agents must equal len(agents)")
    else:
        _validate_user(cfg.get("user"), "user")

    # The heuristic policy decides actions in heuristic mode and in the
    # heuristic arm of every compare run. Exploration only offsets the
    # saturation risk where the runner actually applies policy.curiosity,
    # which run_compare never does.
    heuristic_decides = mode == "heuristic" or runner == "compare"
    exploration_active = curiosity > 0.0 and runner != "compare"
    if heuristic_decides and not exploration_active:
        users = (
            [(f"agents[{index}]", agent) for index, agent in enumerate(cfg["agents"])]
            if "agents" in cfg
            else [("user", cfg["user"])]
        )
        for path, user_obj in users:
            message = _watch_saturation_warning(user_obj, path)
            if message:
                warnings.append(message)

    n_videos = _validate_corpus(cfg)
    if runner == "sweep":
        top_k_grid = cfg.get("top_k_grid")
        rank_alpha_grid = cfg.get("rank_alpha_grid")
        if not isinstance(top_k_grid, list) or not top_k_grid:
            _fail("top_k_grid must be a non-empty list")
        if not isinstance(rank_alpha_grid, list) or not rank_alpha_grid:
            _fail("rank_alpha_grid must be a non-empty list")
        for index, top_k in enumerate(top_k_grid):
            value = _positive_int(top_k, f"top_k_grid[{index}]")
            if value > n_videos:
                _fail(f"top_k_grid[{index}] exceeds corpus size")
        for index, rank_alpha in enumerate(rank_alpha_grid):
            _bounded(rank_alpha, f"rank_alpha_grid[{index}]", 0.0, 1.0)
    else:
        top_k = _positive_int(cfg.get("top_k"), "top_k")
        if top_k > n_videos:
            _fail("top_k must not exceed corpus size")
        _bounded(cfg.get("rank_alpha"), "rank_alpha", 0.0, 1.0)

    if not cfg.get("enable_interest_updates", False):
        dormant = sorted(key for key in cfg if key.startswith("interest_") and key != "interest_normalise")
        if dormant:
            warnings.append("interest update parameters are dormant because updates are disabled")
    if not cfg.get("enable_viewpoint_drift", False) and drift_alpha > 0.0:
        warnings.append("drift_alpha is dormant because viewpoint drift is disabled")
    if "drift_alpha" in cfg and "viewpoint_drift_rate" in cfg:
        warnings.append("viewpoint_drift_rate is a redundant legacy alias; drift_alpha takes precedence")

    return ConfigAudit(warnings=tuple(warnings))
