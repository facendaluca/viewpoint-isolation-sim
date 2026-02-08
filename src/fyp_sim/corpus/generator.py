"""
Deterministic, config-driven generator for synthetic Video corpuses.
"""

from __future__ import annotations

import random
from typing import Any

from fyp_sim.models import Video

try:
    from fyp_sim import taxonomy as default_taxonomy
except ImportError:
    default_taxonomy = None


def _get_weighted_choice(rng: random.Random, weights_map: dict[Any, float]) -> Any:
    """Helper to select a key from a map of value->weight."""
    if not weights_map:
        return None

    # Sort for determinism
    ordered_items = sorted(weights_map.items(), key=str)
    choices = [k for k, _ in ordered_items]
    weights = [w for _, w in ordered_items]

    # Handle implicit normalization
    total = sum(weights)
    if total <= 0:
        return rng.choice(choices)

    return rng.choices(choices, weights=weights, k=1)[0]


def _sample_duration(rng: random.Random, cfg: dict[str, Any]) -> int:
    """Sample duration based on distribution config."""
    dist_type = cfg.get("dist", "uniform")
    params = cfg.get("params", {})
    min_val = cfg.get("min", 15)
    max_val = cfg.get("max", 300)

    val = 0.0

    if dist_type == "lognormal":
        mu = params.get("mean", 4.0)  # Default around 55s
        sigma = params.get("sigma", 0.5)
        val = rng.lognormvariate(mu, sigma)
    elif dist_type == "normal":
        mu = params.get("mean", 60.0)
        sigma = params.get("std", 20.0)
        val = rng.gauss(mu, sigma)
    elif dist_type == "choice":
        # Discrete set of durations
        options = params.get("values", [15, 30, 60])
        val = float(rng.choice(options))
    else:
        # Uniform
        val = rng.uniform(min_val, max_val)

    # Clamp
    msg_val = max(min_val, min(max_val, val))
    return int(round(msg_val))


def generate_video_corpus(
    n: int, seed: int, cfg: dict[str, Any], taxonomy: Any = None
) -> list[Video]:
    """
    Generate a deterministic list of N Video objects based on config.

    Args:
        n: Number of videos to generate.
        seed: Random seed for reproducibility.
        cfg: Configuration dict containing generator settings.
        taxonomy: Optional override for taxonomy module.

    Returns:
        List of Video objects.

    Raises:
        ValueError: If n < 0.
    """
    if n < 0:
        raise ValueError(f"n must be non-negative, got {n}")
    if n == 0:
        return []

    # Local RNG for determinism
    rng = random.Random(seed)

    # Resolve taxonomy
    tx = taxonomy if taxonomy else default_taxonomy

    # Parse config sections
    gen_cfg = cfg.get("generator", {})

    # Topic config
    topic_weights = gen_cfg.get("topic", {}).get("weights", {})
    if not topic_weights and tx and hasattr(tx, "TOPIC_CATEGORIES"):
        # Default to uniform over taxonomy if no weights provided
        topic_weights = {t: 1.0 for t in tx.TOPIC_CATEGORIES}

    # Validate topics against taxonomy if possible
    if tx and hasattr(tx, "TOPIC_CATEGORIES"):
        allowed = set(tx.TOPIC_CATEGORIES)
        for topic in topic_weights.keys():
            if topic not in allowed:
                raise ValueError(f"Topic '{topic}' not found in taxonomy")

    # Sentiment config (map score -> weight)
    # If empty, default to uniform 3-point scale
    sentiment_weights = gen_cfg.get("sentiment", {}).get("weights")
    if not sentiment_weights:
        sentiment_weights = {-1.0: 1.0, 0.0: 1.0, 1.0: 1.0}

    # Ensure keys are floats
    sentiment_weights = {float(k): v for k, v in sentiment_weights.items()}

    # Viewpoint config (map score -> weight)
    # If empty, default to uniform 3-point scale
    viewpoint_weights = gen_cfg.get("viewpoint", {}).get("weights")
    if not viewpoint_weights:
        viewpoint_weights = {0.0: 1.0, 0.5: 1.0, 1.0: 1.0}

    # Ensure keys are floats
    viewpoint_weights = {float(k): v for k, v in viewpoint_weights.items()}

    # Duration config
    duration_cfg = gen_cfg.get("duration", {})

    # Tags config
    tags_cfg = gen_cfg.get("tags", {})
    vocab = tags_cfg.get("vocab", [])
    per_video_cfg = tags_cfg.get("per_video", {})
    min_tags = per_video_cfg.get("min", 0)
    max_tags = per_video_cfg.get("max", 5)

    # Generate videos
    videos = []
    for i in range(n):
        # 1. Topic
        topic = _get_weighted_choice(rng, topic_weights)
        if topic is None:
            # Fallback if no topics and no taxonomy
            topic = "undefined"

        # 2. Sentiment
        sentiment = _get_weighted_choice(rng, sentiment_weights)

        # 3. Viewpoint
        viewpoint = _get_weighted_choice(rng, viewpoint_weights)

        # 4. Duration
        duration = _sample_duration(rng, duration_cfg)

        # 5. Tags
        video_tags = []
        if vocab:
            # Determine count
            # Simple uniform for now, could add Poisson later config permitting
            count = rng.randint(min_tags, max_tags)
            if count > 0:
                # Sample without replacement for unique tags per video
                # If count > len(vocab), cap it
                k = min(count, len(vocab))
                # Sort vocab for deterministic sampling
                sorted_vocab = sorted(vocab)
                video_tags = rng.sample(sorted_vocab, k)

        # Create object
        # ID strategy: f"vid_{i}" or just int? Model says `video_id: int`
        # Using integer ID starting from 0 (or large number? usually 0-indexed in list)
        # To avoid valid IDs being mistaken for defaults, let's just use i

        vid = Video(
            video_id=i,
            topic_category=topic,
            viewpoint_score=viewpoint,
            sentiment_score=sentiment,
            duration_s=duration,
            tags=tuple(video_tags),
        )
        videos.append(vid)

    return videos
