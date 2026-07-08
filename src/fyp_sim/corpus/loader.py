"""
Factory for loading or generating video corpus based on configuration.
"""

from __future__ import annotations

from typing import Any

from fyp_sim.models import Video

from .generator import generate_video_corpus


def _load_from_list(video_list: list[dict[str, Any]]) -> list[Video]:
    """Parse list of dicts into list of Video objects."""
    pool: list[Video] = []
    for v in video_list:
        tags = tuple(v.get("tags", []))
        pool.append(
            Video(
                video_id=int(v["video_id"]),
                topic_category=str(v["topic_category"]),
                viewpoint_score=float(v["viewpoint_score"]),
                sentiment_score=float(v["sentiment_score"]),
                duration_s=int(v["duration_s"]),
                tags=tags,
            )
        )
    return pool


def build_corpus(cfg: dict[str, Any]) -> list[Video]:
    """
    Build the video corpus from configuration.

    Supports two modes:
    1. "file" (default): Loads explicit list from `video_pool` key.
    2. "generated": Generates synthetic corpus using `corpus` config.

    Args:
        cfg: Full experiment configuration dictionary.

    Returns:
        List of Video objects.

    Raises:
        ValueError: If source is unknown or required config is missing.
    """
    corpus_cfg = cfg.get("corpus", {})
    source = corpus_cfg.get("source", "file")

    if source == "file":
        # Backward compatibility: use the explicit "video_pool" list from the config
        if "video_pool" not in cfg:
            raise ValueError("Config key 'video_pool' required for source='file'")

        return _load_from_list(cfg["video_pool"])

    elif source == "generated":
        if "n_videos" not in corpus_cfg:
            raise ValueError("Config key 'corpus.n_videos' required for source='generated'")

        n = int(corpus_cfg["n_videos"])

        # corpus.seed is deliberately separate from the simulation "seeds" list:
        # one fixed corpus seed keeps the same video pool across every seed in a batch.
        seed = corpus_cfg.get("seed", 42)

        gen_cfg = {"generator": corpus_cfg.get("generator", {})}

        return generate_video_corpus(n, seed, gen_cfg)

    else:
        raise ValueError(f"Unknown corpus source: {source}")
