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
        # Backward compatibility: use "video_pool" list directly from config
        # (This mimics previous behavior where config JSON contained the full list)
        if "video_pool" not in cfg:
            # It's possible the user intends to use "file" but provided no pool.
            # However, existing scripts crashed if key was missing or empty anyway.
            # Let's return empty or let it error naturally?
            # existing scripts did: `for v in cfg["video_pool"]` -> KeyError if missing.
            # We'll mimic that or provide a nicer error.
            if "video_pool" not in cfg:
                # If implied source=file but no video_pool, it's an invalid config for file mode.
                # But maybe they just want an empty pool? Unlikely.
                raise ValueError("Config key 'video_pool' required for source='file'")

        return _load_from_list(cfg["video_pool"])

    elif source == "generated":
        # Required keys for generation
        if "n_videos" not in corpus_cfg:
            raise ValueError("Config key 'corpus.n_videos' required for source='generated'")

        n = int(corpus_cfg["n_videos"])

        # Seed: prefer `corpus.seed`, fallback to `seed` (root), else error/default?
        # The prompt suggested: config["corpus"]["n_videos"] and config["seed"]
        # run_batch.py has "seeds" list (multi-seed).
        # run_sweep.py has "seeds" list.
        # The generator needs a SPECIFIC seed for the corpus.
        # If we use the simulation seed, the corpus changes per simulation seed?
        # Usually we want a fixed corpus for all seeds in a batch, OR varying.
        # "Output must be deterministic for same (N, seed, config)"
        # If the user wants a fixed corpus across the batch, they should provide `corpus.seed`.
        # If they want it to vary, they might not set it?
        # Let's look for `corpus.seed`. If missing, we can use a fixed default or
        # raise error to force reproducibility.
        # Prompt said: "Decide where n and seed live: Recommend: config['corpus']['n_videos'] and config['seed']"
        # BUT `config["seed"]` doesn't exist in `experiment_baseline.json`, it has `"seeds": [...]`.
        # So we probably want a fixed seed for the corpus generation to ensure all runs in the batch use the SAME corpus.
        # So `corpus.seed` is best.

        seed = corpus_cfg.get("seed", 42)  # Default fixed seed if not specified

        # Generator config
        gen_cfg = {"generator": corpus_cfg.get("generator", {})}

        return generate_video_corpus(n, seed, gen_cfg)

    else:
        raise ValueError(f"Unknown corpus source: {source}")
