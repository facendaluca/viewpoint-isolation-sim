"""
Interest update logic

- interest_vector is a single map over BOTH topic categories and tags (0.0-1.0 affinity).
- Updates are deterministic and bounded to [0.0, 1.0].
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fyp_sim.models import User, Video


def _clamp01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def update_interest_vector(
    *,
    user: User,
    video: Video,
    watch_time_s: int,
    topic_alpha: float = 0.10,
    tag_alpha: float = 0.05,
    decay: float = 0.02,
    normalise: bool = False,
    prune_below: float = 0.001,
) -> None:
    """Update user's interest_vector using implicit feedback (watch time).

    Rule:
    - Decay applied to all keys each update (controls saturation + forgetting)
    - Then bump topic/tags by alpha * watched_fraction
    - Optional normalisation (keeps total mass bounded)
    - Optional pruning of tiny values to keep dict small.
    """
    if watch_time_s is None or watch_time_s <= 0:
        return
    if video.duration_s <= 0:
        return

    frac = _clamp01(watch_time_s / float(video.duration_s))

    # 1) Decay existing interests
    if decay < 0.0 or decay >= 1.0:
        raise ValueError("decay must be in [0.0, 1.0]")
    if decay > 0.0 and user.interest_vector:
        for k in list(user.interest_vector.keys()):
            user.interest_vector[k] = float(user.interest_vector[k]) * (1.0 - decay)

    # 2) Bump helpers
    def bump(key: str, alpha: float) -> None:
        cur = float(user.interest_vector.get(key, 0.0))
        user.interest_vector[key] = _clamp01(cur + alpha * frac)

    bump(video.topic_category, topic_alpha)
    for tag in video.tags:
        bump(tag, tag_alpha)

    # 3) Prune tiny values
    if prune_below > 0.0:
        for k in list(user.interest_vector.keys()):
            if float(user.interest_vector[k]) < prune_below:
                del user.interest_vector[k]

    # 4) Optional normalisation
    if normalise and user.interest_vector:
        total = sum(float(v) for v in user.interest_vector.values())
        if total > 0.0:
            for k in list(user.interest_vector.keys()):
                user.interest_vector[k] = _clamp01(float(user.interest_vector[k]) / total)
