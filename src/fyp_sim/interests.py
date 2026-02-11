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


def update_interest_vector(*, user: User, video: Video, watch_time_s: int) -> None:
    """Update user's interest_vector using implicit feedback (watch time).

    Rule:
    - Ignore non-positive watch_time_s or non-positive duration.
    - Convert watch_time_s to a 0..1 fraction of duration.
    - Apply a small learning rate to the topic category and a smaller one to tags.
    - Cap affinities into [0.0, 1.0].
    """
    if watch_time_s is None or watch_time_s <= 0:
        return
    if video.duration_s <= 0:
        return

    frac = _clamp01(watch_time_s / float(video.duration_s))

    # Learning rates (small to avoid instant saturation)
    topic_alpha = 0.10
    tag_alpha = 0.05

    def bump(key: str, alpha: float) -> None:
        cur = float(user.interest_vector.get(key, 0.0))
        user.interest_vector[key] = _clamp01(cur + alpha * frac)

    # Topic update
    bump(video.topic_category, topic_alpha)

    # Tag updates (if any)
    for tag in video.tags:
        bump(tag, tag_alpha)
