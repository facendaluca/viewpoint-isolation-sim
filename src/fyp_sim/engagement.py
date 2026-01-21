from __future__ import annotations

import random

from fyp_sim.models import User, UserAction, Video
from fyp_sim.policy import decide_action


def watch_time_seconds(user: User, video: Video, rng: random.Random) -> int:
    """Return watch time (seconds) as an implicit feedback proxy.
    Mapping:
        - AVOID -> 0s
        - SAMPLE -> 3-5s (capped by video duration)
        - WATCH -> majority of the video (70-95% of duration)

    Note: the action comes from 'decide_action', which incorporates phenotype _ interest (topic + tags) _ sentiment gating.
    so a sampler can WATCH if interest is very high, etc.
    """

    if video.duration_s < 0:
        raise ValueError("video.duration_seconds must be >= 0")

    action = decide_action(user, video)

    if action == UserAction.AVOID:
        return 0

    if action == UserAction.SAMPLE:
        return min(video.duration_s, rng.randint(3, 5))

    # WATCH: majority of the video
    if video.duration_s == 0:
        return 0
    fraction = rng.uniform(0.70, 0.95)

    return min(video.duration_s, max(1, int(round(video.duration_s * fraction))))
