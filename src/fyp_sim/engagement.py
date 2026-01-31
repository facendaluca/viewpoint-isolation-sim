from __future__ import annotations

import random

from fyp_sim.models import User, UserAction, Video
from fyp_sim.policy import decide_action


def watch_time_seconds(user: User, video: Video, rng: random.Random) -> int:
    """Return watch time (seconds) as an implicit feedback proxy.

    Mapping (capped by video duration):
        - AVOID -> 0 seconds
        - SAMPLE -> 3-5 seconds (brief skim)
        - WATCH -> 70-95% of duration (majority watched)

    Notes:
        - Action comes from 'decide_action' (phenotype + interest (topic/tags) + sentiment gating).
        - This function is intentionally simple and stochastic but reproducible given rng seed.
    """

    if video.duration_s < 0:
        raise ValueError("video.duration_s must be >= 0")

    action = decide_action(user, video)

    if action == UserAction.AVOID:
        return 0

    if action == UserAction.SAMPLE:
        return min(video.duration_s, rng.randint(3, 5))

    # WATCH: sample a "majority watched" fraction to reflect partial completion behaviour.
    if video.duration_s == 0:
        return 0
    fraction = rng.uniform(0.70, 0.95)

    return min(video.duration_s, max(1, int(round(video.duration_s * fraction))))
