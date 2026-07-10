from __future__ import annotations

import random

from fyp_sim.models import User, UserAction, UserPhenotype, Video
from fyp_sim.simulation.engine import run_simulation


class FixedDecider:
    """Always returns the same action (no meta)."""

    last_meta = None

    def __init__(self, action: UserAction):
        self.action = action

    def decide_next_action(self, user: User, video: Video) -> UserAction:  # noqa: ARG002
        return self.action


def _user() -> User:
    return User(
        phenotype=UserPhenotype.WATCHER,
        viewpoint_score=0.5,
        interest_vector={"sports": 0.9, "news": 0.8, "music": 0.7},
        sentiment_threshold=-1.0,
    )


def _pool() -> list[Video]:
    topics = ["sports", "news", "music", "sports", "news", "music"]
    return [Video(i, topic, 0.5, 0.5, 60) for i, topic in enumerate(topics, start=1)]


def _run(action: UserAction, *, exposure_seed: int = 123, engagement_seed: int | None = 7):
    engagement_rng = None if engagement_seed is None else random.Random(engagement_seed)
    return run_simulation(
        user=_user(),
        video_pool=_pool(),
        steps=50,
        rng=random.Random(exposure_seed),
        engagement_rng=engagement_rng,
        top_k=3,
        rank_alpha=0.5,
        decider=FixedDecider(action),
    )


def test_separate_streams_keep_exposure_identical_across_arms():
    # WATCH consumes a watch-time draw every step, AVOID consumes none. With a
    # separate engagement stream that difference must not move exposure.
    a = _run(UserAction.WATCH)
    b = _run(UserAction.AVOID)
    assert [r.video_id for r in a] == [r.video_id for r in b]


def test_engagement_seed_does_not_change_exposure_order():
    a = _run(UserAction.WATCH, engagement_seed=7)
    b = _run(UserAction.WATCH, engagement_seed=8)
    assert [r.video_id for r in a] == [r.video_id for r in b]
    # Different engagement seeds should still change watch times, proving the
    # second stream is really the one feeding watch_time_seconds.
    assert [r.watch_time_s for r in a] != [r.watch_time_s for r in b]


def test_shared_stream_lets_watch_time_shift_exposure():
    # Documents the problem the split fixes: with one shared rng, arms that act
    # differently drift apart in what they even get shown.
    a = _run(UserAction.WATCH, engagement_seed=None)
    b = _run(UserAction.AVOID, engagement_seed=None)
    assert [r.video_id for r in a] != [r.video_id for r in b]
