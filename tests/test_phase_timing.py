import random

from fyp_sim.benchmarks.phase_timing import PhaseTimer
from fyp_sim.models import User, UserAction, UserPhenotype, Video
from fyp_sim.simulation.engine import run_simulation


class DummyDecider:
    def __init__(self) -> None:
        self.last_meta = None

    def decide_next_action(self, user: User, video: Video) -> UserAction:
        return UserAction.WATCH


def chooser_first(user: User, pool: list[Video], rng, *, top_k: int, rank_alpha: float) -> Video:
    return pool[0]


def watch_time_const(user: User, v: Video, rng) -> int:
    return 5


def test_phase_timer_records_expected_counts() -> None:
    user = User(
        phenotype=UserPhenotype.WATCHER,
        viewpoint_score=0.5,
        interest_vector={},
        sentiment_threshold=0.0,
    )
    pool = [
        Video(
            video_id=1,
            topic_category="topic",
            viewpoint_score=0.2,
            sentiment_score=0.0,
            duration_s=10,
        ),
        Video(
            video_id=2,
            topic_category="topic",
            viewpoint_score=0.8,
            sentiment_score=0.0,
            duration_s=20,
        ),
    ]

    timer = PhaseTimer()
    _ = run_simulation(
        user=user,
        video_pool=pool,
        steps=7,
        rng=random.Random(123),
        top_k=1,
        rank_alpha=0.5,
        drift_alpha=0.0,
        chooser=chooser_first,
        watch_time_fn=watch_time_const,
        decider=DummyDecider(),
        enable_interest_updates=False,
        enable_viewpoint_drift=False,
        viewpoint_drift_rate=0.0,
        phase_tracer=timer,
    )

    assert timer.timings.counts["generate_feed"] == 7
    assert timer.timings.counts["simulate_interaction"] == 7
    assert timer.timings.counts["update_state"] == 7
    assert timer.timings.counts["log_append"] == 7
