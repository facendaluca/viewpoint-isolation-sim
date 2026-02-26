import random

import pytest

from fyp_sim.models import User, UserAction, UserPhenotype, Video
from fyp_sim.simulation.engine import run_simulation
from fyp_sim.simulation.viewpoint_drift import apply_viewpoint_drift


class AlwaysDecide:
    """Deterministic decider stub that matches engine's expected interface."""

    def __init__(self, action: UserAction) -> None:
        self._action = action

    def decide_next_action(self, user: User, video: Video) -> UserAction:
        _ = (user, video)
        return self._action


def choose_first(
    user: User,
    video_pool: list[Video],
    rng: random.Random,
    *,
    top_k: int,
    rank_alpha: float,
) -> Video:
    """Chooser matching engine signature; deterministic for tests."""
    _ = (user, rng, top_k, rank_alpha)
    return video_pool[0]


@pytest.mark.parametrize(
    "action, weight",
    [
        (UserAction.WATCH, 1.0),
        (UserAction.SAMPLE, 0.2),
        (UserAction.AVOID, 0.0),
    ],
)
def test_apply_viewpoint_drift_matches_chapter3_update_rule(
    action: UserAction, weight: float
) -> None:
    """
    Chapter 3 drift rule:
        new = old + alpha * weight(action) * (target - old)
    """
    old = 0.2
    target = 0.8
    alpha = 0.5

    expected = old + alpha * weight * (target - old)
    got = apply_viewpoint_drift(old, target, drift_rate=alpha, action=action)

    assert got == pytest.approx(expected, abs=1e-12)


def test_engine_logs_viewpoint_pre_post_and_uses_video_target_when_drift_enabled() -> None:
    user = User(
        phenotype=UserPhenotype.WATCHER,
        viewpoint_score=0.2,
        interest_vector={},
        sentiment_threshold=0.0,
    )
    video = Video(
        video_id=1,
        topic_category="topic",
        viewpoint_score=0.8,
        sentiment_score=0.0,
        duration_s=30,
        tags=(),
    )

    logs = run_simulation(
        user=user,
        video_pool=[video],
        steps=1,
        rng=random.Random(123),
        top_k=1,
        rank_alpha=0.0,
        chooser=choose_first,
        watch_time_fn=lambda _u, _v, _rng: 0,  # remove unrelated randomness
        decider=AlwaysDecide(UserAction.WATCH),
        enable_viewpoint_drift=True,
        viewpoint_drift_rate=0.5,
    )

    assert len(logs) == 1
    row = logs[0]

    assert row.user_viewpoint_pre == pytest.approx(0.2, abs=1e-12)
    assert row.video_viewpoint_score == pytest.approx(0.8, abs=1e-12)

    expected_post = apply_viewpoint_drift(
        row.user_viewpoint_pre,
        row.video_viewpoint_score,
        drift_rate=0.5,
        action=UserAction.WATCH,
    )
    assert row.user_viewpoint_post == pytest.approx(expected_post, abs=1e-12)


def test_engine_drift_is_deterministic_given_seed_and_inputs() -> None:
    def run_once() -> float:
        user = User(
            phenotype=UserPhenotype.WATCHER,
            viewpoint_score=0.2,
            interest_vector={},
            sentiment_threshold=0.0,
        )
        video = Video(
            video_id=1,
            topic_category="topic",
            viewpoint_score=0.8,
            sentiment_score=0.0,
            duration_s=30,
            tags=(),
        )

        logs = run_simulation(
            user=user,
            video_pool=[video],
            steps=1,
            rng=random.Random(999),
            top_k=1,
            rank_alpha=0.0,
            chooser=choose_first,
            watch_time_fn=lambda _u, _v, _rng: 0,
            decider=AlwaysDecide(UserAction.SAMPLE),
            enable_viewpoint_drift=True,
            viewpoint_drift_rate=0.5,
        )
        return logs[0].user_viewpoint_post

    assert run_once() == pytest.approx(run_once(), abs=1e-12)


@pytest.mark.parametrize(
    "old,target,alpha",
    [
        (0.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 1.0, 1.0),
        (1.0, 0.0, 1.0),
        (0.2, 0.8, 0.5),
        (0.9, 0.1, 0.25),
    ],
)
@pytest.mark.parametrize("action", [UserAction.WATCH, UserAction.SAMPLE, UserAction.AVOID])
def test_drift_output_is_bounded_for_valid_inputs(
    old: float,
    target: float,
    alpha: float,
    action: UserAction,
) -> None:
    new = apply_viewpoint_drift(old, target, drift_rate=alpha, action=action)
    assert 0.0 <= new <= 1.0
