import random
from types import SimpleNamespace
from typing import Any

import pytest

import fyp_sim.simulation.engine as engine
from fyp_sim.models import UserAction

# TODO: Refactor test by implementing pytest fixtures (massively reduces boilerplate)


class DummyDecider:
    def __init__(self, action: UserAction):
        self._action = action
        self.last_meta = None

    def decide_next_action(self, user, video):
        return self._action


def chooser_first(user, pool, rng, *, top_k, rank_alpha):
    return pool[0]


def watch_time_const(user, video, rng):
    return 10


def test_viewpoint_drift_disabled_is_noop_and_logs_pre_post_equal(monkeypatch):
    # Make engine dependencies deterministic and independent of full model behaviour
    monkeypatch.setattr(engine, "interest_score", lambda user, v: 0.0)
    monkeypatch.setattr(engine, "viewpoint_distance", lambda a, b: abs(a - b))
    monkeypatch.setattr(
        engine,
        "running_mean",
        lambda prev, t, x: x if t == 0 else (prev * t + x) / (t + 1),
    )

    user: Any = SimpleNamespace(viewpoint_score=0.2, interest_vector={})
    video: Any = SimpleNamespace(video_id=1, viewpoint_score=0.9, topic_category="topic", duration_s=30)

    logs = engine.run_simulation(
        user=user,
        video_pool=[video],
        steps=3,
        rng=random.Random(0),
        chooser=chooser_first,
        watch_time_fn=watch_time_const,
        decider=DummyDecider(UserAction.WATCH),
        enable_viewpoint_drift=False,
        rank_alpha=0.3,
        viewpoint_drift_rate=0.5,
    )

    assert user.viewpoint_score == pytest.approx(0.2)
    for row in logs:
        assert row.user_viewpoint_pre == pytest.approx(0.2)
        assert row.user_viewpoint_post == pytest.approx(0.2)
        assert row.vii_t == pytest.approx(abs(0.2 - 0.9))


def test_viewpoint_drift_enabled_moves_toward_target_and_chains_state(monkeypatch):
    monkeypatch.setattr(engine, "interest_score", lambda user, v: 0.0)
    monkeypatch.setattr(engine, "viewpoint_distance", lambda a, b: abs(a - b))
    monkeypatch.setattr(
        engine,
        "running_mean",
        lambda prev, t, x: x if t == 0 else (prev * t + x) / (t + 1),
    )

    user: Any = SimpleNamespace(viewpoint_score=0.2, interest_vector={})
    video: Any = SimpleNamespace(video_id=1, viewpoint_score=1.0, topic_category="topic", duration_s=30)

    logs = engine.run_simulation(
        user=user,
        video_pool=[video],
        steps=3,
        rng=random.Random(0),
        chooser=chooser_first,
        watch_time_fn=watch_time_const,
        decider=DummyDecider(UserAction.WATCH),
        rank_alpha=0.3,
        enable_viewpoint_drift=True,
        viewpoint_drift_rate=0.2,  # WATCH => k=0.2
    )

    # Pre of step i+1 should equal post of step i (no other code mutates viewpoint).
    for i in range(len(logs) - 1):
        assert logs[i + 1].user_viewpoint_pre == pytest.approx(logs[i].user_viewpoint_post)

    # Each step should reduce distance to target (monotronic approach for constant target).
    for row in logs:
        d_pre = abs(video.viewpoint_score - row.user_viewpoint_pre)
        d_post = abs(video.viewpoint_score - row.user_viewpoint_post)
        assert d_post <= d_pre + 1e-12

    assert user.viewpoint_score == pytest.approx(logs[-1].user_viewpoint_post)


def test_viewpoint_drift_applies_when_only_drift_alpha_is_passed(monkeypatch):
    # Regression: the engine used to key the drift branch off viewpoint_drift_rate
    # only, so callers passing just drift_alpha silently got no drift.
    monkeypatch.setattr(engine, "interest_score", lambda user, v: 0.0)
    monkeypatch.setattr(engine, "viewpoint_distance", lambda a, b: abs(a - b))
    monkeypatch.setattr(
        engine,
        "running_mean",
        lambda prev, t, x: x if t == 0 else (prev * t + x) / (t + 1),
    )

    user: Any = SimpleNamespace(viewpoint_score=0.2, interest_vector={})
    video: Any = SimpleNamespace(video_id=1, viewpoint_score=1.0, topic_category="topic", duration_s=30)

    logs = engine.run_simulation(
        user=user,
        video_pool=[video],
        steps=3,
        rng=random.Random(0),
        chooser=chooser_first,
        watch_time_fn=watch_time_const,
        decider=DummyDecider(UserAction.WATCH),
        rank_alpha=0.3,
        enable_viewpoint_drift=True,
        drift_alpha=0.2,  # no viewpoint_drift_rate on purpose
    )

    assert user.viewpoint_score > 0.2
    for row in logs:
        assert row.user_viewpoint_post >= row.user_viewpoint_pre


def test_viewpoint_drift_enabled_but_avoid_action_is_noop(monkeypatch):
    monkeypatch.setattr(engine, "interest_score", lambda user, v: 0.0)
    monkeypatch.setattr(engine, "viewpoint_distance", lambda a, b: abs(a - b))
    monkeypatch.setattr(
        engine,
        "running_mean",
        lambda prev, t, x: x if t == 0 else (prev * t + x) / (t + 1),
    )

    user: Any = SimpleNamespace(viewpoint_score=0.3, interest_vector={})
    video: Any = SimpleNamespace(video_id=1, viewpoint_score=0.9, topic_category="topic", duration_s=30)

    logs = engine.run_simulation(
        user=user,
        video_pool=[video],
        steps=3,
        rng=random.Random(0),
        chooser=chooser_first,
        watch_time_fn=watch_time_const,
        decider=DummyDecider(UserAction.AVOID),
        rank_alpha=0.3,
        enable_viewpoint_drift=True,
        viewpoint_drift_rate=0.8,
    )

    assert user.viewpoint_score == pytest.approx(0.3)
    for row in logs:
        assert row.user_viewpoint_pre == pytest.approx(0.3)
        assert row.user_viewpoint_post == pytest.approx(0.3)
