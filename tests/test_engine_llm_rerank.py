from __future__ import annotations

import random

from fyp_sim.agents.deciders import LLMDecider
from fyp_sim.models import User, UserAction, UserPhenotype, Video
from fyp_sim.simulation.engine import run_simulation, select_video_llm_slate


class ScriptedDecider:
    """Fixed action per video_id; records which videos it was asked about."""

    def __init__(self, actions: dict[int, UserAction], default: UserAction = UserAction.AVOID):
        self.actions = actions
        self.default = default
        self.asked: list[int] = []
        self.last_meta = None

    def decide_next_action(self, user: User, video: Video) -> UserAction:  # noqa: ARG002
        self.asked.append(video.video_id)
        return self.actions.get(video.video_id, self.default)


def _user() -> User:
    return User(
        phenotype=UserPhenotype.WATCHER,
        viewpoint_score=0.5,
        interest_vector={"sports": 0.9, "news": 0.8, "music": 0.7, "travel": 0.1},
        sentiment_threshold=-1.0,
    )


def _pool() -> list[Video]:
    topics = ["sports", "news", "music", "travel", "travel", "travel"]
    return [Video(i, topic, 0.5, 0.5, 30) for i, topic in enumerate(topics, start=1)]


def test_rerank_asks_decider_only_about_the_slate():
    decider = ScriptedDecider({}, default=UserAction.SAMPLE)
    select_video_llm_slate(
        _user(), _pool(), random.Random(0), top_k=3, rank_alpha=0.5, decider=decider
    )
    # Top-3 by heuristic interest: sports, news, music. Never the full pool.
    assert sorted(decider.asked) == [1, 2, 3]


def test_rerank_lets_decider_actions_steer_exposure():
    # The decider avoids the two heuristically strongest candidates, so exposure
    # should shift to the one it watches (rank_alpha=1.0 -> action-only scoring).
    decider = ScriptedDecider({1: UserAction.AVOID, 2: UserAction.AVOID, 3: UserAction.WATCH})
    picks = set()
    for seed in range(10):
        video, action, _ = select_video_llm_slate(
            _user(), _pool(), random.Random(seed), top_k=3, rank_alpha=1.0, decider=decider
        )
        picks.add(video.video_id)
        assert action == UserAction.WATCH
    assert picks == {3}


def test_run_simulation_llm_rerank_reuses_slate_decision():
    decider = ScriptedDecider({1: UserAction.AVOID, 2: UserAction.AVOID, 3: UserAction.WATCH})
    steps = 4
    logs = run_simulation(
        user=_user(),
        video_pool=_pool(),
        steps=steps,
        rng=random.Random(1),
        top_k=3,
        rank_alpha=1.0,
        decider=decider,
        llm_rerank=True,
    )
    assert [r.video_id for r in logs] == [3] * steps
    assert all(r.action == "Watch" for r in logs)
    # One decider call per slate candidate; no second call for the chosen video.
    assert len(decider.asked) == 3 * steps


def test_run_simulation_llm_rerank_logs_meta_of_chosen_candidate():
    class FakeClient:
        def complete(self, prompt: str, *, timeout_s: float) -> str:  # noqa: ARG002
            return '{"action": "Watch", "confidence": 0.8}'

    decider = LLMDecider(prompt_id="decision_v1", client=FakeClient(), timeout_s=1.0)
    logs = run_simulation(
        user=_user(),
        video_pool=_pool(),
        steps=2,
        rng=random.Random(0),
        top_k=2,
        rank_alpha=0.5,
        decider=decider,
        llm_rerank=True,
    )
    for r in logs:
        assert r.policy_mode == "llm"
        assert r.llm_valid is True
        assert r.llm_action == "Watch"
        assert r.llm_confidence == 0.8
