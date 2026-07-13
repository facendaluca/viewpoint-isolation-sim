from __future__ import annotations

import random

from fyp_sim.agents.deciders import HeuristicDecider
from fyp_sim.models import User, UserPhenotype, Video
from src.scripts.run_compare import _run_simulation_compat


def _user() -> User:
    return User(
        phenotype=UserPhenotype.WATCHER,
        viewpoint_score=0.5,
        interest_vector={"sports": 0.6},
        sentiment_threshold=-1.0,
    )


def _pool() -> list[Video]:
    return [Video(1, "sports", 0.5, 0.5, 30)]


def _run(user: User, interest_kwargs: dict | None) -> None:
    _run_simulation_compat(
        user=user,
        video_pool=_pool(),
        steps=5,
        rng=random.Random(0),
        top_k=1,
        rank_alpha=0.3,
        drift_alpha=0.0,
        enable_viewpoint_drift=False,
        decider=HeuristicDecider(),
        engagement_rng=random.Random(1),
        interest_kwargs=interest_kwargs,
    )


def test_compare_forwards_interest_updates_when_enabled():
    # Watcher with interest 0.6 watches this video every step, so enabling
    # interest updates must move the interest vector.
    user = _user()
    _run(user, {"enable_interest_updates": True})
    assert user.interest_vector["sports"] != 0.6


def test_compare_leaves_state_untouched_by_default():
    user = _user()
    _run(user, None)
    assert user.interest_vector["sports"] == 0.6
