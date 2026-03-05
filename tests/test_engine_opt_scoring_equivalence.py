from fyp_sim.models import User, UserPhenotype, Video
from fyp_sim.policy import decide_action, interest_score
from fyp_sim.simulation.engine import engagement_proxy
from fyp_sim.simulation.engine_opt import (
    _decide_action_from_interest,
    _interest_score_once,
    _video_score_opt,
)


def test_opt_interest_matches_policy() -> None:
    user = User(UserPhenotype.WATCHER, 0.5, {"a": 0.2, "t": 0.7, "x": 0.9}, 0.0)
    v = Video(
        video_id=1,
        topic_category="t",
        viewpoint_score=0.2,
        sentiment_score=0.0,
        duration_s=10,
        tags=("x",),
    )
    assert _interest_score_once(user, v) == interest_score(user, v)


def test_opt_action_matches_policy() -> None:
    user = User(UserPhenotype.SAMPLER, 0.5, {"t": 0.3}, -0.2)
    v = Video(
        video_id=1, topic_category="t", viewpoint_score=0.2, sentiment_score=0.0, duration_s=10
    )
    i = interest_score(user, v)
    assert _decide_action_from_interest(
        phenotype=user.phenotype,
        sentiment_threshold=user.sentiment_threshold,
        video_sentiment=v.sentiment_score,
        interest=i,
    ) == decide_action(user, v)


def test_opt_video_score_matches_engine_video_score_definition() -> None:
    user = User(UserPhenotype.WATCHER, 0.5, {"t": 0.6}, 0.0)
    v = Video(
        video_id=1, topic_category="t", viewpoint_score=0.2, sentiment_score=0.0, duration_s=50
    )
    rank_alpha = 0.5
    max_d = 100

    # definition from engine.video_score: (1-a)*interest + a*engagement_proxy(decide_action(...))
    a = decide_action(user, v)
    e = engagement_proxy(a, v, max_duration=max_d)
    i = interest_score(user, v)
    expected = (1.0 - rank_alpha) * i + rank_alpha * e

    got = _video_score_opt(user, v, rank_alpha=rank_alpha, max_duration=max_d)
    assert got == expected
