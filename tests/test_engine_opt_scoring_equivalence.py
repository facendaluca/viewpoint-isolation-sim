from fyp_sim.models import User, UserAction, UserPhenotype, Video
from fyp_sim.policy import decide_action, interest_score, predicted_action
from fyp_sim.simulation.engine import engagement_proxy, video_score
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


def test_opt_action_matches_predicted_policy() -> None:
    # Ranking uses the platform's predicted_action, so that is the reference
    # the optimised shortcut has to match (not the user's realised decide_action).
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
    ) == predicted_action(user, v)


def test_opt_video_score_matches_engine_video_score_definition() -> None:
    user = User(UserPhenotype.WATCHER, 0.5, {"t": 0.6}, 0.0)
    v = Video(
        video_id=1, topic_category="t", viewpoint_score=0.2, sentiment_score=0.0, duration_s=50
    )
    rank_alpha = 0.5
    max_d = 100

    # definition from engine.video_score:
    # (1-a)*interest + a*engagement_proxy(predicted_action(...))
    a = predicted_action(user, v)
    e = engagement_proxy(a, v, max_duration=max_d)
    i = interest_score(user, v)
    expected = (1.0 - rank_alpha) * i + rank_alpha * e

    assert video_score(user, v, rank_alpha=rank_alpha, max_duration=max_d) == expected
    assert _video_score_opt(user, v, rank_alpha=rank_alpha, max_duration=max_d) == expected


def test_engines_score_identically_when_watcher_refuses_a_tag_hook() -> None:
    # The platform's model falls for the tag hook (predicted Watch) while the
    # watcher's topic-taste rule refuses the off-topic video (realised Avoid).
    # Both engines must rank this candidate with the predicted action and agree
    # exactly; scoring it with the realised Avoid would zero the engagement term.
    user = User(UserPhenotype.WATCHER, 0.5, {"meme": 0.9}, -0.2)
    v = Video(
        video_id=9,
        topic_category="diy_life_hacks",
        viewpoint_score=0.5,
        sentiment_score=0.0,
        duration_s=40,
        tags=("meme",),
    )

    assert predicted_action(user, v) == UserAction.WATCH
    assert decide_action(user, v) == UserAction.AVOID

    max_d = 100
    e_watch = engagement_proxy(UserAction.WATCH, v, max_duration=max_d)
    for rank_alpha in (0.0, 0.3, 1.0):
        baseline = video_score(user, v, rank_alpha=rank_alpha, max_duration=max_d)
        opt = _video_score_opt(user, v, rank_alpha=rank_alpha, max_duration=max_d)
        assert baseline == opt
        assert baseline == (1.0 - rank_alpha) * 0.9 + rank_alpha * e_watch
