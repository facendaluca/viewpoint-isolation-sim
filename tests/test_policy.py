from fyp_sim.models import User, UserAction, UserPhenotype, Video
from fyp_sim.policy import decide_action


def test_watcher_watches_high_interest():
    u = User(
        phenotype=UserPhenotype.WATCHER,
        viewpoint_score=0.5,
        interest_vector={"politics": 0.7},
        sentiment_threshold=-0.5,
    )
    v = Video(1, "politics", 0.6, 0.0, 30)
    assert decide_action(u, v) == UserAction.WATCH


def test_sampler_avoids_low_interest():
    u = User(
        phenotype=UserPhenotype.SAMPLER,
        viewpoint_score=0.5,
        interest_vector={"politics": 0.1},
        sentiment_threshold=-0.5,
    )
    v = Video(2, "politics", 0.6, 0.0, 30)
    assert decide_action(u, v) == UserAction.AVOID


def test_sentiment_gate_triggers_avoid():
    u = User(
        phenotype=UserPhenotype.WATCHER,
        viewpoint_score=0.5,
        interest_vector={"politics": 0.7},
        sentiment_threshold=-0.2,
    )
    v = Video(3, "politics", 0.6, -0.9, 30)  # sentiment_score < threshold
    assert decide_action(u, v) == UserAction.AVOID
