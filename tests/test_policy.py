import pytest

from fyp_sim.models import User, UserAction, UserPhenotype, Video
from fyp_sim.policy import decide_action

ALL_PHENOTYPES = [UserPhenotype.WATCHER, UserPhenotype.SAMPLER, UserPhenotype.AVOIDER]


def _make_user(phenotype: UserPhenotype, interest_vector: dict[str, float]) -> User:
    return User(
        phenotype=phenotype,
        viewpoint_score=0.5,
        interest_vector=interest_vector,
        sentiment_threshold=-0.2,
    )


@pytest.mark.parametrize("phenotype", ALL_PHENOTYPES)
def test_sentiment_gate_overrides_high_interest(phenotype: UserPhenotype):
    u = _make_user(phenotype, {"politics": 1.0})
    v = Video(1, "politics", 0.6, -0.9, 30)  # sentiment_score < threshold
    assert decide_action(u, v) == UserAction.AVOID


@pytest.mark.parametrize("phenotype", ALL_PHENOTYPES)
def test_everyone_watches_high_topic_interest(phenotype: UserPhenotype):
    # 0.9 clears every phenotype's watch threshold, including the sampler's 0.7
    u = _make_user(phenotype, {"politics": 0.9})
    v = Video(2, "politics", 0.6, 0.0, 30)
    assert decide_action(u, v) == UserAction.WATCH


@pytest.mark.parametrize("phenotype", ALL_PHENOTYPES)
def test_tag_interest_counts_like_topic_interest(phenotype: UserPhenotype):
    # Interest can come from a tag, not just the topic category
    u = _make_user(phenotype, {"politics": 0.1, "meme": 0.9})
    v = Video(3, "politics", 0.6, 0.0, 30, tags=("meme",))
    assert decide_action(u, v) == UserAction.WATCH


@pytest.mark.parametrize(
    ("interest", "expected"),
    [
        (0.50, UserAction.WATCH),  # clear match
        (0.20, UserAction.SAMPLE),  # adjacent / uncertain
        (0.19, UserAction.AVOID),  # clearly irrelevant
    ],
)
def test_watcher_thresholds(interest: float, expected: UserAction):
    u = _make_user(UserPhenotype.WATCHER, {"politics": interest})
    v = Video(4, "politics", 0.6, 0.0, 30)
    assert decide_action(u, v) == expected


@pytest.mark.parametrize(
    ("interest", "expected"),
    [
        (0.70, UserAction.WATCH),  # strong match, worth a full watch
        (0.69, UserAction.SAMPLE),  # just under the bar, keep exploring
        (0.00, UserAction.SAMPLE),  # samplers sample even with zero interest
    ],
)
def test_sampler_thresholds(interest: float, expected: UserAction):
    u = _make_user(UserPhenotype.SAMPLER, {"politics": interest})
    v = Video(5, "politics", 0.6, 0.0, 30)
    assert decide_action(u, v) == expected


def test_sampler_samples_with_no_interest_entry_at_all():
    # An empty interest vector still means sample, as long as sentiment is safe
    u = _make_user(UserPhenotype.SAMPLER, {})
    v = Video(6, "politics", 0.6, 0.0, 30)
    assert decide_action(u, v) == UserAction.SAMPLE


@pytest.mark.parametrize(
    ("interest", "expected"),
    [
        (0.50, UserAction.WATCH),  # clear match
        (0.49, UserAction.AVOID),  # avoiders never sample
    ],
)
def test_avoider_thresholds(interest: float, expected: UserAction):
    u = _make_user(UserPhenotype.AVOIDER, {"politics": interest})
    v = Video(7, "politics", 0.6, 0.0, 30)
    assert decide_action(u, v) == expected
