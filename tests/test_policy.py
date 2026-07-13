import pytest

from fyp_sim.models import User, UserAction, UserPhenotype, Video
from fyp_sim.policy import decide_action, predicted_action

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


@pytest.mark.parametrize("phenotype", [UserPhenotype.SAMPLER, UserPhenotype.AVOIDER])
def test_tag_interest_counts_like_topic_interest(phenotype: UserPhenotype):
    # For samplers and avoiders, interest can come from a tag, not just the topic
    u = _make_user(phenotype, {"politics": 0.1, "meme": 0.9})
    v = Video(3, "politics", 0.6, 0.0, 30, tags=("meme",))
    assert decide_action(u, v) == UserAction.WATCH


def test_watcher_refuses_tag_hooked_video_outside_taste():
    # A watcher's established taste is anchored on topics: a hooky tag on an
    # off-taste topic does not earn a watch, it gets refused.
    u = _make_user(UserPhenotype.WATCHER, {"politics": 0.1, "meme": 0.9})
    v = Video(3, "politics", 0.6, 0.0, 30, tags=("meme",))
    assert decide_action(u, v) == UserAction.AVOID


def test_watcher_refusal_is_hidden_from_the_platform_model():
    # The ranker's predicted_action still sees the tag hook as a Watch, which is
    # exactly why the platform can serve videos the watcher then refuses.
    u = _make_user(UserPhenotype.WATCHER, {"politics": 0.1, "meme": 0.9})
    v = Video(3, "politics", 0.6, 0.0, 30, tags=("meme",))
    assert predicted_action(u, v) == UserAction.WATCH
    assert decide_action(u, v) == UserAction.AVOID


@pytest.mark.parametrize(
    ("topic_affinity", "expected"),
    [
        (0.50, UserAction.WATCH),  # clear match inside taste
        (0.20, UserAction.WATCH),  # slight interest still earns a watch
        (0.19, UserAction.AVOID),  # topic outside the established taste
    ],
)
def test_watcher_taste_floor(topic_affinity: float, expected: UserAction):
    u = _make_user(UserPhenotype.WATCHER, {"politics": topic_affinity})
    v = Video(4, "politics", 0.6, 0.0, 30)
    assert decide_action(u, v) == expected


def test_watcher_never_samples():
    # Watchers commit or refuse; sampling is the sampler's behaviour.
    for affinity in (0.0, 0.19, 0.20, 0.49, 0.50, 1.0):
        u = _make_user(UserPhenotype.WATCHER, {"politics": affinity})
        v = Video(4, "politics", 0.6, 0.0, 30)
        assert decide_action(u, v) in {UserAction.WATCH, UserAction.AVOID}


@pytest.mark.parametrize(
    ("interest", "expected"),
    [
        (0.50, UserAction.WATCH),  # clear match
        (0.20, UserAction.SAMPLE),  # adjacent / uncertain
        (0.19, UserAction.AVOID),  # clearly irrelevant
    ],
)
def test_predicted_watcher_brackets(interest: float, expected: UserAction):
    # The platform's engagement model keeps the original interest brackets.
    u = _make_user(UserPhenotype.WATCHER, {"politics": interest})
    v = Video(4, "politics", 0.6, 0.0, 30)
    assert predicted_action(u, v) == expected


@pytest.mark.parametrize("phenotype", [UserPhenotype.SAMPLER, UserPhenotype.AVOIDER])
@pytest.mark.parametrize("interest", [0.0, 0.19, 0.20, 0.49, 0.50, 0.69, 0.70, 1.0])
def test_predicted_matches_realised_for_samplers_and_avoiders(
    phenotype: UserPhenotype, interest: float
):
    u = _make_user(phenotype, {"politics": interest})
    v = Video(4, "politics", 0.6, 0.0, 30)
    assert predicted_action(u, v) == decide_action(u, v)


def test_predicted_action_respects_sentiment_gate():
    u = _make_user(UserPhenotype.WATCHER, {"politics": 1.0})
    v = Video(1, "politics", 0.6, -0.9, 30)
    assert predicted_action(u, v) == UserAction.AVOID


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
