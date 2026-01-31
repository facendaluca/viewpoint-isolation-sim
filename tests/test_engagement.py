import random

import pytest

from fyp_sim.engagement import watch_time_seconds
from fyp_sim.models import User, UserPhenotype, Video


def test_avoider_is_zero_seconds():
    rng = random.Random(0)
    u = User(UserPhenotype.AVOIDER, 0.5, {"politics": 0.7}, -0.2)
    v = Video(1, "politics", 0.6, 0.0, 25)

    assert watch_time_seconds(u, v, rng) == 0


def test_sampler_samples_when_not_interested():
    rng = random.Random(0)
    u = User(UserPhenotype.SAMPLER, 0.5, {"politics": 0.2}, -0.2)
    v = Video(2, "politics", 0.6, 0.0, 25)

    t = watch_time_seconds(u, v, rng)
    assert 3 <= t <= 5


def test_sampler_watches_majority_when_interested_via_tag():
    rng = random.Random(0)
    # High tag intesrest should lead to WATCH action for sampler
    u = User(UserPhenotype.SAMPLER, 0.5, {"politics": 0.1, "meme": 0.9}, -0.2)
    v = Video(2, "politics", 0.6, 0.0, 25, tags=("meme",))

    t = watch_time_seconds(u, v, rng)
    assert int(0.70 * 25) <= t <= 25


def test_sampling_is_capped_by_duration():
    rng = random.Random(0)
    u = User(UserPhenotype.SAMPLER, 0.5, {"politics": 0.2}, -0.2)
    v_short = Video(3, "politics", 0.6, 0.0, 2)

    t = watch_time_seconds(u, v_short, rng)
    assert 0 <= t <= 2


def test_negative_duration_duration_raises_value_error():
    rng = random.Random(0)
    u = User(UserPhenotype.WATCHER, 0.5, {"politics": 0.7}, -0.2)
    v = Video(4, "politics", 0.6, 0.0, -10)

    with pytest.raises(ValueError, match=r"^video\.duration_s must be >= 0$"):
        watch_time_seconds(u, v, rng)
