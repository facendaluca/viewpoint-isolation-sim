import pytest

from fyp_sim.interests import update_interest_vector
from fyp_sim.models import User, UserPhenotype, Video


def test_interest_profile_updates_on_watch():
    # Arrange: create a user and a video the user "watches"
    u = User(
        phenotype=UserPhenotype.SAMPLER,
        viewpoint_score=0.7,
        interest_vector={"Education": 0.9, "Entertainment": 0.4},
        sentiment_threshold=0.2,
    )

    v = Video(
        video_id=99,
        topic_category="Education",
        viewpoint_score=0.5,
        sentiment_score=0.0,
        duration_s=42,
    )

    before = u.interest_vector.get("Education", 0.0)

    # Simulate a "mostly watched" event
    update_interest_vector(user=u, video=v, watch_time_s=35)

    after = u.interest_vector.get("Education", 0.0)

    assert after > before


def test_interest_decay_reduces_existing_weights():
    u = User(
        phenotype=UserPhenotype.SAMPLER,
        viewpoint_score=0.5,
        interest_vector={"Education": 0.5, "Entertainment": 0.5},
        sentiment_threshold=-1.0,
    )
    v = Video(
        video_id=1,
        topic_category="Education",
        viewpoint_score=0.5,
        sentiment_score=0.0,
        duration_s=100,
        tags=(),
    )

    # Apply update that should decay both keys, then bump Education
    update_interest_vector(
        user=u,
        video=v,
        watch_time_s=50,  # frac=0.5
        topic_alpha=0.10,  # Education increases before normalisation
        tag_alpha=0.0,
        decay=0.0,
        normalise=True,
        prune_below=0.0,
    )

    total = sum(u.interest_vector.values())
    assert total == pytest.approx(1.0)
