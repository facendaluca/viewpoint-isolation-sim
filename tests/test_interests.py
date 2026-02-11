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
