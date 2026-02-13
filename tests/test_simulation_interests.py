import random

from fyp_sim.models import User, UserPhenotype, Video
from fyp_sim.simulation.engine import choose_video_max_interest, run_simulation


def test_simulation_updates_interest_profile():
    rng = random.Random(42)

    user = User(
        phenotype=UserPhenotype.WATCHER,  # should reliably WATCH high-interest content
        viewpoint_score=0.5,
        interest_vector={"Education": 0.9, "Entertainment": 0.0},
        sentiment_threshold=-1.0,  # don't block anything by sentiment
    )

    pool = [
        Video(
            video_id=1,
            topic_category="Education",
            viewpoint_score=0.5,
            sentiment_score=0.0,
            duration_s=40,
        ),
        Video(
            video_id=2,
            topic_category="Entertainment",
            viewpoint_score=0.5,
            sentiment_score=0.0,
            duration_s=40,
        ),
    ]

    before = user.interest_vector.get("Education", 0.0)

    logs = run_simulation(
        user=user,
        video_pool=pool,
        steps=10,
        rng=rng,
        chooser=choose_video_max_interest,
        top_k=2,
        alpha=0.0,
        enable_interest_updates=True,
    )

    after = user.interest_vector.get("Education", 0.0)

    assert any(row.action == "Watch" for row in logs)

    assert after > before
